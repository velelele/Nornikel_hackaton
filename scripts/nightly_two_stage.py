from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config_manager import ConfigManager, AppConfig
from backend.theme_sharding import get_ingestion_profile, iter_supported_documents
from scripts.progress_utils import TerminalProgress


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(cmd: list[str], *, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(str(x) for x in cmd))
    print(f"[log] {log_path}")
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(str(x) for x in cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return proc.wait()


def _extend(cmd: list[str], values: Iterable[str], flag: str) -> None:
    for value in values:
        if value:
            cmd.extend([flag, value])


def _count_documents(root: Path, config: AppConfig, *, limit: int | None = None) -> int:
    count = 0
    for _path, _rel in iter_supported_documents(root, config, limit=limit):
        count += 1
    return count


def _check_model_endpoint(base_url: str, *, timeout: float = 4.0) -> tuple[bool, str]:
    base = base_url.rstrip("/") + "/"
    url = urljoin(base, "models")
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local operator endpoint check
            status = getattr(resp, "status", 200)
            if 200 <= int(status) < 300:
                return True, f"{url} OK"
            return False, f"{url} returned HTTP {status}"
    except URLError as exc:
        return False, f"{url} unavailable: {exc}"
    except Exception as exc:
        return False, f"{url} check failed: {exc}"


def _preflight(
    *,
    root: Path,
    config_path: Path,
    stage1_profile: str,
    stage2_profile: str,
    stage1_limit: int | None,
    retrieval_only: bool,
    skip_model_check: bool,
    strict_model_check: bool,
) -> tuple[AppConfig, dict[str, object]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not config_path.exists():
        errors.append(f"Config not found: {config_path}")
    if not root.exists():
        errors.append(f"Document root not found: {root}")
    for script in (ROOT / "scripts" / "stage1_fast_fill.py", ROOT / "scripts" / "stage2_full_build.py"):
        if not script.exists():
            errors.append(f"Required script not found: {script}")
    if errors:
        raise SystemExit("\n".join(errors))

    config = ConfigManager(config_path).load()
    try:
        stage1 = get_ingestion_profile(config, stage1_profile)
    except Exception as exc:
        errors.append(f"Stage 1 profile not found or invalid: {stage1_profile}: {exc}")
        stage1 = {}
    try:
        stage2 = get_ingestion_profile(config, stage2_profile)
    except Exception as exc:
        errors.append(f"Stage 2 profile not found or invalid: {stage2_profile}: {exc}")
        stage2 = {}

    doc_count = _count_documents(root, config, limit=stage1_limit)
    if doc_count <= 0:
        warnings.append("No supported documents were found under the selected root.")

    model_status = "skipped"
    if not skip_model_check and not retrieval_only:
        llm_ok, llm_msg = _check_model_endpoint(config.llm_base_url)
        emb_ok, emb_msg = _check_model_endpoint(config.embedding_base_url)
        model_status = f"llm={llm_msg}; embedding={emb_msg}"
        if not llm_ok or not emb_ok:
            msg = "Model endpoint preflight failed: " + model_status
            if strict_model_check:
                errors.append(msg)
            else:
                warnings.append(msg)

    graph_mode = str(stage2.get("graph_mode") or "").strip().lower()
    compressed_graph = graph_mode in {"compressed_graph", "compressed_kg", "compressed_lightrag"} and bool(stage2.get("build_runtime_index", False))
    if compressed_graph:
        if str(stage2.get("compressed_runtime_doc_mode") or "chunk") != "chunk":
            warnings.append("compressed_runtime_doc_mode is not 'chunk'; previous answerability was best with one selected chunk per LightRAG document.")
        if not bool(getattr(config, "lightrag_extraction_prompt_hardening", True)):
            warnings.append("rag.extraction_prompt_hardening=false; local LLMs may produce Markdown/pipe output that LightRAG parser drops.")
        if not bool(getattr(config, "lightrag_extraction_output_repair", True)):
            warnings.append("rag.extraction_output_repair=false; malformed entity/relation rows will be dropped and fallback will be more frequent.")
        if not bool(stage2.get("compressed_doc_type_aware", True)):
            warnings.append("compressed_doc_type_aware is disabled; presentation/article boilerplate may slow LightRAG extraction.")
        if float(stage2.get("compressed_min_chunk_quality", 0.0) or 0.0) > 0.35:
            warnings.append("compressed_min_chunk_quality is high; useful short technical chunks can be filtered out.")
        if not bool(stage2.get("clear_runtime", False)):
            warnings.append("clear_runtime=false; stale LightRAG runtime may hijack answers.")
        if int(stage2.get("max_parallel_themes") or config.topic_max_parallel_themes or 1) < 2:
            warnings.append("max_parallel_themes < 2; 24GB nightly build will not use parallel theme processing.")
        if int(stage2.get("max_candidate_chunks_per_theme") or 0) <= 0:
            warnings.append("max_candidate_chunks_per_theme is disabled; very large themes can create slow compressed-graph selection.")
        if int(stage2.get("max_chunks_per_document_for_graph") or 0) > 5 and bool(stage2.get("compressed_dynamic_document_limits", True)):
            warnings.append("max_chunks_per_document_for_graph > 5; dynamic limits will still reduce short docs but medium docs may be slower.")

    report = {
        "root": str(root),
        "config": str(config_path),
        "stage1_profile": stage1_profile,
        "stage2_profile": stage2_profile,
        "documents_found": doc_count,
        "stage1": stage1,
        "stage2": stage2,
        "model_status": model_status,
        "warnings": warnings,
        "errors": errors,
    }
    print("[preflight]")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if errors:
        raise SystemExit("\n".join(errors))
    return config, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nightly two-stage ◈NiCo ingestion: Stage 1 durable fill, then Stage 2 compressed graph build."
    )
    parser.add_argument("root", type=Path, help="Root directory with source documents")
    parser.add_argument("--config", type=Path, default=ROOT / "config.nightly.24gb.yaml")
    parser.add_argument("--stage1-profile", default="nightly_stage1_24gb")
    parser.add_argument("--stage2-profile", default="nightly_compressed_answerable_24gb")
    parser.add_argument("--stage1-limit", type=int, default=None)
    parser.add_argument("--stage2-max-themes", type=int, default=None)
    parser.add_argument("--theme-id", action="append", default=[], help="Restrict Stage 2 to a theme id; can be repeated")
    parser.add_argument("--force-stage2", action="store_true", help="Rebuild Stage 2 even if themes already have target status")
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-ingest existing Stage 1 sources instead of skipping them")
    parser.add_argument("--retrieval-only", action="store_true", help="Use deterministic retrieval_kg Stage 2, no LightRAG runtime")
    parser.add_argument("--full-lightrag", action="store_true", help="Use full LightRAG Stage 2 profile; recommended only with --theme-id")
    parser.add_argument("--dry-run", action="store_true", help="Run preflight and print Stage 1/Stage 2 commands without executing them")
    parser.add_argument("--skip-model-check", action="store_true", help="Skip /models check for local OpenAI-compatible LLM endpoints")
    parser.add_argument("--strict-model-check", action="store_true", help="Fail preflight if model endpoints are unavailable")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "data" / "nightly_runs")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars in child Stage 1/Stage 2 scripts")
    args = parser.parse_args()

    config_path = args.config.resolve()
    root = args.root.resolve()

    stage2_profile = args.stage2_profile
    if args.retrieval_only:
        stage2_profile = "nightly_retrieval_kg_24gb"
    if args.full_lightrag:
        stage2_profile = "nightly_full_lightrag_24gb"

    config, _report = _preflight(
        root=root,
        config_path=config_path,
        stage1_profile=args.stage1_profile,
        stage2_profile=stage2_profile,
        stage1_limit=args.stage1_limit,
        retrieval_only=args.retrieval_only,
        skip_model_check=args.skip_model_check,
        strict_model_check=args.strict_model_check,
    )

    run_dir = args.log_dir / _timestamp()
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    py = sys.executable
    stage1_cmd = [
        py,
        str(ROOT / "scripts" / "stage1_fast_fill.py"),
        str(root),
        "--config",
        str(config_path),
        "--profile",
        args.stage1_profile,
        "--report",
        str(run_dir / "stage1_report.csv"),
    ]
    if args.stage1_limit is not None:
        stage1_cmd.extend(["--limit", str(args.stage1_limit)])
    if not args.no_skip_existing:
        stage1_cmd.append("--skip-existing")
    if args.no_progress:
        stage1_cmd.append("--no-progress")

    stage2_cmd = [
        py,
        str(ROOT / "scripts" / "stage2_full_build.py"),
        "--config",
        str(config_path),
        "--profile",
        stage2_profile,
    ]
    if args.force_stage2:
        stage2_cmd.append("--force")
    if args.stage2_max_themes is not None:
        stage2_cmd.extend(["--max-themes", str(args.stage2_max_themes)])
    _extend(stage2_cmd, args.theme_id, "--theme-id")
    if args.no_progress:
        stage2_cmd.append("--no-progress")

    print("[nightly] effective commands:")
    print("stage1:", " ".join(str(x) for x in stage1_cmd))
    print("stage2:", " ".join(str(x) for x in stage2_cmd))
    if args.dry_run:
        print("[dry-run] Commands were not executed.")
        return 0

    nightly_progress = TerminalProgress("Nightly stages", total=2, enabled=not args.no_progress)
    nightly_progress.update(0, "Stage 1 starting", force=True)
    rc = _run(stage1_cmd, log_path=run_dir / "stage1.log", env=env)
    if rc != 0:
        nightly_progress.update(1, "Stage 1 failed", force=True)
        print(f"Stage 1 failed with exit code {rc}. Stage 2 will not be started.")
        return rc

    nightly_progress.update(1, "Stage 1 done; Stage 2 starting", force=True)
    rc = _run(stage2_cmd, log_path=run_dir / "stage2.log", env=env)
    if rc != 0:
        nightly_progress.update(2, "Stage 2 failed", force=True)
        print(f"Stage 2 failed with exit code {rc}.")
        return rc

    summary = {
        "created_at": _timestamp(),
        "root": str(root),
        "config": str(config_path),
        "stage1_profile": args.stage1_profile,
        "stage2_profile": stage2_profile,
        "logs": str(run_dir),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "nightly_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    nightly_progress.finish("Stage 1 + Stage 2 done")
    print(f"Nightly two-stage run finished. Logs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
