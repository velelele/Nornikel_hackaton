from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from backend.config_manager import ConfigManager
from backend.knowledge_store import jsonl_append, utc_now_iso
from backend.theme_sharding import _READINESS_ORDER, ThemeShardManager, get_ingestion_profile
from scripts.progress_utils import TerminalProgress


def _theme_rank(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("collection") or ""), str(row.get("theme_name") or ""), str(row.get("theme_id") or ""))


def _apply_profile_runtime_overrides(config: object, profile: dict[str, Any]) -> dict[str, bool]:
    """Apply runtime-only profile overrides that are not represented in Stage 2 CLI args.

    This is intentionally narrow: it is used to disable experimental LightRAG
    extraction prompt/output repair for answerability-oriented compressed profiles
    without changing global config defaults.
    """
    applied: dict[str, bool] = {}
    if "lightrag_extraction_prompt_hardening" in profile:
        value = bool(profile.get("lightrag_extraction_prompt_hardening"))
        setattr(config, "lightrag_extraction_prompt_hardening", value)
        applied["lightrag_extraction_prompt_hardening"] = value
    if "lightrag_extraction_output_repair" in profile:
        value = bool(profile.get("lightrag_extraction_output_repair"))
        setattr(config, "lightrag_extraction_output_repair", value)
        applied["lightrag_extraction_output_repair"] = value
    return applied


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 for ◈NiCo: rebuild graph artifacts from existing topic-sharded knowledge_store. "
            "Default mode builds retrieval KG over knowledge_store without LightRAG. Use --graph-mode compressed_kg with --build-runtime-index only for explicit LightRAG."
        )
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--profile", default="overnight_retrieval_kg", help="Profile from config.yaml; default: overnight_retrieval_kg")
    parser.add_argument("--theme-id", action="append", default=[], help="Theme id to rebuild. Can be passed multiple times. Default: all themes")
    parser.add_argument("--max-themes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--clear-runtime", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rebuild even themes already marked full_kg_ready")
    parser.add_argument("--no-wait", action="store_true", help="Only enqueue LightRAG work and exit; not recommended for night build")
    parser.add_argument("--build-runtime-index", action="store_true", help="Force LightRAG runtime build. Use only for selected full_kg themes.")
    parser.add_argument("--poll-interval", type=float, default=None)
    parser.add_argument("--timeout-sec", type=float, default=None, help="Per-track timeout; 0 means no timeout")
    parser.add_argument("--graph-mode", choices=["vector_only", "cheap_kg", "retrieval_kg", "compressed_kg", "compressed_graph", "compressed_lightrag", "full_kg"], default=None)
    parser.add_argument("--max-chunks-per-document", type=int, default=None, help="For compressed_kg: selected chunks per document")
    parser.add_argument("--min-kg-score", type=float, default=None, help="For compressed_kg: minimum chunk KG score after first chunk per doc")
    parser.add_argument("--compressed-min-chunk-quality", type=float, default=None, help="For compressed_kg: suppress low-quality boilerplate chunks below this document-type-aware quality score")
    parser.add_argument("--compressed-runtime-doc-mode", choices=["chunk", "grouped", "source"], default=None, help="For compressed_kg runtime: how selected chunks are converted into LightRAG documents")
    parser.add_argument("--dry-run", action="store_true", help="Print selected themes and effective settings, but do not build Stage 2")
    parser.add_argument("--no-doc-type-aware", action="store_true", help="Disable document-type-aware filtering for compressed_kg")
    parser.add_argument("--max-chunks-per-theme", type=int, default=None, help="For compressed_kg: hard cap for selected chunks sent to LightRAG; 0/None means no cap")
    parser.add_argument("--max-candidate-chunks-per-theme", type=int, default=None, help="For compressed_kg: rough preselection cap before expensive scoring; default from profile, e.g. 1000")
    parser.add_argument("--max-parallel-themes", type=int, default=None, help="Number of Stage 2 theme builds to run concurrently; uses ThemeShardManager semaphore")
    parser.add_argument("--no-dynamic-document-limits", action="store_true", help="Disable short-doc=3 / long-doc=6 dynamic selected chunk limits")
    parser.add_argument("--compressed-short-document-max-chunks", type=int, default=None, help="Compressed KG selected chunks for short documents; default 3")
    parser.add_argument("--compressed-long-document-max-chunks", type=int, default=None, help="Compressed KG selected chunks for long documents; default 6")
    parser.add_argument("--skip-graph-metrics", action="store_true")
    parser.add_argument("--skip-theme-embeddings", action="store_true")
    parser.add_argument("--no-router", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress bar")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = ConfigManager(args.config).load()
    profile = get_ingestion_profile(config, args.profile)
    runtime_overrides = _apply_profile_runtime_overrides(config, profile)
    manager = ThemeShardManager(ROOT, config)

    batch_size = int(args.batch_size or profile.get("batch_size_docs") or config.topic_batch_size_docs)
    requested_graph_mode = str(args.graph_mode or profile.get("graph_mode") or "retrieval_kg").strip().lower()
    compressed_lightrag_alias = requested_graph_mode in {"compressed_graph", "compressed_lightrag", "compressed_lightrag_kg", "compressed_kg_lightrag"}
    graph_mode = "compressed_kg" if compressed_lightrag_alias else requested_graph_mode
    build_runtime_index = bool(args.build_runtime_index or (True if compressed_lightrag_alias else profile.get("build_runtime_index", False)))
    wait = bool(build_runtime_index and not args.no_wait and profile.get("wait", True))
    poll_interval = float(args.poll_interval if args.poll_interval is not None else profile.get("poll_interval", 10.0))
    timeout_sec = float(args.timeout_sec if args.timeout_sec is not None else profile.get("timeout_sec", 0.0))
    clear_runtime = bool(args.clear_runtime or profile.get("clear_runtime", False))
    if graph_mode == "compressed_kg" and not build_runtime_index and not compressed_lightrag_alias:
        graph_mode = "retrieval_kg"
    if graph_mode in {"cheap_kg", "vector_only", "retrieval_kg"}:
        build_runtime_index = False
    max_chunks_per_document = int(args.max_chunks_per_document or profile.get("max_chunks_per_document_for_graph") or 8)
    min_kg_score = float(args.min_kg_score if args.min_kg_score is not None else profile.get("min_kg_score", 0.15))
    compressed_min_chunk_quality = float(args.compressed_min_chunk_quality if args.compressed_min_chunk_quality is not None else profile.get("compressed_min_chunk_quality", 0.20))
    compressed_doc_type_aware = bool(False if args.no_doc_type_aware else profile.get("compressed_doc_type_aware", True))
    compressed_runtime_doc_mode = str(args.compressed_runtime_doc_mode or profile.get("compressed_runtime_doc_mode") or profile.get("compressed_lightrag_doc_mode") or "chunk").strip().lower()
    if compressed_runtime_doc_mode == "source":
        compressed_runtime_doc_mode = "grouped"
    max_chunks_per_theme = args.max_chunks_per_theme if args.max_chunks_per_theme is not None else profile.get("max_chunks_per_theme", 0)
    max_chunks_per_theme = None if not max_chunks_per_theme or int(max_chunks_per_theme) <= 0 else int(max_chunks_per_theme)
    max_candidate_chunks_per_theme = args.max_candidate_chunks_per_theme if args.max_candidate_chunks_per_theme is not None else profile.get("max_candidate_chunks_per_theme", 1000)
    max_candidate_chunks_per_theme = None if not max_candidate_chunks_per_theme or int(max_candidate_chunks_per_theme) <= 0 else int(max_candidate_chunks_per_theme)
    max_parallel_themes = int(args.max_parallel_themes or profile.get("max_parallel_themes") or config.topic_max_parallel_themes or 1)
    compressed_dynamic_document_limits = bool(False if args.no_dynamic_document_limits else profile.get("compressed_dynamic_document_limits", True))
    compressed_short_document_max_chunks = int(args.compressed_short_document_max_chunks or profile.get("compressed_short_document_max_chunks") or 3)
    compressed_long_document_max_chunks = int(args.compressed_long_document_max_chunks or profile.get("compressed_long_document_max_chunks") or 6)
    target_status = str(profile.get("target_status") or ({"cheap_kg": "cheap_kg_ready", "compressed_kg": "compressed_kg_ready", "retrieval_kg": "retrieval_kg_ready", "full_kg": "full_kg_ready"}.get(graph_mode, "search_ready")))

    rows = sorted(manager.list_themes(), key=_theme_rank)
    selected_ids = set(args.theme_id or [])
    if selected_ids:
        rows = [row for row in rows if str(row.get("theme_id") or "") in selected_ids]
    elif not args.force:
        target_rank = _READINESS_ORDER.get(target_status, _READINESS_ORDER["compressed_kg_ready"])
        rows = [row for row in rows if _READINESS_ORDER.get(str(row.get("status") or "not_ready"), 0) < target_rank]
    if args.max_themes is not None:
        rows = rows[: max(0, args.max_themes)]

    effective = {
        "profile": args.profile,
        "graph_mode": graph_mode,
        "build_runtime_index": build_runtime_index,
        "wait": wait,
        "clear_runtime": clear_runtime,
        "target_status": target_status,
        "batch_size": batch_size,
        "poll_interval": poll_interval,
        "timeout_sec": timeout_sec,
        "max_chunks_per_document": max_chunks_per_document,
        "min_kg_score": min_kg_score,
        "max_chunks_per_theme": max_chunks_per_theme,
        "max_candidate_chunks_per_theme": max_candidate_chunks_per_theme,
        "max_parallel_themes": max_parallel_themes,
        "compressed_runtime_doc_mode": compressed_runtime_doc_mode,
        "compressed_doc_type_aware": compressed_doc_type_aware,
        "compressed_min_chunk_quality": compressed_min_chunk_quality,
        "compressed_dynamic_document_limits": compressed_dynamic_document_limits,
        "compressed_short_document_max_chunks": compressed_short_document_max_chunks,
        "compressed_long_document_max_chunks": compressed_long_document_max_chunks,
        "runtime_overrides": runtime_overrides,
        "selected_theme_count": len(rows),
        "selected_themes": [str(row.get("theme_id")) for row in rows],
    }
    print("[stage2] effective settings:")
    print(json.dumps(effective, ensure_ascii=False, indent=2))
    if args.dry_run:
        await manager.shutdown()
        return

    run_id = "stage2-full-build:" + utc_now_iso()
    results: list[dict[str, Any]] = []
    ok = failed = 0
    done_count = 0
    progress = TerminalProgress("Stage 2 themes", total=len(rows), enabled=not args.no_progress)
    progress.update(0, f"selected {len(rows)} themes; parallel={max_parallel_themes}; mode={graph_mode}", force=True)
    try:
        if not rows:
            progress.log("No themes selected for stage2.")
        theme_ids = [str(row.get("theme_id") or "") for row in rows if row.get("theme_id")]
        row_by_theme = {str(row.get("theme_id") or ""): row for row in rows if row.get("theme_id")}

        async def on_start(index: int, theme_id: str) -> None:
            row = row_by_theme.get(theme_id, {})
            progress.log(f"[{index + 1}/{len(theme_ids)}] stage2 theme={theme_id} status={row.get('status')} graph_mode={graph_mode} batch_size={batch_size} build_runtime_index={build_runtime_index} wait={wait}")
            progress.update(done_count, f"running {theme_id}", force=True)

        async def on_done(index: int, theme_id: str, result: dict[str, Any]) -> None:
            nonlocal ok, failed, results, done_count
            if result.get("ok"):
                ok += 1
                short = {
                    "theme_id": theme_id,
                    "status": result.get("status"),
                    "chunks": result.get("chunks"),
                    "graph_mode": result.get("graph_mode") or graph_mode,
                    "track_ids": result.get("track_ids"),
                    "processed_count": result.get("processed_count"),
                    "failed_count": result.get("failed_count"),
                    "timeout_count": result.get("timeout_count"),
                    "compressed_plan": result.get("compressed_plan"),
                }
                progress.log(json.dumps(short, ensure_ascii=False, indent=2))
                results.append({"theme_id": theme_id, "ok": True, **short})
            else:
                failed += 1
                progress.log(f"    ERROR theme={theme_id}: {result.get('error')}")
                results.append({"theme_id": theme_id, "ok": False, "error": result.get("error")})
            done_count += 1
            progress.update(done_count, f"done {theme_id}; ok={ok}; failed={failed}", force=True)

        await manager.rebuild_themes_runtime(
            theme_ids,
            max_parallel_themes=max_parallel_themes,
            on_theme_start=on_start,
            on_theme_done=on_done,
            clear_runtime=clear_runtime,
            batch_size=batch_size,
            wait=wait,
            poll_interval=poll_interval,
            timeout_sec=timeout_sec,
            target_status=target_status,
            graph_mode=graph_mode,
            max_chunks_per_document_for_graph=max_chunks_per_document,
            min_kg_score=min_kg_score,
            max_chunks_per_theme=max_chunks_per_theme,
            max_candidate_chunks_per_theme=max_candidate_chunks_per_theme,
            compressed_runtime_doc_mode=compressed_runtime_doc_mode,
            compressed_min_chunk_quality=compressed_min_chunk_quality,
            compressed_doc_type_aware=compressed_doc_type_aware,
            compressed_dynamic_document_limits=compressed_dynamic_document_limits,
            compressed_short_document_max_chunks=compressed_short_document_max_chunks,
            compressed_long_document_max_chunks=compressed_long_document_max_chunks,
            build_runtime_index=build_runtime_index,
            compute_graph_metrics=False,
        )
    finally:
        if not args.skip_graph_metrics and bool(profile.get("rebuild_graph_metrics_after", True)):
            progress.log("[post] rebuilding graph metrics...")
            try:
                manager.compute_theme_graph_metrics(theme_ids=[str(row.get("theme_id")) for row in rows if row.get("theme_id")])
            except Exception as exc:
                progress.log(f"[warn] graph metrics failed: {exc}")

        if not args.skip_theme_embeddings and bool(profile.get("rebuild_theme_embeddings_after", True)):
            progress.log("[post] rebuilding theme embeddings...")
            try:
                await manager.rebuild_theme_embeddings([str(row.get("theme_id")) for row in rows if row.get("theme_id")])
            except Exception as exc:
                progress.log(f"[warn] theme embeddings failed: {exc}")

        if not args.no_router and bool(profile.get("build_global_router_after", True)):
            progress.log("[post] rebuilding global router...")
            try:
                manager.catalog.build_router()
            except Exception as exc:
                progress.log(f"[warn] global router failed: {exc}")

        jsonl_append(
            manager.knowledge_root / "global" / "two_stage_runs.jsonl",
            {
                "run_id": run_id,
                "stage": "stage2_build",
                "profile": args.profile,
                "selected_themes": [str(row.get("theme_id")) for row in rows],
                "batch_size": batch_size,
                "graph_mode": graph_mode,
                "max_chunks_per_document": max_chunks_per_document,
                "min_kg_score": min_kg_score,
                "compressed_runtime_doc_mode": compressed_runtime_doc_mode,
                "compressed_min_chunk_quality": compressed_min_chunk_quality,
                "compressed_doc_type_aware": compressed_doc_type_aware,
                "runtime_overrides": runtime_overrides,
                "max_chunks_per_theme": max_chunks_per_theme,
                "wait": wait,
                "clear_runtime": clear_runtime,
                "build_runtime_index": build_runtime_index,
                "ok": ok,
                "failed": failed,
                "results": results,
                "created_at": utc_now_iso(),
            },
        )
        progress.finish(f"done ok={ok} failed={failed}")
        await manager.shutdown()

    print(f"Done stage2: ok={ok}, failed={failed}, wait={wait}")


if __name__ == "__main__":
    asyncio.run(main())
