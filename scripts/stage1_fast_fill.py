from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from backend.config_manager import ConfigManager
from backend.domain.document_processor import process_document
from backend.domain.numeric_extractor import get_numeric_extractor
from backend.knowledge_store import jsonl_append, jsonl_read, utc_now_iso
from backend.theme_sharding import ThemeShardManager, get_ingestion_profile, iter_supported_documents
from scripts.progress_utils import TerminalProgress


SUPPORTED_REPORT_FIELDS = [
    "idx",
    "status",
    "file_path",
    "relative_path",
    "theme_id",
    "collection",
    "theme_name",
    "year",
    "source_type",
    "confidence",
    "chunks",
    "numeric_facts",
    "error",
]


def _bool_profile(profile: dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(profile.get(key, default))


def _collect_existing_original_paths(manager: ThemeShardManager) -> set[str]:
    existing: set[str] = set()
    root = manager.themes_store_root
    if not root.exists():
        return existing
    for theme_dir in root.iterdir():
        if not theme_dir.is_dir():
            continue
        for row in jsonl_read(theme_dir / "sources.jsonl"):
            original_path = str(row.get("original_path") or "").replace("\\", "/")
            if original_path:
                existing.add(original_path)
    return existing


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 for ◈NiCo: fast recursive fill of durable knowledge_store. "
            "By default it skips LightRAG runtime indexing so large corpora do not overload Ollama."
        )
    )
    parser.add_argument("root", type=Path, help="Root directory with arbitrary nested document tree")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--profile", default="fast_fill", help="Ingestion profile from config.yaml; default: fast_fill")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Skip files whose original_path is already present in any theme store")
    parser.add_argument("--runtime-index", action="store_true", help="Also enqueue chunks into LightRAG runtime index; slower and not recommended for huge first fill")
    parser.add_argument("--graph-metrics", action="store_true", help="Rebuild graph metrics after stage 1")
    parser.add_argument("--theme-embeddings", action="store_true", help="Build theme profile embeddings after stage 1; requires Ollama embeddings")
    parser.add_argument("--no-router", action="store_true", help="Do not rebuild global router after stage 1")
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "knowledge_store" / "global" / "stage1_fast_fill_report.csv")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress bar")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = ConfigManager(args.config).load()
    profile = get_ingestion_profile(config, args.profile)
    manager = ThemeShardManager(ROOT, config)
    extractor = get_numeric_extractor()

    build_runtime_index = bool(args.runtime_index or profile.get("build_runtime_index", False))
    compute_graph_during = bool(profile.get("compute_graph_metrics_during_ingest", False))
    status_if_no_runtime = str(profile.get("status") or "parsed_ready")
    existing_paths = _collect_existing_original_paths(manager) if args.skip_existing else set()
    docs = list(iter_supported_documents(args.root.resolve(), config, limit=args.limit))
    progress = TerminalProgress("Stage 1 documents", total=len(docs), enabled=not args.no_progress)
    progress.update(0, f"selected {len(docs)} supported documents", force=True)
    report_rows: list[dict[str, Any]] = []

    run_id = "stage1-fast-fill:" + utc_now_iso()
    ok = skipped = failed = 0
    try:
        for idx, (path, rel) in enumerate(docs, start=1):
            progress.update(idx - 1, f"processing {rel}")
            if rel in existing_paths:
                skipped += 1
                progress.log(f"[{idx}/{len(docs)}] SKIP existing: {rel}")
                report_rows.append({"idx": idx, "status": "skipped", "file_path": str(path), "relative_path": rel})
                progress.update(idx, f"skipped {rel}", force=True)
                continue

            quick_theme = manager.resolve_theme(rel)
            progress.log(f"[{idx}/{len(docs)}] {rel}")
            progress.log(f"    pre-resolve: theme={quick_theme.theme_id} confidence={quick_theme.confidence} evidence={','.join(quick_theme.evidence)}")
            try:
                processed = await asyncio.to_thread(process_document, path, original_name=rel)
                facts = []
                for obj in processed.objects:
                    facts.extend(extractor.extract(
                        obj.text,
                        source_name=processed.source_name,
                        object_id=obj.object_id,
                        object_type=obj.object_type,
                        metadata={**obj.metadata, "source_hash": processed.source_hash},
                    ))
                result = await manager.ingest_processed_document(
                    processed,
                    facts,
                    original_path=rel,
                    build_runtime_index=build_runtime_index,
                    compute_graph_metrics=compute_graph_during,
                    status_if_no_runtime=status_if_no_runtime,
                )
                theme = result["theme"]
                ok += 1
                progress.log(
                    "    -> "
                    f"stage1 theme={theme['theme_id']} collection={theme.get('collection')} "
                    f"year={theme.get('year')} runtime_index={result.get('runtime_index_requested')} "
                    f"chunks={len(processed.objects)} facts={len(facts)} track={result.get('track_id') or '-'}"
                )
                progress.update(idx, f"ok {theme.get('theme_id')} chunks={len(processed.objects)} facts={len(facts)}", force=True)
                report_rows.append({
                    "idx": idx,
                    "status": "ok",
                    "file_path": str(path),
                    "relative_path": rel,
                    "theme_id": theme.get("theme_id"),
                    "collection": theme.get("collection"),
                    "theme_name": theme.get("theme_name"),
                    "year": theme.get("year"),
                    "source_type": theme.get("source_type"),
                    "confidence": theme.get("confidence"),
                    "chunks": len(processed.objects),
                    "numeric_facts": len(facts),
                    "error": "",
                })
            except Exception as exc:
                failed += 1
                progress.log(f"    ERROR: {exc}")
                progress.update(idx, f"failed {rel}", force=True)
                report_rows.append({"idx": idx, "status": "failed", "file_path": str(path), "relative_path": rel, "error": str(exc)})
    finally:
        if args.graph_metrics or _bool_profile(profile, "rebuild_graph_metrics_after", False):
            progress.log("[post] rebuilding graph metrics...")
            try:
                manager.compute_theme_graph_metrics()
            except Exception as exc:
                progress.log(f"[warn] graph metrics failed: {exc}")

        if args.theme_embeddings or _bool_profile(profile, "rebuild_theme_embeddings_after", False):
            progress.log("[post] rebuilding theme embeddings...")
            try:
                await manager.rebuild_theme_embeddings()
            except Exception as exc:
                progress.log(f"[warn] theme embeddings failed: {exc}")

        if not args.no_router and _bool_profile(profile, "build_global_router_after", True):
            progress.log("[post] rebuilding global router...")
            try:
                manager.catalog.build_router()
            except Exception as exc:
                progress.log(f"[warn] global router failed: {exc}")

        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=SUPPORTED_REPORT_FIELDS)
                writer.writeheader()
                writer.writerows(report_rows)
            progress.log(f"[report] {args.report}")

        jsonl_append(
            manager.knowledge_root / "global" / "two_stage_runs.jsonl",
            {
                "run_id": run_id,
                "stage": "stage1_fast_fill",
                "profile": args.profile,
                "root": str(args.root),
                "runtime_index": build_runtime_index,
                "ok": ok,
                "skipped": skipped,
                "failed": failed,
                "report": str(args.report),
                "created_at": utc_now_iso(),
            },
        )
        progress.finish(f"done ok={ok} skipped={skipped} failed={failed}")
        await manager.shutdown()

    print(f"Done stage1: ok={ok}, skipped={skipped}, failed={failed}, runtime_index={build_runtime_index}")


if __name__ == "__main__":
    asyncio.run(main())
