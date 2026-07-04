from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from backend.config_manager import ConfigManager
from backend.domain.document_processor import process_document
from backend.domain.numeric_extractor import get_numeric_extractor
from backend.theme_sharding import ThemeShardManager, get_ingestion_profile, iter_supported_documents


async def main() -> None:
    parser = argparse.ArgumentParser(description="Recursively ingest root_dir into topic-sharded ◈NiCo indexes.")
    parser.add_argument("root", type=Path, help="Root directory with arbitrary nested structure")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--profile", default="balanced", help="balanced/safe/fast_smoke/fast_fill")
    parser.add_argument("--no-runtime-index", action="store_true", help="Skip LightRAG runtime indexing and only fill durable knowledge_store")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = ConfigManager(args.config).load()
    profile = get_ingestion_profile(config, args.profile)
    manager = ThemeShardManager(ROOT, config)
    extractor = get_numeric_extractor()

    ok = 0
    failed = 0
    try:
        for idx, (path, rel) in enumerate(iter_supported_documents(args.root.resolve(), config, limit=args.limit), start=1):
            quick_theme = manager.resolve_theme(rel)
            print(f"[{idx}] {rel}")
            print(f"    pre-resolve: theme={quick_theme.theme_id} confidence={quick_theme.confidence} evidence={','.join(quick_theme.evidence)}")
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
                    build_runtime_index=bool(profile.get("build_runtime_index", True)) and not args.no_runtime_index,
                    compute_graph_metrics=bool(profile.get("compute_graph_metrics_during_ingest", config.graph_metrics_enabled)),
                    status_if_no_runtime=str(profile.get("status") or "parsed_ready"),
                )
                theme = result["theme"]
                print(
                    "    -> "
                    f"theme={theme['theme_id']} collection={theme.get('collection')} "
                    f"year={theme.get('year')} source_type={theme.get('source_type')} "
                    f"confidence={theme.get('confidence')} runtime_index={result.get('runtime_index_requested')} chunks={len(processed.objects)} facts={len(facts)} "
                    f"track={result.get('track_id')}"
                )
                ok += 1
            except Exception as exc:
                failed += 1
                print(f"    ERROR: {exc}")
    finally:
        await manager.shutdown()

    print(f"Done: ok={ok}, failed={failed}")


if __name__ == "__main__":
    asyncio.run(main())
