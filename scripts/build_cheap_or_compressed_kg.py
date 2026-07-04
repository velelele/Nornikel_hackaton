from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config_manager import ConfigManager
from backend.compressed_kg import build_cheap_kg, build_compressed_kg_plan
from backend.knowledge_store import KnowledgeStore
from backend.theme_sharding import ThemeShardManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic cheap/compressed KG artifacts without LightRAG/Ollama.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--theme-id", action="append", default=[])
    parser.add_argument("--mode", choices=["cheap_kg", "compressed_kg"], default="compressed_kg")
    parser.add_argument("--max-chunks-per-document", type=int, default=8)
    parser.add_argument("--min-kg-score", type=float, default=0.15)
    parser.add_argument("--compressed-min-chunk-quality", type=float, default=0.20)
    parser.add_argument("--no-doc-type-aware", action="store_true")
    parser.add_argument("--max-chunks-per-theme", type=int, default=0)
    parser.add_argument("--max-candidate-chunks-per-theme", type=int, default=1000)
    parser.add_argument("--no-dynamic-document-limits", action="store_true")
    parser.add_argument("--compressed-short-document-max-chunks", type=int, default=3)
    parser.add_argument("--compressed-long-document-max-chunks", type=int, default=6)
    args = parser.parse_args()

    config = ConfigManager(args.config).load()
    manager = ThemeShardManager(ROOT, config)
    rows = manager.list_themes()
    selected = set(args.theme_id or [])
    if selected:
        rows = [row for row in rows if str(row.get("theme_id") or "") in selected]

    results = {}
    for row in rows:
        theme_id = str(row.get("theme_id") or "")
        if not theme_id:
            continue
        store = manager.theme_store(theme_id)
        if args.mode == "cheap_kg":
            payload = build_cheap_kg(store)
            manager.mark_theme_status(theme_id, "cheap_kg_ready")
            results[theme_id] = {"status": "cheap_kg_ready", "counts": payload.get("counts")}
        else:
            max_theme = None if args.max_chunks_per_theme <= 0 else args.max_chunks_per_theme
            payload = build_compressed_kg_plan(
                store,
                max_chunks_per_document=args.max_chunks_per_document,
                min_kg_score=args.min_kg_score,
                max_chunks_per_theme=max_theme,
                max_candidate_chunks_per_theme=None if args.max_candidate_chunks_per_theme <= 0 else args.max_candidate_chunks_per_theme,
                min_chunk_quality=args.compressed_min_chunk_quality,
                doc_type_aware=not args.no_doc_type_aware,
                dynamic_document_limits=not args.no_dynamic_document_limits,
                short_document_max_chunks=args.compressed_short_document_max_chunks,
                long_document_max_chunks=args.compressed_long_document_max_chunks,
            )
            manager.mark_theme_status(theme_id, "cheap_kg_ready")
            results[theme_id] = {k: v for k, v in payload.items() if k != "rows"}
    print(json.dumps({"mode": args.mode, "themes": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
