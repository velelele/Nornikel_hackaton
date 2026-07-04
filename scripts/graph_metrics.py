from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config_manager import ConfigManager, resolve_project_path
from backend.graph_embedding_intelligence import write_graph_metrics
from backend.knowledge_store import KnowledgeStore, jsonl_read
from backend.theme_sharding import ThemeCatalog, ThemeInfo


def _theme_dirs(knowledge_root: Path, theme_id: str | None) -> list[Path]:
    themes_root = knowledge_root / "themes"
    if not themes_root.exists():
        return []
    if theme_id:
        # Theme dir is slugified theme_id in the current implementation.
        direct = themes_root / theme_id
        if direct.exists():
            return [direct]
        matches = []
        for path in themes_root.iterdir():
            if not path.is_dir():
                continue
            store = KnowledgeStore(path)
            first = next(iter(jsonl_read(store.sources_path)), {})
            if str(first.get("theme_id") or path.name) == theme_id:
                matches.append(path)
        return matches
    return sorted(path for path in themes_root.iterdir() if path.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute graph metrics for NiCo theme stores.")
    parser.add_argument("--theme-id", default="", help="Optional exact theme_id. If omitted, all themes are processed.")
    parser.add_argument("--top-n", type=int, default=None)
    args = parser.parse_args()

    config = ConfigManager(ROOT / "config.yaml").load()
    knowledge_root = resolve_project_path(ROOT, config.knowledge_store_dir, "./data/knowledge_store")
    dirs = _theme_dirs(knowledge_root, args.theme_id or None)
    if not dirs:
        print(json.dumps({"updated": 0, "reason": "no theme dirs found"}, ensure_ascii=False, indent=2))
        return 0
    results = {}
    catalog = ThemeCatalog(knowledge_root / "global")
    existing_status = {str(row.get("theme_id")): str(row.get("status") or "parsed_ready") for row in catalog.list_themes()}
    for theme_dir in dirs:
        store = KnowledgeStore(theme_dir, schema_version=config.schema_version, ontology_version=config.ontology_version, app_version=config.app_version)
        first = next(iter(jsonl_read(store.sources_path)), {})
        theme_id = str(first.get("theme_id") or theme_dir.name)
        results[theme_id] = write_graph_metrics(
            store,
            top_n=args.top_n or config.graph_metrics_top_n,
            pagerank_iterations=config.graph_pagerank_iterations,
            betweenness_sample=config.graph_betweenness_sample,
        )
        stats = store.stats()
        entities = jsonl_read(store.entities_path)
        metrics = results[theme_id]
        top_terms = [str(row.get("label") or row.get("term_id")) for row in entities[:60] if row.get("label") or row.get("term_id")]
        stats.update({
            "top_terms": list(dict.fromkeys([*top_terms, *[str(row.get("label") or row.get("node_id")) for row in metrics.get("top_pagerank", [])[:20]]])),
            "top_entities": [str(row.get("label") or row.get("node_id")) for row in metrics.get("top_pagerank", [])[:40]],
            "top_processes": [],
            "graph_metrics": metrics,
        })
        theme = ThemeInfo(
            theme_id=theme_id,
            collection=str(first.get("collection") or "unknown"),
            theme_name=str(first.get("theme_name") or theme_id),
            relative_path="",
            source_name="",
            year=int(first.get("year")) if str(first.get("year") or "").isdigit() else None,
            source_type=str(first.get("source_type") or "unknown"),
            confidence=float(first.get("confidence") or 0.0),
            evidence=list(first.get("evidence") or []),
        )
        catalog.upsert_theme(theme, stats, status=existing_status.get(theme_id, "parsed_ready"))
    print(json.dumps({"updated": len(results), "themes": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
