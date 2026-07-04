from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config_manager import ConfigManager, resolve_project_path
from backend.graph_embedding_intelligence import embed_texts_openai_compatible, route_theme_scores
from backend.theme_sharding import ThemeCatalog, _READINESS_ORDER


async def _run(message: str, *, top_k: int, no_embeddings: bool) -> dict:
    config = ConfigManager(ROOT / "config.yaml").load()
    knowledge_root = resolve_project_path(ROOT, config.knowledge_store_dir, "./data/knowledge_store")
    catalog = ThemeCatalog(knowledge_root / "global")
    query_embedding = None
    embedding_error = ""
    if not no_embeddings and config.theme_embeddings_enabled and config.routing_use_theme_embeddings:
        try:
            query_embedding = (await embed_texts_openai_compatible([message], config=config))[0]
        except Exception as exc:
            embedding_error = str(exc)
    scores = route_theme_scores(
        message,
        catalog.list_themes(),
        global_dir=knowledge_root / "global",
        query_embedding=query_embedding,
        max_themes=top_k,
        min_readiness_rank=_READINESS_ORDER.get("search_ready", 2),
        readiness_order=_READINESS_ORDER,
        min_score=config.routing_min_theme_score,
    )
    return {"message": message, "embedding_error": embedding_error, "routes": scores}


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain NiCo global theme routing for one query.")
    parser.add_argument("message")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-embeddings", action="store_true")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args.message, top_k=args.top_k, no_embeddings=args.no_embeddings)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
