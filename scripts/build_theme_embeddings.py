from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config_manager import ConfigManager
from backend.graph_embedding_intelligence import build_theme_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Build profile embeddings for NiCo theme router.")
    parser.add_argument("--theme-id", action="append", default=[], help="Optional theme_id. Can be supplied multiple times.")
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args()
    config = ConfigManager(ROOT / "config.yaml").load()
    result = asyncio.run(build_theme_embeddings(
        project_root=ROOT,
        config=config,
        theme_ids=args.theme_id or None,
        max_chunks=args.max_chunks or config.theme_embeddings_max_chunks_per_theme,
    ))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
