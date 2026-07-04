from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config_manager import read_project_yaml, resolve_project_path
from backend.knowledge_store import KnowledgeStore


def _cfg_get(data: dict, path: tuple[str, ...], default=None):
    node = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт durable knowledge_store ◈NiCo в версионируемую папку")
    parser.add_argument("--out", type=Path, default=None, help="Папка назначения. По умолчанию data/exports/kb_<timestamp>")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = read_project_yaml(args.config)
    store_dir = resolve_project_path(ROOT, _cfg_get(cfg, ("storage", "knowledge_store_dir")), "./data/knowledge_store")
    store = KnowledgeStore(store_dir)
    out = args.out or (ROOT / "data" / "exports" / f"kb_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    export_dir = store.export_to(out)
    print(f"Экспортировано: {export_dir}")
    print(store.stats())


if __name__ == "__main__":
    main()
