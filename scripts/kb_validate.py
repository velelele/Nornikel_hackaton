from __future__ import annotations

import argparse
import json
import sys
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
    parser = argparse.ArgumentParser(description="Валидация durable knowledge_store ◈NiCo")
    parser.add_argument("--store", type=Path, default=None, help="Путь к knowledge_store. По умолчанию из config.yaml")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = read_project_yaml(args.config)
    store_dir = args.store or resolve_project_path(ROOT, _cfg_get(cfg, ("storage", "knowledge_store_dir")), "./data/knowledge_store")
    store = KnowledgeStore(store_dir)
    result = store.validate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 2)


if __name__ == "__main__":
    main()
