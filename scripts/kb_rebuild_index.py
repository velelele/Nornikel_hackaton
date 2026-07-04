from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config_manager import ConfigManager, read_project_yaml, resolve_project_path
from backend.rag_service import RagService


def _cfg_get(data: dict, path: tuple[str, ...], default=None):
    node = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


async def rebuild(*, clear_runtime: bool, batch_size: int) -> None:
    config_path = ROOT / "config.yaml"
    project_config = read_project_yaml(config_path)
    working_dir = resolve_project_path(ROOT, _cfg_get(project_config, ("rag", "working_dir")), "./data/rag_storage")
    knowledge_store_dir = resolve_project_path(ROOT, _cfg_get(project_config, ("storage", "knowledge_store_dir")), "./data/knowledge_store")

    if clear_runtime and working_dir.exists():
        shutil.rmtree(working_dir)

    config = ConfigManager(config_path).load()
    rag = RagService(working_dir, knowledge_store_dir)
    await rag.ensure_initialized(config)
    try:
        result = await rag.rebuild_runtime_index_from_store(batch_size=batch_size)
        print(result)
    finally:
        await rag.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Пересборка LightRAG runtime index из data/knowledge_store без повторного parsing документов")
    parser.add_argument("--clear-runtime", action="store_true", help="Удалить rag_storage перед пересборкой")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    asyncio.run(rebuild(clear_runtime=args.clear_runtime, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
