from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config_manager import ConfigManager, read_project_yaml, resolve_project_path
from backend.document_loader import SUPPORTED_EXTENSIONS
from backend.domain.document_processor import process_document
from backend.domain.numeric_extractor import get_numeric_extractor
from backend.rag_service import RagService


def _iter_input_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def _extract_archives(input_path: Path, tmp_dir: Path) -> Path:
    if input_path.is_dir():
        return input_path
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as zf:
            zf.extractall(tmp_dir)
        return tmp_dir
    raise ValueError(f"Поддерживается папка или .zip, получено: {input_path}")


def _cfg_get(data: dict, path: tuple[str, ...], default=None):
    node = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


async def ingest(input_path: Path, *, limit: int = 0) -> None:
    config_path = ROOT / "config.yaml"
    project_config = read_project_yaml(config_path)
    working_dir = resolve_project_path(ROOT, _cfg_get(project_config, ("rag", "working_dir")), "./data/rag_storage")
    knowledge_store_dir = resolve_project_path(ROOT, _cfg_get(project_config, ("storage", "knowledge_store_dir")), "./data/knowledge_store")
    config = ConfigManager(config_path).load()
    rag = RagService(working_dir, knowledge_store_dir)
    await rag.ensure_initialized(config)
    extractor = get_numeric_extractor()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            corpus_root = _extract_archives(input_path, Path(tmp))
            files = _iter_input_files(corpus_root)
            if limit > 0:
                files = files[:limit]
            if not files:
                print("Нет поддерживаемых файлов для импорта.")
                return

            print(f"Найдено файлов: {len(files)}")
            for file_idx, file_path in enumerate(files, start=1):
                rel_name = str(file_path.relative_to(corpus_root)) if file_path.is_relative_to(corpus_root) else file_path.name
                print(f"[{file_idx}/{len(files)}] {rel_name}")
                try:
                    processed = process_document(file_path, original_name=rel_name)
                    facts = []
                    for obj in processed.objects:
                        facts.extend(
                            extractor.extract(
                                obj.text,
                                source_name=processed.source_name,
                                object_id=obj.object_id,
                                object_type=obj.object_type,
                                metadata={**obj.metadata, "source_hash": processed.source_hash},
                            )
                        )
                    rag.knowledge_store.upsert_processed_document(
                        processed,
                        facts,
                        original_path=rel_name,
                    )
                    await rag.enqueue_documents_batch([(obj.to_lightrag_text(), obj.citation_name) for obj in processed.objects])
                    print(f"  objects={len(processed.objects)} facts={len(facts)} chars={processed.total_chars}")
                except Exception as exc:
                    print(f"  ERROR: {exc}")
    finally:
        await rag.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Импорт папки или zip-корпуса в ◈NiCo")
    parser.add_argument("input", type=Path, help="Папка с документами или .zip")
    parser.add_argument("--limit", type=int, default=0, help="Ограничить число файлов для smoke/demo")
    args = parser.parse_args()
    asyncio.run(ingest(args.input, limit=args.limit))


if __name__ == "__main__":
    main()
