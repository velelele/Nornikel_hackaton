from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from backend.config_manager import ConfigManager
from backend.domain.document_processor import process_document
from backend.theme_sharding import ThemeShardManager, iter_supported_documents


def _preview_from_document(path: Path, rel: str, max_chars: int) -> tuple[str, int, str]:
    try:
        processed = process_document(path, original_name=rel)
        text = "\n".join(obj.text for obj in processed.objects[:8])[:max_chars]
        return text, len(processed.objects), ""
    except Exception as exc:
        return "", 0, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview ◈NiCo theme resolution before Stage 1 to detect micro-theme fragmentation."
    )
    parser.add_argument("root", type=Path, help="Document root")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview-chars", type=int, default=9000)
    parser.add_argument("--min-docs-per-theme", type=int, default=2)
    parser.add_argument("--min-chunks-per-theme", type=int, default=5)
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "knowledge_store" / "global" / "theme_resolution_preview.csv")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = ConfigManager(args.config).load()
    manager = ThemeShardManager(ROOT, config)

    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"docs": 0, "chunks": 0, "examples": [], "evidence": Counter()})
    rows: list[dict[str, Any]] = []

    for idx, (path, rel) in enumerate(iter_supported_documents(args.root.resolve(), config, limit=args.limit), start=1):
        preview, chunks, error = _preview_from_document(path, rel, args.preview_chars)
        theme = manager.resolve_theme(rel, preview_text=preview)
        g = groups[theme.theme_id]
        g["docs"] += 1
        g["chunks"] += chunks
        if len(g["examples"]) < 3:
            g["examples"].append(rel)
        for ev in theme.evidence:
            g["evidence"][ev] += 1
        rows.append({
            "idx": idx,
            "relative_path": rel,
            "theme_id": theme.theme_id,
            "collection": theme.collection,
            "theme_name": theme.theme_name,
            "confidence": theme.confidence,
            "source_type": theme.source_type,
            "chunks": chunks,
            "evidence": ",".join(theme.evidence),
            "error": error,
        })
        print(f"[{idx}] {rel} -> {theme.theme_id} conf={theme.confidence} chunks={chunks} evidence={','.join(theme.evidence)}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["idx", "relative_path", "theme_id"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nTheme distribution:")
    small = []
    for theme_id, g in sorted(groups.items(), key=lambda kv: (-kv[1]["docs"], kv[0])):
        docs = int(g["docs"])
        chunks = int(g["chunks"])
        examples = "; ".join(g["examples"])
        evidence = ", ".join(f"{k}:{v}" for k, v in g["evidence"].most_common(4))
        mark = ""
        if docs < args.min_docs_per_theme or chunks < args.min_chunks_per_theme:
            mark = "  [SMALL]"
            small.append(theme_id)
        print(f"- {theme_id}: docs={docs}, chunks={chunks}{mark}; evidence={evidence}; examples={examples}")

    if small:
        print("\nWARNING: small themes detected. In coarse mode they should mainly be explicit path/override themes, not filename:auto_theme.")
    print(f"\nReport: {args.report}")


if __name__ == "__main__":
    main()
