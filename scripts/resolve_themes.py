from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from backend.config_manager import ConfigManager
from backend.theme_sharding import ThemeShardManager, iter_supported_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview how ◈NiCo will distribute files into topic shards.")
    parser.add_argument("root", type=Path, help="Root directory with arbitrary nested structure")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "theme_resolution_report.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = ConfigManager(args.config).load()
    manager = ThemeShardManager(ROOT, config)

    rows = []
    for path, rel in iter_supported_documents(args.root.resolve(), config, limit=args.limit):
        theme = manager.resolve_theme(rel)
        row = theme.to_dict()
        row["file_path"] = str(path)
        row["evidence"] = ";".join(theme.evidence)
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_path", "relative_path", "source_name", "collection", "theme_name", "theme_id",
        "year", "source_type", "confidence", "evidence",
    ]
    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resolved {len(rows)} file(s) -> {args.out}")
    by_theme: dict[str, int] = {}
    for row in rows:
        by_theme[row["theme_id"]] = by_theme.get(row["theme_id"], 0) + 1
    for theme_id, count in sorted(by_theme.items(), key=lambda item: (-item[1], item[0]))[:30]:
        print(f"{count:5d}  {theme_id}")


if __name__ == "__main__":
    main()
