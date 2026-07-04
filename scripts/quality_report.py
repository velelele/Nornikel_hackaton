from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config_manager import ConfigManager, resolve_project_path
from backend.graph_embedding_intelligence import build_quality_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NiCo knowledge-store quality report.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    args = parser.parse_args()
    config = ConfigManager(ROOT / "config.yaml").load()
    knowledge_root = resolve_project_path(ROOT, config.knowledge_store_dir, "./data/knowledge_store")
    report = build_quality_report(knowledge_root)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
