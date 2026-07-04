from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config_manager import ConfigManager
from backend.theme_sharding import ThemeShardManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild global theme router for ◈NiCo.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config = ConfigManager(args.config).load()
    manager = ThemeShardManager(ROOT, config)
    payload = manager.catalog.build_router()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
