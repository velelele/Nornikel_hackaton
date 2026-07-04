from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config_manager import ConfigManager
from backend.theme_sharding import ThemeShardManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Show ◈NiCo topic-sharded theme status.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config = ConfigManager(args.config).load()
    manager = ThemeShardManager(ROOT, config)
    rows = manager.list_themes()
    print(f"{'theme_id':40} {'docs':>5} {'chunks':>7} {'facts':>7} {'nodes':>7} {'isol':>6} {'health':>12} {'status':>16}")
    for row in rows:
        stats = row.get("stats") or {}
        
        gm = stats.get('graph_metrics') or {}
        print(
            f"{str(row.get('theme_id'))[:40]:40} "
            f"{int(stats.get('sources') or 0):5d} "
            f"{int(stats.get('chunks') or 0):7d} "
            f"{int(stats.get('numeric_facts') or 0):7d} "
            f"{int(gm.get('node_count') or 0):7d} "
            f"{float(gm.get('isolated_nodes_ratio') or 0):6.2f} "
            f"{str(gm.get('health') or '-'):>12} "
            f"{str(row.get('status') or 'not_ready'):>16}"
        )


if __name__ == "__main__":
    main()
