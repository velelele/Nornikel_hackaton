from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config_manager import ConfigManager
from backend.theme_sharding import ThemeShardManager, get_ingestion_profile


async def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild LightRAG runtime index for one ◈NiCo theme from knowledge_store.")
    parser.add_argument("theme_id")
    parser.add_argument("--clear-runtime", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--profile", default="balanced", help="balanced/safe/overnight_full")
    parser.add_argument("--wait", action="store_true", help="Wait until LightRAG finishes this rebuild")
    parser.add_argument("--poll-interval", type=float, default=None)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config = ConfigManager(args.config).load()
    profile = get_ingestion_profile(config, args.profile)
    manager = ThemeShardManager(ROOT, config)
    try:
        result = await manager.rebuild_theme_runtime(
            args.theme_id,
            clear_runtime=args.clear_runtime,
            batch_size=args.batch_size or int(profile.get("batch_size_docs") or config.topic_batch_size_docs),
            wait=bool(args.wait or profile.get("wait", False)),
            poll_interval=float(args.poll_interval if args.poll_interval is not None else profile.get("poll_interval", 10.0)),
            timeout_sec=float(args.timeout_sec if args.timeout_sec is not None else profile.get("timeout_sec", 0.0)),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
