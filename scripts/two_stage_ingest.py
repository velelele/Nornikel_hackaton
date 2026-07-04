from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convenience wrapper for two-stage ◈NiCo ingestion. "
            "Use 'fast' during the day and 'night' for compressed/cheap KG build over the tree."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fast = sub.add_parser("fast", help="Quick durable fill: parse/chunk/extract facts, skip LightRAG runtime")
    fast.add_argument("root", type=Path)
    fast.add_argument("--limit", type=int, default=None)
    fast.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    fast.add_argument("--skip-existing", action="store_true")
    fast.add_argument("--theme-embeddings", action="store_true")

    night = sub.add_parser("night", help="Complete missing fast fill, then build compressed KG/runtime tree")
    night.add_argument("root", type=Path, help="Root dir is used to complete missing fast fill before full build")
    night.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    night.add_argument("--clear-runtime", action="store_true")
    night.add_argument("--batch-size", type=int, default=None)
    night.add_argument("--max-themes", type=int, default=None)
    night.add_argument("--force", action="store_true")
    night.add_argument("--skip-theme-embeddings", action="store_true")
    night.add_argument("--skip-graph-metrics", action="store_true")
    night.add_argument("--graph-mode", choices=["vector_only", "cheap_kg", "retrieval_kg", "compressed_kg", "full_kg"], default=None)
    night.add_argument("--max-chunks-per-document", type=int, default=None)
    night.add_argument("--min-kg-score", type=float, default=None)
    night.add_argument("--max-chunks-per-theme", type=int, default=None)

    both = sub.add_parser("both", help="Run fast fill over the full tree and then compressed KG/runtime build")
    both.add_argument("root", type=Path)
    both.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    both.add_argument("--clear-runtime", action="store_true")
    both.add_argument("--batch-size", type=int, default=None)
    both.add_argument("--graph-mode", choices=["vector_only", "cheap_kg", "retrieval_kg", "compressed_kg", "full_kg"], default=None)
    both.add_argument("--max-chunks-per-document", type=int, default=None)
    both.add_argument("--min-kg-score", type=float, default=None)
    both.add_argument("--max-chunks-per-theme", type=int, default=None)

    args = parser.parse_args()

    if args.command == "fast":
        cmd = [sys.executable, str(ROOT / "scripts" / "stage1_fast_fill.py"), str(args.root), "--config", str(args.config)]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.theme_embeddings:
            cmd.append("--theme-embeddings")
        _run(cmd)
        return

    if args.command == "night":
        # First complete durable store for all missing files. This makes the night
        # command safe even if the day command used --limit 50.
        cmd1 = [
            sys.executable,
            str(ROOT / "scripts" / "stage1_fast_fill.py"),
            str(args.root),
            "--config",
            str(args.config),
            "--skip-existing",
        ]
        _run(cmd1)
        cmd2 = [sys.executable, str(ROOT / "scripts" / "stage2_full_build.py"), "--config", str(args.config)]
        if args.clear_runtime:
            cmd2.append("--clear-runtime")
        if args.batch_size is not None:
            cmd2 += ["--batch-size", str(args.batch_size)]
        if args.max_themes is not None:
            cmd2 += ["--max-themes", str(args.max_themes)]
        if args.graph_mode:
            cmd2 += ["--graph-mode", args.graph_mode]
        if args.max_chunks_per_document is not None:
            cmd2 += ["--max-chunks-per-document", str(args.max_chunks_per_document)]
        if args.min_kg_score is not None:
            cmd2 += ["--min-kg-score", str(args.min_kg_score)]
        if args.max_chunks_per_theme is not None:
            cmd2 += ["--max-chunks-per-theme", str(args.max_chunks_per_theme)]
        if args.force:
            cmd2.append("--force")
        if args.skip_theme_embeddings:
            cmd2.append("--skip-theme-embeddings")
        if args.skip_graph_metrics:
            cmd2.append("--skip-graph-metrics")
        _run(cmd2)
        return

    if args.command == "both":
        cmd1 = [sys.executable, str(ROOT / "scripts" / "stage1_fast_fill.py"), str(args.root), "--config", str(args.config)]
        _run(cmd1)
        cmd2 = [sys.executable, str(ROOT / "scripts" / "stage2_full_build.py"), "--config", str(args.config)]
        if args.clear_runtime:
            cmd2.append("--clear-runtime")
        if args.batch_size is not None:
            cmd2 += ["--batch-size", str(args.batch_size)]
        if args.graph_mode:
            cmd2 += ["--graph-mode", args.graph_mode]
        if args.max_chunks_per_document is not None:
            cmd2 += ["--max-chunks-per-document", str(args.max_chunks_per_document)]
        if args.min_kg_score is not None:
            cmd2 += ["--min-kg-score", str(args.min_kg_score)]
        if args.max_chunks_per_theme is not None:
            cmd2 += ["--max-chunks-per-theme", str(args.max_chunks_per_theme)]
        _run(cmd2)
        return


if __name__ == "__main__":
    main()
