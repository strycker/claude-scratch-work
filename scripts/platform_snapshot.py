#!/usr/bin/env python3
"""
platform_snapshot.py — export, restore, or list offline dev snapshots of the
platform checkpoint namespace (``data/checkpoints/platform/``).

DEV/OFFLINE CONVENIENCE ONLY. A snapshot is a frozen copy of the platform
checkpoints captured at a stated moment — never live data, never loaded
automatically. See ``docs/offline_snapshots.md`` for the full guide.

Usage:
    python scripts/platform_snapshot.py export [--names daily_raw monthly_raw ...] \
        [--max-mb 10] [--daily-start 2015-01-01] [--compression zstd] [--snapshot-dir PATH]
    python scripts/platform_snapshot.py restore [--names ...] [--snapshot-dir PATH]
    python scripts/platform_snapshot.py list [--snapshot-dir PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from trading_crab_lib.platform.snapshots import (  # noqa: E402
    DEFAULT_SNAPSHOT_NAMES,
    SNAPSHOT_DIR,
    export_snapshot,
    read_manifest,
    restore_snapshot,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export, restore, or list offline dev snapshots of the platform checkpoints.",
    )
    parser.add_argument("--verbose", action="store_true", help="Set logging to DEBUG.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--snapshot-dir", type=Path, default=None,
        help=f"Snapshot location (default: {SNAPSHOT_DIR}).",
    )
    common.add_argument(
        "--names", nargs="+", default=None,
        help=f"Checkpoint names (default: {' '.join(DEFAULT_SNAPSHOT_NAMES)}).",
    )

    export_p = sub.add_parser(
        "export", parents=[common], help="Export checkpoints to the snapshot location."
    )
    export_p.add_argument(
        "--max-mb", type=float, default=10.0,
        help="Refuse to write a snapshot set larger than this total size, in MB (default: 10.0).",
    )
    export_p.add_argument(
        "--daily-start", default=None,
        help="Truncate daily_raw to this lower date bound (e.g. 2015-01-01) before writing.",
    )
    export_p.add_argument(
        "--compression", default="zstd",
        help="Parquet compression codec (default: zstd; falls back to snappy if unavailable).",
    )

    sub.add_parser("restore", parents=[common], help="Restore checkpoints from the snapshot location.")
    sub.add_parser("list", parents=[common], help="Show the snapshot manifest.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "export":
        out_dir = export_snapshot(
            names=args.names,
            snapshot_dir=args.snapshot_dir,
            max_mb=args.max_mb,
            daily_start=args.daily_start,
            compression=args.compression,
        )
        print(f"Exported snapshot to: {out_dir}")
        return 0

    if args.command == "restore":
        manifest = read_manifest(snapshot_dir=args.snapshot_dir)
        captured_at = manifest.get("captured_at", "unknown capture time")
        restored = restore_snapshot(names=args.names, snapshot_dir=args.snapshot_dir)
        print("=" * 78)
        print("RESTORED CHECKPOINT(S) FROM OFFLINE SNAPSHOT — THIS IS NOT LIVE DATA")
        print(f"Snapshot captured at: {captured_at}")
        print(f"Checkpoints restored: {', '.join(restored) if restored else '(none — nothing matched)'}")
        print("=" * 78)
        return 0

    if args.command == "list":
        manifest = read_manifest(snapshot_dir=args.snapshot_dir)
        if not manifest:
            print("No snapshot manifest found.")
            return 0
        print(f"Captured at: {manifest.get('captured_at', 'unknown')}")
        print(f"Git commit:  {manifest.get('git_commit', 'unknown')}")
        for name, meta in manifest.get("names", {}).items():
            print(
                f"  {name}: {meta.get('rows', '?')} rows x {meta.get('columns', '?')} cols "
                f"({meta.get('index_start', '?')} -> {meta.get('index_end', '?')})"
            )
        if "source_provenance" in manifest:
            print("Splice source provenance included (see manifest.json for detail).")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
