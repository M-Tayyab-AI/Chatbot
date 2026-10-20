"""Refresh trigger A (CLI): re-fetch the 7 pages and re-index only if something changed.

Usage:  python scripts/ingest.py [--strategy structured|recursive] [--force]
"""
from __future__ import annotations

import argparse
import logging

from placidway_bot.ingest.pipeline import refresh


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="structured", choices=["structured", "recursive"])
    parser.add_argument("--force", action="store_true", help="re-parse and re-index even if nothing changed")
    args = parser.parse_args()
    report = refresh(strategy=args.strategy, force=args.force)
    print(report.summary())
    for slug, err in report.failed.items():
        print(f"FAILED {slug}: {err}")
    raise SystemExit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
