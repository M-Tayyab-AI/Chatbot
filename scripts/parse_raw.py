"""Parse archived raw HTML of one method into clean JSON + Markdown.

Usage:  python scripts/parse_raw.py requests [--out data/experiments/requests]
Default output is data/experiments/<method>/; the chosen method is later promoted to data/processed/.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from placidway_bot.config import DATA_DIR, RAW_DIR, load_sources
from placidway_bot.ingest.parsers.pages import parse_page
from placidway_bot.ingest.render import to_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["requests", "selenium", "selenium_en"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out or DATA_DIR / "experiments" / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in load_sources():
        raw = RAW_DIR / args.method / f"{src.slug}.html"
        fetched_at = datetime.fromtimestamp(raw.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        doc = parse_page(raw.read_text(encoding="utf-8"), src, args.method, fetched_at)
        (out_dir / f"{src.slug}.json").write_text(json.dumps(doc.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / f"{src.slug}.md").write_text(to_markdown(doc), encoding="utf-8")
        chars = sum(len(s.text) for s in doc.sections)
        print(f"{src.slug:40s} sections={len(doc.sections):3d} faqs={len(doc.faqs):2d} prices={len(doc.prices):3d} "
              f"videos={len(doc.videos):3d} chars={chars:6d} warnings={len(doc.warnings)}")


if __name__ == "__main__":
    main()
