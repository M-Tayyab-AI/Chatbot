"""Fetch the raw HTML of every source with one scraping method and archive it under data/raw/<method>/.

Usage:  python scripts/fetch_raw.py requests
(The Selenium method was evaluated and removed — see docs/scraper_comparison.md; its archived HTML can still be
re-parsed with scripts/parse_raw.py selenium.)
"""
from __future__ import annotations

import argparse
import json
import logging
import time

from placidway_bot.config import RAW_DIR, load_sources


def build_scraper(method: str):
    if method == "requests":
        from placidway_bot.ingest.scrapers.requests_bs4 import RequestsScraper

        return RequestsScraper()
    raise SystemExit(f"unknown method {method!r}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["requests"])
    args = parser.parse_args()

    out_dir = RAW_DIR / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    scraper = build_scraper(args.method)
    timings: dict[str, float] = {}
    try:
        for src in load_sources():
            start = time.perf_counter()
            html = scraper.get_html(src.url)
            timings[src.slug] = round(time.perf_counter() - start, 2)
            (out_dir / f"{src.slug}.html").write_text(html, encoding="utf-8")
            logging.info("%-40s %7d chars  %.2fs", src.slug, len(html), timings[src.slug])
    finally:
        scraper.close()
    (out_dir / "_timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
