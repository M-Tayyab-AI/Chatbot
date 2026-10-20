"""Ingestion + refresh: fetch -> parse -> hash-diff -> write processed data -> rebuild index -> atomic switch.

Both refresh triggers (CLI `scripts/ingest.py` and the admin button in the UI) call `refresh()`.
Unchanged pages are skipped; unchanged chunks reuse cached embeddings, so a no-op refresh costs no API calls.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from placidway_bot.config import INDEX_DIR, PROCESSED_DIR, RAW_DIR, Source, load_sources
from placidway_bot.ingest.chunker import STRATEGIES
from placidway_bot.ingest.models import PageDoc
from placidway_bot.ingest.parsers.pages import parse_page
from placidway_bot.ingest.render import to_markdown
from placidway_bot.ingest.scrapers.requests_bs4 import RequestsScraper

log = logging.getLogger(__name__)
MANIFEST = INDEX_DIR / "manifest.json"


@dataclass
class RefreshReport:
    started_at: str
    finished_at: str = ""
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    index_rebuilt: bool = False
    active_index: str = ""
    n_chunks: int = 0

    def summary(self) -> str:
        return (f"changed={len(self.changed)} unchanged={len(self.unchanged)} failed={len(self.failed)} "
                f"index_rebuilt={self.index_rebuilt} chunks={self.n_chunks} active={self.active_index}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"pages": {}}


def load_processed(processed_dir: Path = PROCESSED_DIR) -> list[PageDoc]:
    return [PageDoc.from_dict(json.loads((processed_dir / f"{s.slug}.json").read_text(encoding="utf-8")))
            for s in load_sources() if (processed_dir / f"{s.slug}.json").exists()]


def _write_doc(doc: PageDoc, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / f"{doc.slug}.json").write_text(json.dumps(doc.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    (processed_dir / f"{doc.slug}.md").write_text(to_markdown(doc), encoding="utf-8")


def fetch_and_parse(sources: tuple[Source, ...], scraper, report: RefreshReport, manifest: dict,
                    processed_dir: Path, raw_dir: Path, force: bool) -> None:
    for src in sources:
        try:
            html = scraper.get_html(src.url)
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{src.slug}.html").write_text(html, encoding="utf-8")
            doc = parse_page(html, src, scraper.name)
        except Exception as exc:  # noqa: BLE001 - one bad page must not break the whole refresh
            log.exception("refresh: %s failed", src.slug)
            report.failed[src.slug] = f"{type(exc).__name__}: {exc}"
            continue
        previous = manifest["pages"].get(src.slug, {})
        if not force and previous.get("content_hash") == doc.content_hash and (processed_dir / f"{src.slug}.json").exists():
            report.unchanged.append(src.slug)
            previous["checked_at"] = doc.fetched_at
            continue
        _write_doc(doc, processed_dir)
        report.changed.append(src.slug)
        manifest["pages"][src.slug] = {"url": src.url, "content_hash": doc.content_hash, "fetched_at": doc.fetched_at,
                                       "checked_at": doc.fetched_at, "changed_at": doc.fetched_at,
                                       "warnings": doc.warnings}


def rebuild_index(strategy: str, voyage, processed_dir: Path, index_root: Path, previous: str | None):
    from placidway_bot.retrieval.store import build_index, delete_index

    chunks = [c for doc in load_processed(processed_dir) for c in STRATEGIES[strategy](doc)]
    name = f"kb_{strategy}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    build_index(name, chunks, voyage, root=index_root)      # built next to the live one ...
    if previous and previous != name:
        delete_index(previous, root=index_root)             # ... and the old one removed only after success
    return name, len(chunks)


def refresh(strategy: str = "structured", force: bool = False, scraper=None, voyage=None,
            processed_dir: Path = PROCESSED_DIR, index_root: Path = INDEX_DIR,
            raw_dir: Path = RAW_DIR / "requests") -> RefreshReport:
    report = RefreshReport(started_at=_now())
    manifest_path = index_root / "manifest.json"
    manifest = read_manifest(manifest_path)
    scraper = scraper or RequestsScraper()
    try:
        fetch_and_parse(load_sources(), scraper, report, manifest, processed_dir, raw_dir, force)
    finally:
        scraper.close()

    needs_index = bool(report.changed) or manifest.get("strategy") != strategy or not manifest.get("active_index")
    if needs_index:
        if voyage is None:
            from placidway_bot.retrieval.voyage_service import VoyageService
            voyage = VoyageService()
        name, n = rebuild_index(strategy, voyage, processed_dir, index_root, manifest.get("active_index"))
        manifest.update(active_index=name, strategy=strategy, n_chunks=n, indexed_at=_now())
        report.index_rebuilt = True
    report.active_index, report.n_chunks = manifest.get("active_index", ""), manifest.get("n_chunks", 0)
    report.finished_at = manifest["refreshed_at"] = _now()
    manifest["last_report"] = {"changed": report.changed, "unchanged": report.unchanged, "failed": report.failed}
    index_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("refresh done: %s", report.summary())
    return report
