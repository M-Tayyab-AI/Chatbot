"""Refresh behaviour with a fake scraper and fake embeddings (no network, no API keys)."""
from __future__ import annotations

from placidway_bot.config import load_sources
from placidway_bot.ingest.pipeline import read_manifest, refresh
from placidway_bot.retrieval.store import load_active_index


class FakeScraper:
    name = "requests"

    def __init__(self, pages: dict[str, str]):
        self.pages = pages

    def get_html(self, url: str) -> str:
        if url not in self.pages:
            raise ConnectionError("site down")
        return self.pages[url]

    def close(self) -> None:
        pass


def _run(tmp_path, pages, voyage, **kw):
    return refresh(scraper=FakeScraper(pages), voyage=voyage, processed_dir=tmp_path / "processed",
                   index_root=tmp_path / "index", raw_dir=tmp_path / "raw", **kw)


def test_refresh_only_reindexes_when_content_changes(tmp_path, package_html, fake_voyage):
    urls = [s.url for s in load_sources()]
    pages = {u: package_html for u in urls}

    first = _run(tmp_path, pages, fake_voyage)
    assert len(first.changed) == 7 and first.index_rebuilt
    first_index = first.active_index

    second = _run(tmp_path, pages, fake_voyage)
    assert second.changed == [] and len(second.unchanged) == 7 and not second.index_rebuilt
    assert second.active_index == first_index

    pages[urls[2]] = package_html.replace("$18,995", "$19,500")       # one page changes its price
    third = _run(tmp_path, pages, fake_voyage)
    assert third.changed == [load_sources()[2].slug] and third.index_rebuilt
    index = load_active_index(tmp_path / "index")
    assert any("$19,500" in c.text for c in index.chunks)
    assert read_manifest(tmp_path / "index" / "manifest.json")["active_index"] == third.active_index


def test_failed_page_keeps_previous_content(tmp_path, package_html, fake_voyage):
    urls = [s.url for s in load_sources()]
    _run(tmp_path, {u: package_html for u in urls}, fake_voyage)
    down = {u: package_html for u in urls[1:]}                           # first page unreachable
    report = _run(tmp_path, down, fake_voyage)
    assert list(report.failed) == [load_sources()[0].slug]
    assert (tmp_path / "processed" / f"{load_sources()[0].slug}.json").exists()
