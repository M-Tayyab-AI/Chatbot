"""Method A: plain HTTP GET of the server-rendered HTML (no JavaScript)."""
from __future__ import annotations

from placidway_bot.ingest.fetcher import PoliteFetcher


class RequestsScraper:
    name = "requests"

    def __init__(self, fetcher: PoliteFetcher | None = None):
        self.fetcher = fetcher or PoliteFetcher()

    def get_html(self, url: str) -> str:
        return self.fetcher.fetch(url)

    def close(self) -> None:
        self.fetcher.session.close()
