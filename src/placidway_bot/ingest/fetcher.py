"""Polite HTTP fetcher: robots.txt check, identifying User-Agent, ~1 req/s, retries with backoff."""
from __future__ import annotations

import logging
import time
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests

from placidway_bot.config import get_settings

log = logging.getLogger(__name__)


class RobotsDisallowedError(RuntimeError):
    pass


class PoliteFetcher:
    def __init__(self, user_agent: str | None = None, delay: float | None = None, max_retries: int = 3):
        settings = get_settings()
        self.user_agent = user_agent or settings.crawler_user_agent
        self.delay = settings.crawl_delay_seconds if delay is None else delay
        self.timeout = settings.request_timeout_seconds
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent, "Accept-Language": "en"})
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request = 0.0

    def _throttle(self) -> None:
        wait = self.delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _robots_for(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            self._throttle()
            resp = self.session.get(f"{origin}/robots.txt", timeout=self.timeout)
            parser = RobotFileParser()
            parser.parse(resp.text.splitlines() if resp.ok else [])
            self._robots[origin] = parser
        return self._robots[origin]

    def is_allowed(self, url: str) -> bool:
        return self._robots_for(url).can_fetch(self.user_agent, url)

    def fetch(self, url: str) -> str:
        """Return the HTML of `url`. Raises RobotsDisallowedError or requests.HTTPError."""
        if not self.is_allowed(url):
            raise RobotsDisallowedError(f"robots.txt disallows {url}")
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                resp.encoding = resp.encoding or "utf-8"
                return resp.text
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                retryable = not isinstance(exc, requests.HTTPError) or (
                    exc.response is not None and (exc.response.status_code >= 500 or exc.response.status_code == 429)
                )
                if attempt == self.max_retries or not retryable:
                    raise
                backoff = 2**attempt
                log.warning("fetch %s failed (%s), retry %d in %ss", url, exc, attempt, backoff)
                time.sleep(backoff)
        raise AssertionError("unreachable")
