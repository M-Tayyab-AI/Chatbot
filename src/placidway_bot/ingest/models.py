"""Clean, structured representation of one source page. This (not HTML) is what gets chunked."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class Section:
    heading: str
    level: int                      # 1 = page intro (no heading), 2 = h2, 3 = h3 ...
    path: list[str]                 # parent headings, outermost first (for chunk context)
    lines: list[str] = field(default_factory=list)   # paragraphs, "- list items", "Col: val | Col: val" rows

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class FAQ:
    question: str
    answer: str
    section: str


@dataclass
class PriceRecord:
    item: str                       # what is priced, e.g. "Out-Patient Program" / "Ozone Therapy"
    price: float                    # lower / single amount
    currency: str
    price_text: str                 # verbatim from the page, e.g. "$18,995" or "$6,200 - $7,200"
    qualifier: str                  # page context: "total cost", "starting from", "additional cost", "typical range"
    section: str
    price_high: float | None = None
    clinic: str | None = None
    location: str | None = None     # "Tijuana, Mexico"
    country: str | None = None
    details: str | None = None      # duration, sessions, inclusions, notes


@dataclass
class VideoRecord:
    title: str
    description: str
    category: str | None
    youtube_url: str | None


@dataclass
class PageDoc:
    slug: str
    url: str
    page_type: str
    title: str                      # <title>: used as the human source name in citations
    h1: str
    extraction_method: str
    fetched_at: str
    treatment: str | None = None
    clinic: str | None = None
    location: str | None = None
    published: str | None = None
    modified: str | None = None
    reviewed_by: str | None = None
    sections: list[Section] = field(default_factory=list)
    faqs: list[FAQ] = field(default_factory=list)
    prices: list[PriceRecord] = field(default_factory=list)
    videos: list[VideoRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    content_hash: str = ""

    def compute_hash(self) -> str:
        """Fingerprint of the *content* only (ignores fetch time), used by refresh to skip unchanged pages."""
        payload = asdict(self)
        for volatile in ("fetched_at", "content_hash", "extraction_method", "warnings"):
            payload.pop(volatile)
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PageDoc":
        data = dict(data)
        data["sections"] = [Section(**s) for s in data.get("sections", [])]
        data["faqs"] = [FAQ(**f) for f in data.get("faqs", [])]
        data["prices"] = [PriceRecord(**p) for p in data.get("prices", [])]
        data["videos"] = [VideoRecord(**v) for v in data.get("videos", [])]
        return cls(**data)
