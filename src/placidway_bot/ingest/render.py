"""PageDoc -> human-readable Markdown (for reviewers; the bot reads the JSON)."""
from __future__ import annotations

from placidway_bot.ingest.models import PageDoc


def _price_line(p) -> str:
    where = ", ".join(x for x in (p.clinic, p.location) if x)
    extra = f" — {p.details}" if p.details else ""
    price = p.price_text if p.currency in p.price_text else f"{p.price_text} {p.currency}"
    return f"| {p.item} | {price} | {p.qualifier} | {where} |{extra}"


def to_markdown(doc: PageDoc) -> str:
    out = [f"# {doc.h1 or doc.title}", "",
           f"- **Source:** {doc.url}", f"- **Page type:** {doc.page_type}", f"- **Page title:** {doc.title}"]
    for label, value in (("Treatment", doc.treatment), ("Clinic", doc.clinic), ("Location", doc.location),
                         ("Published", doc.published), ("Modified", doc.modified), ("Medically reviewed by", doc.reviewed_by)):
        if value:
            out.append(f"- **{label}:** {value}")
    out += [f"- **Fetched:** {doc.fetched_at} ({doc.extraction_method})", f"- **Content hash:** {doc.content_hash}", ""]

    if doc.prices:
        out += ["## Prices (structured)", "", "| Item | Price | Context | Clinic / location |", "|---|---|---|---|"]
        out += [_price_line(p) for p in doc.prices]
        out.append("")
    for sec in doc.sections:
        hashes = "#" * min(max(sec.level, 2), 6)
        out += [f"{hashes} {sec.heading}", "", sec.text, ""]
    if doc.videos:
        out += ["## Videos", ""]
        out += [f"- **{v.title}** ({v.category or 'video'}) — {v.description} {v.youtube_url or ''}".rstrip() for v in doc.videos]
        out.append("")
    if doc.warnings:
        out += ["## Extraction warnings", ""] + [f"- {w}" for w in doc.warnings]
    return "\n".join(out).rstrip() + "\n"
