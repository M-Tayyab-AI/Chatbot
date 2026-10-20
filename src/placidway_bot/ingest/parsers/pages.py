"""Page-type specific parsing on top of the generic base. Entry point: parse_page()."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from placidway_bot.config import Source
from placidway_bot.ingest.models import PageDoc, PriceRecord, VideoRecord
from placidway_bot.ingest.parsers import base

# The compare page embeds per-listing notes (e.g. "1 session" vs "5 sessions") in its inline data that are
# not shown in the visible table. They are essential for a fair comparison; toggle here if excluded.
INCLUDE_LISTING_NOTES = True


def _title(soup: BeautifulSoup) -> str:
    return base.norm(soup.title.get_text(" ")) if soup.title else ""


# --------------------------------------------------------------------------- page types
def _package(doc: PageDoc, main, meta: base.HeaderMeta) -> None:
    doc.prices += base.table_prices(main, doc.clinic, doc.location)
    doc.prices += base.labelled_list_prices(
        doc.sections, r"optional", "optional procedure, additional cost", doc.clinic, doc.location)
    doc.prices += base.labelled_list_prices(
        doc.sections, r"excluded", "NOT included in the package price (extra cost)", doc.clinic, doc.location)
    if meta.header_price:
        doc.prices.insert(0, PriceRecord(
            item=f"{doc.treatment} package", price=base.parse_amount(meta.header_price), currency="USD",
            price_text=meta.header_price, qualifier=meta.header_price_label or "package price",
            section="Page header", clinic=doc.clinic, location=doc.location))


def _price_list(doc: PageDoc, main, jsonld: list[dict]) -> None:
    """HTML table is primary; JSON-LD Product offers fill empty cells (one row has no name in HTML)."""
    rows = base.table_prices(main, doc.clinic, doc.location)
    products = [j for j in jsonld if j.get("@type") == "Product"]
    for i, rec in enumerate(rows):
        if i >= len(products):
            break
        ld_name = products[i].get("name", "").strip()
        ld_price = float(products[i].get("offers", {}).get("price", 0) or 0)
        if not rec.item or rec.item == rec.qualifier:
            rec.item = ld_name
            doc.warnings.append(f"price row {i + 1}: name missing in HTML, filled from JSON-LD as {ld_name!r}")
        if ld_price and rec.price != ld_price:
            doc.warnings.append(f"price row {i + 1}: HTML {rec.price} != JSON-LD {ld_price}")
    doc.prices += rows


def _price_compare(doc: PageDoc, main, raw_html: str) -> None:
    doc.prices += base.table_prices(main, None, None, skip_ids={"center_pricing_table"})
    m = re.search(r"var center_data = (\[.*?\]);\s*\n", raw_html, re.S)
    if not m:
        doc.warnings.append("inline center_data not found; clinic listings parsed from visible table only")
        return
    visible = [base.norm(tr.get_text(" ")) for tr in main.select("#center_pricing_table tbody tr")]
    for i, row in enumerate(json.loads(m.group(1))):
        city, country = row.get("city", "").strip(), row.get("country", "").strip()
        modified = re.search(r"Modified date ([A-Z][a-z]{2} \d{1,2}, \d{4})", visible[i]) if i < len(visible) else None
        details = []
        if INCLUDE_LISTING_NOTES and (row.get("notes") or "").strip():
            details.append(f"Listing note: {base.norm(row['notes'])}")
        if modified:
            details.append(f"Price listing modified {modified.group(1)}")
        doc.prices.append(PriceRecord(
            item=row.get("subtreatment_name", "").strip(), price=float(row["dollar_cost"]), currency="USD",
            price_text=f"${row['dollar_cost']}", qualifier="Standard price (clinic listing)",
            section="Which Clinics Abroad Offer the Best Alternative Medicine Packages?",
            clinic=row.get("center_name", "").strip(), location=f"{city}, {country}", country=country,
            details="; ".join(details) or None))


def _videos(doc: PageDoc, soup: BeautifulSoup, main, jsonld: list[dict]) -> None:
    descriptions = {
        base.norm(j.get("name", "")): base.clean_jsonld_text(
            re.sub(r"^Quick answer: what this video shows", "", j.get("description", "")))
        for j in jsonld if j.get("@type") == "VideoObject"
    }
    for card in soup.select(f"{base.MAIN_SELECTOR} .related-videos-card"):
        title_el = card.select_one(".title")
        if not title_el:
            continue
        title = base.norm(title_el.get_text(" "))
        iframe = card.find("iframe")
        src = (iframe.get("data-src") or iframe.get("src") or "") if iframe else ""
        yt = re.search(r"youtube\.com/embed/([\w-]+)", src)
        desc_parts = [base.norm(p) for p in card.select_one(".desc").get_text("|", strip=True).split("|")]
        desc_parts = [p for p in desc_parts if p and p != title and p != "Read More"]
        doc.videos.append(VideoRecord(
            title=title, description=descriptions.get(title, ""),
            category=desc_parts[0] if desc_parts else None,
            youtube_url=f"https://www.youtube.com/watch?v={yt.group(1)}" if yt else None))
    # the card grid is represented by VideoRecords, not as one giant text section
    for grid in main.select("section.related-videos"):
        grid.decompose()


# --------------------------------------------------------------------------- entry point
def parse_page(html: str, source: Source, method: str, fetched_at: str | None = None) -> PageDoc:
    soup = base.make_soup(html)
    jsonld = base.load_jsonld(soup)
    meta = base.parse_header(soup)
    doc = PageDoc(
        slug=source.slug, url=source.url, page_type=source.page_type, title=_title(soup), h1=meta.h1,
        extraction_method=method, fetched_at=fetched_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        treatment=source.treatment, clinic=source.clinic, location=meta.location,
        published=meta.published, modified=meta.modified, reviewed_by=meta.reviewed_by)

    main = base.select_main(soup)
    if source.page_type == "clinic_videos":
        _videos(doc, soup, main, jsonld)          # needs iframes, so before noise stripping
    main = base.strip_noise(main)

    doc.sections = base.build_sections(base.linearise(main), intro_heading=doc.h1 or doc.title)
    aside = soup.select_one(base.ASIDE_SELECTOR)
    if aside is not None and not aside.find_parent("main"):
        doc.sections += base.build_sections(base.linearise(base.strip_noise(aside)), intro_heading="About Medical Center")
    doc.faqs = base.extract_faqs(doc.sections)

    if source.page_type == "package":
        _package(doc, main, meta)
    elif source.page_type == "price_list":
        _price_list(doc, main, jsonld)
    elif source.page_type == "price_compare":
        _price_compare(doc, main, html)
    elif source.page_type == "clinic_profile":
        doc.prices += base.table_prices(main, doc.clinic, doc.location)

    doc.content_hash = doc.compute_hash()
    return doc
