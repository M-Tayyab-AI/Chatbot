"""Generic HTML -> clean structured text, shared by every page-type parser.

Pipeline: pick main content -> strip noise -> linearise into ordered blocks -> group blocks into
heading-based sections -> derive FAQs and table prices. Page-type parsers add what is specific to them.
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from placidway_bot.ingest.models import FAQ, PriceRecord, Section

MAIN_SELECTOR = "main.article-dt-desc"

# Removed before extraction. The related-listing widgets (outside <main>) carry OTHER clinics' prices and must
# never leak. Sliders *inside* <main> hold this clinic's own reviews, so only their JS-made clones are dropped.
NOISE_SELECTORS = [
    "script", "style", "noscript", "iframe", "svg", "button", "form", "input", "select", "textarea", "template",
    "[id$='_popup']", ".social-popup-main", ".request-vido-popup-main", ".article-social-section",
    ".related-widgets-row", ".related-packages-sec", ".related-videos-card .image-wrapper", ".splide__slide--clone",
    "a.custom_link",
    "#cta-form", ".cta-form-container", ".pw-stats-section", ".dt-pages-bottom-widget",
]
ASIDE_SELECTOR = ".pack-about-aside"   # "About Medical Center" sidebar: facts about THIS clinic (address, languages)
NOISE_HEADINGS = re.compile(r"^(related experiences:?|location)$", re.I)
# UI chrome that survives as text; never knowledge. Also the aside's giant "all videos" link list.
UI_NOISE_LINE = re.compile(r"^(- )?(view center reviews|write a review|read more|copy link|share this listing|chat with center"
                           r"|medical center videos:)", re.I)
FAQ_HEADING = re.compile(r"\bFAQs?\b|frequently asked", re.I)
DROP_COLUMNS = {"action", "actions"}

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
BLOCK_TAGS = HEADING_TAGS | {"p", "ul", "ol", "table", "dl", "details", "blockquote", "div", "section", "article",
                             "aside", "header", "footer", "figure", "li", "tr"}

PRICE_RE = re.compile(r"(?:US\$|\$|USD\s?)\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s?USD)?", re.I)
PRICE_RANGE_RE = re.compile(
    r"((?:US\$|\$)\s?(?:\d{1,3}(?:,\d{3})+|\d+))\s*(?:-|–|to)\s*((?:US\$|\$)?\s?(?:\d{1,3}(?:,\d{3})+|\d+))(?:\s?USD)?",
    re.I,
)


# --------------------------------------------------------------------------- text helpers
def clean_jsonld_text(text: str) -> str:
    """JSON-LD strings carry HTML entities and headings glued to the next sentence ("MexicoReceiving")."""
    text = html.unescape(html.unescape(text))
    text = re.sub(r"([a-z.?!])([A-Z][a-z])", r"\1 \2", text)
    return norm(text)


def norm(text: str) -> str:
    """Collapse whitespace and fix the spacing artefacts get_text(' ') leaves around punctuation."""
    text = text.replace("\xa0", " ").replace("​", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text


def parse_amount(text: str) -> float | None:
    match = PRICE_RE.search(text) or re.search(r"(\d{1,3}(?:,\d{3})+|\d+)", text)
    return float(match.group(1).replace(",", "")) if match else None


def find_prices(text: str) -> list[str]:
    return [m.group(0).strip() for m in PRICE_RE.finditer(text)]


# --------------------------------------------------------------------------- soup helpers
def make_soup(html: str) -> BeautifulSoup:
    # html5lib (the HTML5 spec algorithm browsers use): these pages have unbalanced tags. lxml nests the
    # sidebar (quote form, site stats) inside <main>; html.parser does not auto-close <p> before headings.
    return BeautifulSoup(html, "html5lib")


def load_jsonld(soup: BeautifulSoup) -> list[dict]:
    """All JSON-LD objects on the page, with lists and @graph flattened."""
    out: list[dict] = []

    def add(obj) -> None:
        if isinstance(obj, list):
            for item in obj:
                add(item)
        elif isinstance(obj, dict):
            out.append(obj)
            if "@graph" in obj:
                add(obj["@graph"])

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            add(json.loads(tag.string or tag.get_text()))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


@dataclass
class HeaderMeta:
    h1: str = ""
    location: str | None = None
    published: str | None = None
    modified: str | None = None
    reviewed_by: str | None = None
    header_price: str | None = None          # e.g. "$18,995"
    header_price_label: str | None = None    # e.g. "Price starting from"


def parse_header(soup: BeautifulSoup) -> HeaderMeta:
    meta = HeaderMeta()
    h1 = soup.find("h1")
    meta.h1 = norm(h1.get_text(" ")) if h1 else ""
    header = soup.select_one(".pack-title-price-header")
    if not header:
        return meta
    parts = [norm(p) for p in header.get_text("|", strip=True).split("|") if norm(p)]
    text = " | ".join(parts)
    if meta.h1 in parts:
        nxt = parts.index(meta.h1) + 1
        if nxt < len(parts) and re.fullmatch(r"[A-Z][\w .'-]+, [A-Z][\w .'-]+", parts[nxt]):
            meta.location = parts[nxt]
    date = r"([A-Z][a-z]{2} \d{1,2}, \d{4})"
    if m := re.search(rf"Publish date: {date}", text):
        meta.published = m.group(1)
    if m := re.search(rf"Modified date: {date}", text):
        meta.modified = m.group(1)
    if m := re.search(r"Medically reviewed by: \| ([^|]+)", text):
        meta.reviewed_by = m.group(1).strip()
    if m := re.search(r"(\$[\d,]+) \| (Price starting from)", text):
        meta.header_price, meta.header_price_label = m.group(1), m.group(2)
    return meta


def select_main(soup: BeautifulSoup) -> Tag:
    main = soup.select_one(MAIN_SELECTOR) or soup.find("main")
    if main is None:
        raise ValueError("main content container not found")
    return main


def strip_noise(main: Tag) -> Tag:
    for selector in NOISE_SELECTORS:
        for el in main.select(selector):
            el.decompose()
    for comment in main.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for heading in main.find_all(list(HEADING_TAGS)):
        if NOISE_HEADINGS.match(norm(heading.get_text(" "))):
            heading.decompose()
    _flatten_review_cards(main)
    return main


def _flatten_review_cards(root: Tag) -> None:
    """A review card renders text first and the reviewer's name heading after it, so heading-based grouping
    would attach each review to the *previous* reviewer. Collapse every card into one attributed paragraph."""
    for card in root.select(".review-card"):
        body, name = card.select_one(".review-body"), card.select_one(".reviewer-name")
        if body is None or name is None:
            continue
        date = card.select_one(".review-date")
        rating = card.select_one("[data-rating]")
        meta = ", ".join(x for x in (norm(date.get_text(" ")) if date else "",
                                     f"{rating['data-rating']}/5" if rating else "") if x)
        para = BeautifulSoup("", "html.parser").new_tag("p")
        para.string = f"Patient review by {norm(name.get_text(' '))}{f' ({meta})' if meta else ''}: {norm(body.get_text(' '))}"
        card.replace_with(para)


# --------------------------------------------------------------------------- linearisation
Block = tuple[str, int, str]   # (kind: heading|text|item|row, heading level, text)


def table_rows(table: Tag) -> tuple[list[str], list[list[str]]]:
    """Return (headers, rows) with the action column dropped. Cells keep their visible text."""
    rows = [[norm(c.get_text(" ")) for c in tr.find_all(["th", "td"])] for tr in table.find_all("tr")]
    rows = [r for r in rows if any(r)]
    if not rows:
        return [], []
    headers, body = rows[0], rows[1:]
    keep = [i for i, h in enumerate(headers) if h.lower() not in DROP_COLUMNS]
    headers = [headers[i] for i in keep]
    body = [[r[i] if i < len(r) else "" for i in keep] for r in body]
    return headers, body


def _row_line(headers: list[str], row: list[str]) -> str:
    return " | ".join(f"{h}: {v}" for h, v in zip(headers, row) if v)


def _has_block_child(tag: Tag) -> bool:
    return any(isinstance(c, Tag) and c.name in BLOCK_TAGS for c in tag.descendants)


def linearise(root: Tag) -> list[Block]:
    blocks: list[Block] = []
    inline: list[str] = []

    def flush() -> None:
        text = norm(" ".join(inline))
        inline.clear()
        if text:
            blocks.append(("text", 0, text))

    def walk(el: Tag) -> None:
        for child in el.children:
            if isinstance(child, NavigableString):
                inline.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name
            if name not in BLOCK_TAGS:
                if name == "br":
                    flush()
                else:
                    inline.append(child.get_text(" "))
                continue
            flush()
            if name in HEADING_TAGS:
                text = norm(child.get_text(" "))
                if text:
                    blocks.append(("heading", int(name[1]), text))
            elif name in {"p", "blockquote"}:
                text = norm(child.get_text(" "))
                if text:
                    blocks.append(("text", 0, text))
            elif name in {"ul", "ol"}:
                for li in child.find_all("li", recursive=False):
                    text = norm(li.get_text(" "))
                    if text:
                        blocks.append(("item", 0, text))
            elif name == "table":
                headers, rows = table_rows(child)
                for row in rows:
                    line = _row_line(headers, row)
                    if line:
                        blocks.append(("row", 0, line))
            elif name == "dl":
                for dt in child.find_all("dt"):
                    dd = dt.find_next_sibling("dd")
                    if dd:
                        blocks.append(("item", 0, f"{norm(dt.get_text(' '))}: {norm(dd.get_text(' '))}"))
            elif name == "details":
                summary = child.find("summary")
                if summary:
                    blocks.append(("heading", 5, norm(summary.get_text(" "))))
                    summary.extract()
                walk(child)
                flush()
            elif _has_block_child(child):
                walk(child)
                flush()
            else:
                text = norm(child.get_text(" "))
                if text:
                    blocks.append(("text", 0, text))
        # trailing inline text belongs to the current container
    walk(root)
    flush()
    return blocks


def build_sections(blocks: list[Block], intro_heading: str) -> list[Section]:
    intro = Section(heading=intro_heading, level=1, path=[])
    sections: list[Section] = [intro]
    stack: list[Section] = [intro]          # open ancestors, outermost first
    current = intro
    for kind, level, text in blocks:
        if kind == "heading":
            # A heading with no text and no sub-headings (e.g. a video-testimonial title) would vanish;
            # keep it as a bullet of its parent section instead.
            if not current.lines and level <= current.level and current is not intro:
                parent = next((s for s in reversed(stack) if s.level < current.level), intro)
                parent.lines.append(f"- {current.heading}")
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            current = Section(heading=text, level=level, path=[s.heading for s in stack[1:]])
            sections.append(current)
            stack.append(current)
        else:
            line = f"- {text}" if kind == "item" else text
            if UI_NOISE_LINE.match(line):
                continue
            if not current.lines or current.lines[-1] != line:   # drop exact consecutive duplicates
                current.lines.append(line)
    if not current.lines and current is not intro:
        next((s for s in reversed(stack[:-1]) if s.level < current.level), intro).lines.append(f"- {current.heading}")
    return [s for s in sections if s.lines]


def extract_faqs(sections: list[Section]) -> list[FAQ]:
    faqs: list[FAQ] = []
    for sec in sections:
        in_faq = any(FAQ_HEADING.search(h) for h in sec.path)
        if in_faq and sec.heading.rstrip().endswith("?"):
            question = re.sub(r"^\d+\.\s*", "", sec.heading).strip()
            faqs.append(FAQ(question=question, answer=sec.text, section=sec.path[-1]))
    return faqs


# --------------------------------------------------------------------------- prices from tables
def _section_of(tag: Tag) -> str:
    heading = tag.find_previous(["h2", "h3"])
    return norm(heading.get_text(" ")) if heading else ""


def _table_note(table: Tag) -> str | None:
    """A caveat paragraph right after a table ("Note: ...", "Prices are starting estimates ...")."""
    container = table
    while container.find_next_sibling() is None and container.parent is not None:
        container = container.parent
    nxt = container.find_next_sibling()
    text = norm(nxt.get_text(" ")) if nxt is not None and nxt.name == "p" else ""
    return text if re.match(r"^(note|prices are)", text, re.I) else None


def table_prices(main: Tag, clinic: str | None, location: str | None, skip_ids: set[str] = frozenset()) -> list[PriceRecord]:
    """Turn every priced table row into an atomic PriceRecord, keeping the column header as context."""
    records: list[PriceRecord] = []
    for table in main.find_all("table"):
        if table.get("id") in skip_ids:
            continue
        headers, rows = table_rows(table)
        if not headers:
            continue
        # Every price on these pages is "$..." or "... USD"; EUR is never displayed in the priced tables.
        currency = "USD"
        lower = [h.lower() for h in headers]
        low_i = next((i for i, h in enumerate(lower) if h.startswith("low")), None)
        high_i = next((i for i, h in enumerate(lower) if h.startswith("high")), None)
        section = _section_of(table)
        note = _table_note(table)
        for row in rows:
            price_cols = [i for i, v in enumerate(row) if PRICE_RE.search(v) and re.match(r"^\s*(US\$|\$|\d)", v)]
            if not price_cols:
                continue
            other = [i for i in range(len(row)) if i not in price_cols and row[i]]
            if low_i is not None and high_i is not None and low_i in price_cols:
                lo, hi = row[low_i], row[high_i]
                item = row[other[0]] if other else headers[low_i]
                details = "; ".join([f"{headers[i]}: {row[i]}" for i in other[1:]] + ([note] if note else []))
                records.append(PriceRecord(
                    item=item, price=parse_amount(lo), price_high=parse_amount(hi), currency=currency,
                    price_text=f"{lo} - {hi}", qualifier=f"{headers[low_i]} / {headers[high_i]} range",
                    section=section, clinic=clinic, location=location, details=details or None))
                continue
            p = price_cols[0]
            first_is_price = p == 0
            item = headers[p] if first_is_price or not other else row[other[0]]
            detail_cols = other if first_is_price else other[1:]
            details = "; ".join([f"{headers[i]}: {row[i]}" for i in detail_cols if row[i] != item] + ([note] if note else []))
            records.append(PriceRecord(
                item=item, price=parse_amount(row[p]), currency=currency, price_text=PRICE_RE.search(row[p]).group(0).strip(),
                qualifier=headers[p], section=section, clinic=clinic, location=location, details=details or None))
    return records


def labelled_list_prices(sections: list[Section], heading_pattern: str, qualifier: str,
                         clinic: str | None, location: str | None) -> list[PriceRecord]:
    """'- Name: $2,600 USD' or '- Name: $6,200 - $7,200' list items under a matching heading."""
    records: list[PriceRecord] = []
    for sec in sections:
        if not re.search(heading_pattern, sec.heading, re.I):
            continue
        for line in sec.lines:
            if not line.startswith("- ") or ":" not in line:
                continue
            label, _, rest = line[2:].partition(":")
            rng = PRICE_RANGE_RE.search(rest)
            single = PRICE_RE.search(rest)
            if not (rng or single):
                continue
            if rng:
                lo, hi = rng.group(1), rng.group(2)
                hi = hi if hi.strip().startswith("$") else f"${hi.strip()}"
                price, price_high, text = parse_amount(lo), parse_amount(hi), rng.group(0).strip()
            else:
                price, price_high, text = parse_amount(single.group(0)), None, single.group(0).strip()
            records.append(PriceRecord(
                item=label.strip(), price=price, price_high=price_high, currency="USD", price_text=text,
                qualifier=qualifier, section=sec.heading, clinic=clinic, location=location,
                details=rest.strip() if len(rest.strip()) > len(text) + 4 else None))
    return records
