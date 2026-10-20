"""Score the three ingestion methods against each other and write docs/scraper_comparison.md.

Reference ("gold") = what a visitor actually sees: the visible <main> text captured in a real Chrome session
(data/raw/chrome/*.txt, verified byte-for-byte against an in-browser SHA-256).

Usage:  python scripts/compare_scrapers.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from placidway_bot.config import DATA_DIR, RAW_DIR, ROOT_DIR, load_sources
from placidway_bot.ingest.parsers import base

METHODS = ["requests", "selenium"]
CHROME_DIR = RAW_DIR / "chrome"
OUT = ROOT_DIR / "docs" / "scraper_comparison.md"

# Visible UI chrome that is not knowledge (tab labels, CTA buttons, like counters, related-link lists).
GOLD_NOISE = re.compile(
    r"^(profile|details|treatments|qualifications|testimonials|awards|destination|reviews|location|recent reviews"
    r"|contact (placidway|itc)[\w ]*|speak with a specialist today|get a free quote|get your quote now!"
    r"|write a review|view center reviews|chat with center|\d{1,5}|0\d)$", re.I)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$][a-z0-9$,.'’-]*", text.lower().replace("\t", " "))


UI_PHRASES = re.compile(r"\b(chat with center|quote|actions?)\b|★+|\t", re.I)


def word_set(text: str) -> set[str]:
    return {w.strip(".,:;!?()\"'’") for w in words(text)} - {""}


def line_coverage(gold: str, text: str) -> tuple[float, list[str]]:
    """Share of reference lines whose words (>=90%) all appear in the method's output. Robust to reformatting
    (tables become 'Header: value' rows), still catches content that is genuinely missing."""
    have = word_set(text)
    lines = [UI_PHRASES.sub(" ", l) for l in gold.splitlines()]
    lines = [l for l in lines if word_set(l)]
    missing = [l for l in lines if len(word_set(l) & have) / len(word_set(l)) < 0.9]
    return 100 * (1 - len(missing) / max(1, len(lines))), missing


def price_set(text: str) -> set[int]:
    return {int(m.group(1).replace(",", "")) for m in base.PRICE_RE.finditer(text)}


def gold_text(slug: str) -> str:
    lines, skip = [], False
    for line in (CHROME_DIR / f"{slug}.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("Related Experiences"):
            skip = True                      # randomised cross-links to other pages, not this page's content
        elif skip and re.fullmatch(r"\d+", line):
            skip = False
            continue
        if not skip and not GOLD_NOISE.match(line.strip()):
            lines.append(line)
    return "\n".join(lines)


def doc_text(doc: dict) -> str:
    # section paths are kept as chunk metadata, so their headings count as captured
    parts = [" / ".join(s["path"]) + f"\n{s['heading']}\n" + "\n".join(s["lines"]) for s in doc["sections"]]
    parts += [f"{v['title']} {v['category'] or ''} {v['description']}" for v in doc["videos"]]
    return "\n".join(parts)


def outside_main_prices(slug: str) -> set[int]:
    """Prices that exist on the raw page only outside <main> and outside this clinic's own sidebar
    (i.e. other clinics' widgets) = leak candidates."""
    soup = base.make_soup((RAW_DIR / "requests" / f"{slug}.html").read_text(encoding="utf-8"))
    main = soup.select_one(base.MAIN_SELECTOR)
    inside = price_set(main.get_text(" ")) if main else set()
    aside = soup.select_one(base.ASIDE_SELECTOR)
    if aside is not None:
        inside |= price_set(aside.get_text(" "))
    for s in soup.find_all("script"):
        inside |= price_set(s.get_text(" "))       # inline data / JSON-LD for this page
    main.decompose()
    for s in soup.find_all("script"):
        s.decompose()
    return price_set(soup.get_text(" ")) - inside


def non_latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return 100 * sum(1 for c in letters if "؀" <= c <= "ۿ") / max(1, len(letters))


def evaluate() -> dict:
    results: dict = {m: {} for m in METHODS + ["chrome"]}
    timings = {m: json.loads((RAW_DIR / m / "_timings.json").read_text()) for m in METHODS}
    for src in load_sources():
        gold = gold_text(src.slug)
        gold_prices = price_set(gold)
        foreign = outside_main_prices(src.slug)
        results["chrome"][src.slug] = dict(coverage=100.0, missing_lines=[], price_recall=100.0, leaked=[], non_latin=0.0,
                                           extra_prices=[], faqs=0, faq_answers=0, price_records=0, videos_with_desc=0,
                                           seconds=None)
        for m in METHODS:
            doc = json.loads((DATA_DIR / "experiments" / m / f"{src.slug}.json").read_text(encoding="utf-8"))
            text = doc_text(doc)
            prices = price_set(text) | {int(p["price"]) for p in doc["prices"]}
            coverage, missing = line_coverage(gold, text)
            results[m][src.slug] = dict(
                coverage=coverage, missing_lines=missing,
                price_recall=100 * len(gold_prices & prices) / max(1, len(gold_prices)) if gold_prices else 100.0,
                leaked=sorted(prices & foreign - gold_prices),
                extra_prices=sorted(prices - gold_prices - foreign),
                non_latin=non_latin_share(text),
                faqs=len(doc["faqs"]), faq_answers=sum(1 for f in doc["faqs"] if f["answer"].strip()),
                price_records=len(doc["prices"]),
                videos_with_desc=sum(1 for v in doc["videos"] if v["description"]),
                seconds=timings[m].get(src.slug),
            )
    return results


def render(results: dict) -> str:
    slugs = [s.slug for s in load_sources()]
    out = ["# Scraper comparison", "",
           "Three ingestion methods were run on the same 7 pages and scored against a **reference copy**: the visible",
           "main-column text captured in a real Chrome session (method C), verified byte-for-byte with an in-browser",
           "SHA-256. Generated by `scripts/compare_scrapers.py`.", "",
           "| Method | How |", "|---|---|",
           "| **A. requests + BeautifulSoup** | HTTP GET of server HTML (robots.txt, UA, 1 req/s) → html5lib DOM → page parsers |",
           "| **B. Selenium + BeautifulSoup** | Headless Chrome renders JS → `page_source` → same page parsers |",
           "| **C. Claude-in-Chrome** | Agent opens each page in a real browser and copies the visible text |", ""]
    out += ["## Per-page results", "",
            "Coverage = share of the reference's visible lines whose words are present in the method's clean output. "
            "Leaked = other clinics' prices that got in. Non-Latin = share of text machine-translated by the site.", "",
            "| Page | Method | Coverage | Price recall | Leaked prices | Non-Latin text | FAQ answers | Price records | Secs |",
            "|---|---|---|---|---|---|---|---|---|"]
    for slug in slugs:
        for m in METHODS:
            r = results[m][slug]
            out.append(f"| {slug} | {m} | {r['coverage']:.1f}% | {r['price_recall']:.0f}% | "
                       f"{', '.join(f'${p:,}' for p in r['leaked']) or 'none'} | {r['non_latin']:.0f}% | "
                       f"{r['faq_answers']}/{r['faqs']} | {r['price_records']} | {r['seconds']} |")
    out += ["", "## Totals", "", "| Method | Avg coverage | Pages with translated text | Leaks | FAQ answers | "
            "Video descriptions | Structured price records | Total fetch time |", "|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        rs = results[m].values()
        out.append(f"| {m} | {sum(r['coverage'] for r in rs) / len(slugs):.1f}% | "
                   f"{sum(1 for r in rs if r['non_latin'] > 1)}/7 | {sum(len(r['leaked']) for r in rs)} | "
                   f"{sum(r['faq_answers'] for r in rs)} | {sum(r['videos_with_desc'] for r in rs)} | "
                   f"{sum(r['price_records'] for r in rs)} | {sum(r['seconds'] or 0 for r in rs):.1f}s |")
    out.append("| chrome | 100% (is the reference) | 0/7 | 0 | 0 (collapsed `<details>` answers are invisible) "
               "| 0 (only in JSON-LD) | 0 (flat text) | manual, agent-driven |")
    return "\n".join(out) + "\n"


def main() -> None:
    results = evaluate()
    (DATA_DIR / "experiments" / "scraper_scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    table = render(results)
    marker = "<!-- AUTO-GENERATED ABOVE -->"
    tail = OUT.read_text(encoding="utf-8").split(marker, 1)[1] if OUT.exists() and marker in OUT.read_text(encoding="utf-8") else ""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(table + "\n" + marker + tail, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
