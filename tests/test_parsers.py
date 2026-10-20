from __future__ import annotations

import re

from placidway_bot.config import Source
from placidway_bot.ingest.parsers.pages import parse_page

TAG = re.compile(r"<[a-zA-Z/][^>]*>")


def all_text(doc) -> str:
    return "\n".join([s.heading + "\n" + s.text for s in doc.sections] + [p.price_text + " " + (p.details or "")
                                                                          for p in doc.prices])


def test_output_is_plain_text_without_noise(package_html, package_source):
    text = all_text(parse_page(package_html, package_source, "requests"))
    assert not TAG.search(text)
    for noise in ("Book a consultation", "Request your Free Quote", "Related Experiences", "Dental Implants", "1267"):
        assert noise not in text


def test_other_clinics_prices_never_leak(package_html, package_source):
    doc = parse_page(package_html, package_source, "requests")
    assert "24,000" not in all_text(doc)
    assert all(p.price != 24000 for p in doc.prices)


def test_unclosed_paragraph_does_not_swallow_headings(package_html, package_source):
    headings = [s.heading for s in parse_page(package_html, package_source, "requests").sections]
    assert "Cost of Treatment" in headings
    assert "Optional Specialized Procedures (Additional Cost)" in headings


def test_header_metadata_and_starting_price(package_html, package_source):
    doc = parse_page(package_html, package_source, "requests")
    assert doc.location == "Tijuana, Mexico" and doc.published == "Mar 02, 2026"
    first = doc.prices[0]
    assert (first.price, first.qualifier) == (18995, "Price starting from")


def test_table_price_keeps_column_context_and_note(package_html, package_source):
    rec = next(p for p in parse_page(package_html, package_source, "requests").prices if p.item == "Out-Patient Program")
    assert rec.price == 18995 and rec.currency == "USD" and rec.qualifier == "Total Cost (USD)"
    assert "Duration: 3 Weeks" in rec.details and "Note: The In-Patient program" in rec.details


def test_list_prices_single_range_and_excluded(package_html, package_source):
    prices = {p.item: p for p in parse_page(package_html, package_source, "requests").prices}
    assert prices["LAK"].price == 2600
    assert (prices["Intratumoral Immunotherapy"].price, prices["Intratumoral Immunotherapy"].price_high) == (6200, 7200)
    assert prices["Intratumoral Immunotherapy"].price_text == "$6,200 - $7,200 USD"
    assert prices["Blood Products"].price == 462 and "NOT included" in prices["Blood Products"].qualifier


def test_faq_question_and_answer_kept_together(package_html, package_source):
    faqs = parse_page(package_html, package_source, "requests").faqs
    assert [(f.question, f.answer) for f in faqs] == [("Is English spoken at the clinic?", "Yes, all staff speak English.")]


def test_review_attributed_to_the_right_person(package_html, package_source):
    text = all_text(parse_page(package_html, package_source, "requests"))
    assert "Patient review by Jana C (Jul 05 2023, 5/5): Great care." in text


def test_content_hash_ignores_fetch_time_but_tracks_content(package_html, package_source):
    a = parse_page(package_html, package_source, "requests", fetched_at="2026-01-01T00:00:00+00:00")
    b = parse_page(package_html, package_source, "requests", fetched_at="2026-02-02T00:00:00+00:00")
    changed = parse_page(package_html.replace("$18,995", "$19,500"), package_source, "requests")
    assert a.content_hash == b.content_hash
    assert changed.content_hash != a.content_hash


def test_price_list_missing_name_filled_from_jsonld(price_list_html):
    src = Source(slug="test-prices", page_type="price_list", url="https://example.test/prices", clinic="ITC")
    doc = parse_page(price_list_html, src, "requests")
    assert [p.item for p in doc.prices] == ["Breast Cancer", "Cancer Treatment"]
    assert any("filled from JSON-LD" in w for w in doc.warnings)
    assert all("Chat with Center" not in s.text for s in doc.sections)
