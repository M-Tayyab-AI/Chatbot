from __future__ import annotations

from placidway_bot.ingest.chunker import chunk_recursive, chunk_structured
from placidway_bot.ingest.parsers.pages import parse_page


def test_structured_chunks_carry_context_header_and_source(package_html, package_source):
    chunks = chunk_structured(parse_page(package_html, package_source, "requests"))
    assert chunks
    for c in chunks:
        assert c.text.startswith("[Treatment package: Alternative Adrenal Cancer Treatment")
        assert "Tijuana, Mexico" in c.text.splitlines()[0]
        assert c.source_url == package_source.url and c.page_title == "Adrenal Package Title"


def test_each_price_is_an_atomic_chunk_with_currency_and_context(package_html, package_source):
    prices = [c for c in chunk_structured(parse_page(package_html, package_source, "requests")) if c.kind == "price"]
    out_patient = next(c for c in prices if "Out-Patient Program" in c.text)
    assert "$18,995 USD" in out_patient.text and "Total Cost (USD)" in out_patient.text
    assert "Duration: 3 Weeks" in out_patient.text


def test_faq_question_and_answer_in_one_chunk(package_html, package_source):
    faq = [c for c in chunk_structured(parse_page(package_html, package_source, "requests")) if c.kind == "faq"]
    assert len(faq) == 1
    assert "Q: 1. Is English spoken at the clinic?" in faq[0].text and "A: Yes, all staff speak English." in faq[0].text


def test_recursive_respects_size_limit(package_html, package_source):
    doc = parse_page(package_html, package_source, "requests")
    assert all(len(c.text) <= 1000 for c in chunk_recursive(doc))
    assert all(len(c.text) <= 120 for c in chunk_recursive(doc, size=120, overlap=20))
