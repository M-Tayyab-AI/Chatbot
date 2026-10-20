"""Data contract for the real processed knowledge base (data/processed/*.json)."""
from __future__ import annotations

import re

import pytest

from placidway_bot.config import PROCESSED_DIR, load_sources
from placidway_bot.ingest.pipeline import load_processed

pytestmark = pytest.mark.skipif(not any(PROCESSED_DIR.glob("*.json")), reason="run scripts/ingest.py first")
TAG = re.compile(r"<[a-zA-Z/][^>]*>")
OTHER_CLINIC_PRICES = {24000, 33400, 4500}       # seen only in related-listing widgets around ITC pages


@pytest.fixture(scope="module")
def docs():
    return {d.slug: d for d in load_processed()}


def _text(doc) -> str:
    return "\n".join([s.heading + "\n" + s.text for s in doc.sections] + [v.title + v.description for v in doc.videos])


def test_all_seven_pages_present(docs):
    assert set(docs) == {s.slug for s in load_sources()}


def test_plain_english_text_only(docs):
    for doc in docs.values():
        text = _text(doc)
        assert not TAG.search(text), doc.slug
        assert not re.search(r"[؀-ۿ]", text), f"{doc.slug} contains machine-translated text"


def test_no_other_clinic_prices_on_itc_pages(docs):
    for slug, doc in docs.items():
        if doc.clinic:
            assert not {int(p.price) for p in doc.prices} & OTHER_CLINIC_PRICES, slug


def test_package_prices(docs):
    for slug in ("itc-adenocarcinoma-package", "itc-adrenal-package", "itc-anal-package"):
        prices = {(p.item, p.price) for p in docs[slug].prices}
        assert ("Out-Patient Program", 18995) in prices and ("In-Patient Program", 30000) in prices
        assert len(docs[slug].faqs) == 8


def test_price_list_and_comparison(docs):
    assert [p.item for p in docs["itc-price-list"].prices].count("Cancer Treatment") == 1
    compare = {(p.country, p.item, p.price) for p in docs["alternative-medicine-price-comparison"].prices if p.country}
    assert ("Turkey", "Ozone Therapy", 110) in compare and ("Mexico", "Ozone Therapy", 1000) in compare
    assert len(docs["alternative-medicine-price-comparison"].faqs) == 12


def test_videos(docs):
    videos = docs["itc-clinic-videos"].videos
    assert len(videos) == 124 and all(v.youtube_url for v in videos)
