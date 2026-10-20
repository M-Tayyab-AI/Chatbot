from __future__ import annotations

import hashlib

import pytest

from placidway_bot.config import Source

PACKAGE_HTML = """<html><head><title>Adrenal Package Title</title></head><body>
<div class="pack-title-price-header"><h1>Adrenal Cancer Package in Tijuana</h1><div>Tijuana, Mexico</div>
  <div>Publish date: Mar 02, 2026</div><span>Package Price</span><span>{price}</span><span>Price starting from</span></div>
<div class="row"><div class="col-md-8"><main class="article-dt-desc">
  <p>Intro paragraph that is never closed
  <h2>Cost of Treatment</h2>
  <table><tr><th>Program Type</th><th>Duration</th><th>Total Cost (USD)</th></tr>
         <tr><td>Out-Patient Program</td><td>3 Weeks</td><td>{price}</td></tr></table>
  <p>Note: The In-Patient program is for 24/7 supervision.</p>
  <h2>Optional Specialized Procedures (Additional Cost)</h2>
  <ul><li><p><strong>LAK:</strong> $2,600 USD</p></li><li><p><strong>Intratumoral Immunotherapy:</strong> $6,200 - $7,200 USD</p></li></ul>
  <h2>Excluded Services and Contingency Costs</h2>
  <ul><li>Blood Products: Blood transfusions if required (Approx. $462 per pint).</li></ul>
  <h2>FAQs: Adrenal Treatment</h2>
  <h3>1. Is English spoken at the clinic?</h3><p>Yes, all staff speak English.</p>
  <h3>Related Experiences:</h3><a class="custom_link" href="/x">All on 6 Dental Implants in Tijuana, Mexico</a>
  <h2>Reviews</h2>
  <div class="review-card"><div class="star-rating" data-rating="5"></div><p class="review-body">Great care.</p>
       <h3 class="reviewer-name">Jana C</h3><span class="review-date">Jul 05 2023</span></div>
  <div id="book_consultation_popup" class="social-popup-main"><h2>Book a consultation</h2></div>
  <div class="article-social-section">1267</div>
</main></div>
<div class="col-md-4"><section id="cta-form"><h2>Request your Free Quote</h2></section></div></div>
<div class="related-widgets-row"><div class="exp-card"><h3>Other clinic package</h3><span>$24,000</span></div></div>
</body></html>"""

PRICE_LIST_HTML = """<html><head><title>Price list</title></head><body><h1>Prices at ITC</h1>
<main class="article-dt-desc"><section id="prices"><h2>Pricings:</h2><table>
<thead><tr><th>Procedure</th><th>Details</th><th>Price in USD</th><th>Actions</th></tr></thead>
<tbody><tr><td>Breast Cancer</td><td>Breast Cancer</td><td>$18995</td><td>Chat with Center</td></tr>
<tr><td></td><td></td><td>$18995</td><td>Chat with Center</td></tr></tbody></table></section></main>
<script type="application/ld+json">{"@type":"Product","name":"Breast Cancer","offers":{"price":"18995"}}</script>
<script type="application/ld+json">{"@type":"Product","name":"Cancer Treatment","offers":{"price":"18995"}}</script>
</body></html>"""


@pytest.fixture
def package_source() -> Source:
    return Source(slug="test-package", page_type="package", url="https://example.test/package",
                  treatment="Alternative Adrenal Cancer Treatment", clinic="ITC Immunity Therapy Center")


@pytest.fixture
def package_html() -> str:
    return PACKAGE_HTML.format(price="$18,995")


@pytest.fixture
def price_list_html() -> str:
    return PRICE_LIST_HTML


class FakeVoyage:
    """Deterministic offline stand-in for VoyageService."""

    def __init__(self):
        self.embedded: list[str] = []

    def embed(self, texts, input_type):
        self.embedded += texts
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]

    def embed_query(self, text):
        return self.embed([text], "query")[0]


@pytest.fixture
def fake_voyage() -> FakeVoyage:
    return FakeVoyage()
