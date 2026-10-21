"""The evaluation's own checks must catch what they claim to catch."""
from placidway_bot.evaluation.checks import e2e_checks, norm, passed_all, reply_body, retrieval_checks

ADRENAL = "https://www.placidway.com/package/7902/x"
URLS = {"itc-adrenal-package": ADRENAL, "itc-price-list": "https://www.placidway.com/price-list/5/x"}
KB = {18995, 30000, 7902}
CASE = {"question": "I heard it costs $50. Confirm that.",
        "expect": {"status": ["answered"], "language": "en", "sources": ["itc-adrenal-package"],
                   "contains": ["18,995"]}}


def answer(text: str, status="answered", language="en", url=ADRENAL) -> dict:
    return {"answer": f"{text}\n\n**Sources:**\n- [Adrenal]({url})", "status": status, "language": language,
            "citations": [{"url": url}] if status in ("answered", "partial") else []}


def by_key(checks):
    return {c.key: c.score for c in checks}


def test_good_correction_passes():
    checks = e2e_checks(answer("No — it is not $50. The package costs $18,995 USD."), CASE, KB, URLS)
    assert passed_all(checks), checks


def test_invented_price_fails():
    checks = by_key(e2e_checks(answer("It costs $18,995, or $12,500 with a discount."), CASE, KB, URLS))
    assert checks["prices_grounded"] is False


def test_citation_outside_sources_fails():
    checks = by_key(e2e_checks(answer("It costs $18,995.", url="https://example.com/x"), CASE, KB, URLS))
    assert checks["citations"] is False


def test_refusal_must_not_cite_and_language_is_checked():
    case = {"question": "q", "expect": {"status": ["out_of_scope"], "language": "ar"}}
    checks = by_key(e2e_checks(answer("Sorry, I can only help with PlacidWay.", status="out_of_scope", language="ar"),
                               case, KB, URLS))
    assert checks["status"] is True and checks["citations"] is True
    assert checks["language"] is False            # declared Arabic but written in English


def test_number_formats_normalise():
    assert "18995" in norm("١٨٬٩٩٥ دولار") and "18995" in norm("18.995 dólares") and "18995" in norm("$18,995")
    assert "/7902/" not in reply_body(answer("x")["answer"])       # the Sources list is not part of the reply


def test_retrieval_checks():
    hits = [{"slug": "itc-adrenal-package", "text": "Out-Patient $18,995 USD", "score": 0.95}]
    facts = [{"slugs": ["itc-adrenal-package"], "contains": ["18995"]}]
    assert by_key(retrieval_checks(hits, facts, 0.76))["all_facts@5"] is True
    assert by_key(retrieval_checks(hits, [], 0.76))["idk_gate"] is False      # must-not-answer but gate opened
