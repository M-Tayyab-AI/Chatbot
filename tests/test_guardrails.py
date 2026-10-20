from __future__ import annotations

from placidway_bot.guardrails.citations import citation_issues
from placidway_bot.guardrails.price_check import money_amounts, unsupported_prices

SOURCE = "Out-Patient Program | Duration: 3 Weeks | Total Cost (USD): $18,995. In-Patient Program $30,000."


def test_amounts_across_formats_and_languages():
    assert money_amounts("Cuesta 18.995 USD; el internado $30,000") == {18995, 30000}
    assert money_amounts("التكلفة ١٨٬٩٩٥ دولار") == {18995}
    assert money_amounts("قیمت 18,995 ڈالر ہے") == {18995}


def test_non_prices_are_ignored():
    assert money_amounts("3 weeks, 24/7 care, 5 sessions, reviewed in 2026") == set()


def test_sourced_prices_pass_and_invented_prices_fail():
    assert unsupported_prices("The Out-Patient Program costs $18,995 USD.", [SOURCE]) == []
    assert unsupported_prices("It costs about $17,500.", [SOURCE]) == [17500]
    assert unsupported_prices("Both programs total $48,995.", [SOURCE]) == [48995]   # no adding up


def test_correcting_a_user_claim_is_allowed():
    answer = "That doesn't match the page: it is not $50, the Out-Patient Program is $18,995 USD."
    assert unsupported_prices(answer, [SOURCE], user_text="I heard it costs $50") == []


def test_citations_must_exist_and_be_present():
    assert citation_issues(["S1"], ["S1", "S2"], True) == []
    assert citation_issues(["S9"], ["S1", "S2"], True)
    assert citation_issues([], ["S1"], True)
    assert citation_issues([], ["S1"], False) == []
