"""Deterministic evaluation checks (no LLM judge): used by scripts/run_eval.py as LangSmith evaluators.

End-to-end checks look at the bot's final reply for one test case:
    status      the graph took the expected path (answered / partial / booking / out_of_scope / medical / not_found)
    prices      every money amount in the reply appears somewhere in the knowledge base or in the user's question
    citations   cited pages are among the 7 sources and include an expected page; non-answers cite nothing
    facts       required strings ("18,995", "30,000") are in the reply (thousands separators / digit scripts ignored)
    language    the reply is in the language the user wrote in
    lead_offer  booking / quote turns offer the lead form
Retrieval checks score the hybrid + rerank result against eval/retrieval_labels.yaml facts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from placidway_bot.guardrails.price_check import all_numbers, money_amounts, normalise

ANSWER_STATUSES = {"answered", "partial"}
ARABIC_SCRIPT = re.compile(r"[؀-ۿ]")


@dataclass
class Check:
    key: str
    score: float | bool | None   # pass/fail, or a fraction for retrieval metrics; None = not applicable
    comment: str = ""

    @property
    def failed(self) -> bool:
        return self.score is not None and float(self.score) < 1.0


def norm(text: str) -> str:
    """Lowercase, Western digits, no thousands separators: '$18,995' / '١٨٬٩٩٥' -> '$18995' / '18995'."""
    text = normalise(text).lower()
    text = re.sub(r"(?<=\d)[,.٬ ](?=\d{3}\b)", "", text)
    return re.sub(r"\s+", " ", text)


def reply_body(answer: str) -> str:
    """The reply without its trailing Sources list (links carry ids such as /package/7903 that are not prices)."""
    return re.split(r"\n\*\*[^\n*]+:\*\*\n- \[", answer, maxsplit=1)[0]


# ---------------------------------------------------------------------------------------------- end-to-end checks
def check_status(outputs: dict, expect: dict) -> Check:
    want = expect["status"]
    return Check("status", outputs.get("status") in want, f"got {outputs.get('status')}, expected {' or '.join(want)}")


def check_prices(outputs: dict, question: str, kb_numbers: set[int]) -> Check:
    amounts = money_amounts(reply_body(outputs.get("answer", "")))
    invented = sorted(a for a in amounts if a not in kb_numbers and a not in all_numbers(question))
    return Check("prices_grounded", not invented,
                 f"not on any page: {', '.join(f'{a:,}' for a in invented)}" if invented
                 else f"{len(amounts)} amount(s), all on the pages")


def check_citations(outputs: dict, expect: dict, urls_by_slug: dict[str, str]) -> Check:
    cited = [c["url"] for c in outputs.get("citations", [])]
    allowed = set(urls_by_slug.values())
    if outside := [u for u in cited if u not in allowed]:
        return Check("citations", False, f"cites a URL outside the 7 sources: {outside}")
    if outputs.get("status") not in ANSWER_STATUSES:
        return Check("citations", not cited, "no answer, no citations" if not cited else "non-answer cites sources")
    if not cited:
        return Check("citations", False, "answer without a source")
    wanted = {urls_by_slug[s] for s in expect.get("sources", [])}
    if wanted and not wanted & set(cited):
        return Check("citations", False, f"none of the expected pages cited: {sorted(expect['sources'])}")
    return Check("citations", True, f"{len(cited)} source(s)")


def check_facts(outputs: dict, expect: dict) -> Check:
    needed = expect.get("contains", [])
    if not needed:
        return Check("facts", None, "n/a")
    body = norm(reply_body(outputs.get("answer", "")))
    missing = [s for s in needed if not any(norm(alt) in body for alt in s.split("|"))]   # "a|b" = either spelling
    return Check("facts", not missing, f"missing: {missing}" if missing else f"all {len(needed)} present")


def check_language(outputs: dict, expect: dict) -> Check:
    want, got = expect.get("language", "en"), outputs.get("language", "")
    has_arabic = bool(ARABIC_SCRIPT.search(reply_body(outputs.get("answer", ""))))
    script_ok = has_arabic if want in ("ar", "ur") else not has_arabic
    return Check("language", got == want and script_ok, f"detected {got}, expected {want}")


def check_lead_offer(outputs: dict, expect: dict) -> Check:
    if "lead_offer" not in expect:
        return Check("lead_offer", None, "n/a")
    return Check("lead_offer", bool(outputs.get("lead_offer")) == expect["lead_offer"],
                 f"lead form offered: {bool(outputs.get('lead_offer'))}")


def e2e_checks(outputs: dict, case: dict, kb_numbers: set[int], urls_by_slug: dict[str, str]) -> list[Check]:
    expect = case["expect"]
    return [check_status(outputs, expect), check_prices(outputs, case["question"], kb_numbers),
            check_citations(outputs, expect, urls_by_slug), check_facts(outputs, expect),
            check_language(outputs, expect), check_lead_offer(outputs, expect)]


def passed_all(checks: list[Check]) -> bool:
    return not any(c.failed for c in checks)


# ---------------------------------------------------------------------------------------------- retrieval checks
def satisfies(chunk: dict, fact: dict) -> bool:
    """A retrieved chunk proves a labelled fact when it is from an expected page and holds every required string."""
    return chunk["slug"] in fact["slugs"] and all(norm(s) in norm(chunk["text"]) for s in fact["contains"])


def retrieval_checks(hits: list[dict], facts: list[dict], threshold: float, k: int = 5) -> list[Check]:
    top1 = hits[0]["score"] if hits else 0.0
    if not facts:                                  # must stay unanswered: the gate has to say "I don't know"
        return [Check("idk_gate", top1 < threshold, f"top-1 {top1:.3f} must be below {threshold}")]
    found = [any(satisfies(h, f) for h in hits[:k]) for f in facts]
    first = next((r + 1 for r, h in enumerate(hits) if any(satisfies(h, f) for f in facts)), None)
    return [Check(f"fact_recall@{k}", sum(found) / len(found), f"{sum(found)}/{len(found)} facts in top-{k}"),
            Check(f"all_facts@{k}", all(found)),
            Check(f"mrr@{len(hits)}", 1 / first if first else 0.0, f"first relevant rank: {first}"),
            Check("idk_gate", top1 >= threshold, f"top-1 {top1:.3f} must reach {threshold}")]
