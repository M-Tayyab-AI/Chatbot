"""Deterministic price guardrail: every money amount in an answer must exist in the cited source chunks.

Works across languages: "$18,995", "18.995 USD" (Spanish), "١٨٬٩٩٥ دولار" (Arabic-Indic digits) all normalise to 18995.
Amounts the user typed themselves are allowed so the bot can *correct* a claim ("it is not $50, it is $18,995").
"""
from __future__ import annotations

import re

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
CURRENCY = r"(?:US\$|\$|€|USD|EUR|dollars?|d[oó]lares|دولار\w*|ڈالر)"
NUMBER = r"\d{1,3}(?:[,.\u066c\u00a0]\d{3})+|\d+"
MONEY = re.compile(rf"{CURRENCY}\s?(?P<a>{NUMBER})|(?P<b>{NUMBER})\s?{CURRENCY}", re.IGNORECASE)
BARE_NUMBER = re.compile(rf"(?<![\w/]){NUMBER}(?![\w/])")


def _value(raw: str) -> int:
    return int(re.sub(r"[,.\u066c\u00a0]", "", raw))


def normalise(text: str) -> str:
    return text.translate(ARABIC_DIGITS)


def money_amounts(text: str) -> set[int]:
    """Currency-attached amounts plus bare numbers >= 1000 that are not plausible years."""
    text = normalise(text)
    amounts = {_value(m.group("a") or m.group("b")) for m in MONEY.finditer(text)}
    for m in BARE_NUMBER.finditer(text):
        value = _value(m.group(0))
        if value >= 1000 and not 1900 <= value <= 2100:
            amounts.add(value)
    return amounts


def all_numbers(text: str) -> set[int]:
    return {_value(m.group(0)) for m in BARE_NUMBER.finditer(normalise(text))}


def unsupported_prices(answer: str, source_texts: list[str], user_text: str = "") -> list[int]:
    """Amounts in `answer` that appear in neither the cited sources nor the user's own message."""
    allowed = set().union(*(all_numbers(t) for t in source_texts)) if source_texts else set()
    allowed |= all_numbers(user_text)
    return sorted(a for a in money_amounts(answer) if a not in allowed)
