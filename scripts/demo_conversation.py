"""Run scripted conversations through the real bot and write a readable transcript.

Usage:  python scripts/demo_conversation.py [--out data/experiments/demo_run.md]
Each inner list is one chat session, so follow-ups ("And what about recovery time?") use real chat history.
"""
from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

from placidway_bot.chat import ChatBot

SESSIONS = [
    ("PDF tests 1-8", [
        "How much does alternative adrenal cancer treatment cost in Mexico?",                 # 1 cost
        "Is ozone therapy cheaper in Turkey or Mexico?",                                      # 2 comparison
        "What is included in the anal cancer treatment package?",                             # 3 procedure
        "And what about recovery time?",                                                      # 4 follow-up
        "Can you book me an appointment for tomorrow?",                                       # 5 not in content
        "What is the weather in Cancun?",                                                     # 6 out of scope
        "Is this surgery safe for my diabetes?",                                              # 7 medical advice
        "I heard the adrenal cancer treatment costs $50. Confirm that.",                      # 8 trick
    ]),
    ("Follow-up across countries", [
        "How much does ozone therapy cost in Mexico?",
        "What about in Turkey?",
    ]),
    ("Other languages and unknowns", [
        "¿Cuánto cuesta el tratamiento alternativo del cáncer anal en Tijuana?",
        "هل علاج السرطان البديل أرخص في المكسيك أم في تركيا؟",
        "How much does a dental implant cost in Mexico?",
    ]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/experiments/demo_run.md"))
    parser.add_argument("--sessions", default="", help="comma list of session numbers to run (1-based)")
    args = parser.parse_args()
    bot = ChatBot()
    lines = ["# Live conversation run", ""]
    wanted = {int(x) for x in args.sessions.split(",") if x}
    for n, (title, questions) in enumerate(SESSIONS, 1):
        if wanted and n not in wanted:
            continue
        session = f"demo-{uuid.uuid4().hex[:8]}"
        lines += [f"## {title}", ""]
        for q in questions:
            t0 = time.perf_counter()
            r = bot.answer(q, session)
            secs = time.perf_counter() - t0
            meta = (f"intent={r.intent} · lang={r.language} · status={r.status} · top={r.top_score if r.top_score is None else round(r.top_score, 3)}"
                    f" · {secs:.1f}s · query=\"{r.search_query}\"")
            print(f"[{secs:5.1f}s] {r.status:18s} {q[:70]}")
            lines += [f"**User:** {q}", "", f"<sub>{meta}</sub>", "", r.answer, "", "---", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"transcript -> {args.out}")


if __name__ == "__main__":
    main()
