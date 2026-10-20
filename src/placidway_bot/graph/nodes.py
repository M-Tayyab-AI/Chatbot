"""Graph nodes. Each node reads the state and returns a partial update; routing lives in build.py."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from placidway_bot.config import Settings
from placidway_bot.graph import prompts, templates
from placidway_bot.graph.state import Analysis, ChatState, GroundedAnswer
from placidway_bot.guardrails.citations import citation_issues
from placidway_bot.guardrails.price_check import unsupported_prices
from placidway_bot.services.gap_log import GapLogger

log = logging.getLogger(__name__)
MAX_ATTEMPTS = 2                 # first answer + one rewrite after a failed fact check
HISTORY_CHARS = 700              # per earlier message given to the analyzer
# statuses that mean "the knowledge base could not answer" -> logged for content-gap review
GAP_STATUSES = {"not_found", "validation_failed", "out_of_scope", "medical_no_content", "error"}


def guess_language(text: str) -> str:
    """Cheap script-based guess, used only when the LLM analysis is unavailable."""
    if re.search(r"[ٹڈڑںھہے]", text):     # Urdu-only letters
        return "ur"
    if re.search(r"[؀-ۿ]", text):
        return "ar"
    if re.search(r"[¿¡ñáéíóú]", text.lower()) or re.search(r"(cuánto|cuesta|tratamiento|precio|qué)", text.lower()):
        return "es"
    return "en"


@dataclass
class Services:
    analyzer: object             # Runnable: messages -> Analysis
    answerer: object             # Runnable: messages -> GroundedAnswer
    retriever: object            # KnowledgeRetriever-like: .search(query) -> RetrievalResult
    gap_logger: GapLogger
    settings: Settings


class Nodes:
    def __init__(self, services: Services):
        self.s = services

    @property
    def links(self) -> dict:
        return {"quote": self.s.settings.quote_url, "contact": self.s.settings.contact_url}

    # ------------------------------------------------------------------ analyze: rewrite + language + route
    def analyze(self, state: ChatState) -> dict:
        question = state["question"]
        history = state.get("messages", [])[:-1][-self.s.settings.history_messages:]
        msgs = [SystemMessage(prompts.ANALYZE_SYSTEM)]
        for m in history:
            content = str(m.content)[:HISTORY_CHARS]
            msgs.append(HumanMessage(content) if m.type == "human" else AIMessage(content))
        msgs.append(HumanMessage(question))
        try:
            analysis: Analysis = self.s.analyzer.invoke(msgs)
        except Exception:  # noqa: BLE001 - degrade to a plain knowledge lookup rather than failing the turn
            log.exception("analyze failed; falling back to raw question")
            analysis = Analysis(search_query=question, language=guess_language(question), intent="knowledge")
        return {"search_query": analysis.search_query or question,
                "language": (analysis.language or "en").lower()[:2],
                "intent": analysis.intent, "asserted_price": analysis.asserted_price}

    # ------------------------------------------------------------------ retrieve
    def retrieve(self, state: ChatState) -> dict:
        result = self.s.retriever.search(state["search_query"])
        context = [dict(ref=f"S{i + 1}", chunk_id=h.chunk.chunk_id, text=h.chunk.text, source_url=h.chunk.source_url,
                        page_title=h.chunk.page_title, section=h.chunk.section, score=round(h.score, 4))
                   for i, h in enumerate(result.hits)]
        return {"context": context, "top_score": result.top_score, "confident": result.confident}

    # ------------------------------------------------------------------ generate
    def generate(self, state: ChatState) -> dict:
        prompt = prompts.answer_user_prompt(state["question"], state["search_query"], state["language"],
                                            state.get("asserted_price"), state["context"], state.get("issues", []))
        attempts = state.get("attempts", 0) + 1
        try:
            draft: GroundedAnswer = self.s.answerer.invoke([SystemMessage(prompts.ANSWER_SYSTEM), HumanMessage(prompt)])
        except Exception:  # noqa: BLE001
            log.exception("generate failed")
            return {"draft": None, "attempts": attempts, "status": "error"}
        return {"draft": draft.model_dump(), "attempts": attempts}

    # ------------------------------------------------------------------ validate (deterministic guardrails)
    def validate(self, state: ChatState) -> dict:
        draft = state.get("draft")
        if not draft or not draft["answer_found"]:
            return {"issues": []}
        refs = {c["ref"]: c for c in state["context"]}
        issues = citation_issues(draft["cited_sources"], list(refs), draft["answer_found"])
        cited_texts = [refs[r]["text"] for r in draft["cited_sources"] if r in refs]
        user_text = f"{state['question']} {state.get('asserted_price') or ''}"
        bad = unsupported_prices(draft["answer"], cited_texts, user_text)
        if bad:
            issues.append("These amounts are not in the sources you cited: "
                          + ", ".join(f"{b:,}" for b in bad) + ". Use only prices that appear in the cited sources.")
        return {"issues": issues}

    # ------------------------------------------------------------------ fixed replies (no retrieval / no answer)
    def respond_fixed(self, state: ChatState) -> dict:
        intent = state.get("intent", "knowledge")
        if intent in ("greeting", "out_of_scope", "booking", "quote_request"):
            status, key = intent, intent
        elif state.get("status") == "error":
            status, key = "error", "error"
        elif intent == "medical_advice":
            status, key = "medical_no_content", "medical"
        elif state.get("draft") and state.get("issues"):
            status, key = "validation_failed", "not_found"
        else:
            status, key = "not_found", "not_found"
        answer = templates.text(key, state.get("language", "en"), **self.links)
        update = {"status": status, "answer": answer, "citations": [],
                  "lead_offer": intent in ("booking", "quote_request"), "messages": [AIMessage(answer)]}
        if key == "medical" and not state.get("disclaimer_shown"):
            update["disclaimer_shown"] = True            # the medical reply already is the disclaimer
        self._log_gap(state, status)
        return update

    # ------------------------------------------------------------------ finalize a grounded answer
    def finalize(self, state: ChatState) -> dict:
        draft, lang = state["draft"], state.get("language", "en")
        refs = {c["ref"]: c for c in state["context"]}
        citations, seen = [], set()
        for r in draft["cited_sources"]:
            c = refs.get(r)
            if c and c["source_url"] not in seen:
                seen.add(c["source_url"])
                citations.append({"title": c["page_title"], "url": c["source_url"], "section": c["section"]})
        parts = [draft["answer"].strip()]
        status = "answered" if draft["fully_answered"] else "partial"
        if status == "partial":
            parts.append(templates.text("partial_cta", lang, **self.links))
        update: dict = {}
        if (draft["medical_topic"] or state.get("intent") == "medical_advice") and not state.get("disclaimer_shown"):
            parts.append(templates.text("disclaimer", lang, **self.links))
            update["disclaimer_shown"] = True
        sources = "\n".join(f"- [{c['title']}]({c['url']})" + (f" — {c['section']}" if c["section"] else "")
                            for c in citations)
        parts.append(f"**{templates.text('sources', lang)}:**\n{sources}")
        answer = "\n\n".join(parts)
        return update | {"status": status, "answer": answer, "citations": citations,
                         "lead_offer": state.get("intent") == "quote_request", "messages": [AIMessage(answer)]}

    # ------------------------------------------------------------------ helpers
    def _log_gap(self, state: ChatState, status: str) -> None:
        if status in GAP_STATUSES:
            self.s.gap_logger.log(session_id=state.get("session_id", ""), question=state["question"],
                                  search_query=state.get("search_query", ""), reason=status,
                                  language=state.get("language", "en"), top_score=state.get("top_score"))
