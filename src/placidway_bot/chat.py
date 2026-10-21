"""Public entry point: `answer(question, session_id)`. The UI calls only ChatBot / answer()."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langgraph.checkpoint.memory import InMemorySaver

from placidway_bot.config import get_settings
from placidway_bot.graph.build import build_graph
from placidway_bot.graph.nodes import Services
from placidway_bot.graph.state import Analysis, GroundedAnswer

log = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    answer: str                          # markdown, includes sources / disclaimer / CTA
    citations: list[dict] = field(default_factory=list)   # [{title, url, section}]
    status: str = ""                     # answered | partial | not_found | out_of_scope | booking | ...
    intent: str = ""
    language: str = "en"
    lead_offer: bool = False             # UI shows the name/email form
    top_score: float | None = None
    search_query: str = ""


@dataclass
class Runtime:
    services: Services
    checkpointer: object
    lead_store: object
    persistent: bool                     # True when chats/leads/gaps live in Neon


def build_runtime() -> Runtime:
    """Real services. Uses Neon when DATABASE_URL is set; otherwise in-memory chats and local JSONL logs."""
    from placidway_bot.llm import configure_tracing, structured_model
    from placidway_bot.retrieval.retriever import KnowledgeRetriever
    from placidway_bot.services.gap_log import JsonlGapLogger
    from placidway_bot.services.leads import JsonlLeadStore

    settings = get_settings()
    configure_tracing(settings)
    checkpointer, gap_logger, lead_store, persistent = InMemorySaver(), JsonlGapLogger(), JsonlLeadStore(), False
    if settings.database_url:
        try:
            from placidway_bot.services.db import NeonGapLogger, NeonLeadStore, get_pool, make_checkpointer
            pool = get_pool()
            checkpointer, gap_logger, lead_store = make_checkpointer(pool), NeonGapLogger(pool), NeonLeadStore(pool)
            persistent = True
        except Exception:  # noqa: BLE001 - a database outage must not take the chatbot down
            log.exception("Neon unavailable; falling back to in-memory chat history and local logs")
    services = Services(
        analyzer=structured_model(Analysis, settings, model=settings.gemini_analyze_model, thinking_level="low"),
        answerer=structured_model(GroundedAnswer, settings, fallback_model=settings.gemini_fallback_model or None),
        retriever=KnowledgeRetriever(settings=settings),
        gap_logger=gap_logger,
        settings=settings,
    )
    return Runtime(services, checkpointer, lead_store, persistent)


# fields that belong to one turn and must not leak into the next one
_TURN_RESET = dict(context=[], draft=None, attempts=0, issues=[], status="", answer="", citations=[],
                   lead_offer=False, top_score=None, confident=False, asserted_price=None)


class ChatBot:
    def __init__(self, services: Services | None = None, checkpointer=None, lead_store=None):
        if services is None:
            runtime = build_runtime()
            services, checkpointer, lead_store = runtime.services, runtime.checkpointer, runtime.lead_store
            self.persistent = runtime.persistent
        else:
            self.persistent = False
        self.services = services
        self.lead_store = lead_store
        self.graph = build_graph(services, checkpointer=checkpointer or InMemorySaver())

    @staticmethod
    def _config(session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    def answer(self, question: str, session_id: str, history: list[tuple[str, str]] | None = None) -> ChatResponse:
        """`history` seeds earlier (role, text) turns into a new session (evaluation of follow-up questions)."""
        question = question.strip()
        state = self.graph.invoke(
            {"messages": [*(history or []), ("user", question)], "question": question, "session_id": session_id,
             **_TURN_RESET},
            config=self._config(session_id),
        )
        return ChatResponse(answer=state["answer"], citations=state.get("citations", []), status=state["status"],
                            intent=state.get("intent", ""), language=state.get("language", "en"),
                            lead_offer=state.get("lead_offer", False), top_score=state.get("top_score"),
                            search_query=state.get("search_query", ""))

    def history(self, session_id: str) -> list[tuple[str, str]]:
        snapshot = self.graph.get_state(self._config(session_id))
        return [("user" if m.type == "human" else "assistant", str(m.content))
                for m in snapshot.values.get("messages", [])]

    def save_lead(self, session_id: str, name: str, email: str, phone: str = "", message: str = "") -> None:
        values = self.graph.get_state(self._config(session_id)).values
        self.lead_store.save(session_id=session_id, name=name.strip(), email=email.strip(), phone=phone.strip(),
                             message=message.strip(), language=values.get("language"),
                             last_question=values.get("question"))

    def reload_index(self) -> None:
        """Call after a refresh switched the active index."""
        self.services.retriever.reload()


_bot: ChatBot | None = None


def answer(question: str, session_id: str) -> ChatResponse:
    global _bot
    if _bot is None:
        _bot = ChatBot()
    return _bot.answer(question, session_id)
