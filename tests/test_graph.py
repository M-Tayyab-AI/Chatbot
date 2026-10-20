"""Graph routing and guardrails with fake LLM / retriever (no network, no API keys)."""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from placidway_bot.chat import ChatBot
from placidway_bot.config import Settings
from placidway_bot.graph.nodes import Services
from placidway_bot.graph.state import Analysis, GroundedAnswer
from placidway_bot.ingest.chunker import Chunk
from placidway_bot.retrieval.hybrid import Hit
from placidway_bot.retrieval.retriever import RetrievalResult

URL = "https://www.placidway.com/package/7902/x"
CHUNK = Chunk(chunk_id="adrenal::structured::002", slug="itc-adrenal-package", source_url=URL,
              page_title="Alternative Adrenal Cancer Treatment in Tijuana", page_type="package",
              section="Cost of Adrenal Cancer Treatment", kind="section", strategy="structured",
              text="[Treatment package: Alternative Adrenal Cancer Treatment]\nOut-Patient Program 3 Weeks $18,995")


class Scripted:
    """Runnable stand-in: returns queued outputs and records the messages it was called with."""

    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    def invoke(self, messages):
        self.calls.append(messages)
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, Exception):
            raise out
        return out


class FakeRetriever:
    def __init__(self, score: float):
        self.score, self.queries = score, []

    def search(self, query, k=None):
        self.queries.append(query)
        return RetrievalResult(query=query, hits=[Hit(CHUNK, self.score)], threshold=0.76)


class ListGapLogger:
    def __init__(self):
        self.rows = []

    def log(self, **row):
        self.rows.append(row)


def analysis(intent="knowledge", query="adrenal cancer treatment cost", lang="en", price=None):
    return Analysis(search_query=query, language=lang, intent=intent, asserted_price=price)


def grounded(answer="The Out-Patient Program costs $18,995 USD for 3 weeks.", found=True, full=True, cited=("S1",),
             medical=False):
    return GroundedAnswer(answer_found=found, fully_answered=full, answer=answer, cited_sources=list(cited),
                          medical_topic=medical)


def bot(analyzer, answerer, score=0.95):
    gaps, retriever = ListGapLogger(), FakeRetriever(score)
    services = Services(analyzer=analyzer, answerer=answerer, retriever=retriever, gap_logger=gaps,
                        settings=Settings(_env_file=None))
    return ChatBot(services=services, checkpointer=InMemorySaver()), gaps, retriever


def test_grounded_answer_has_sources():
    chat, gaps, _ = bot(Scripted(analysis()), Scripted(grounded()))
    r = chat.answer("How much is the adrenal package?", "s1")
    assert r.status == "answered" and "$18,995" in r.answer
    assert r.citations == [{"title": CHUNK.page_title, "url": URL, "section": CHUNK.section}]
    assert f"]({URL})" in r.answer and not gaps.rows


def test_low_relevance_says_i_dont_know_and_logs_gap():
    chat, gaps, _ = bot(Scripted(analysis(query="dental implant cost mexico")), Scripted(grounded()), score=0.3)
    r = chat.answer("How much is a dental implant in Mexico?", "s1")
    assert r.status == "not_found" and "request-info" in r.answer and r.citations == []
    assert gaps.rows[0]["reason"] == "not_found" and gaps.rows[0]["top_score"] == 0.3


def test_invented_price_is_retried_then_blocked():
    answerer = Scripted(grounded(answer="It costs $17,500."))
    chat, gaps, _ = bot(Scripted(analysis()), answerer)
    r = chat.answer("How much is the adrenal package?", "s1")
    assert len(answerer.calls) == 2                                   # one rewrite with the fact-check feedback
    assert "17,500" in answerer.calls[1][1].content                     # feedback names the bad amount
    assert r.status == "validation_failed" and "17,500" not in r.answer
    assert gaps.rows[0]["reason"] == "validation_failed"


def test_retry_can_fix_the_answer():
    answerer = Scripted(grounded(answer="It costs $17,500."), grounded())
    chat, _, _ = bot(Scripted(analysis()), answerer)
    assert chat.answer("How much is the adrenal package?", "s1").status == "answered"


def test_wrong_user_price_can_be_corrected():
    fix = grounded(answer="That doesn't match the page: not $50 — the Out-Patient Program is $18,995 USD.")
    chat, _, _ = bot(Scripted(analysis(price="$50")), Scripted(fix))
    r = chat.answer("I heard it costs $50. Confirm that.", "s1")
    assert r.status == "answered" and "$18,995" in r.answer


def test_fixed_routes_skip_retrieval():
    for intent, status in [("out_of_scope", "out_of_scope"), ("booking", "booking"), ("greeting", "greeting")]:
        chat, _, retriever = bot(Scripted(analysis(intent=intent)), Scripted(grounded()))
        r = chat.answer("question", "s1")
        assert r.status == status and retriever.queries == []
    chat, _, _ = bot(Scripted(analysis(intent="booking")), Scripted(grounded()))
    assert chat.answer("Book me for tomorrow", "s1").lead_offer is True


def test_quote_request_still_answers_from_pages_and_offers_lead_form():
    chat, _, retriever = bot(Scripted(analysis(intent="quote_request")), Scripted(grounded()))
    r = chat.answer("How much is the adrenal package? Please send me a quote", "s1")
    assert r.status == "answered" and "$18,995" in r.answer and r.lead_offer and retriever.queries


def test_answers_in_user_language():
    chat, _, _ = bot(Scripted(analysis(intent="out_of_scope", lang="es")), Scripted(grounded()))
    assert "Lo siento" in chat.answer("¿Qué tiempo hace en Cancún?", "s1").answer


def test_medical_question_without_content_recommends_doctor():
    chat, gaps, _ = bot(Scripted(analysis(intent="medical_advice")), Scripted(grounded()), score=0.4)
    r = chat.answer("Is this surgery safe for my diabetes?", "s1")
    assert r.status == "medical_no_content" and "doctor" in r.answer


def test_medical_disclaimer_only_once_per_session():
    med = grounded(answer="The page lists possible side effects; discuss with your doctor.", medical=True)
    chat, _, _ = bot(Scripted(analysis()), Scripted(med))
    first, second = chat.answer("Are there side effects?", "s1"), chat.answer("Any risks?", "s1")
    assert "Medical note" in first.answer and "Medical note" not in second.answer
    assert "Medical note" in chat.answer("Any risks?", "other-session").answer


def test_follow_up_sees_previous_turns():
    analyzer = Scripted(analysis())
    chat, _, _ = bot(analyzer, Scripted(grounded()))
    chat.answer("What is included in the anal cancer package?", "s1")
    chat.answer("And what about recovery time?", "s1")
    second_call = [m.content for m in analyzer.calls[1]]
    assert any("anal cancer package" in c for c in second_call)        # earlier question passed as history
    assert len(chat.history("s1")) == 4


def test_llm_failure_degrades_gracefully():
    chat, gaps, _ = bot(Scripted(RuntimeError("boom")), Scripted(RuntimeError("down")))
    r = chat.answer("How much is the adrenal package?", "s1")
    assert r.status == "error" and "contact" in r.answer.lower()
