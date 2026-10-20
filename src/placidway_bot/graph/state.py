"""Graph state and the two structured LLM outputs."""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Intent = Literal["knowledge", "medical_advice", "booking", "quote_request", "out_of_scope", "greeting"]


class Analysis(BaseModel):
    """Output of the analyze node (rewrite + route in one call)."""
    search_query: str = Field(description="The user's latest question rewritten as ONE standalone ENGLISH search "
                                          "query, resolving references to earlier messages (e.g. 'what about "
                                          "recovery?' -> 'recovery time after the anal cancer treatment package').")
    language: str = Field(description="ISO 639-1 code of the language the user wrote their latest message in, "
                                      "e.g. en, es, ar, ur.")
    intent: Intent
    asserted_price: str | None = Field(default=None, description="A price the user states as fact or asks you to "
                                                                 "confirm (e.g. '$50'), copied verbatim; else null.")


class GroundedAnswer(BaseModel):
    """Output of the generate node."""
    answer_found: bool = Field(description="True only if the SOURCES contain information that answers the question "
                                           "(fully or partly).")
    fully_answered: bool = Field(description="False if part of the question is not covered by the SOURCES.")
    answer: str = Field(description="The reply to the user, in the requested language, using only facts from the "
                                    "SOURCES. No URLs and no source ids in this text.")
    cited_sources: list[str] = Field(description="Ids of the sources the answer relies on, e.g. ['S1', 'S3'].")
    medical_topic: bool = Field(description="True ONLY if the question is about health effects, treatment safety, "
                                            "risks, side effects, eligibility or suitability for a condition. False "
                                            "for questions about prices, inclusions, logistics or travel.")


class ChatState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # full conversation (persisted per session)
    session_id: str
    question: str
    # set by analyze
    search_query: str
    language: str
    intent: str
    asserted_price: str | None
    # set by retrieve
    context: list[dict]          # [{ref, chunk_id, text, source_url, page_title, section, score}]
    top_score: float | None
    confident: bool              # top rerank score >= relevance threshold
    # set by generate / validate
    draft: dict | None
    attempts: int
    issues: list[str]
    # set by finalize
    status: str                  # answered | partial | not_found | validation_failed | out_of_scope | booking ...
    answer: str
    citations: list[dict]
    lead_offer: bool
    disclaimer_shown: bool       # persists across turns: the medical note is shown once per session
