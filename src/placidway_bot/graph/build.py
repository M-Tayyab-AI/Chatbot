"""LangGraph wiring.

    START -> analyze -> (knowledge | medical_advice) -> retrieve -> (confident) -> generate -> validate -> finalize
                     \\-> greeting | out_of_scope | booking | quote_request -> respond_fixed      ^        |
                                          retrieve (not confident) -> respond_fixed               |  retry once
                                          validate (no answer / still failing) -> respond_fixed  <-+--------+
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from placidway_bot.graph.nodes import MAX_ATTEMPTS, Nodes, Services
from placidway_bot.graph.state import ChatState

# quote_request also retrieves: if the pages answer the question we answer it *and* offer the lead form, so a
# misrouted price question can never be swallowed by a canned reply.
RETRIEVAL_INTENTS = {"knowledge", "medical_advice", "quote_request"}


def route_after_analyze(state: ChatState) -> str:
    return "retrieve" if state["intent"] in RETRIEVAL_INTENTS else "respond_fixed"


def route_after_retrieve(state: ChatState) -> str:
    return "generate" if state.get("confident") else "respond_fixed"


def route_after_validate(state: ChatState) -> str:
    draft = state.get("draft")
    if state.get("status") == "error" or not draft or not draft["answer_found"]:
        return "respond_fixed"
    if state.get("issues"):
        return "generate" if state.get("attempts", 0) < MAX_ATTEMPTS else "respond_fixed"
    return "finalize"


def build_graph(services: Services, checkpointer=None):
    nodes = Nodes(services)
    g = StateGraph(ChatState)
    g.add_node("analyze", nodes.analyze)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("generate", nodes.generate)
    g.add_node("validate", nodes.validate)
    g.add_node("finalize", nodes.finalize)
    g.add_node("respond_fixed", nodes.respond_fixed)

    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", route_after_analyze, ["retrieve", "respond_fixed"])
    g.add_conditional_edges("retrieve", route_after_retrieve, ["generate", "respond_fixed"])
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", route_after_validate, ["finalize", "generate", "respond_fixed"])
    g.add_edge("finalize", END)
    g.add_edge("respond_fixed", END)
    return g.compile(checkpointer=checkpointer)
