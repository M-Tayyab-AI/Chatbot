"""Production retrieval entry point: hybrid search (BM25 + Voyage vectors) -> reranker -> "I don't know" gate.

Configuration (chosen by evaluation, see docs/chunking_comparison.md): structured chunks, Cohere rerank-v4.0-pro,
relevance threshold 0.76. Retrieval should receive an English, standalone query (the graph's rewrite node does that).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from placidway_bot.config import INDEX_DIR, Settings, get_settings
from placidway_bot.retrieval.hybrid import Hit, retrieve
from placidway_bot.retrieval.store import Index, load_active_index


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit] = field(default_factory=list)
    threshold: float = 0.0

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

    @property
    def confident(self) -> bool:
        """False -> the knowledge base does not answer this question; the bot must say so."""
        return self.top_score >= self.threshold


def build_reranker(settings: Settings, voyage):
    if settings.reranker == "cohere":
        from placidway_bot.retrieval.cohere_reranker import CohereReranker
        return CohereReranker()
    return voyage                                   # VoyageService also implements .rerank()


class KnowledgeRetriever:
    def __init__(self, index: Index | None = None, voyage=None, reranker=None, settings: Settings | None = None,
                 index_root=INDEX_DIR):
        self.settings = settings or get_settings()
        if voyage is None:
            from placidway_bot.retrieval.voyage_service import VoyageService
            voyage = VoyageService()
        self.voyage = voyage
        self.reranker = reranker or build_reranker(self.settings, voyage)
        self.index_root = index_root
        self.index = index or load_active_index(index_root)

    def reload(self) -> None:
        """Pick up the index a refresh just switched to."""
        self.index = load_active_index(self.index_root)

    def search(self, query: str, k: int | None = None) -> RetrievalResult:
        hits = retrieve(self.index, self.voyage, query, mode="hybrid_rerank",
                        k=k or self.settings.retrieval_top_k, reranker=self.reranker)
        return RetrievalResult(query=query, hits=hits, threshold=self.settings.relevance_threshold)
