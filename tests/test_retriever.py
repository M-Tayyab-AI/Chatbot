"""KnowledgeRetriever with offline fakes: the reranker's score decides whether the bot may answer."""
from __future__ import annotations

from placidway_bot.config import Settings
from placidway_bot.ingest.chunker import chunk_structured
from placidway_bot.ingest.parsers.pages import parse_page
from placidway_bot.retrieval.hybrid import VECTOR_SAFETY_NET, rerank_pool, rrf_fuse, Hit
from placidway_bot.retrieval.retriever import KnowledgeRetriever
from placidway_bot.retrieval.store import build_index
from placidway_bot.retrieval.voyage_service import RerankHit


class KeywordReranker:
    """Scores 0.95 for documents containing the keyword, 0.2 otherwise."""

    def __init__(self, keyword: str):
        self.keyword = keyword

    def rerank(self, query, documents, top_k):
        scored = [RerankHit(i, 0.95 if self.keyword in d else 0.2) for i, d in enumerate(documents)]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:top_k]


def _retriever(tmp_path, package_html, package_source, fake_voyage, keyword):
    chunks = chunk_structured(parse_page(package_html, package_source, "requests"))
    index = build_index("test_kb", chunks, fake_voyage, root=tmp_path)
    settings = Settings(relevance_threshold=0.76, retrieval_top_k=3, _env_file=None)
    return KnowledgeRetriever(index=index, voyage=fake_voyage, reranker=KeywordReranker(keyword), settings=settings)


def test_confident_when_reranker_finds_the_answer(tmp_path, package_html, package_source, fake_voyage):
    result = _retriever(tmp_path, package_html, package_source, fake_voyage, "Out-Patient").search("outpatient cost")
    assert result.confident and result.top_score == 0.95
    assert "Out-Patient" in result.hits[0].chunk.text and len(result.hits) == 3


def test_not_confident_below_threshold(tmp_path, package_html, package_source, fake_voyage):
    result = _retriever(tmp_path, package_html, package_source, fake_voyage, "dental implant").search("dental implant")
    assert not result.confident and result.top_score == 0.2


def test_rerank_pool_always_contains_vector_top_hits(package_html, package_source):
    chunks = chunk_structured(parse_page(package_html, package_source, "requests"))
    keyword_noise = [Hit(c, 1.0) for c in chunks[1:]]          # BM25 floods the fused list ...
    vector = [Hit(chunks[0], 0.9)]                               # ... while the vector's best is elsewhere
    pool = rerank_pool(rrf_fuse(keyword_noise * 3), vector)
    assert chunks[0].chunk_id in {h.chunk.chunk_id for h in pool}
    assert VECTOR_SAFETY_NET >= 1
