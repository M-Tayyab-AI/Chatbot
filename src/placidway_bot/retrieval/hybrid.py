"""Retrieval: BM25 + vector search fused with Reciprocal Rank Fusion, then Voyage rerank-3."""
from __future__ import annotations

from dataclasses import dataclass

from placidway_bot.ingest.chunker import Chunk
from placidway_bot.retrieval.store import Index, tokenize
from placidway_bot.retrieval.voyage_service import VoyageService

RRF_K = 60
CANDIDATES = 30       # per retriever, before fusion
RERANK_POOL = 20      # fused candidates sent to the reranker
VECTOR_SAFETY_NET = 10  # vector top-N always added to the rerank pool


@dataclass
class Hit:
    chunk: Chunk
    score: float      # retriever-specific: bm25 score, cosine similarity, RRF score or rerank relevance (0-1)


def bm25_search(index: Index, query: str, k: int = CANDIDATES) -> list[Hit]:
    scores = index.bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [Hit(index.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def vector_search(index: Index, query_embedding: list[float], k: int = CANDIDATES) -> list[Hit]:
    res = index.collection.query(query_embeddings=[query_embedding], n_results=min(k, len(index.chunks)))
    by_id = index.by_id
    return [Hit(by_id[cid], 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]


def rrf_fuse(*rankings: list[Hit], k: int = RRF_K) -> list[Hit]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            scores[hit.chunk.chunk_id] = scores.get(hit.chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunks[hit.chunk.chunk_id] = hit.chunk
    return [Hit(chunks[cid], s) for cid, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def rerank_pool(fused: list[Hit], vector: list[Hit]) -> list[Hit]:
    """Fused top candidates plus the vector top-N as a safety net: for a query BM25 cannot read (Spanish, Arabic)
    keyword noise can push the vector's best chunks out of the fused list before the reranker ever sees them."""
    pool, seen = [], set()
    for hit in [*fused[:RERANK_POOL], *vector[:VECTOR_SAFETY_NET]]:
        if hit.chunk.chunk_id not in seen:
            seen.add(hit.chunk.chunk_id)
            pool.append(hit)
    return pool


def rerank(reranker, query: str, pool: list[Hit], top_k: int) -> list[Hit]:
    """`reranker` is anything with .rerank(query, documents, top_k) -> [RerankHit] (Voyage or Cohere)."""
    ranked = reranker.rerank(query, [h.chunk.text for h in pool], top_k=min(top_k, len(pool)))
    return [Hit(pool[r.index].chunk, r.score) for r in ranked]


def retrieve(index: Index, voyage: VoyageService, query: str, mode: str = "hybrid_rerank", k: int = 5,
             reranker=None) -> list[Hit]:
    """mode: bm25 | vector | hybrid | hybrid_rerank. The reranker defaults to Voyage rerank-3."""
    if mode == "bm25":
        return bm25_search(index, query)[:k]
    vec = vector_search(index, voyage.embed_query(query))
    if mode == "vector":
        return vec[:k]
    fused = rrf_fuse(bm25_search(index, query), vec)
    if mode == "hybrid":
        return fused[:k]
    return rerank(reranker or voyage, query, rerank_pool(fused, vec), top_k=k)
