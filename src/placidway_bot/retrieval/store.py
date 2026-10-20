"""Knowledge-base index: Chroma (vectors + metadata) and BM25 (keywords) over the same chunks."""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from placidway_bot.config import INDEX_DIR
from placidway_bot.ingest.chunker import Chunk
from placidway_bot.retrieval.voyage_service import VoyageService

STOPWORDS = set("""a an and are as at be by can do does for from has have how i in is it its me my of on or
our that the their this to was what when where which who will with you your""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; numbers lose thousands separators so "$18,995" matches "18995"."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())
    return [t for t in re.findall(r"[a-z0-9]+|[^\W\d_]+", text) if t not in STOPWORDS]


@dataclass
class Index:
    name: str
    chunks: list[Chunk]
    collection: chromadb.Collection
    bm25: BM25Okapi

    @property
    def by_id(self) -> dict[str, Chunk]:
        return {c.chunk_id: c for c in self.chunks}


def _client(root: Path) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(root / "chroma"))


def build_index(name: str, chunks: list[Chunk], voyage: VoyageService, root: Path = INDEX_DIR) -> Index:
    """(Re)build one named index from scratch. Embeddings come from the Voyage cache when unchanged."""
    embeddings = voyage.embed([c.text for c in chunks], "document")
    client = _client(root)
    try:
        client.delete_collection(name)
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    collection.add(ids=[c.chunk_id for c in chunks], documents=[c.text for c in chunks],
                   embeddings=embeddings, metadatas=[c.metadata() for c in chunks])
    out = root / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "chunks.json").write_text(json.dumps([c.__dict__ for c in chunks], ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    return Index(name, chunks, collection, BM25Okapi([tokenize(c.text) for c in chunks]))


def delete_index(name: str, root: Path = INDEX_DIR) -> None:
    try:
        _client(root).delete_collection(name)
    except Exception:  # noqa: BLE001 - already gone
        pass
    shutil.rmtree(root / name, ignore_errors=True)


def load_active_index(root: Path = INDEX_DIR) -> Index:
    """The index the latest successful refresh switched to (see ingest/pipeline.py)."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return load_index(manifest["active_index"], root)


def load_index(name: str, root: Path = INDEX_DIR) -> Index:
    chunks = [Chunk(**c) for c in json.loads((root / name / "chunks.json").read_text(encoding="utf-8"))]
    collection = _client(root).get_collection(name)
    return Index(name, chunks, collection, BM25Okapi([tokenize(c.text) for c in chunks]))
