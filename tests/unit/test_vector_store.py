"""Unit tests for vector store core behavior."""

from __future__ import annotations

from types import MethodType

import numpy as np
import pytest

pytest.importorskip("snowballstemmer")

from core.document_processor import DocumentChunk
from core.vector_store import VectorStore


class FakeIndex:
    """Minimal FAISS-like index for deterministic unit tests."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    def add(self, embeddings: np.ndarray) -> None:
        if self.vectors.size == 0:
            self.vectors = embeddings.copy()
        else:
            self.vectors = np.vstack([self.vectors, embeddings])

    def search(self, query_vec: np.ndarray, k: int):
        if self.vectors.size == 0:
            return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
        scores = self.vectors @ query_vec[0]
        order = np.argsort(scores)[::-1][:k]
        return np.array([scores[order]], dtype=np.float32), np.array([order], dtype=np.int64)

    def reconstruct(self, idx: int) -> np.ndarray:
        return self.vectors[idx]


def _fake_encode(self, texts: list[str]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for text in texts:
        lower = text.lower()
        alpha = lower.count("alpha")
        beta = lower.count("beta")
        vec = np.array([max(alpha, 1), max(beta, 1)], dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        vectors.append(vec)
    return np.vstack(vectors).astype(np.float32)


def _build_vs_for_unit_tests(monkeypatch) -> VectorStore:
    vs = VectorStore()
    vs._dimension = 2
    vs._index = FakeIndex(2)
    vs._ids = []
    vs._texts = []
    vs._metadatas = []
    vs._encode = MethodType(_fake_encode, vs)
    monkeypatch.setattr(vs, "_save", lambda: None)
    monkeypatch.setattr(vs, "_build_bm25_index", lambda: None)
    return vs


def test_add_and_query_with_similarity_threshold(monkeypatch):
    """Added chunks should be retrievable with semantic ranking and filtering."""
    vs = _build_vs_for_unit_tests(monkeypatch)
    chunks = [
        DocumentChunk(
            text="alpha alpha ethics",
            metadata={"source": "f1.pdf", "chunk_index": 0, "filename": "f1.pdf", "course": "CTIS 363"},
        ),
        DocumentChunk(
            text="beta beta microservice",
            metadata={"source": "f2.pdf", "chunk_index": 0, "filename": "f2.pdf", "course": "CTIS 465"},
        ),
    ]
    vs.add_chunks(chunks)
    results = vs.query(query_text="alpha ethics", n_results=2, course_filter="CTIS 363")
    assert len(results) == 1
    assert results[0]["metadata"]["filename"] == "f1.pdf"
    assert results[0]["distance"] <= 0.5


def test_delete_by_course_uses_predicate(monkeypatch):
    """Delete-by-course should route through predicate-based deletion."""
    vs = _build_vs_for_unit_tests(monkeypatch)
    called: dict[str, object] = {}

    def fake_delete(predicate):
        called["predicate"] = predicate

    monkeypatch.setattr(vs, "_delete_where", fake_delete)
    vs.delete_by_course("CTIS 363")
    predicate = called["predicate"]
    assert predicate({"course": "CTIS 363-1"}) is True
    assert predicate({"course": "CTIS 465-1"}) is False


def test_hybrid_search_returns_semantic_when_bm25_empty(monkeypatch):
    """Hybrid search should fall back to semantic results if BM25 is empty."""
    vs = _build_vs_for_unit_tests(monkeypatch)
    monkeypatch.setattr(vs, "query", lambda **kwargs: [{"id": "1", "text": "alpha", "metadata": {}, "distance": 0.2}])
    monkeypatch.setattr(vs, "bm25_search", lambda *args, **kwargs: [])
    results = vs.hybrid_search(query="alpha", n_results=5)
    assert len(results) == 1
    assert results[0]["id"] == "1"
