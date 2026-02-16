"""Unit tests for hybrid retrieval sufficiency decisions."""

from __future__ import annotations

import pytest

from bot.services import rag_service
from bot.state import STATE


class FakeStore:
    """Simple hybrid-search stub returning prepared retrieval rows."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.last_query: str | None = None
        self.last_n_results: int | None = None
        self.last_course_filter: str | None = None

    def hybrid_search(self, query: str, n_results: int, course_filter: str):
        self.last_query = query
        self.last_n_results = n_results
        self.last_course_filter = course_filter
        return self.rows


@pytest.mark.asyncio
async def test_retrieve_sufficient_context(monkeypatch):
    """Retrieval should report sufficient context with >=2 high-similarity chunks."""
    rows = [
        {"id": "1", "text": "A", "distance": 0.20, "metadata": {"filename": "a.pdf"}},
        {"id": "2", "text": "B", "distance": 0.30, "metadata": {"filename": "b.pdf"}},
        {"id": "3", "text": "C", "distance": 0.55, "metadata": {"filename": "c.pdf"}},
    ]
    monkeypatch.setattr(STATE, "vector_store", FakeStore(rows))
    result = await rag_service.retrieve_context(query="polimorfizm", course_id="CTIS 363")
    assert result.has_sufficient_context is True
    assert len(result.chunks) == 2
    assert result.confidence > 0.70


@pytest.mark.asyncio
async def test_retrieve_insufficient_context(monkeypatch):
    """Retrieval should report insufficient context when high-similarity chunks are too few."""
    rows = [
        {"id": "1", "text": "A", "distance": 0.20, "metadata": {"filename": "a.pdf"}},
        {"id": "2", "text": "B", "distance": 0.60, "metadata": {"filename": "b.pdf"}},
    ]
    monkeypatch.setattr(STATE, "vector_store", FakeStore(rows))
    result = await rag_service.retrieve_context(query="kuantum", course_id="CTIS 363")
    assert result.has_sufficient_context is False
    assert len(result.chunks) == 1


@pytest.mark.asyncio
async def test_retrieve_returns_correct_top_k(monkeypatch):
    """Retrieval call should pass the requested top_k to vector store."""
    rows = [
        {"id": "1", "text": "A", "distance": 0.20, "metadata": {"filename": "a.pdf"}},
        {"id": "2", "text": "B", "distance": 0.25, "metadata": {"filename": "b.pdf"}},
        {"id": "3", "text": "C", "distance": 0.30, "metadata": {"filename": "c.pdf"}},
    ]
    store = FakeStore(rows)
    monkeypatch.setattr(STATE, "vector_store", store)
    result = await rag_service.retrieve_context(query="oop", course_id="CTIS 363", top_k=2)
    assert store.last_n_results == 2
    assert store.last_query == "oop"
    assert store.last_course_filter == "CTIS 363"
    assert len(result.chunks) == 3


@pytest.mark.asyncio
async def test_retrieve_filters_below_threshold(monkeypatch):
    """Threshold should filter out low-similarity rows."""
    rows = [
        {"id": "1", "text": "A", "distance": 0.20, "metadata": {"filename": "a.pdf"}},  # 0.80
        {"id": "2", "text": "B", "distance": 0.41, "metadata": {"filename": "b.pdf"}},  # 0.59
        {"id": "3", "text": "C", "distance": 0.30, "metadata": {"filename": "c.pdf"}},  # 0.70
    ]
    monkeypatch.setattr(STATE, "vector_store", FakeStore(rows))
    result = await rag_service.retrieve_context(query="oop", course_id="CTIS 363", threshold=0.65)
    assert [chunk.chunk_id for chunk in result.chunks] == ["1", "3"]
    assert result.has_sufficient_context is True


@pytest.mark.asyncio
async def test_retrieve_empty_store(monkeypatch):
    """Missing vector store should return an insufficient empty result."""
    monkeypatch.setattr(STATE, "vector_store", None)
    result = await rag_service.retrieve_context(query="anything", course_id="CTIS 363")
    assert result.chunks == []
    assert result.has_sufficient_context is False
    assert result.confidence == 0.0
