"""Unit tests for hybrid retrieval sufficiency decisions."""

from __future__ import annotations

import pytest

from bot.services import rag_service
from bot.state import STATE


class FakeStore:
    """Simple hybrid-search stub returning prepared retrieval rows."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def hybrid_search(self, query: str, n_results: int, course_filter: str):
        _ = (query, n_results, course_filter)
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
