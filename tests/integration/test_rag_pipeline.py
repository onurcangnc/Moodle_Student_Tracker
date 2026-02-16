"""Integration tests for chat-first retrieval sufficiency flow."""

from __future__ import annotations

import pytest

from bot.services import rag_service
from bot.state import STATE


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_context_with_course_filter(vector_store, monkeypatch):
    """Retrieve context should keep only filtered course chunks above threshold."""
    monkeypatch.setattr(STATE, "vector_store", vector_store)
    result = await rag_service.retrieve_context(query="privacy", course_id="CTIS 363", top_k=5, threshold=0.65)
    assert result.has_sufficient_context is False
    assert len(result.chunks) == 1
