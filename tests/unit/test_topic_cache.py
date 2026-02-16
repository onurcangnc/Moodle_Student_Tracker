"""Unit tests for course topic cache behavior."""

from __future__ import annotations

import pytest

from bot.services.topic_cache import TopicCache
from bot.state import STATE


class FakeVectorStore:
    """Vector store stub exposing metadata/text arrays."""

    def __init__(self):
        self._metadatas = [
            {"course": "CTIS 363", "section": "Kalitim", "filename": "oop.pdf"},
            {"course": "CTIS 363", "topic": "Polimorfizm", "filename": "oop.pdf"},
            {"course": "CTIS 465", "section": "Docker", "filename": "cloud.pdf"},
        ]
        self._texts = [
            "Kalim sayesinde siniflar tekrar kullanilir.",
            "Polimorfizm birden fazla davranis saglar.",
            "Docker container mantigi ile calisir.",
        ]


@pytest.mark.asyncio
async def test_topic_cache(monkeypatch):
    """Topic cache should build and return deduplicated course topics."""
    cache = TopicCache()
    monkeypatch.setattr(STATE, "vector_store", FakeVectorStore())
    topics = await cache.get_topics("CTIS 363")
    assert "Kalitim" in topics
    assert "Polimorfizm" in topics
    assert all("Docker" not in t for t in topics)


@pytest.mark.asyncio
async def test_cache_refresh(monkeypatch):
    """Refresh should rebuild topic list from updated metadata."""
    cache = TopicCache()
    store = FakeVectorStore()
    monkeypatch.setattr(STATE, "vector_store", store)

    await cache.refresh("CTIS 363")
    topics = await cache.get_topics("CTIS 363")
    assert "Kalitim" in topics

    store._metadatas.append({"course": "CTIS 363", "chapter": "Design Patterns"})
    store._texts.append("Design patterns maintainability saglar.")
    await cache.refresh("CTIS 363")
    updated = await cache.get_topics("CTIS 363")
    assert "Design Patterns" in updated


@pytest.mark.asyncio
async def test_cache_empty_course(monkeypatch):
    """Courses without matching chunks should produce an empty list."""
    cache = TopicCache()
    monkeypatch.setattr(STATE, "vector_store", FakeVectorStore())
    topics = await cache.get_topics("NON_EXISTENT")
    assert topics == []
