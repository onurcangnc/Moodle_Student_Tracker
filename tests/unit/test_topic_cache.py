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
