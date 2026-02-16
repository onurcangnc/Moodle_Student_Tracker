"""Unit tests for RAG service wrappers."""

from __future__ import annotations

from types import SimpleNamespace

from bot.services import rag_service


def test_detect_active_course_delegates_to_legacy(monkeypatch):
    """Service should delegate course detection to legacy implementation."""
    fake_legacy = SimpleNamespace(detect_active_course=lambda message, user_id: f"{message}:{user_id}")
    monkeypatch.setattr(rag_service, "_legacy", lambda: fake_legacy)
    assert rag_service.detect_active_course("ctis", 7) == "ctis:7"


def test_hybrid_search_returns_empty_when_vector_store_missing(monkeypatch):
    """Service should return empty list when vector store is not initialized."""
    fake_legacy = SimpleNamespace(vector_store=None)
    monkeypatch.setattr(rag_service, "_legacy", lambda: fake_legacy)
    assert rag_service.hybrid_search("privacy") == []


def test_hybrid_search_delegates_with_filters(monkeypatch):
    """Service should call legacy vector store hybrid search with parameters."""

    class FakeStore:
        def hybrid_search(self, query, n_results, course_filter):
            return [{"id": "1", "text": query, "metadata": {"course": course_filter}, "distance": 0.2}]

    fake_legacy = SimpleNamespace(vector_store=FakeStore())
    monkeypatch.setattr(rag_service, "_legacy", lambda: fake_legacy)
    results = rag_service.hybrid_search("docker", n_results=3, course_filter="CTIS 465")
    assert len(results) == 1
    assert results[0]["text"] == "docker"
    assert results[0]["metadata"]["course"] == "CTIS 465"
