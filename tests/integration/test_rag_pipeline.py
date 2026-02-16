"""Integration tests for service-level RAG pipeline behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot.services import rag_service


@pytest.mark.integration
def test_hybrid_search_with_course_filter(vector_store, monkeypatch):
    """RAG service should return filtered hybrid results for a given course."""
    fake_legacy = SimpleNamespace(vector_store=vector_store)
    monkeypatch.setattr(rag_service, "_legacy", lambda: fake_legacy)
    results = rag_service.hybrid_search("privacy", n_results=5, course_filter="CTIS 363")
    assert results
    assert all("CTIS 363" in r["metadata"]["course"] for r in results)


@pytest.mark.integration
def test_build_file_summaries_context_delegation(monkeypatch):
    """File summary context generation should preserve selected file filters."""
    fake_legacy = SimpleNamespace(
        _build_file_summaries_context=lambda selected_files=None, course=None: f"{selected_files}|{course}"
    )
    monkeypatch.setattr(rag_service, "_legacy", lambda: fake_legacy)
    output = rag_service.build_file_summaries_context(selected_files=["a.pdf"], course="CTIS 363")
    assert output == "['a.pdf']|CTIS 363"
