"""Shared pytest fixtures for unit, integration, and e2e tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@dataclass(slots=True)
class SampleDocument:
    """Simple document fixture model."""

    course: str
    filename: str
    text: str


@pytest.fixture
def sample_documents() -> list[SampleDocument]:
    """Sample document set used across retrieval-oriented tests."""
    return [
        SampleDocument(course="CTIS 363", filename="ethics_intro.pdf", text="privacy surveillance ethics"),
        SampleDocument(course="CTIS 465", filename="microservices.pdf", text="docker api gateway service"),
        SampleDocument(course="EDEB 201", filename="novel_notes.pdf", text="dogu bati catismasi roman"),
    ]


@pytest.fixture
def mock_telegram_update():
    """Mock Telegram Update object with async reply interface."""
    msg = SimpleNamespace(reply_text=AsyncMock())
    user = SimpleNamespace(id=123456)
    return SimpleNamespace(
        update_id=1,
        effective_user=user,
        effective_message=msg,
        message=msg,
    )


@pytest.fixture
def vector_store(tmp_path):
    """In-memory vector store fixture with deterministic hybrid search behavior."""

    class InMemoryVectorStore:
        def __init__(self):
            self.store_dir = tmp_path
            self.items: list[dict] = []

        def add(self, text: str, filename: str, course: str, distance: float = 0.2) -> None:
            self.items.append(
                {
                    "id": f"{filename}:{len(self.items)}",
                    "text": text,
                    "metadata": {"filename": filename, "course": course},
                    "distance": distance,
                }
            )

        def query(self, query_text: str, n_results: int = 5, course_filter: str | None = None):
            q = query_text.lower()
            filtered = [
                i
                for i in self.items
                if (course_filter is None or course_filter.lower() in i["metadata"]["course"].lower())
                and any(tok in i["text"].lower() for tok in q.split())
            ]
            return filtered[:n_results]

        def hybrid_search(self, query: str, n_results: int = 5, course_filter: str | None = None):
            return self.query(query_text=query, n_results=n_results, course_filter=course_filter)

    vs = InMemoryVectorStore()
    vs.add("privacy surveillance ethics", "ethics_intro.pdf", "CTIS 363", distance=0.10)
    vs.add("docker api gateway service", "microservices.pdf", "CTIS 465", distance=0.18)
    return vs
