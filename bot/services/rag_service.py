"""RAG service wrappers.

Encapsulates retrieval-related operations and keeps handler modules free from
direct retrieval pipeline details.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def _legacy():
    from bot import legacy

    return legacy


def detect_active_course(user_message: str, user_id: int) -> str | None:
    """Detect active course from user message and chat history context."""
    return _legacy().detect_active_course(user_message, user_id)


def build_smart_query(user_message: str, history: list[dict[str, Any]]) -> str:
    """Build retrieval query using current message plus recent context."""
    return _legacy().build_smart_query(user_message, history)


def build_file_summaries_context(
    selected_files: list[str] | None = None,
    course: str | None = None,
) -> str:
    """Build context from pre-generated file summaries."""
    return _legacy()._build_file_summaries_context(selected_files=selected_files, course=course)


def get_relevant_files(query: str, course: str | None = None, top_k: int = 5) -> list[str]:
    """Return likely relevant filenames for a user query."""
    return _legacy()._get_relevant_files(query=query, course=course, top_k=top_k)


def hybrid_search(query: str, n_results: int = 15, course_filter: str | None = None) -> list[dict[str, Any]]:
    """Run hybrid retrieval and emit lightweight performance logs."""
    legacy = _legacy()
    if legacy.vector_store is None:
        return []
    start = time.perf_counter()
    results = legacy.vector_store.hybrid_search(query=query, n_results=n_results, course_filter=course_filter)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "Hybrid search completed",
        extra={
            "elapsed_ms": round(elapsed_ms, 2),
            "result_count": len(results),
            "course_filter": course_filter,
        },
    )
    return results
