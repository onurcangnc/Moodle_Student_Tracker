"""Hybrid retrieval service used by the chat-first learning flow."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from bot.config import CONFIG
from bot.state import STATE

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Chunk:
    """Retrieved chunk payload used for downstream LLM prompting."""

    chunk_id: str
    text: str
    similarity: float
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Result of retrieval stage with confidence and sufficiency signal."""

    chunks: list[Chunk]
    confidence: float
    has_sufficient_context: bool


def _similarity_from_distance(distance: float) -> float:
    """Convert distance metric to similarity score in [0, 1]."""
    value = 1.0 - distance
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


async def retrieve_context(
    query: str,
    course_id: str,
    top_k: int = CONFIG.rag_top_k,
    threshold: float = CONFIG.rag_similarity_threshold,
) -> RetrievalResult:
    """
    Retrieve relevant chunks for the query and evaluate sufficiency.

    Sufficiency rule:
    - at least CONFIG.rag_min_chunks chunks
    - each included chunk similarity >= threshold
    """
    store = STATE.vector_store
    if store is None:
        return RetrievalResult(chunks=[], confidence=0.0, has_sufficient_context=False)

    started = time.perf_counter()
    try:
        raw_results = await asyncio.to_thread(
            store.hybrid_search,
            query,
            top_k,
            course_id,
        )
    except (AttributeError, RuntimeError, ValueError, OSError) as exc:
        logger.error("Retrieval failed", exc_info=True, extra={"course_id": course_id, "error": str(exc)})
        return RetrievalResult(chunks=[], confidence=0.0, has_sufficient_context=False)

    selected: list[Chunk] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        similarity = _similarity_from_distance(float(item.get("distance", 1.0)))
        if similarity < threshold:
            continue
        selected.append(
            Chunk(
                chunk_id=str(item.get("id", "")),
                text=str(item.get("text", "")),
                similarity=similarity,
                metadata=dict(item.get("metadata", {})),
            )
        )

    confidence = (sum(chunk.similarity for chunk in selected) / len(selected)) if selected else 0.0
    has_sufficient = len(selected) >= CONFIG.rag_min_chunks

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "Retrieval completed",
        extra={
            "course_id": course_id,
            "query_len": len(query),
            "top_k": top_k,
            "threshold": threshold,
            "returned": len(raw_results),
            "selected": len(selected),
            "confidence": round(confidence, 3),
            "has_sufficient_context": has_sufficient,
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )
    return RetrievalResult(chunks=selected, confidence=confidence, has_sufficient_context=has_sufficient)
