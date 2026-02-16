"""Topic cache for fast guidance-mode suggestions per course."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from bot.state import STATE

logger = logging.getLogger(__name__)


def _extract_first_sentence(text: str) -> str:
    """Extract a compact sentence-like topic candidate from text."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    sentence = re.split(r"[.!?\n]", cleaned, maxsplit=1)[0].strip()
    if len(sentence) < 12:
        return ""
    return sentence[:120]


@dataclass(slots=True)
class TopicCache:
    """
    Cache topic labels for each course.

    Topics are rebuilt from vector metadata when refreshed.
    """

    _topics: dict[str, list[str]] = field(default_factory=dict)

    async def get_topics(self, course_id: str) -> list[str]:
        """Return cached topics for course, refreshing lazily on first access."""
        key = course_id.casefold().strip()
        if key not in self._topics:
            await self.refresh(course_id)
        return list(self._topics.get(key, []))

    async def refresh(self, course_id: str) -> None:
        """Rebuild course topic list from indexed chunk metadata."""
        store = STATE.vector_store
        key = course_id.casefold().strip()
        if store is None:
            self._topics[key] = []
            return

        extracted: list[str] = []
        seen: set[str] = set()
        metadatas = getattr(store, "_metadatas", [])
        texts = getattr(store, "_texts", [])
        for idx, meta in enumerate(metadatas):
            if not isinstance(meta, dict):
                continue
            meta_course = str(meta.get("course", "")).casefold().strip()
            if key and key not in meta_course:
                continue

            for field_name in ("topic", "week", "chapter", "section"):
                value = meta.get(field_name)
                if value is None:
                    continue
                candidate = str(value).strip()
                candidate_key = candidate.casefold()
                if candidate and candidate_key not in seen:
                    seen.add(candidate_key)
                    extracted.append(candidate)

            if idx < len(texts):
                fallback = _extract_first_sentence(str(texts[idx]))
                fallback_key = fallback.casefold()
                if fallback and fallback_key not in seen:
                    seen.add(fallback_key)
                    extracted.append(fallback)

        self._topics[key] = extracted[:15]
        logger.info(
            "Topic cache refreshed",
            extra={"course_id": course_id, "topic_count": len(self._topics[key])},
        )


TOPIC_CACHE = TopicCache()
