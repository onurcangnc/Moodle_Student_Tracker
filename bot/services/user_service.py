"""User service for active course selection, rate limiting, and upload session flags."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from bot.config import CONFIG
from bot.services.conversation_memory import ConversationMemory
from bot.state import STATE

logger = logging.getLogger(__name__)
MEMORY = ConversationMemory(
    max_messages=CONFIG.memory_max_messages,
    ttl_minutes=CONFIG.memory_ttl_minutes,
)


@dataclass(frozen=True, slots=True)
class CourseSelection:
    """Canonical representation of a selectable course."""

    course_id: str
    short_name: str
    display_name: str


def _normalize(text: str) -> str:
    """Normalize strings for robust course matching."""
    lowered = text.casefold()
    return re.sub(r"\s+", " ", lowered).strip()


def list_courses() -> list[CourseSelection]:
    """Return available courses from Moodle cache or indexed vector metadata."""
    courses: list[CourseSelection] = []
    seen: set[str] = set()

    llm = STATE.llm
    if llm is not None and isinstance(getattr(llm, "moodle_courses", None), list):
        for item in llm.moodle_courses:
            if not isinstance(item, dict):
                continue
            display_name = str(item.get("fullname", "")).strip()
            short_name = str(item.get("shortname", "")).strip()
            if not display_name:
                continue
            key = _normalize(display_name)
            if key in seen:
                continue
            seen.add(key)
            courses.append(
                CourseSelection(
                    course_id=display_name,
                    short_name=short_name or display_name.split()[0],
                    display_name=display_name,
                )
            )

    if courses:
        return courses

    store = STATE.vector_store
    if store is None:
        return []

    for meta in getattr(store, "_metadatas", []):
        course = str(meta.get("course", "")).strip()
        if not course:
            continue
        key = _normalize(course)
        if key in seen:
            continue
        seen.add(key)
        courses.append(
            CourseSelection(
                course_id=course,
                short_name=course.split()[0],
                display_name=course,
            )
        )

    return courses


def find_course(query: str) -> CourseSelection | None:
    """Find a course by short name or full name using case-insensitive matching."""
    if not query.strip():
        return None

    target = _normalize(query)
    candidates = list_courses()
    if not candidates:
        return None

    exact = next(
        (c for c in candidates if _normalize(c.short_name) == target or _normalize(c.display_name) == target), None
    )
    if exact is not None:
        return exact

    partial = next(
        (c for c in candidates if target in _normalize(c.short_name) or target in _normalize(c.display_name)),
        None,
    )
    return partial


def set_active_course(user_id: int, course_id: str) -> None:
    """Set active course for the user session."""
    STATE.active_courses[user_id] = course_id


def get_active_course(user_id: int) -> CourseSelection | None:
    """Get active course selection for user if available."""
    active_id = STATE.active_courses.get(user_id)
    if active_id is None:
        return None

    for course in list_courses():
        if _normalize(course.course_id) == _normalize(active_id):
            return course

    # Keep stale value if course list is temporarily unavailable.
    return CourseSelection(course_id=active_id, short_name=active_id.split()[0], display_name=active_id)


def clear_active_course(user_id: int) -> None:
    """Clear active course for a user."""
    STATE.active_courses.pop(user_id, None)


def check_rate_limit(user_id: int) -> bool:
    """Return whether message rate is within configured per-window limits."""
    now = time.time()
    window_start = now - CONFIG.rate_limit_window
    timestamps = STATE.rate_limit_windows.setdefault(user_id, [])
    timestamps[:] = [ts for ts in timestamps if ts >= window_start]
    if len(timestamps) >= CONFIG.rate_limit_max:
        logger.warning("Rate limit exceeded", extra={"user_id": user_id})
        return False
    timestamps.append(now)
    return True


def begin_upload_session(user_id: int) -> None:
    """Mark user as awaiting a document upload."""
    STATE.pending_upload_users.add(user_id)


def clear_upload_session(user_id: int) -> None:
    """Clear upload-awaiting state for user."""
    STATE.pending_upload_users.discard(user_id)


def is_upload_session_active(user_id: int) -> bool:
    """Return whether user is currently expected to upload a document."""
    return user_id in STATE.pending_upload_users


def add_conversation_turn(user_id: int, role: str, content: str) -> None:
    """Record one turn in short-lived conversation memory."""
    MEMORY.add(user_id=user_id, role=role, content=content)


def get_conversation_history(user_id: int) -> list[dict[str, str]]:
    """Return recent conversation turns for user."""
    return MEMORY.get_history(user_id)


def clear_conversation_history(user_id: int) -> None:
    """Clear short-lived conversation history for user."""
    MEMORY.clear(user_id)
