"""Validation helpers for user text and course selection inputs."""

from __future__ import annotations


def is_non_empty_text(value: str | None) -> bool:
    """Return whether message text contains a non-whitespace payload."""
    return bool(value and value.strip())


def normalize_course_query(value: str) -> str:
    """Normalize course query by collapsing whitespace and stripping ends."""
    return " ".join(value.split()).strip()
