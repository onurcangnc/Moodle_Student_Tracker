"""User/session service wrappers.

Centralizes user-scoped state, history, and throttling operations.
"""

from __future__ import annotations

from typing import Any


def _legacy():
    from bot import legacy

    return legacy


def get_user_state(user_id: int) -> dict[str, Any]:
    """Return mutable user state for the current session."""
    return _legacy()._get_user_state(user_id)


def reset_reading_mode(state: dict[str, Any]) -> None:
    """Reset reading mode attributes on a state dictionary."""
    _legacy()._reset_reading_mode(state)


def start_reading_mode(state: dict[str, Any], filename: str, display_name: str, total: int) -> None:
    """Initialize reading mode attributes for a state dictionary."""
    _legacy()._start_reading_mode(state, filename, display_name, total)


def check_rate_limit(user_id: int) -> bool:
    """Return whether a user request passes rate limiting."""
    return _legacy()._check_rate_limit(user_id)


def get_conversation_history(user_id: int, limit: int = 5) -> list[dict[str, Any]]:
    """Fetch recent conversation history for the user."""
    return _legacy().get_conversation_history(user_id=user_id, limit=limit)


def save_to_history(
    user_id: int,
    role: str,
    content: str,
    active_course: str | None = None,
) -> None:
    """Persist a user/assistant turn into history storage."""
    _legacy().save_to_history(user_id=user_id, role=role, content=content, active_course=active_course)
