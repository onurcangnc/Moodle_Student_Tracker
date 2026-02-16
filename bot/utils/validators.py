"""Input and intent validation helpers."""

from __future__ import annotations


def _legacy():
    from bot import legacy

    return legacy


def is_continue_command(message: str) -> bool:
    """Return whether a message requests continuing the current explanation."""
    return _legacy()._is_continue_command(message)


def is_test_command(message: str) -> bool:
    """Return whether a message requests quiz/test mode."""
    return _legacy()._is_test_command(message)


def needs_topic_menu(message: str) -> bool:
    """Return whether a message should trigger topic-file selection menu."""
    return _legacy()._needs_topic_menu(message)
