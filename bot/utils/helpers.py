"""General helper wrappers kept for compatibility and modular call sites."""

from __future__ import annotations


def _legacy():
    from bot import legacy

    return legacy


def reading_batch(filename: str, position: int) -> list[dict]:
    """Return a reading batch from file chunks."""
    return _legacy()._get_reading_batch(filename=filename, position=position)


def reading_buttons(state: dict, show_back: bool = False, completed: bool = False):
    """Return inline keyboard for reading navigation."""
    return _legacy()._reading_buttons(state=state, show_back=show_back, completed=completed)


def topic_menu(course_name: str, files: list[dict]):
    """Return formatted topic menu header and keyboard."""
    return _legacy()._format_topic_menu(course_name=course_name, files=files)
