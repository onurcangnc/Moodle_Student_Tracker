"""
Shared helper functions for tools.
===================================
Common utilities used across multiple tool modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.state import ServiceContainer

from bot.services import user_service

__all__ = ["DAY_NAMES_TR", "resolve_course", "course_matches"]


DAY_NAMES_TR = {
    0: "Pazartesi",
    1: "Salı",
    2: "Çarşamba",
    3: "Perşembe",
    4: "Cuma",
    5: "Cumartesi",
    6: "Pazar",
}


def resolve_course(args: dict, user_id: int, key: str = "course_filter") -> str | None:
    """Resolve course name from args or active course."""
    name = args.get(key)
    if not name:
        active = user_service.get_active_course(user_id)
        name = active.course_id if active else None
    return name


def course_matches(course_name: str, filter_term: str) -> bool:
    """
    Simple case-insensitive substring match for agentic LLM outputs.

    The LLM handles understanding user intent and extracts proper identifiers.
    This tool just does simple filtering - no complex pattern matching needed.

    Examples (LLM extracts right side from user query):
    - User: "Auditte kaç saat" -> LLM: "Auditing" or "CTIS 474"
    - User: "tarih dersi" -> LLM: "HCIV" or "History"
    """
    return filter_term.lower() in course_name.lower()
