"""
Shared helper functions for tools.
===================================
Common utilities used across multiple tool modules.
"""

from __future__ import annotations

import re
import unicodedata
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


_COURSE_CODE_RE = re.compile(r"([A-Za-z]{2,})\s*(\d+)")


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


def _course_code(text: str) -> str | None:
    """Extract course code (e.g. 'EDEB201') from any format.

    'EDEB 201-2' -> 'edeb201'   (section suffix '-2' dropped)
    'EDEB201-2'  -> 'edeb201'
    'EDEB 201 Introduction to Turkish Fiction' -> 'edeb201'
    'CTIS 474'   -> 'ctis474'
    'Edeb'       -> None (no number)
    """
    m = _COURSE_CODE_RE.search(text)
    if not m:
        return None
    return f"{m.group(1).lower()}{m.group(2)}"


def course_matches(course_name: str, filter_term: str) -> bool:
    """Match course filter against cached course name.

    Moodle exposes sections ('EDEB 201-2'), STARS stores the base code
    ('EDEB 201'). LLM can emit either format. Strategy:

      1. Course-code extraction: if both sides yield a code and they match
         on dept+number (ignoring section suffix), it's the same course.
      2. Normalized substring fallback: for plain-text queries like
         'Edeb' or 'auditing' that don't have a number.
    """
    norm_filter = _normalize(filter_term)
    if not norm_filter:
        return False

    filter_code = _course_code(filter_term)
    course_code = _course_code(course_name)
    if filter_code and course_code and filter_code == course_code:
        return True

    return norm_filter in _normalize(course_name)
