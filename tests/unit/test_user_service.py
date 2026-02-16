"""Unit tests for user state helpers in simplified chat flow."""

from __future__ import annotations

from types import SimpleNamespace

from bot.services import user_service
from bot.state import STATE


def test_set_and_get_active_course(monkeypatch):
    """Active course should round-trip through user service."""
    monkeypatch.setattr(user_service, "list_courses", lambda: [])
    user_service.set_active_course(5, "CTIS 363")
    course = user_service.get_active_course(5)
    assert course is not None
    assert course.course_id == "CTIS 363"


def test_upload_session_flags():
    """Upload session flags should be toggleable."""
    user_id = 99
    user_service.begin_upload_session(user_id)
    assert user_service.is_upload_session_active(user_id)
    user_service.clear_upload_session(user_id)
    assert not user_service.is_upload_session_active(user_id)


def test_rate_limit(monkeypatch):
    """Rate limiter should block requests beyond configured window size."""
    from bot.config import CONFIG

    user_id = 77
    STATE.rate_limit_windows[user_id] = []
    monkeypatch.setattr(user_service, "time", type("T", (), {"time": staticmethod(lambda: 1000.0)}))
    allowed = [user_service.check_rate_limit(user_id) for _ in range(CONFIG.rate_limit_max + 1)]
    assert allowed.count(False) >= 1


def test_find_course_matches_partial_name(monkeypatch):
    """find_course should match by partial short or display name."""
    llm_stub = SimpleNamespace(
        moodle_courses=[
            {"shortname": "CTIS 363", "fullname": "CTIS 363 Ethics"},
            {"shortname": "CTIS 465", "fullname": "CTIS 465 Cloud"},
        ]
    )
    monkeypatch.setattr(STATE, "llm", llm_stub)
    monkeypatch.setattr(STATE, "vector_store", None)

    match = user_service.find_course("ethics")
    assert match is not None
    assert match.course_id == "CTIS 363 Ethics"


def test_list_courses_uses_vector_store_fallback(monkeypatch):
    """list_courses should use vector metadata when Moodle list is unavailable."""
    store = SimpleNamespace(
        _metadatas=[
            {"course": "CTIS 300"},
            {"course": "CTIS 300"},
            {"course": "CTIS 400"},
        ]
    )
    monkeypatch.setattr(STATE, "llm", None)
    monkeypatch.setattr(STATE, "vector_store", store)

    courses = user_service.list_courses()
    assert {item.course_id for item in courses} == {"CTIS 300", "CTIS 400"}
