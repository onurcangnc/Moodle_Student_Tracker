"""Unit tests for user state helpers in simplified chat flow."""

from __future__ import annotations

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
    user_id = 77
    STATE.rate_limit_windows[user_id] = []
    monkeypatch.setattr(user_service, "time", type("T", (), {"time": staticmethod(lambda: 1000.0)}))
    allowed = [user_service.check_rate_limit(user_id) for _ in range(31)]
    assert allowed.count(False) >= 1
