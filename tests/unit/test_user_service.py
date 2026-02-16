"""Unit tests for user service wrappers."""

from __future__ import annotations

from types import SimpleNamespace

from bot.services import user_service


def test_get_user_state_delegates(monkeypatch):
    """User state getter should forward to legacy user-state factory."""
    fake_legacy = SimpleNamespace(_get_user_state=lambda uid: {"uid": uid, "socratic_mode": True})
    monkeypatch.setattr(user_service, "_legacy", lambda: fake_legacy)
    assert user_service.get_user_state(99)["uid"] == 99


def test_check_rate_limit_delegates(monkeypatch):
    """Rate limit check should forward to legacy limiter."""
    fake_legacy = SimpleNamespace(_check_rate_limit=lambda uid: uid == 1)
    monkeypatch.setattr(user_service, "_legacy", lambda: fake_legacy)
    assert user_service.check_rate_limit(1) is True
    assert user_service.check_rate_limit(2) is False


def test_save_to_history_delegates(monkeypatch):
    """History writes should call legacy save with named arguments."""
    calls: list[dict] = []

    def fake_save(**kwargs):
        calls.append(kwargs)

    fake_legacy = SimpleNamespace(save_to_history=fake_save)
    monkeypatch.setattr(user_service, "_legacy", lambda: fake_legacy)
    user_service.save_to_history(user_id=5, role="user", content="hello", active_course="CTIS 363")
    assert calls[0]["user_id"] == 5
    assert calls[0]["content"] == "hello"
