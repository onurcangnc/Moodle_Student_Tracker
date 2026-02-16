"""Unit tests for validator utilities."""

from __future__ import annotations

from types import SimpleNamespace

from bot.utils import validators


def test_is_continue_command_delegates(monkeypatch):
    """Continue validator should delegate to legacy matcher."""
    fake_legacy = SimpleNamespace(_is_continue_command=lambda message: message == "devam")
    monkeypatch.setattr(validators, "_legacy", lambda: fake_legacy)
    assert validators.is_continue_command("devam")
    assert not validators.is_continue_command("hayir")


def test_is_test_command_delegates(monkeypatch):
    """Quiz validator should delegate to legacy matcher."""
    fake_legacy = SimpleNamespace(_is_test_command=lambda message: "test" in message)
    monkeypatch.setattr(validators, "_legacy", lambda: fake_legacy)
    assert validators.is_test_command("beni test et")
    assert not validators.is_test_command("anlat")


def test_needs_topic_menu_delegates(monkeypatch):
    """Topic menu validator should delegate to legacy heuristic."""
    fake_legacy = SimpleNamespace(_needs_topic_menu=lambda message: message.endswith("calisalim"))
    monkeypatch.setattr(validators, "_legacy", lambda: fake_legacy)
    assert validators.needs_topic_menu("edeb calisalim")
    assert not validators.needs_topic_menu("kvkk nedir")
