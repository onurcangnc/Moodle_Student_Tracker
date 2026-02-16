"""Unit tests for formatter utilities."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from bot.utils import formatters


def test_format_for_telegram_delegates(monkeypatch):
    """Formatter utility should delegate message normalization."""
    fake_legacy = SimpleNamespace(format_for_telegram=lambda text: text.upper())
    monkeypatch.setattr(formatters, "_legacy", lambda: fake_legacy)
    assert formatters.format_for_telegram("abc") == "ABC"


def test_format_source_footer_delegates(monkeypatch):
    """Source footer utility should delegate to legacy footer logic."""
    fake_legacy = SimpleNamespace(_format_source_footer=lambda chunks, source_type: f"{source_type}:{len(chunks)}")
    monkeypatch.setattr(formatters, "_legacy", lambda: fake_legacy)
    assert formatters.format_source_footer([{"id": 1}], "reading") == "reading:1"


def test_send_long_message_delegates(monkeypatch):
    """Long-message sender should forward async call to legacy implementation."""
    captured: dict[str, object] = {}

    async def fake_send(update, text, reply_markup=None, parse_mode=None):
        captured["update"] = update
        captured["text"] = text
        captured["reply_markup"] = reply_markup
        captured["parse_mode"] = parse_mode

    fake_legacy = SimpleNamespace(send_long_message=fake_send)
    monkeypatch.setattr(formatters, "_legacy", lambda: fake_legacy)
    asyncio.run(formatters.send_long_message(update="u", text="hello", parse_mode="HTML"))
    assert captured["update"] == "u"
    assert captured["text"] == "hello"
    assert captured["parse_mode"] == "HTML"
