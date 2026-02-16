"""Unit tests for global Telegram error middleware."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.middleware import error_handler


class DummyUpdate:
    """Minimal Update-like test double."""

    def __init__(self, user_id: int = 123):
        self.update_id = 77
        self.effective_user = SimpleNamespace(id=user_id)
        self.effective_message = SimpleNamespace(reply_text=AsyncMock())


@pytest.mark.asyncio
async def test_global_handler_logs_exception(monkeypatch):
    """Unhandled exceptions should be logged through middleware logger."""
    update = DummyUpdate()
    context = SimpleNamespace(error=RuntimeError("boom"))

    logger_mock = MagicMock()
    monkeypatch.setattr(error_handler, "Update", DummyUpdate)
    monkeypatch.setattr(error_handler, "logger", logger_mock)

    await error_handler.global_error_handler(update, context)

    assert logger_mock.exception.called
    first_call = logger_mock.exception.call_args_list[0]
    assert "Unhandled exception in Telegram update pipeline" in first_call.args[0]


@pytest.mark.asyncio
async def test_global_handler_sends_user_message(monkeypatch):
    """Middleware should send a generic error message to the user."""
    update = DummyUpdate()
    context = SimpleNamespace(error=RuntimeError("boom"))
    monkeypatch.setattr(error_handler, "Update", DummyUpdate)
    monkeypatch.setattr(error_handler, "logger", MagicMock())

    await error_handler.global_error_handler(update, context)

    update.effective_message.reply_text.assert_awaited_once_with("Bir hata oluştu. Lütfen tekrar deneyin.")
