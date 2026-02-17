"""Unit tests for agentic message handler routing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.handlers import messages


def _build_update(text: str = "Polimorfizm nedir?") -> SimpleNamespace:
    message = SimpleNamespace(text=text, reply_text=AsyncMock())
    user = SimpleNamespace(id=12345)
    return SimpleNamespace(effective_message=message, effective_user=user)


@pytest.mark.asyncio
async def test_handler_rate_limited(monkeypatch):
    """Rate-limited users should get rejection message."""
    update = _build_update()
    context = SimpleNamespace()

    monkeypatch.setattr(messages.user_service, "record_user_activity", lambda user_id: None)
    monkeypatch.setattr(messages.user_service, "check_rate_limit", lambda user_id: False)

    await messages.handle_message(update, context)

    update.effective_message.reply_text.assert_awaited_once()
    payload = update.effective_message.reply_text.await_args_list[0].args[0]
    assert "hızlı" in payload.lower() or "dakika" in payload.lower()


@pytest.mark.asyncio
async def test_handler_routes_to_agent(monkeypatch):
    """Messages should be routed to agent_service.handle_agent_message."""
    update = _build_update("Ödevlerim ne zaman?")
    context = SimpleNamespace()

    monkeypatch.setattr(messages.user_service, "record_user_activity", lambda user_id: None)
    monkeypatch.setattr(messages.user_service, "check_rate_limit", lambda user_id: True)

    agent_mock = AsyncMock(return_value="Yaklaşan ödevleriniz...")
    monkeypatch.setattr(messages, "handle_agent_message", agent_mock)

    await messages.handle_message(update, context)

    agent_mock.assert_awaited_once_with(user_id=12345, user_text="Ödevlerim ne zaman?")
    update.effective_message.reply_text.assert_awaited_once_with(
        "Yaklaşan ödevleriniz...", parse_mode="Markdown"
    )


@pytest.mark.asyncio
async def test_handler_empty_message_ignored(monkeypatch):
    """Empty messages should be silently ignored."""
    update = _build_update(text="   ")
    context = SimpleNamespace()

    monkeypatch.setattr(messages.user_service, "record_user_activity", lambda user_id: None)
    monkeypatch.setattr(messages.user_service, "check_rate_limit", lambda user_id: True)

    agent_mock = AsyncMock()
    monkeypatch.setattr(messages, "handle_agent_message", agent_mock)

    await messages.handle_message(update, context)

    agent_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_handler_no_user_ignored():
    """Messages without effective_user should be silently ignored."""
    message = SimpleNamespace(text="hello", reply_text=AsyncMock())
    update = SimpleNamespace(effective_message=message, effective_user=None)

    await messages.handle_message(update, SimpleNamespace())
    message.reply_text.assert_not_awaited()
