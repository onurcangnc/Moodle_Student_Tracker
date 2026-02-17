"""Unit tests for Telegram command handlers (agentic: /start + /upload only)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.handlers import commands


def _build_update(user_id: int | None = 123) -> SimpleNamespace:
    """Build Update-like object with async reply mock."""
    message = SimpleNamespace(reply_text=AsyncMock())
    user = None if user_id is None else SimpleNamespace(id=user_id)
    return SimpleNamespace(effective_user=user, effective_message=message)


@pytest.mark.asyncio
async def test_post_init_sets_bot_commands():
    """post_init should register visible command list."""
    app = SimpleNamespace(bot=SimpleNamespace(set_my_commands=AsyncMock()))
    await commands.post_init(app)
    app.bot.set_my_commands.assert_awaited_once()
    # Should only register 2 commands (start, upload)
    registered = app.bot.set_my_commands.await_args.args[0]
    assert len(registered) == 2


@pytest.mark.asyncio
async def test_cmd_start_replies_welcome():
    """start command should send welcome guidance."""
    update = _build_update()
    await commands.cmd_start(update, SimpleNamespace())
    update.effective_message.reply_text.assert_awaited_once()
    payload = update.effective_message.reply_text.await_args_list[0].args[0]
    assert "Merhaba" in payload


@pytest.mark.asyncio
async def test_cmd_upload_denied(monkeypatch):
    """upload command should stop when admin gate fails."""
    update = _build_update()
    monkeypatch.setattr(commands, "admin_only", AsyncMock(return_value=False))
    begin_mock = AsyncMock()
    monkeypatch.setattr(commands.user_service, "begin_upload_session", begin_mock)
    await commands.cmd_upload(update, SimpleNamespace())
    begin_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_cmd_upload_enables_session(monkeypatch):
    """upload command should enable upload mode for admins."""
    update = _build_update(user_id=44)
    monkeypatch.setattr(commands, "admin_only", AsyncMock(return_value=True))
    called: dict[str, int] = {}
    monkeypatch.setattr(commands.user_service, "begin_upload_session", lambda user_id: called.setdefault("id", user_id))
    await commands.cmd_upload(update, SimpleNamespace())
    assert called["id"] == 44
    update.effective_message.reply_text.assert_awaited_once()


def test_register_command_handlers():
    """Command registration should add expected two handlers (start + upload)."""
    added = []
    app = SimpleNamespace(add_handler=lambda handler: added.append(handler))
    commands.register_command_handlers(app)
    assert len(added) == 2
