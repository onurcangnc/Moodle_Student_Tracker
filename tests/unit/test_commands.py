"""Unit tests for Telegram command handlers."""

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


@pytest.mark.asyncio
async def test_cmd_start_replies_welcome():
    """start command should send welcome guidance."""
    update = _build_update()
    await commands.cmd_start(update, SimpleNamespace())
    update.effective_message.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_cmd_help_replies_usage():
    """help command should send short usage text."""
    update = _build_update()
    await commands.cmd_help(update, SimpleNamespace())
    update.effective_message.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_cmd_courses_returns_when_user_missing(monkeypatch):
    """courses command should no-op when effective user is missing."""
    update = _build_update(user_id=None)
    monkeypatch.setattr(commands.user_service, "list_courses", lambda: [])
    await commands.cmd_courses(update, SimpleNamespace(args=[]))
    update.effective_message.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_cmd_courses_no_courses(monkeypatch):
    """courses command should report empty course set."""
    update = _build_update()
    monkeypatch.setattr(commands.user_service, "list_courses", lambda: [])
    await commands.cmd_courses(update, SimpleNamespace(args=[]))
    update.effective_message.reply_text.assert_awaited_once_with("Henuz yuklu kurs bulunamadi.")


@pytest.mark.asyncio
async def test_cmd_courses_set_active_not_found(monkeypatch):
    """courses command should report non-matching selection queries."""
    update = _build_update()
    monkeypatch.setattr(commands.user_service, "list_courses", lambda: [SimpleNamespace(course_id="x")])
    monkeypatch.setattr(commands.user_service, "find_course", lambda query: None)
    await commands.cmd_courses(update, SimpleNamespace(args=["CTIS", "999"]))
    update.effective_message.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_cmd_courses_set_active_success(monkeypatch):
    """courses command should set active course and confirm selection."""
    update = _build_update(user_id=77)
    match = SimpleNamespace(course_id="CTIS 363", display_name="CTIS 363 Ethics")
    monkeypatch.setattr(commands.user_service, "list_courses", lambda: [match])
    monkeypatch.setattr(commands.user_service, "find_course", lambda query: match)

    called: dict[str, object] = {}

    def _set_active(user_id: int, course_id: str) -> None:
        called["user_id"] = user_id
        called["course_id"] = course_id

    monkeypatch.setattr(commands.user_service, "set_active_course", _set_active)
    await commands.cmd_courses(update, SimpleNamespace(args=["CTIS", "363"]))

    assert called == {"user_id": 77, "course_id": "CTIS 363"}
    update.effective_message.reply_text.assert_awaited_once_with("Aktif kurs secildi: CTIS 363 Ethics")


@pytest.mark.asyncio
async def test_cmd_courses_lists_with_active_marker(monkeypatch):
    """courses command should mark active course in list output."""
    update = _build_update(user_id=5)
    courses = [
        SimpleNamespace(course_id="CTIS 363", short_name="CTIS", display_name="CTIS 363 Ethics"),
        SimpleNamespace(course_id="CTIS 465", short_name="CTIS", display_name="CTIS 465 Cloud"),
    ]
    monkeypatch.setattr(commands.user_service, "list_courses", lambda: courses)
    monkeypatch.setattr(commands.user_service, "get_active_course", lambda user_id: courses[0])
    await commands.cmd_courses(update, SimpleNamespace(args=[]))
    payload = update.effective_message.reply_text.await_args_list[0].args[0]
    assert "* CTIS | CTIS 363 Ethics" in payload
    assert "- CTIS | CTIS 465 Cloud" in payload


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


@pytest.mark.asyncio
async def test_cmd_stats_without_store(monkeypatch):
    """stats command should report unavailable vector store."""
    update = _build_update()
    monkeypatch.setattr(commands, "admin_only", AsyncMock(return_value=True))
    monkeypatch.setattr(commands.STATE, "vector_store", None)
    await commands.cmd_stats(update, SimpleNamespace())
    update.effective_message.reply_text.assert_awaited_once_with("Vector store henuz hazir degil.")


@pytest.mark.asyncio
async def test_cmd_stats_with_store(monkeypatch):
    """stats command should include basic operational counters."""
    update = _build_update()
    monkeypatch.setattr(commands, "admin_only", AsyncMock(return_value=True))
    monkeypatch.setattr(
        commands.STATE,
        "vector_store",
        SimpleNamespace(get_stats=lambda: {"total_chunks": 20, "unique_courses": 2, "unique_files": 8}),
    )
    monkeypatch.setattr(commands.STATE, "active_courses", {1: "a", 2: "b"})
    monkeypatch.setattr(commands.STATE, "pending_upload_users", {1})
    await commands.cmd_stats(update, SimpleNamespace())
    payload = update.effective_message.reply_text.await_args_list[0].args[0]
    assert "- Toplam chunk: 20" in payload
    assert "- Kurs sayisi: 2" in payload
    assert "- Dosya sayisi: 8" in payload


def test_register_command_handlers():
    """Command registration should add expected five handlers."""
    added = []
    app = SimpleNamespace(add_handler=lambda handler: added.append(handler))
    commands.register_command_handlers(app)
    assert len(added) == 5
