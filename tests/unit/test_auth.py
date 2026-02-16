"""Unit tests for authorization helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import TelegramError

from bot.middleware import auth


def test_is_admin_user_owner_bypass(monkeypatch):
    """Owner id zero should allow all users."""
    monkeypatch.setattr(auth, "CONFIG", SimpleNamespace(owner_id=0, admin_ids=frozenset()))
    assert auth.is_admin_user(999) is True


def test_is_admin_user_owner_and_admin_list(monkeypatch):
    """Configured owner and admin IDs should be authorized."""
    monkeypatch.setattr(auth, "CONFIG", SimpleNamespace(owner_id=42, admin_ids=frozenset({7, 8})))
    assert auth.is_admin_user(42) is True
    assert auth.is_admin_user(7) is True
    assert auth.is_admin_user(9) is False


@pytest.mark.asyncio
async def test_admin_only_handles_missing_user():
    """Missing effective user should return False."""
    update = SimpleNamespace(effective_user=None, effective_message=None)
    assert await auth.admin_only(update) is False


@pytest.mark.asyncio
async def test_admin_only_allows_admin(monkeypatch):
    """Authorized user should pass without sending denial."""
    monkeypatch.setattr(auth, "CONFIG", SimpleNamespace(owner_id=10, admin_ids=frozenset()))
    msg = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=10), effective_message=msg)
    assert await auth.admin_only(update) is True
    msg.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_only_denies_and_replies(monkeypatch):
    """Unauthorized user should receive denial message."""
    monkeypatch.setattr(auth, "CONFIG", SimpleNamespace(owner_id=10, admin_ids=frozenset()))
    msg = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=99), effective_message=msg)
    assert await auth.admin_only(update) is False
    msg.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_admin_only_denies_when_reply_fails(monkeypatch):
    """Reply failure should still return False without raising."""
    monkeypatch.setattr(auth, "CONFIG", SimpleNamespace(owner_id=10, admin_ids=frozenset()))
    msg = SimpleNamespace(reply_text=AsyncMock(side_effect=TelegramError("network")))
    update = SimpleNamespace(effective_user=SimpleNamespace(id=99), effective_message=msg)
    assert await auth.admin_only(update) is False
