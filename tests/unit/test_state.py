"""Unit tests for global bot state container defaults."""

from __future__ import annotations

from bot.state import BotState


def test_bot_state_defaults():
    """BotState should initialize empty mutable containers."""
    state = BotState()
    assert state.moodle is None
    assert state.vector_store is None
    assert state.active_courses == {}
    assert state.pending_upload_users == set()
    assert state.rate_limit_windows == {}
    assert state.file_summaries == {}
