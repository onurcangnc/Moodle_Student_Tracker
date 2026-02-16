"""Unit tests for short-term conversation memory behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from bot.services.conversation_memory import ConversationMemory


def test_conversation_memory_ttl_and_max_messages():
    """Memory should enforce max_messages and expire entries after TTL."""
    now = datetime(2026, 2, 16, 12, 0, tzinfo=UTC)

    def fake_now():
        return now

    memory = ConversationMemory(max_messages=5, ttl_minutes=30, now_provider=fake_now)
    user_id = 42
    for i in range(7):
        memory.add(user_id, role="user", content=f"m{i}")

    history = memory.get_history(user_id)
    assert len(history) == 5
    assert history[0]["content"] == "m2"
    assert history[-1]["content"] == "m6"

    now = now + timedelta(minutes=31)
    assert memory.get_history(user_id) == []
