"""Unit tests for short-term conversation memory behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bot.services.conversation_memory import ConversationMemory


def test_conversation_memory_ttl_and_max_messages():
    """Memory should enforce max_messages and expire entries after TTL."""
    now = datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)

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


def test_memory_max_messages_limit():
    """Oldest messages should be dropped once max_messages is exceeded."""
    now = datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)

    def fake_now():
        return now

    memory = ConversationMemory(max_messages=3, ttl_minutes=30, now_provider=fake_now)
    user_id = 99
    for idx in range(5):
        memory.add(user_id, role="user", content=f"m{idx}")

    history = memory.get_history(user_id)
    assert [entry["content"] for entry in history] == ["m2", "m3", "m4"]


def test_memory_ttl_expiry():
    """Memory should expire after TTL inactivity window."""
    now = datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)

    def fake_now():
        return now

    memory = ConversationMemory(max_messages=5, ttl_minutes=30, now_provider=fake_now)
    memory.add(1, role="user", content="ilk mesaj")
    assert memory.get_history(1) != []

    now = now + timedelta(minutes=31)
    assert memory.get_history(1) == []


def test_memory_clear():
    """Clear should remove all memory for the target user."""
    now = datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)
    memory = ConversationMemory(max_messages=5, ttl_minutes=30, now_provider=lambda: now)
    memory.add(1, role="user", content="x")
    memory.clear(1)
    assert memory.get_history(1) == []


def test_memory_separate_users():
    """Separate users should maintain independent memory buckets."""
    now = datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)
    memory = ConversationMemory(max_messages=5, ttl_minutes=30, now_provider=lambda: now)
    memory.add(1, role="user", content="u1")
    memory.add(2, role="user", content="u2")

    assert memory.get_history(1) == [{"role": "user", "content": "u1"}]
    assert memory.get_history(2) == [{"role": "user", "content": "u2"}]
