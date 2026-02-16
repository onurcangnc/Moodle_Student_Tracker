"""Short-lived conversation memory used for follow-up question continuity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Callable


@dataclass(slots=True)
class _MemoryBucket:
    """Per-user message bucket with last access timestamp."""

    messages: list[dict[str, str]]
    updated_at: datetime


class ConversationMemory:
    """
    Keep the last N messages per user.

    Buckets expire after `ttl_minutes` of inactivity.
    """

    def __init__(
        self,
        max_messages: int = 5,
        ttl_minutes: int = 30,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize memory with bounded size and TTL."""
        self.max_messages = max_messages
        self.ttl = timedelta(minutes=ttl_minutes)
        self._now = now_provider or (lambda: datetime.now(UTC))
        self._storage: dict[int, _MemoryBucket] = {}

    def _is_expired(self, bucket: _MemoryBucket, now: datetime) -> bool:
        """Return whether bucket exceeded inactivity TTL."""
        return (now - bucket.updated_at) > self.ttl

    def add(self, user_id: int, role: str, content: str) -> None:
        """Append a new message and enforce max history size."""
        now = self._now()
        bucket = self._storage.get(user_id)
        if bucket is None or self._is_expired(bucket, now):
            bucket = _MemoryBucket(messages=[], updated_at=now)
            self._storage[user_id] = bucket

        bucket.messages.append({"role": role, "content": content})
        if len(bucket.messages) > self.max_messages:
            bucket.messages = bucket.messages[-self.max_messages :]
        bucket.updated_at = now

    def get_history(self, user_id: int) -> list[dict[str, str]]:
        """Return non-expired message history for a user."""
        now = self._now()
        bucket = self._storage.get(user_id)
        if bucket is None:
            return []
        if self._is_expired(bucket, now):
            self._storage.pop(user_id, None)
            return []
        bucket.updated_at = now
        return list(bucket.messages)

    def clear(self, user_id: int) -> None:
        """Remove memory bucket for user."""
        self._storage.pop(user_id, None)
