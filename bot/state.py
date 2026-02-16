"""Shared bot runtime state.

Contains mutable process-level state previously kept as scattered globals.
Compatibility sync helpers allow gradual migration from legacy globals.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.document_processor import DocumentProcessor
    from core.llm_engine import LLMEngine
    from core.memory import MemoryManager
    from core.moodle_client import MoodleClient
    from core.stars_client import StarsClient
    from core.sync_engine import SyncEngine
    from core.vector_store import VectorStore
    from core.webmail_client import WebmailClient


@dataclass(slots=True)
class BotState:
    """Container for mutable bot state shared across modules."""

    moodle: MoodleClient | None = None
    processor: DocumentProcessor | None = None
    vector_store: VectorStore | None = None
    llm: LLMEngine | None = None
    sync_engine: SyncEngine | None = None
    memory: MemoryManager | None = None
    stars_client: StarsClient | None = None
    webmail_client: WebmailClient | None = None

    last_sync_time: str = "Henüz yapılmadı"
    last_sync_new_files: int = 0
    known_assignment_ids: set[str] = field(default_factory=set)
    sync_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_stars_notification: float = 0.0
    prev_stars_snapshot: dict = field(default_factory=dict)

    conversation_history: dict[int, dict] = field(default_factory=dict)
    user_state: dict[int, dict] = field(default_factory=dict)
    file_summaries: dict[str, dict] = field(default_factory=dict)


STATE = BotState()


def sync_from_legacy(legacy_module: object) -> BotState:
    """Mirror legacy module globals into the shared `STATE` object."""
    STATE.moodle = getattr(legacy_module, "moodle", None)
    STATE.processor = getattr(legacy_module, "processor", None)
    STATE.vector_store = getattr(legacy_module, "vector_store", None)
    STATE.llm = getattr(legacy_module, "llm", None)
    STATE.sync_engine = getattr(legacy_module, "sync_engine", None)
    STATE.memory = getattr(legacy_module, "memory", None)
    STATE.stars_client = getattr(legacy_module, "stars_client", None)
    STATE.webmail_client = getattr(legacy_module, "webmail_client", None)

    STATE.last_sync_time = getattr(legacy_module, "last_sync_time", STATE.last_sync_time)
    STATE.last_sync_new_files = getattr(legacy_module, "last_sync_new_files", STATE.last_sync_new_files)
    STATE.known_assignment_ids = getattr(legacy_module, "known_assignment_ids", STATE.known_assignment_ids)
    STATE.sync_lock = getattr(legacy_module, "sync_lock", STATE.sync_lock)
    STATE.last_stars_notification = getattr(
        legacy_module,
        "last_stars_notification",
        STATE.last_stars_notification,
    )
    STATE.prev_stars_snapshot = getattr(
        legacy_module,
        "_prev_stars_snapshot",
        STATE.prev_stars_snapshot,
    )
    STATE.conversation_history = getattr(legacy_module, "conversation_history", STATE.conversation_history)
    STATE.user_state = getattr(legacy_module, "_user_state", STATE.user_state)
    STATE.file_summaries = getattr(legacy_module, "file_summaries", STATE.file_summaries)
    return STATE
