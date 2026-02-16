"""Shared bot runtime state container for modular application services."""

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

    active_courses: dict[int, str] = field(default_factory=dict)
    pending_upload_users: set[int] = field(default_factory=set)
    rate_limit_windows: dict[int, list[float]] = field(default_factory=dict)
    user_last_seen: dict[int, float] = field(default_factory=dict)
    conversation_history: dict[int, list[dict[str, str]]] = field(default_factory=dict)
    file_summaries: dict[str, dict] = field(default_factory=dict)
    started_at_monotonic: float = 0.0
    startup_version: str = "unknown"


STATE = BotState()
