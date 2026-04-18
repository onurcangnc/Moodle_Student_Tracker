"""
Service Container for Dependency Injection.
============================================
Implements ISP (Interface Segregation) and DIP (Dependency Inversion).

ServiceContainer replaces the old monolithic STATE singleton with typed
dependencies that can be injected into services.

For backwards compatibility, STATE alias is provided but deprecated.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.services.llm_router import LLMRouter
    from bot.services.tools import ToolRegistry
    from core.document_processor import DocumentProcessor
    from core.llm_engine import LLMEngine
    from core.memory import MemoryManager
    from core.moodle_client import MoodleClient
    from core.stars_client import StarsClient
    from core.sync_engine import SyncEngine
    from core.vector_store import VectorStore
    from core.webmail_client import WebmailClient


@dataclass(slots=True)
class ServiceContainer:
    """
    Dependency Injection container for bot services.

    Groups dependencies by domain for Interface Segregation:
    - Core services: moodle, vector_store, llm, sync_engine
    - External services: stars, webmail
    - Agent services: tool_registry, llm_router
    - Runtime state: active_courses, sync_lock, etc.

    Use typed access via container.moodle instead of STATE.moodle.
    """

    # ─── Core Services ───────────────────────────────────────────────────────────
    moodle: MoodleClient | None = None
    processor: DocumentProcessor | None = None
    vector_store: VectorStore | None = None
    llm: LLMEngine | None = None
    sync_engine: SyncEngine | None = None
    memory: MemoryManager | None = None

    # ─── External Services ───────────────────────────────────────────────────────
    stars: StarsClient | None = None
    webmail: WebmailClient | None = None

    # ─── Agent Services (NEW) ────────────────────────────────────────────────────
    tool_registry: ToolRegistry | None = None
    llm_router: LLMRouter | None = None

    # ─── Sync State ──────────────────────────────────────────────────────────────
    last_sync_time: str = "Henüz yapılmadı"
    last_sync_new_files: int = 0
    sync_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ─── STARS State ─────────────────────────────────────────────────────────────
    last_stars_notification: float = 0.0
    prev_stars_snapshot: dict = field(default_factory=dict)

    # ─── User State ──────────────────────────────────────────────────────────────
    active_courses: dict[int, str] = field(default_factory=dict)
    pending_upload_users: set[int] = field(default_factory=set)
    rate_limit_windows: dict[int, list[float]] = field(default_factory=dict)
    user_last_seen: dict[int, float] = field(default_factory=dict)

    # ─── Runtime State ───────────────────────────────────────────────────────────
    file_summaries: dict[str, dict] = field(default_factory=dict)
    started_at_monotonic: float = 0.0
    startup_version: str = "unknown"
    last_update_received: float = 0.0
    last_poll_healthcheck: float = 0.0

    # ─── Backwards Compatibility Properties ──────────────────────────────────────
    @property
    def stars_client(self) -> StarsClient | None:
        """Backwards compatibility alias for stars."""
        return self.stars

    @stars_client.setter
    def stars_client(self, value: StarsClient | None) -> None:
        self.stars = value

    @property
    def webmail_client(self) -> WebmailClient | None:
        """Backwards compatibility alias for webmail."""
        return self.webmail

    @webmail_client.setter
    def webmail_client(self, value: WebmailClient | None) -> None:
        self.webmail = value

    @property
    def known_assignment_ids(self) -> set[str]:
        """Deprecated: Use cache_db for persistence."""
        # Return empty set - actual data is now in SQLite
        return set()

    @known_assignment_ids.setter
    def known_assignment_ids(self, value: set[str]) -> None:
        # No-op - data is now persisted in SQLite
        pass


# Backwards compatibility alias
BotState = ServiceContainer

# Global singleton instance
STATE = ServiceContainer()
