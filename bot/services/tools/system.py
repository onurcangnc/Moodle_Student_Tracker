"""
System Tools — Bot stats and database access.
==============================================
Tools: get_stats, query_db
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import TYPE_CHECKING, Any

from bot.services.tools import BaseTool

if TYPE_CHECKING:
    from bot.state import ServiceContainer

logger = logging.getLogger(__name__)

__all__ = ["get_system_tools"]


class GetStatsTool(BaseTool):
    """Get bot statistics."""

    @property
    def name(self) -> str:
        return "get_stats"

    @property
    def description(self) -> str:
        return "Bot istatistikleri. 'sistem durumu', 'kaç dosya var', 'stats' gibi isteklerde çağır."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        store = services.vector_store
        if store is None:
            return "Vector store hazır değil."

        stats = store.get_stats()
        uptime = int(time.monotonic() - services.started_at_monotonic)
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)

        from bot.services.summary_service import list_summaries

        summaries = list_summaries()

        return (
            f"Toplam chunk: {stats.get('total_chunks', 0)}\n"
            f"Kurs sayısı: {stats.get('unique_courses', 0)}\n"
            f"Dosya sayısı: {stats.get('unique_files', 0)}\n"
            f"Kaynak özetleri: {len(summaries)}\n"
            f"Aktif kullanıcı: {len(services.active_courses)}\n"
            f"Uptime: {hours}s {minutes}dk {seconds}sn\n"
            f"Versiyon: {services.startup_version}"
        )


class QueryDbTool(BaseTool):
    """Execute read-only SQL query on cache DB."""

    @property
    def name(self) -> str:
        return "query_db"

    @property
    def description(self) -> str:
        return (
            "Cache veritabanında SQL sorgusu çalıştırır (sadece SELECT). "
            "Tablolar: emails, json_cache, conversations. Debug/analiz için."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL SELECT sorgusu",
                },
            },
            "required": ["sql"],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        sql = (args.get("sql") or "").strip()
        if not sql:
            return "SQL sorgusu belirtilmedi."

        sql_upper = sql.upper().lstrip()
        if not sql_upper.startswith("SELECT"):
            return "Sadece SELECT sorguları desteklenir. INSERT/UPDATE/DELETE yasaktır."

        _BLOCKED = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "ATTACH", "DETACH", "PRAGMA"]
        for keyword in _BLOCKED:
            if keyword in sql_upper:
                return f"Güvenlik: '{keyword}' içeren sorgular yasaktır."

        db_path = "data/cache.db"
        try:
            conn = sqlite3.connect(db_path, timeout=5)
            conn.execute("PRAGMA query_only = ON")
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(50)
            conn.close()

            if not rows:
                return "Sorgu sonucu boş."

            lines = [" | ".join(columns)]
            lines.append("-" * len(lines[0]))
            for row in rows:
                lines.append(" | ".join(str(v)[:200] if v is not None else "NULL" for v in row))

            result = "\n".join(lines)
            if len(rows) == 50:
                result += "\n... (50 satır limiti)"
            return result

        except sqlite3.Error as exc:
            return f"SQL hatası: {exc}"


def get_system_tools() -> list[BaseTool]:
    """Factory function returning all system tools."""
    return [
        GetStatsTool(),
        QueryDbTool(),
    ]
