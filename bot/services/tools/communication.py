"""
Communication Tools — Email access.
====================================
Tools: get_emails, get_email_detail
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from bot.services.tools import BaseTool
from core import cache_db

if TYPE_CHECKING:
    from bot.state import ServiceContainer

logger = logging.getLogger(__name__)

__all__ = ["get_communication_tools"]


class GetEmailsTool(BaseTool):
    """Get AIRS/DAIS emails from SQLite cache."""

    @property
    def name(self) -> str:
        return "get_emails"

    @property
    def description(self) -> str:
        return (
            "AIRS/DAIS e-postaları. scope: 'unread' (okunmamış), 'auto' (önce okunmamış, yoksa son), 'recent'. "
            "keyword: gönderen veya konu filtresi. 'maillerim', 'AIRS duyuruları' gibi isteklerde çağır."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Kaç mail gösterilsin (varsayılan: 5)",
                },
                "scope": {
                    "type": "string",
                    "enum": ["unread", "auto", "recent"],
                    "description": "unread/auto/recent (varsayılan: auto)",
                },
                "keyword": {
                    "type": "string",
                    "description": "Gönderen veya konu filtresi (opsiyonel)",
                },
            },
            "required": [],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        email_count = cache_db.get_email_count()
        if email_count == 0:
            return "Mail cache henüz doldurulmadı. Birkaç saniye bekleyip tekrar dene."

        count = args.get("count", 5)
        scope = args.get("scope", "auto")
        keyword = args.get("keyword", "") or args.get("sender_filter", "")

        if keyword:
            mails = cache_db.search_emails(keyword, limit=max(count, 50))
        elif scope == "unread":
            mails = cache_db.get_unread_emails()
        elif scope == "auto":
            mails = cache_db.get_unread_emails()
            if not mails:
                mails = cache_db.get_emails(count) or []
        else:
            mails = cache_db.get_emails(count) or []

        mails = mails[:count]

        if not mails:
            return "AIRS/DAIS e-postası bulunamadı."

        lines = []
        for m in mails:
            subject = m.get("subject", "Konusuz")
            from_addr = m.get("from", "")
            date = m.get("date", "")
            body = m.get("body_preview", "")
            source = m.get("source", "")
            lines.append(
                f"📧 [{source}] {subject}\n"
                f"  Kimden: {from_addr}\n"
                f"  Tarih: {date}\n"
                f"  Özet: {body[:200]}{'...' if len(body) > 200 else ''}"
            )

        return "\n\n".join(lines)


class GetEmailDetailTool(BaseTool):
    """Get full content of matching emails."""

    @property
    def name(self) -> str:
        return "get_email_detail"

    @property
    def description(self) -> str:
        return (
            "Mail detayını getirir. keyword: arama terimi (konu veya gönderen). "
            "'X mailinin detayı', 'şu maili oku' gibi isteklerde çağır."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Arama terimi (konu veya gönderen)",
                },
                "count": {
                    "type": "integer",
                    "description": "Kaç mail gösterilsin (varsayılan: 5)",
                },
            },
            "required": ["keyword"],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        keyword = args.get("keyword", "") or args.get("email_subject", "")
        if not keyword:
            return "Mail arama terimi belirtilmedi."

        count = args.get("count", 5)

        mails = cache_db.search_emails(keyword, limit=count)
        if not mails:
            return f"'{keyword}' ile eşleşen mail bulunamadı."

        lines = []
        for m in mails:
            body = m.get("body_full") or m.get("body_preview", "")
            lines.append(
                f"📧 *{m.get('subject', 'Konusuz')}*\n"
                f"Kimden: {m.get('from', '')}\n"
                f"Tarih: {m.get('date', '')}\n\n"
                f"{body}"
            )

        return "\n\n---\n\n".join(lines)


def get_communication_tools() -> list[BaseTool]:
    """Factory function returning all communication tools."""
    return [
        GetEmailsTool(),
        GetEmailDetailTool(),
    ]
