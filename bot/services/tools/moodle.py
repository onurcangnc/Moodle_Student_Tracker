"""
Moodle Tools — Moodle API access.
=================================
Tools: get_moodle_materials, get_assignments, list_courses, set_active_course
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from bot.services import user_service
from bot.services.tools import BaseTool
from bot.services.tools.helpers import resolve_course

if TYPE_CHECKING:
    from bot.state import ServiceContainer

logger = logging.getLogger(__name__)

__all__ = ["get_moodle_tools"]


class GetMoodleMaterialsTool(BaseTool):
    """Get materials directly from Moodle API."""

    @property
    def name(self) -> str:
        return "get_moodle_materials"

    @property
    def description(self) -> str:
        return (
            "Moodle'dan kursun materyal/kaynak listesini doğrudan Moodle API'sinden getirir. "
            "'Moodle'da ne var', 'en güncel materyaller', 'haftalık içerik' gibi isteklerde kullan."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "course_filter": {
                    "type": "string",
                    "description": "Kurs adı (opsiyonel)",
                },
            },
            "required": [],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        moodle = services.moodle
        if moodle is None:
            return "Moodle bağlantısı hazır değil."

        course_name = resolve_course(args, user_id)

        try:
            courses = await asyncio.to_thread(moodle.get_courses)
        except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
            logger.error("Moodle courses fetch failed: %s", exc, exc_info=True)
            return f"Moodle'a bağlanılamadı: {exc}"

        target = None
        if course_name:
            cn_lower = course_name.lower()
            for c in courses:
                if cn_lower in c.fullname.lower() or cn_lower in c.shortname.lower():
                    target = c
                    break

        if not target and courses:
            target = courses[0]

        if not target:
            return "Kurs bulunamadı."

        try:
            text = await asyncio.to_thread(moodle.get_course_topics_text, target)
        except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
            logger.error("Moodle topics fetch failed: %s", exc, exc_info=True)
            return f"Moodle içeriği alınamadı: {exc}"

        if not text:
            return f"'{target.fullname}' kursunda içerik bulunamadı."

        if len(text) > 3000:
            text = text[:3000] + "\n\n[... kısaltıldı ...]"

        return text


class GetAssignmentsTool(BaseTool):
    """Get Moodle assignments with filtering."""

    @property
    def name(self) -> str:
        return "get_assignments"

    @property
    def description(self) -> str:
        return (
            "Moodle ödevleri. filter: 'upcoming' (yaklaşan), 'overdue' (gecikmiş), 'all'. "
            "keyword: ders/ödev adı filtresi. 'ödevlerim', 'teslim tarihleri' gibi isteklerde çağır."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "enum": ["upcoming", "overdue", "all"],
                    "description": "upcoming (varsayılan), overdue veya all",
                },
                "keyword": {
                    "type": "string",
                    "description": "Ders veya ödev adı filtresi (opsiyonel)",
                },
            },
            "required": [],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        moodle = services.moodle
        if moodle is None:
            return "Moodle bağlantısı hazır değil."

        filter_mode = args.get("filter", "upcoming")
        keyword = args.get("keyword", "").strip()
        now_ts = time.time()

        try:
            if filter_mode == "all" or keyword:
                assignments = await asyncio.to_thread(moodle.get_assignments)
            else:
                assignments = await asyncio.to_thread(moodle.get_upcoming_assignments, 14)
        except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
            logger.error("Assignment fetch failed: %s", exc, exc_info=True)
            return f"Ödev bilgileri alınamadı: {exc}"

        if filter_mode == "overdue":
            assignments = [
                a for a in (assignments or [])
                if not a.submitted and a.due_date and a.due_date < now_ts
            ]

        if keyword and assignments:
            kw_lower = keyword.lower()
            assignments = [a for a in assignments if kw_lower in f"{a.course_name} {a.name}".lower()]

        if not assignments:
            labels = {"upcoming": "Yaklaşan", "overdue": "Süresi geçmiş", "all": "Hiç"}
            if keyword:
                return f"'{keyword}' ile eşleşen ödev bulunamadı."
            return f"{labels.get(filter_mode, 'Yaklaşan')} ödev bulunamadı."

        lines = []
        for a in assignments:
            status = "✅ Teslim edildi" if a.submitted else "⏳ Teslim edilmedi"
            if hasattr(a, "due_date") and a.due_date and a.due_date > 0:
                due_dt = datetime.fromtimestamp(a.due_date)
                due = due_dt.strftime("%d/%m/%Y %H:%M")
            else:
                due = "Son tarih yok"
            remaining = a.time_remaining if hasattr(a, "time_remaining") else ""
            line = f"• {a.course_name} — {a.name}\n  Tarih: {due} | {status}"
            if remaining and not a.submitted:
                line += f" | Kalan: {remaining}"
            if filter_mode == "overdue":
                line += " | ⚠️ Süresi geçmiş!"
            lines.append(line)

        return "\n".join(lines)


class ListCoursesTool(BaseTool):
    """List available courses."""

    @property
    def name(self) -> str:
        return "list_courses"

    @property
    def description(self) -> str:
        return "Mevcut kursları listeler. 'Hangi dersler var?', 'kurslarım' gibi isteklerde çağır."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        courses = user_service.list_courses()
        if not courses:
            return "Henüz yüklü kurs bulunamadı."

        active = user_service.get_active_course(user_id)
        lines = []
        for c in courses:
            prefix = "▸ " if active and active.course_id == c.course_id else "  "
            lines.append(f"{prefix}{c.short_name} — {c.display_name}")

        return "\n".join(lines)


class SetActiveCourseTool(BaseTool):
    """Set active course for user."""

    @property
    def name(self) -> str:
        return "set_active_course"

    @property
    def description(self) -> str:
        return (
            "Kullanıcının aktif kursunu ayarlar. 'Ethics dersiyle çalışayım', "
            "'CTIS 474'e geç' gibi isteklerde çağır."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "course_name": {
                    "type": "string",
                    "description": "Kurs adı veya kodu",
                },
            },
            "required": ["course_name"],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        course_name = args.get("course_name", "")
        if not course_name:
            return "Kurs adı belirtilmedi."

        match = user_service.find_course(course_name)
        if match is None:
            courses = user_service.list_courses()
            available = ", ".join(c.short_name for c in courses) if courses else "Yok"
            return f"'{course_name}' ile eşleşen kurs bulunamadı. Mevcut kurslar: {available}"

        user_service.set_active_course(user_id, match.course_id)
        if services.llm:
            services.llm.set_active_course(match.course_id)
        return f"Aktif kurs değiştirildi: {match.display_name}"


def get_moodle_tools() -> list[BaseTool]:
    """Factory function returning all Moodle tools."""
    return [
        GetMoodleMaterialsTool(),
        GetAssignmentsTool(),
        ListCoursesTool(),
        SetActiveCourseTool(),
    ]
