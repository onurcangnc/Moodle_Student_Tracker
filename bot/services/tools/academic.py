"""
Academic Tools — STARS data access.
====================================
Tools: get_schedule, get_grades, get_attendance, get_exams, get_transcript, get_letter_grades
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from bot.services.tools import BaseTool
from bot.services.tools.helpers import DAY_NAMES_TR, course_matches
from core import cache_db

if TYPE_CHECKING:
    from bot.state import ServiceContainer

logger = logging.getLogger(__name__)

__all__ = ["get_academic_tools"]


class GetScheduleTool(BaseTool):
    """Get class schedule from STARS cache."""

    @property
    def name(self) -> str:
        return "get_schedule"

    @property
    def description(self) -> str:
        return (
            "Ders programı. 'Bugün derslerim' → today, 'yarın ne var' → tomorrow, "
            "'haftalık' → week. SADECE sorulan dönemi getir. STARS girişi gerektirir."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["today", "tomorrow", "week"],
                    "description": "today/tomorrow/week (varsayılan: today)",
                },
            },
            "required": ["period"],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        schedule = cache_db.get_json("schedule", user_id)

        if not schedule:
            return "Ders programı bulunamadı. STARS session süresi dolmuş olabilir — /start ile tekrar giriş yap."

        period = args.get("period", "today")

        if period in ("today", "tomorrow"):
            now = datetime.now()
            target = now + timedelta(days=1) if period == "tomorrow" else now
            target_day = DAY_NAMES_TR.get(target.weekday(), "")
            schedule = [e for e in schedule if e.get("day", "") == target_day]
            if not schedule:
                return f"{target_day} günü için ders bulunamadı."

        lines = []
        current_day = ""
        for entry in schedule:
            day = entry.get("day", "")
            time_slot = entry.get("time", "")
            course = entry.get("course", "")
            room = entry.get("room", "")
            if day != current_day:
                current_day = day
                lines.append(f"\n*{day}*")
            room_str = f" ({room})" if room else ""
            lines.append(f"  • {time_slot} — {course}{room_str}")

        return "\n".join(lines).strip() if lines else "Ders programı boş."


class GetGradesTool(BaseTool):
    """Get grades from STARS cache."""

    @property
    def name(self) -> str:
        return "get_grades"

    @property
    def description(self) -> str:
        return (
            "Not bilgileri. Spesifik ders sorulursa SADECE o dersi getir. "
            "'Notlarım' → tüm dersler. STARS girişi gerektirir."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "course_filter": {
                    "type": "string",
                    "description": "Ders adı (opsiyonel)",
                },
            },
            "required": [],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        grades = cache_db.get_json("grades", user_id)

        if not grades:
            return "Not bilgisi bulunamadı. STARS session süresi dolmuş olabilir — /start ile tekrar giriş yap."

        course_filter = args.get("course_filter", "")
        if course_filter:
            grades = [g for g in grades if course_matches(g.get("course", ""), course_filter)]
            if not grades:
                return f"'{course_filter}' ile eşleşen kurs notu bulunamadı."

        lines = []
        for course in grades:
            cname = course.get("course", "Bilinmeyen")
            assessments = course.get("assessments", [])
            if not assessments:
                lines.append(f"📚 {cname}: Henüz not girilmemiş")
                continue
            lines.append(f"📚 {cname}:")
            for a in assessments:
                name = a.get("name", "")
                grade = a.get("grade", "")
                atype = a.get("type", "")
                date = a.get("date", "")
                weight = a.get("weight", "")
                extras = []
                if atype:
                    extras.append(atype)
                if date:
                    extras.append(date)
                if weight:
                    extras.append(f"Ağırlık: {weight}")
                extra_str = f" ({', '.join(extras)})" if extras else ""
                lines.append(f"  • {name}: {grade}{extra_str}")

        return "\n".join(lines)


class GetAttendanceTool(BaseTool):
    """Get attendance from STARS cache + syllabus limits."""

    @property
    def name(self) -> str:
        return "get_attendance"

    @property
    def description(self) -> str:
        return (
            "Devamsızlık bilgisi + syllabus'tan max limit + kalan hak hesabı. "
            "STARS'tan mevcut devamsızlık, RAG'den syllabus limiti çeker. "
            "Spesifik ders sorulursa SADECE o dersi getir."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "course_filter": {
                    "type": "string",
                    "description": "Ders adı (opsiyonel)",
                },
            },
            "required": [],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        attendance = cache_db.get_json("attendance", user_id)

        if not attendance:
            return "Devamsızlık bilgisi bulunamadı. STARS session süresi dolmuş olabilir — /start ile tekrar giriş yap."

        course_filter = args.get("course_filter", "")
        if course_filter:
            attendance = [a for a in attendance if course_matches(a.get("course", ""), course_filter)]
            if not attendance:
                return f"'{course_filter}' ile eşleşen kurs devamsızlığı bulunamadı."

        syllabus_limits: dict[str, int] = cache_db.get_json("syllabus_limits", user_id) or {}

        def _calc_missed_hours(records: list[dict]) -> int:
            total_missed = 0
            for r in records:
                raw = r.get("raw", "")
                if "/" in raw:
                    try:
                        parts = raw.replace(" ", "").split("/")
                        attended_h = int(parts[0])
                        total_h = int(parts[1])
                        total_missed += total_h - attended_h
                    except (ValueError, IndexError):
                        pass
            return total_missed

        lines = []
        for cd in attendance:
            cname = cd.get("course", "Bilinmeyen")
            records = cd.get("records", [])
            ratio = cd.get("ratio", "")

            absent_sessions = sum(1 for r in records if not r.get("attended", True))
            hours_absent = _calc_missed_hours(records)

            line = f"📚 {cname}:"
            if ratio:
                line += f" Devam: {ratio}"
            line += f" ({hours_absent} saat devamsız / {absent_sessions} ders)"

            max_hours = syllabus_limits.get(cname)
            if max_hours and max_hours > 0:
                remaining_hours = max(0, max_hours - hours_absent)
                line += f"\n  📋 Syllabus limiti: max {max_hours} saat"
                if remaining_hours > 0:
                    line += f" → {remaining_hours} saat hakkın kaldı ✅"
                else:
                    line += f" → ⚠️ LİMİT AŞILDI! ({hours_absent - max_hours} saat fazla)"
            else:
                try:
                    ratio_num = float(ratio.replace("%", "")) if ratio else 100
                    if ratio_num < 85:
                        line += "\n  ⚠️ Dikkat: Devamsızlık limiti %20'ye yaklaşıyor!"
                except (ValueError, AttributeError):
                    pass

            lines.append(line)

        return "\n".join(lines)


class GetExamsTool(BaseTool):
    """Get exam schedule from STARS cache."""

    @property
    def name(self) -> str:
        return "get_exams"

    @property
    def description(self) -> str:
        return (
            "Sınav takvimi. 'sınavlarım', 'exam schedule', 'ne zaman sınav', "
            "'midterm ne zaman', 'final tarihleri' gibi isteklerde çağır."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "course_filter": {
                    "type": "string",
                    "description": "Ders adı filtresi (opsiyonel)",
                },
            },
            "required": [],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        exams = cache_db.get_json("exams", user_id)

        if not exams:
            return "Sınav takvimi bulunamadı. STARS session süresi dolmuş olabilir — /start ile tekrar giriş yap."

        course_filter = args.get("course_filter", "")
        if course_filter:
            exams = [e for e in exams if course_matches(e.get("course", ""), course_filter)]
            if not exams:
                return f"'{course_filter}' ile eşleşen sınav bulunamadı."

        lines = []
        for e in exams:
            course = e.get("course", "")
            exam_name = e.get("exam_name", "Sınav")
            date = e.get("date", "")
            start_time = e.get("start_time", "")
            time_block = e.get("time_block", "")
            time_remaining = e.get("time_remaining", "")

            line = f"📝 *{course}* — {exam_name}"
            if date:
                line += f"\n  📅 {date}"
                if start_time:
                    line += f", {start_time}"
                elif time_block:
                    line += f", {time_block}"
            if time_remaining:
                line += f"\n  ⏳ {time_remaining}"
            lines.append(line)

        return "\n\n".join(lines)


class GetTranscriptTool(BaseTool):
    """Get academic transcript from STARS."""

    @property
    def name(self) -> str:
        return "get_transcript"

    @property
    def description(self) -> str:
        return (
            "Akademik transkript — tüm dönemler ve dersler. 'transkript', "
            "'geçmiş notlarım', 'transcript', 'tüm derslerim' gibi isteklerde çağır."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        stars = services.stars
        transcript = None

        if stars and stars.is_authenticated(user_id):
            try:
                transcript = await asyncio.to_thread(stars.get_transcript, user_id)
            except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
                logger.warning("Transcript live fetch failed, trying cache: %s", exc)

        if not transcript:
            transcript = cache_db.get_json("transcript", user_id)
            if transcript:
                logger.info("Transcript served from cache for user %d", user_id)

        if not transcript:
            return "Transkript bulunamadı. STARS session süresi dolmuş olabilir — /start ile tekrar giriş yap."

        lines = ["📋 *Transkript*\n"]
        current_semester = ""
        for entry in transcript:
            semester = entry.get("semester", "")
            if semester != current_semester:
                current_semester = semester
                lines.append(f"\n*{semester}*")

            code = entry.get("code", "")
            name = entry.get("name", "")
            grade = entry.get("grade", "")
            credits = entry.get("credits", "")

            line = f"  • {code} {name}"
            if grade:
                line += f" — {grade}"
            if credits:
                line += f" ({credits} kr)"
            lines.append(line)

        return "\n".join(lines)


class GetLetterGradesTool(BaseTool):
    """Get letter grades from STARS."""

    @property
    def name(self) -> str:
        return "get_letter_grades"

    @property
    def description(self) -> str:
        return (
            "Harf notları — dönem bazlı. 'harf notlarım', 'letter grades', "
            "'final notları' gibi isteklerde çağır."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "semester_filter": {
                    "type": "string",
                    "description": "Dönem filtresi (opsiyonel, örn: '2024-2025 Fall')",
                },
            },
            "required": [],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        stars = services.stars
        letter_grades = None

        if stars and stars.is_authenticated(user_id):
            try:
                letter_grades = await asyncio.to_thread(stars.get_letter_grades, user_id)
            except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
                logger.warning("Letter grades live fetch failed, trying cache: %s", exc)

        if not letter_grades:
            letter_grades = cache_db.get_json("letter_grades", user_id)
            if letter_grades:
                logger.info("Letter grades served from cache for user %d", user_id)

        if not letter_grades:
            return "Harf notları bulunamadı. STARS session süresi dolmuş olabilir — /start ile tekrar giriş yap."

        semester_filter = args.get("semester_filter", "")

        lines = ["📊 *Harf Notları*\n"]
        for sem in letter_grades:
            semester = sem.get("semester", "")
            if semester_filter and semester_filter.lower() not in semester.lower():
                continue

            lines.append(f"\n*{semester}*")
            for c in sem.get("courses", []):
                code = c.get("code", "")
                name = c.get("name", "")
                grade = c.get("grade", "")
                lines.append(f"  • {code} {name} — {grade}")

        if len(lines) == 1:
            return f"'{semester_filter}' dönemi ile eşleşen not bulunamadı."

        return "\n".join(lines)


def get_academic_tools() -> list[BaseTool]:
    """Factory function returning all academic tools."""
    return [
        GetScheduleTool(),
        GetGradesTool(),
        GetAttendanceTool(),
        GetExamsTool(),
        GetTranscriptTool(),
        GetLetterGradesTool(),
    ]
