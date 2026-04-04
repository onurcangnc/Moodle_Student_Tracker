"""
Background notification service using PTB job queue.
=====================================================
Every job does two things:
  1. Detect changes → notify owner via Telegram
  2. Write fresh data to SQLite cache (so tool handlers read from cache, not live API)

Job schedule:
  stars_full_sync    — 1 min   (keep-alive + ALL STARS data → cache, near real-time)
  assignment_check   — 10 min  (new assignments + cache refresh)
  email_check        — 5 min   (new mails notification)
  email_cache_sync   — 30 sec  (FULL IMAP → SQLite sync, agent queries instant)
  grades_sync        — 30 min  (new grades NOTIFICATION only — cache already fresh from full_sync)
  attendance_sync    — 60 min  (low attendance alert — cache already fresh from full_sync)
  exam_reminder      — 1 h     (1-day-before exam alerts with room info from mail)
  deadline_reminder  — 30 min  (upcoming deadline alerts)
  session_refresh    — 24 h    (re-login webmail + STARS once per day)
  summary_generation — 60 min  (KATMAN 2 source summaries)
  material_sync      — 30 min  (Moodle → vector store, auto-index new materials)

Architecture: Cache-first reads. User queries read from SQLite (instant).
Background sync updates cache every 30 sec for emails, 1 min for STARS.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timedelta

from telegram.ext import Application, ContextTypes

from bot.config import CONFIG
from bot.state import STATE
from core import cache_db

logger = logging.getLogger(__name__)

OWNER_ID = CONFIG.owner_id

# Attendance warning threshold (%) — fallback when no syllabus limit found
_ATTENDANCE_WARN_THRESHOLD = 85.0

# Notify when this many absence slots remain (syllabus-based tracking)
_ABSENCE_WARN_REMAINING = 3   # ⚠️ warning
_ABSENCE_CRIT_REMAINING = 1   # 🚨 critical

# Regex patterns to extract max absence hours from syllabus text.
# Ordered most-specific → least-specific; first match wins.
_ABSENCE_PATTERNS = [
    # "miss more than 12 hrs of lecture"
    # "miss more than 10-class hours"   ← dash+word before "hours"
    # "miss more than 10 class hours"
    re.compile(r"miss\s+more\s+than\s+(\d+)[^.\n]{0,20}?hours?", re.IGNORECASE),
    # Same but with "hrs" abbreviation: "miss more than 12 hrs"
    re.compile(r"miss\s+more\s+than\s+(\d+)[^.\n]{0,10}?hrs?\b", re.IGNORECASE),
    # "maximum 12 hours of absence"
    re.compile(r"maximum\s+(\d+)\s*hours?", re.IGNORECASE),
    # "absence limit: 12" / "absence limit 12 hours"
    re.compile(r"absence\s+limit[:\s]+(\d+)", re.IGNORECASE),
    # "(\d+) hours of absence"
    re.compile(r"(\d+)\s*hours?\s+of\s+absence", re.IGNORECASE),
    # "less than 19 lecture hours of absence"
    re.compile(r"less\s+than\s+(\d+)\s*(?:lecture\s+)?hours?\s*(?:of\s+)?absence", re.IGNORECASE),
    # Turkish: "devamsızlık hakkı: 12 saat" / "12 saatlik devamsızlık hakkı"
    re.compile(r"devams[ıi]zl[ıi]k\s+hakk[ıi][:\s]+(\d+)\s*saat", re.IGNORECASE),
    re.compile(r"(\d+)\s*saatlik\s+devams[ıi]zl[ıi]k", re.IGNORECASE),
    # Loose: "12 hrs" anywhere near "lecture" in the same line
    re.compile(r"(\d+)\s*hrs?[^.\n]{0,30}lecture", re.IGNORECASE),
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _serialize_assignments(assignments: list) -> list[dict]:
    """Convert assignment objects → JSON-serializable dicts."""
    result = []
    for a in assignments or []:
        result.append({
            "name":          getattr(a, "name", ""),
            "course_name":   getattr(a, "course_name", ""),
            "submitted":     getattr(a, "submitted", False),
            "due_date":      getattr(a, "due_date", None),
            "time_remaining": getattr(a, "time_remaining", ""),
        })
    return result


def _grade_keys(grades: list[dict]) -> set[tuple]:
    """Build set of (course, assessment_name) tuples that have a non-empty grade."""
    keys = set()
    for course in grades or []:
        cname = course.get("course", "")
        for a in course.get("assessments", []):
            if a.get("grade"):
                keys.add((cname, a.get("name", "")))
    return keys


def _attendance_ratios(attendance: list[dict]) -> dict[str, float]:
    """Build {course_name: ratio_float} dict from attendance list."""
    ratios = {}
    for cd in attendance or []:
        cname = cd.get("course", "")
        try:
            ratio = float(cd.get("ratio", "100").replace("%", ""))
        except (ValueError, AttributeError):
            ratio = 100.0
        ratios[cname] = ratio
    return ratios


_COURSE_CODE_RE = re.compile(r"^([A-Z]{2,}\s*\d{3}[A-Z]?)\b")


def _short_course_code(course_name: str) -> str:
    """
    Extract the short course code from a full STARS course name.

    STARS returns long names like "HCIV 201 Science and Technology in History".
    RAG metadata stores the short code "HCIV 201" (or similar prefix).
    The vector_store course_filter checks: filter.lower() in meta["course"].lower()
    So we need the SHORT code so it is a substring of the metadata value.
    """
    m = _COURSE_CODE_RE.match(course_name.strip())
    return m.group(1).strip() if m else course_name


def _extract_syllabus_attendance_limit(course_name: str) -> int | None:
    """
    Dynamically search RAG for any course's syllabus and extract the max
    absence limit. Works for every enrolled course — nothing hardcoded.

    Strategy:
    1. Derive short code from STARS full name (e.g. "HCIV 201 Sci..." → "HCIV 201")
    2. FIRST: Check files with "syllabus" or "course details" in name (Bilkent convention)
    3. Then: Query RAG with course_filter=short_code
    4. Fallback: query without filter but embed course name in query text
    Returns integer hour limit, or None if not found / no syllabus uploaded.
    """
    store = STATE.vector_store
    if store is None:
        return None

    short_code = _short_course_code(course_name)

    def _extract_from_texts(texts: list[str]) -> int | None:
        """Apply absence patterns to combined texts."""
        combined = "\n".join(texts)
        for pattern in _ABSENCE_PATTERNS:
            m = pattern.search(combined)
            if m:
                val = int(m.group(1))
                if 4 <= val <= 50:
                    return val
        return None

    def _search_and_extract(queries: list[tuple[str, str | None]]) -> int | None:
        """Search RAG and extract limit from results."""
        seen_texts: list[str] = []
        for query, cf in queries:
            try:
                hits = store.query(query, n_results=5, course_filter=cf)
                for hit in hits or []:
                    text = hit.get("text", "")
                    if text and text not in seen_texts:
                        seen_texts.append(text)
            except Exception as exc:
                logger.debug("Syllabus RAG query failed for %s: %s", course_name, exc)
        return _extract_from_texts(seen_texts)

    # Step 0: Bilkent convention — first doc is usually syllabus
    # Directly read files named "syllabus*" or "course details*" (case-insensitive)
    try:
        files = store.get_files_for_course(short_code)
        syllabus_files = [
            f for f in files
            if any(kw in f.get("filename", "").lower() for kw in ["syllabus", "course_details", "course details"])
        ]
        for sf in syllabus_files:
            chunks = store.get_file_chunks(sf["filename"], max_chunks=20)
            texts = [c.get("text", "") for c in chunks if c.get("text")]
            limit = _extract_from_texts(texts)
            if limit is not None:
                logger.info(
                    "Syllabus attendance limit found for %s (file=%s): %d h",
                    course_name, sf["filename"], limit,
                )
                return limit
    except Exception as exc:
        logger.debug("Direct syllabus file check failed for %s: %s", course_name, exc)

    # Step 1: Try filtered queries (course-specific results only)
    filtered_queries = [
        ("syllabus attendance absence limit hours miss lecture", short_code),
        ("devamsızlık saat limit hakkı miss", short_code),
        ("minimum requirements qualify final exam", short_code),
        ("course details attendance absence hours miss", short_code),
        ("week 1 attendance policy absence limit", short_code),
    ]
    limit = _search_and_extract(filtered_queries)
    if limit is not None:
        logger.info(
            "Syllabus attendance limit found for %s (code=%s): %d h",
            course_name, short_code, limit,
        )
        return limit

    # Step 2: Fallback to unfiltered queries (course code in query text)
    fallback_queries = [
        (f"{short_code} syllabus attendance miss hours absence limit", None),
        (f"{short_code} course details minimum requirements absence", None),
    ]
    limit = _search_and_extract(fallback_queries)
    if limit is not None:
        logger.info(
            "Syllabus attendance limit found for %s (code=%s, fallback): %d h",
            course_name, short_code, limit,
        )
        return limit

    return None


def _count_absences(records: list[dict]) -> int:
    """Count sessions where the student was absent."""
    return sum(1 for r in records if not r.get("attended", True))


async def _send(context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    try:
        await context.bot.send_message(
            chat_id=OWNER_ID, text=text, parse_mode="Markdown"
        )
    except Exception as exc:
        logger.error("Notification send failed: %s", exc)


# ─── Jobs ─────────────────────────────────────────────────────────────────────

async def _check_new_assignments(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check Moodle for new assignments → notify + cache."""
    moodle = STATE.moodle
    if moodle is None:
        return

    try:
        raw = moodle.get_upcoming_assignments(days=14)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Notification: assignment check failed: %s", exc)
        return

    # Cache refresh — always write even if no new assignments
    serialized = _serialize_assignments(raw)
    cache_db.set_json("assignments", OWNER_ID, serialized)
    logger.debug("Assignments cached: %d entries", len(serialized))

    now = time.time()
    # Detect truly new assignments (not yet seen this session, not expired)
    new_assignments = []
    for a in raw or []:
        # Skip expired assignments
        if hasattr(a, "due_date") and a.due_date > 0 and a.due_date < now:
            continue
        aid = f"{a.course_name}_{a.name}"
        if aid not in STATE.known_assignment_ids:
            STATE.known_assignment_ids.add(aid)
            new_assignments.append(a)

    if not new_assignments:
        return

    lines = ["📋 *Yeni Ödev Bildirimi*\n"]
    for a in new_assignments:
        # Format due date as human-readable
        due_str = "?"
        if hasattr(a, "due_date") and a.due_date > 0:
            due_dt = datetime.fromtimestamp(a.due_date)
            due_str = due_dt.strftime("%d/%m/%Y %H:%M")
        remaining = a.time_remaining if hasattr(a, "time_remaining") else ""
        lines.append(f"• *{a.course_name}* — {a.name}\n  Teslim: {due_str}")
        if remaining:
            lines.append(f"  Kalan: {remaining}")

    await _send(context, "\n".join(lines))
    logger.info("Assignment notification sent: %d new", len(new_assignments))


async def _check_new_emails(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check for new AIRS/DAIS emails → notify + cache."""
    webmail = STATE.webmail_client
    if webmail is None or not webmail.authenticated:
        return

    try:
        new_mails = webmail.check_new_airs_dais()
    except (ConnectionError, RuntimeError, OSError, ValueError, TypeError) as exc:
        logger.error("Notification: email check failed: %s", exc)
        return

    # Full sync — fetch all AIRS/DAIS and cache them
    await _sync_email_cache(context)

    if not new_mails:
        return

    lines = ["📧 *Yeni Mail Bildirimi*\n"]
    for m in new_mails[:5]:
        subject = m.get("subject", "Konusuz")
        source = m.get("source", "")
        date = m.get("date", "")
        lines.append(f"• [{source}] *{subject}*\n  {date}")

    await _send(context, "\n".join(lines))
    logger.info("Email notification sent: %d new mails", len(new_mails))


async def _sync_email_cache(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Full email cache sync — fetches ALL AIRS/DAIS mails from IMAP to SQLite.

    Runs every 30 seconds. Only fetches body for NEW emails (not in cache).
    """
    webmail = STATE.webmail_client
    if webmail is None or not webmail.authenticated:
        return

    try:
        # Get existing UIDs from cache
        cached = cache_db.get_emails(1000) or []
        existing_uids = {m["uid"] for m in cached if m.get("uid")}

        # Sync with IMAP — only fetches body for NEW emails
        new_mails, all_current_uids = await asyncio.to_thread(
            webmail.sync_all_emails, existing_uids
        )

        # Store new emails (mark as unread)
        if new_mails:
            stored = cache_db.store_emails(new_mails, mark_read=False)
            logger.info("Email sync: %d new mails cached", stored)

        # Note: We don't delete emails that disappeared from IMAP (user might have deleted)
        # They'll expire via the normal cleanup job

    except (ConnectionError, RuntimeError, OSError, ValueError, TypeError) as exc:
        logger.warning("Email cache sync failed: %s", exc)


async def _sync_grades(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch grades from STARS → detect new grades → notify + cache."""
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(OWNER_ID):
        return

    # Snapshot previous grades for change detection
    prev = cache_db.get_json("grades", OWNER_ID)
    prev_keys = _grade_keys(prev)

    try:
        grades = await asyncio.to_thread(stars.get_grades, OWNER_ID)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Grades sync failed: %s", exc)
        return

    if not grades:
        return

    # Cache refresh
    cache_db.set_json("grades", OWNER_ID, grades)
    logger.debug("Grades cached: %d courses", len(grades))

    # Notify for new grade entries
    new_keys = _grade_keys(grades)
    truly_new = new_keys - prev_keys
    if not truly_new:
        return

    lines = ["📊 *Yeni Not Girişi*\n"]
    for course in grades:
        cname = course.get("course", "")
        for a in course.get("assessments", []):
            if (cname, a.get("name", "")) in truly_new:
                lines.append(
                    f"• *{cname}* — {a.get('name', '')}: *{a.get('grade', '')}*"
                )

    await _send(context, "\n".join(lines))
    logger.info("Grade notification sent: %d new entries", len(truly_new))


async def _sync_attendance(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Fetch attendance from STARS → detect drops → notify + cache.

    Per-course logic:
    - If syllabus limit found in RAG: hour-based tracking (remaining = limit - absences)
      Notify at ≤3 remaining (⚠️) and ≤1 remaining (🚨).
    - Fallback: ratio-based (notify on first drop below 85%).
    """
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(OWNER_ID):
        return

    # Snapshot previous state for change detection
    prev = cache_db.get_json("attendance", OWNER_ID)
    prev_ratios = _attendance_ratios(prev)
    prev_abs_counts: dict[str, int] = {
        cd.get("course", ""): _count_absences(cd.get("records", []))
        for cd in (prev or [])
    }

    # Load cached syllabus limits {course_name: max_hours}
    syllabus_limits: dict[str, int] = cache_db.get_json("syllabus_limits", OWNER_ID) or {}

    try:
        attendance = await asyncio.to_thread(stars.get_attendance, OWNER_ID)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Attendance sync failed: %s", exc)
        return

    if not attendance:
        return

    # Cache refresh
    cache_db.set_json("attendance", OWNER_ID, attendance)
    logger.debug("Attendance cached: %d courses", len(attendance))

    warnings: list[str] = []

    for cd in attendance:
        course = cd.get("course", "")
        records = cd.get("records", [])
        absent_now = _count_absences(records)
        absent_prev = prev_abs_counts.get(course, 0)

        limit = syllabus_limits.get(course) or None  # 0 = "not found" sentinel → None

        if limit is not None:
            # ── Syllabus-based tracking ──────────────────────────────
            remaining = limit - absent_now
            prev_remaining = limit - absent_prev

            # Notify only when we cross a threshold (not every sync)
            crossed_warn = prev_remaining > _ABSENCE_WARN_REMAINING >= remaining
            crossed_crit = prev_remaining > _ABSENCE_CRIT_REMAINING >= remaining

            if crossed_crit:
                warnings.append(
                    f"🚨 *{course}*: {absent_now}/{limit} saat devamsızlık — "
                    f"yalnızca *{remaining} saat* kaldı! KRİTİK!"
                )
            elif crossed_warn:
                warnings.append(
                    f"⚠️ *{course}*: {absent_now}/{limit} saat devamsızlık — "
                    f"*{remaining} saat* kaldı."
                )
        else:
            # ── Fallback: ratio-based (existing logic) ───────────────
            try:
                ratio = float(cd.get("ratio", "100").replace("%", ""))
            except (ValueError, AttributeError):
                ratio = 100.0
            was_ok = prev_ratios.get(course, 100.0) >= _ATTENDANCE_WARN_THRESHOLD
            now_low = ratio < _ATTENDANCE_WARN_THRESHOLD
            if was_ok and now_low:
                warnings.append(
                    f"⚠️ *{course}*: %{ratio:.1f} devam oranı — limit yaklaşıyor!"
                )

    if not warnings:
        return

    lines = ["⚠️ *Devamsızlık Uyarısı*\n"] + warnings
    await _send(context, "\n".join(lines))
    logger.info("Attendance warning sent: %d courses", len(warnings))


async def _stars_full_sync(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Unified STARS sync: keep-alive + fetch ALL data + cache.
    Runs every 1 minute for near real-time data.

    This replaces separate jobs: keep_alive, sync_schedule, sync_exams.
    Grades/attendance sync jobs still run for NOTIFICATIONS only.
    """
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(OWNER_ID):
        return

    try:
        # 1. Keep session alive
        alive = await asyncio.to_thread(stars.keep_alive, OWNER_ID)
        if not alive:
            logger.warning("STARS keep-alive failed in full_sync")
            return

        # 2. Fetch all data at once
        cache = await asyncio.to_thread(stars.fetch_all_data, OWNER_ID)
        if not cache:
            logger.warning("STARS fetch_all_data returned None")
            return

        # 3. Write everything to SQLite cache
        cache_db.set_json("schedule", OWNER_ID, cache.schedule)
        cache_db.set_json("grades", OWNER_ID, cache.grades)
        cache_db.set_json("attendance", OWNER_ID, cache.attendance)
        cache_db.set_json("exams", OWNER_ID, cache.exams)
        cache_db.set_json("letter_grades", OWNER_ID, cache.letter_grades)
        cache_db.set_json("transcript", OWNER_ID, cache.transcript)
        cache_db.set_json("user_info", OWNER_ID, cache.user_info)

        logger.debug(
            "STARS full sync OK: %d grades, %d attendance, %d exams, %d schedule",
            len(cache.grades), len(cache.attendance), len(cache.exams), len(cache.schedule),
        )
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.warning("STARS full sync error: %s", exc)


async def _sync_exams(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch exams from STARS → cache (no notification). DEPRECATED - kept for compatibility."""
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(OWNER_ID):
        return

    try:
        exams = await asyncio.to_thread(stars.get_exams, OWNER_ID)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Exam sync failed: %s", exc)
        return

    if exams:
        cache_db.set_json("exams", OWNER_ID, exams)
        logger.debug("Exams cached: %d entries", len(exams))


# Regex patterns to extract room info from mail body
_ROOM_PATTERNS = [
    re.compile(r"(?:Room|Salon|Sınıf|Derslik|Hall)[:\s]+([A-Z0-9]+-?\d{1,4}[A-Z]?)", re.IGNORECASE),
    re.compile(r"\b([A-Z]-\d{2,3})\b"),  # e.g. B-201, A-05
    re.compile(r"\b(EA-\d{2,3})\b", re.IGNORECASE),  # e.g. EA-409
]


def _find_exam_room_in_mails(course_code: str) -> str | None:
    """Search cached emails for exam room info matching a course code."""
    emails = cache_db.get_emails(limit=30)
    if not emails:
        return None

    # Exam keywords to identify relevant mails
    exam_kw = re.compile(r"exam|sınav|midterm|final|quiz|qme", re.IGNORECASE)
    code_pattern = re.compile(re.escape(course_code), re.IGNORECASE)

    for mail in emails:
        subject = mail.get("subject", "")
        body = mail.get("body_preview", "") or mail.get("body_full", "")
        text = f"{subject} {body}"

        # Must mention the course code AND an exam keyword
        if not code_pattern.search(text) or not exam_kw.search(text):
            continue

        # Try to extract room
        for pat in _ROOM_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1)

    return None


def _parse_exam_date(exam: dict) -> datetime | None:
    """Parse exam date string into datetime. Handles common STARS formats."""
    date_str = exam.get("date", "")
    if not date_str:
        return None

    # Try common formats
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


async def _check_exam_reminders(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check for exams happening tomorrow and send reminder with room info."""
    exams = cache_db.get_json("exams", OWNER_ID)
    if not exams:
        return

    now = datetime.now()
    tomorrow = now.date() + timedelta(days=1)

    # Track sent reminders to avoid duplicates
    sent: list[str] = cache_db.get_json("exam_reminders_sent", OWNER_ID) or []

    notifications = []
    for exam in exams:
        exam_dt = _parse_exam_date(exam)
        if exam_dt is None or exam_dt.date() != tomorrow:
            continue

        # Build unique key for dedup
        key = f"{exam.get('course', '')}_{exam.get('exam_name', '')}_{exam.get('date', '')}"
        if key in sent:
            continue

        # Extract course code for mail matching (e.g. "CTIS 256" from "CTIS 256 Discrete Structures")
        course = exam.get("course", "")
        code_match = re.match(r"([A-Z]{2,}\s*\d{3})", course)
        course_code = code_match.group(1) if code_match else course

        # Search mails for room info
        room = _find_exam_room_in_mails(course_code)

        line = f"*{course}* — {exam.get('exam_name', 'Sınav')}"
        date_info = exam.get("date", "")
        time_info = exam.get("start_time", "") or exam.get("time_block", "")
        if date_info:
            line += f"\n📅 {date_info}"
            if time_info:
                line += f", {time_info}"
        if room:
            line += f"\n🏫 Salon: {room}"

        notifications.append(line)
        sent.append(key)

    if notifications:
        header = "📝 *Yarın sınavın var!*\n"
        msg = header + "\n\n".join(notifications) + "\n\nBaşarılar!"
        await _send(context, msg)
        cache_db.set_json("exam_reminders_sent", OWNER_ID, sent)
        logger.info("Exam reminder sent: %d exams", len(notifications))


async def _check_deadline_reminders(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remind about assignments due within 24 hours (not expired, not already notified)."""
    moodle = STATE.moodle
    if moodle is None:
        return

    try:
        assignments = moodle.get_upcoming_assignments(days=1)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Notification: deadline reminder check failed: %s", exc)
        return

    now = time.time()
    # Filter: not submitted, not expired, has a deadline
    urgent = [
        a for a in assignments
        if not a.submitted
        and a.due_date > now  # not expired
        and a.due_date > 0    # has a deadline
    ]
    if not urgent:
        return

    # Dedup: track which deadlines we've already notified
    sent: list[str] = cache_db.get_json("deadline_reminders_sent", OWNER_ID) or []
    notifications = []

    for a in urgent:
        key = f"{a.course_name}_{a.name}_{a.due_date}"
        if key in sent:
            continue

        remaining = a.time_remaining if hasattr(a, "time_remaining") else ""
        line = f"• *{a.course_name}* — {a.name}"
        if remaining:
            line += f"\n  Kalan: {remaining}"
        notifications.append(line)
        sent.append(key)

    if not notifications:
        return

    msg = "⏰ *Yaklaşan Deadline'lar (24 saat içinde)*\n\n" + "\n".join(notifications)
    await _send(context, msg)
    cache_db.set_json("deadline_reminders_sent", OWNER_ID, sent)
    logger.info("Deadline reminder sent: %d urgent", len(notifications))


async def _cleanup_old_cache(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Monthly job: delete emails older than 365 days from SQLite."""
    deleted = cache_db.clean_old_emails()
    if deleted:
        logger.info("Weekly cache cleanup: removed %d old emails", deleted)


async def _refresh_sessions(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Hourly re-login for webmail IMAP and STARS to keep sessions fresh."""
    from bot.main import refresh_external_sessions

    try:
        await asyncio.to_thread(refresh_external_sessions)
    except Exception as exc:
        logger.error("Session refresh failed: %s", exc, exc_info=True)


async def _generate_missing_summaries(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Background job: generate KATMAN 2 summaries for files that don't have one."""
    store = STATE.vector_store
    if store is None or STATE.llm is None:
        return

    try:
        from bot.services.summary_service import generate_missing_summaries

        count = await asyncio.to_thread(generate_missing_summaries)
        if count > 0:
            logger.info("Background summary generation: %d new summaries", count)
    except Exception as exc:
        logger.error("Background summary generation failed: %s", exc, exc_info=True)


async def _sync_syllabus_limits(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Daily job: for each course in attendance cache, search RAG for the syllabus
    and extract the max absence limit. Caches results as {course_name: max_hours}.

    Runs once at startup (after 5 min) then every 24h. Results persist in SQLite
    so they survive bot restarts.
    """
    attendance = cache_db.get_json("attendance", OWNER_ID) or []
    if not attendance:
        # No attendance data yet — skip silently
        return

    existing: dict[str, int] = cache_db.get_json("syllabus_limits", OWNER_ID) or {}
    updated = dict(existing)
    found = 0

    for cd in attendance:
        course = cd.get("course", "")
        if not course:
            continue
        # Re-scan even if already cached (syllabus might be uploaded mid-semester)
        limit = await asyncio.to_thread(_extract_syllabus_attendance_limit, course)
        if limit is not None:
            updated[course] = limit
            found += 1
        elif course not in updated:
            # Explicitly mark as "no limit found" so we don't re-scan every 24h
            # Use 0 as sentinel → treated as None in _sync_attendance
            updated[course] = 0

    cache_db.set_json("syllabus_limits", OWNER_ID, updated)
    logger.info(
        "Syllabus limits synced: %d courses checked, %d limits found", len(attendance), found
    )


# ─── Auto Material Sync ───────────────────────────────────────────────────────


async def _auto_sync_materials(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Auto-sync Moodle materials to vector store every 30 minutes.

    Duplicate control:
    1. sync_state.json tracks synced_files (file path → metadata)
    2. vector_store.add_chunks() skips duplicate chunk_ids

    Only new files are downloaded and indexed — existing files are skipped.
    """
    sync_engine = STATE.sync_engine
    moodle = STATE.moodle
    if sync_engine is None or moodle is None:
        return

    # Prevent concurrent syncs
    if STATE.sync_lock.locked():
        logger.debug("Material sync skipped — another sync in progress")
        return

    async with STATE.sync_lock:
        try:
            logger.info("Auto material sync starting...")
            start = time.time()

            # Run sync in thread pool (blocking I/O)
            new_chunks = await asyncio.to_thread(sync_engine.sync_all)

            elapsed = time.time() - start
            logger.info(
                "Auto material sync completed in %.1fs — %s new chunks",
                elapsed,
                new_chunks if new_chunks else 0,
            )

            # Update state for status display
            from datetime import datetime as dt

            STATE.last_sync_time = dt.now().strftime("%Y-%m-%d %H:%M")
            STATE.last_sync_new_files = new_chunks or 0

        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            logger.error("Auto material sync failed: %s", exc)


# Polling watchdog: if no Telegram update received for a long time, self-kill.
# systemd Restart=always will restart the process.
# 12 hours — single-user bot, nobody messages at night; short timeouts cause
# unnecessary restarts → STARS re-login → verification code spam.
_WATCHDOG_TIMEOUT = 43200  # 12 hours
_WATCHDOG_GRACE = 600      # 10 minutes after startup before checking


async def _polling_watchdog(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Kill the process if Telegram polling appears stuck."""
    now = time.monotonic()
    uptime = now - STATE.started_at_monotonic

    # Don't check during startup grace period
    if uptime < _WATCHDOG_GRACE:
        return

    last = STATE.last_update_received
    # If never received any update, use startup time as baseline
    if last == 0.0:
        last = STATE.started_at_monotonic

    silence = now - last
    if silence > _WATCHDOG_TIMEOUT:
        logger.critical(
            "WATCHDOG: No Telegram update for %.0f seconds — killing process for restart",
            silence,
        )
        await _send(context, "⚠️ Bot polling stuck — otomatik restart yapılıyor...")
        os._exit(1)  # noqa: SLF001 — hard kill, systemd restarts


# ─── Registration ─────────────────────────────────────────────────────────────

def register_notification_jobs(app: Application) -> None:
    """Register all periodic background jobs on the PTB job queue."""
    jq = app.job_queue
    if jq is None:
        logger.warning("Job queue not available — notifications disabled")
        return

    jq.run_repeating(
        _check_new_assignments,
        interval=timedelta(seconds=CONFIG.assignment_check_interval),
        first=timedelta(seconds=30),
        name="assignment_check",
    )
    jq.run_repeating(
        _check_new_emails,
        interval=timedelta(minutes=5),
        first=timedelta(seconds=60),
        name="email_check",
    )
    # ═══ Email Cache Sync: 30-second full IMAP sync → SQLite (instant agent queries) ═══
    jq.run_repeating(
        _sync_email_cache,
        interval=timedelta(seconds=30),
        first=timedelta(seconds=15),  # Start early so cache is ready
        name="email_cache_sync",
    )
    jq.run_repeating(
        _sync_grades,
        interval=timedelta(minutes=30),
        first=timedelta(minutes=3),
        name="grades_sync",
    )
    jq.run_repeating(
        _sync_attendance,
        interval=timedelta(minutes=60),
        first=timedelta(minutes=4),
        name="attendance_sync",
    )
    # ═══ STARS Full Sync: 1-minute unified sync (keep-alive + all data + cache) ═══
    jq.run_repeating(
        _stars_full_sync,
        interval=timedelta(minutes=1),
        first=timedelta(seconds=30),
        name="stars_full_sync",
    )
    jq.run_repeating(
        _check_exam_reminders,
        interval=timedelta(hours=1),
        first=timedelta(minutes=10),
        name="exam_reminder",
    )
    jq.run_repeating(
        _check_deadline_reminders,
        interval=timedelta(minutes=30),
        first=timedelta(minutes=2),
        name="deadline_reminder",
    )
    jq.run_repeating(
        _refresh_sessions,
        interval=timedelta(hours=24),
        first=timedelta(hours=23),
        name="session_refresh",
    )
    jq.run_repeating(
        _generate_missing_summaries,
        interval=timedelta(minutes=60),
        first=timedelta(minutes=5),
        name="summary_generation",
    )
    jq.run_repeating(
        _cleanup_old_cache,
        interval=timedelta(weeks=4),  # Monthly — 365-day retention means no rush
        first=timedelta(hours=1),
        name="cache_cleanup",
    )
    jq.run_repeating(
        _sync_syllabus_limits,
        interval=timedelta(hours=24),
        first=timedelta(minutes=5),  # Run soon after startup so limits are ready
        name="syllabus_limits_sync",
    )
    # ═══ Auto Material Sync: 30-minute Moodle → vector store sync ═══
    jq.run_repeating(
        _auto_sync_materials,
        interval=timedelta(minutes=30),
        first=timedelta(minutes=2),  # Quick first sync to catch any new materials
        name="material_sync",
    )
    jq.run_repeating(
        _polling_watchdog,
        interval=timedelta(minutes=5),
        first=timedelta(minutes=10),
        name="polling_watchdog",
    )

    logger.info(
        "Notification jobs registered: stars_full_sync=1m, assignments=10m, emails=5m, "
        "email_cache=30s, grades=30m, attendance=60m, exam_reminder=1h, deadlines=30m, "
        "session=24h, summaries=60m, cache_cleanup=weekly, syllabus_limits=24h, "
        "material_sync=30m, polling_watchdog=5m"
    )
