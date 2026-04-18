"""
SQLite persistent store for agent tool results.
================================================
Design principle: background jobs are the ONLY writers. Tool handlers are
read-only. Cache miss happens ONLY when the table is completely empty
(fresh install, before the first background job run).

Background job refresh intervals (for reference):
  emails      → 5 min   (email_check)
  assignments → 10 min  (assignment_check)
  grades      → 30 min  (grades_sync)
  attendance  → 60 min  (attendance_sync)
  schedule    → 6 h     (schedule_sync)

Cleanup: emails older than CLEANUP_DAYS are removed by a monthly job.
data_cache rows are single key-value entries that get overwritten on each
write — no accumulation, no cleanup needed there.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/cache.db")
CLEANUP_DAYS = 365  # delete emails older than this (1 year retention)

_initialized = False


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    """Create tables. Idempotent — safe to call multiple times."""
    global _initialized
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS emails (
                uid          TEXT PRIMARY KEY,
                subject      TEXT NOT NULL DEFAULT '',
                from_addr    TEXT NOT NULL DEFAULT '',
                date         TEXT NOT NULL DEFAULT '',
                body_preview TEXT NOT NULL DEFAULT '',
                body_full    TEXT NOT NULL DEFAULT '',
                source       TEXT NOT NULL DEFAULT '',
                is_read      INTEGER NOT NULL DEFAULT 0,
                inserted_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_emails_inserted
                ON emails (inserted_at DESC);
            CREATE INDEX IF NOT EXISTS idx_emails_unread
                ON emails (is_read, inserted_at DESC);

            CREATE TABLE IF NOT EXISTS data_cache (
                cache_key  TEXT    NOT NULL,
                user_id    INTEGER NOT NULL,
                json_data  TEXT    NOT NULL,
                updated_at REAL    NOT NULL,
                PRIMARY KEY (cache_key, user_id)
            );
        """)
        # Migration: add is_read column if missing (for existing DBs)
        try:
            conn.execute("ALTER TABLE emails ADD COLUMN is_read INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists
    _initialized = True
    logger.debug("Cache DB initialized at %s", _DB_PATH)


def _ensure_init() -> None:
    if not _initialized:
        init_db()


# ─── Email Cache ──────────────────────────────────────────────────────────────

def store_emails(mails: list[dict], mark_read: bool = True) -> int:
    """Upsert emails into persistent store. Returns number of rows written.

    Args:
        mails: List of email dicts with uid, subject, from, date, body_preview, etc.
        mark_read: If True, marks emails as read. If False, preserves is_read=0 for new.
    """
    if not mails:
        return 0
    _ensure_init()
    now = time.time()
    rows = []
    for m in mails:
        uid = m.get("uid") or f"{m.get('subject','')}:{m.get('from','')}:{m.get('date','')}"
        is_read = 1 if mark_read else m.get("is_read", 0)
        rows.append((
            str(uid),
            m.get("subject", ""),
            m.get("from", ""),
            m.get("date", ""),
            m.get("body_preview", ""),
            m.get("body_full", m.get("body", "")),
            m.get("source", ""),
            is_read,
            now,
        ))
    try:
        with _conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO emails "
                "(uid, subject, from_addr, date, body_preview, body_full, source, is_read, inserted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        logger.debug("Stored %d emails to cache", len(rows))
        return len(rows)
    except sqlite3.Error as exc:
        logger.error("Email cache write failed: %s", exc)
        return 0


def get_emails(limit: int = 20) -> list[dict] | None:
    """Return emails from persistent store ordered by recency.

    Returns None ONLY if the table is empty (fresh install).
    Freshness is guaranteed by the background email_check job — no TTL check here.
    """
    _ensure_init()
    try:
        with _conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
            if count == 0:
                return None  # Empty DB — background job hasn't run yet

            rows = conn.execute(
                "SELECT uid, subject, from_addr, date, body_preview, body_full, source, is_read "
                "FROM emails ORDER BY inserted_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [
            {
                "uid":          r[0],
                "subject":      r[1],
                "from":         r[2],
                "date":         r[3],
                "body_preview": r[4],
                "body_full":    r[5],
                "source":       r[6],
                "is_read":      bool(r[7]),
            }
            for r in rows
        ]
    except sqlite3.Error as exc:
        logger.error("Email cache read failed: %s", exc)
        return None


def get_unread_emails() -> list[dict]:
    """Return all unread emails from cache."""
    _ensure_init()
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT uid, subject, from_addr, date, body_preview, body_full, source "
                "FROM emails WHERE is_read = 0 ORDER BY inserted_at DESC",
            ).fetchall()

        return [
            {
                "uid":          r[0],
                "subject":      r[1],
                "from":         r[2],
                "date":         r[3],
                "body_preview": r[4],
                "body_full":    r[5],
                "source":       r[6],
                "is_read":      False,
            }
            for r in rows
        ]
    except sqlite3.Error as exc:
        logger.error("Email cache read (unread) failed: %s", exc)
        return []


def search_emails(keyword: str, limit: int = 20) -> list[dict]:
    """Search emails by keyword across subject, from, source, date, and body.

    Matches the full keyword as a single LIKE substring, then falls back to
    OR-combined token matching if nothing is found. Searches body_full so
    content buried deep in long messages is still discoverable.
    """
    _ensure_init()
    if not keyword:
        return get_emails(limit) or []

    keyword = keyword.strip()
    if not keyword:
        return get_emails(limit) or []

    fields = ("subject", "from_addr", "source", "date", "body_preview", "body_full")
    field_clause = " OR ".join(f"{f} LIKE ? COLLATE NOCASE" for f in fields)

    try:
        with _conn() as conn:
            # Pass 1: match the entire keyword as a single substring.
            full_pattern = f"%{keyword}%"
            rows = conn.execute(
                f"""
                SELECT uid, subject, from_addr, date, body_preview, body_full, source, is_read
                FROM emails
                WHERE {field_clause}
                ORDER BY inserted_at DESC
                LIMIT ?
                """,
                (*[full_pattern] * len(fields), limit),
            ).fetchall()

            # Pass 2: fall back to OR-combined per-token match (any token in any field).
            if not rows:
                tokens = [t for t in keyword.split() if t]
                if len(tokens) > 1:
                    token_conditions = []
                    params: list[str] = []
                    for tok in tokens:
                        pattern = f"%{tok}%"
                        token_conditions.append(f"({field_clause})")
                        params.extend([pattern] * len(fields))
                    where_clause = " OR ".join(token_conditions)
                    rows = conn.execute(
                        f"""
                        SELECT uid, subject, from_addr, date, body_preview, body_full, source, is_read
                        FROM emails
                        WHERE {where_clause}
                        ORDER BY inserted_at DESC
                        LIMIT ?
                        """,
                        (*params, limit),
                    ).fetchall()

        return [
            {
                "uid":          r[0],
                "subject":      r[1],
                "from":         r[2],
                "date":         r[3],
                "body_preview": r[4],
                "body_full":    r[5],
                "source":       r[6],
                "is_read":      bool(r[7]),
            }
            for r in rows
        ]
    except sqlite3.Error as exc:
        logger.error("Email search failed: %s", exc)
        return []


def mark_emails_read(uids: list[str]) -> int:
    """Mark specific emails as read. Returns count updated."""
    if not uids:
        return 0
    _ensure_init()
    try:
        with _conn() as conn:
            placeholders = ",".join("?" * len(uids))
            cur = conn.execute(
                f"UPDATE emails SET is_read = 1 WHERE uid IN ({placeholders})",
                uids,
            )
            return cur.rowcount
    except sqlite3.Error as exc:
        logger.error("Email mark_read failed: %s", exc)
        return 0


def get_email_count() -> int:
    """Return total email count in cache."""
    _ensure_init()
    try:
        with _conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
    except sqlite3.Error:
        return 0


def clean_old_emails(days: int = CLEANUP_DAYS) -> int:
    """Delete emails older than `days` days. Returns number of rows deleted."""
    _ensure_init()
    cutoff = time.time() - days * 86400
    try:
        with _conn() as conn:
            cur = conn.execute("DELETE FROM emails WHERE inserted_at < ?", (cutoff,))
            deleted = cur.rowcount
        if deleted:
            logger.info("Email cleanup: deleted %d emails older than %d days", deleted, days)
        return deleted
    except sqlite3.Error as exc:
        logger.error("Email cleanup failed: %s", exc)
        return 0


# ─── Generic JSON Store (grades, attendance, schedule, assignments) ───────────

def get_json(cache_key: str, user_id: int) -> Any | None:
    """Return stored data for this key/user.

    Returns None ONLY if the key has never been written (fresh install).
    Freshness is guaranteed by background sync jobs — no TTL check here.
    """
    _ensure_init()
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT json_data FROM data_cache WHERE cache_key=? AND user_id=?",
                (cache_key, user_id),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        logger.error("Cache read failed [%s/%s]: %s", cache_key, user_id, exc)
        return None


def set_json(cache_key: str, user_id: int, data: Any) -> None:
    """Overwrite stored data for this key/user."""
    _ensure_init()
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO data_cache "
                "(cache_key, user_id, json_data, updated_at) VALUES (?, ?, ?, ?)",
                (cache_key, user_id, json.dumps(data, ensure_ascii=False), time.time()),
            )
        logger.debug("Cache set [%s/%s]", cache_key, user_id)
    except (sqlite3.Error, TypeError) as exc:
        logger.error("Cache write failed [%s/%s]: %s", cache_key, user_id, exc)


# ─── Student Profile ────────────────────────────────────────────────────────


def get_student_profile(user_id: int) -> dict:
    """Get student profile with preferences and behavior data."""
    profile = get_json("student_profile", user_id)
    if profile is None:
        # Default profile
        return {
            "course_queries": {},  # {course_name: count}
            "preferred_lang": "tr",  # detected over time
            "query_count": 0,
            "last_topics": [],  # recent 5 topics asked about
            "style_hints": [],  # e.g., ["prefers_detail", "asks_followups"]
        }
    return profile


def update_student_profile(user_id: int, updates: dict) -> None:
    """Update specific fields in student profile."""
    profile = get_student_profile(user_id)
    profile.update(updates)
    set_json("student_profile", user_id, profile)


def track_query(user_id: int, course: str | None = None, topic: str | None = None) -> None:
    """Track a query for profile building."""
    profile = get_student_profile(user_id)

    # Increment query count
    profile["query_count"] = profile.get("query_count", 0) + 1

    # Track course queries
    if course:
        courses = profile.get("course_queries", {})
        courses[course] = courses.get(course, 0) + 1
        profile["course_queries"] = courses

    # Track recent topics (keep last 5)
    if topic:
        topics = profile.get("last_topics", [])
        if topic not in topics:
            topics.insert(0, topic)
            profile["last_topics"] = topics[:5]

    set_json("student_profile", user_id, profile)


def get_profile_context(user_id: int) -> str:
    """Build context string from profile for system prompt."""
    profile = get_student_profile(user_id)

    parts = []

    # Favorite courses
    course_queries = profile.get("course_queries", {})
    if course_queries:
        top_courses = sorted(course_queries.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_courses:
            names = [c[0] for c in top_courses]
            parts.append(f"Sık sorduğu dersler: {', '.join(names)}")

    # Recent topics
    topics = profile.get("last_topics", [])
    if topics:
        parts.append(f"Son ilgilendiği konular: {', '.join(topics[:3])}")

    # Query count (engagement level)
    count = profile.get("query_count", 0)
    if count > 50:
        parts.append("Aktif kullanıcı (50+ sorgu)")
    elif count > 10:
        parts.append("Düzenli kullanıcı")

    if not parts:
        return ""

    return "\n📊 Öğrenci Profili:\n" + "\n".join(f"  • {p}" for p in parts)
