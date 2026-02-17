"""
SQLite cache for agent tool results.
=====================================
Provides TTL-based caching for IMAP emails, STARS grades/attendance/schedule.
Background sync jobs write here; tool handlers read from here first.

TTLs:
  emails     → 5 min  (matches email_check notification job)
  grades     → 2 h
  attendance → 2 h
  schedule   → 24 h
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

_TTL: dict[str, int] = {
    "emails":      5 * 60,       # refreshed every 5 min by email_check job
    "assignments": 15 * 60,      # refreshed every 10 min by assignment_check job
    "grades":      2 * 3600,     # refreshed every 30 min by grades_sync job
    "attendance":  2 * 3600,     # refreshed every 60 min by attendance_sync job
    "schedule":    24 * 3600,    # refreshed every 6 h by schedule_sync job
}

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
                inserted_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_emails_inserted
                ON emails (inserted_at DESC);

            CREATE TABLE IF NOT EXISTS data_cache (
                cache_key  TEXT    NOT NULL,
                user_id    INTEGER NOT NULL,
                json_data  TEXT    NOT NULL,
                updated_at REAL    NOT NULL,
                PRIMARY KEY (cache_key, user_id)
            );
        """)
    _initialized = True
    logger.debug("Cache DB initialized at %s", _DB_PATH)


def _ensure_init() -> None:
    if not _initialized:
        init_db()


# ─── Email Cache ──────────────────────────────────────────────────────────────

def store_emails(mails: list[dict]) -> int:
    """Upsert emails into cache. Returns number of rows stored."""
    if not mails:
        return 0
    _ensure_init()
    now = time.time()
    rows = []
    for i, m in enumerate(mails):
        # Prefer explicit uid; fall back to stable hash of (subject, from, date)
        uid = m.get("uid") or f"{m.get('subject','')}:{m.get('from','')}:{m.get('date','')}"
        rows.append((
            str(uid),
            m.get("subject", ""),
            m.get("from", ""),
            m.get("date", ""),
            m.get("body_preview", ""),
            m.get("body_full", m.get("body", "")),
            m.get("source", ""),
            now,
        ))
    try:
        with _conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO emails "
                "(uid, subject, from_addr, date, body_preview, body_full, source, inserted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        logger.debug("Stored %d emails to cache", len(rows))
        return len(rows)
    except sqlite3.Error as exc:
        logger.error("Email cache write failed: %s", exc)
        return 0


def get_emails(limit: int = 20) -> list[dict] | None:
    """Return cached emails if fresh (within TTL), else None (cache miss).

    Returns emails ordered by date descending (newest first).
    """
    _ensure_init()
    ttl = _TTL["emails"]
    cutoff = time.time() - ttl
    try:
        with _conn() as conn:
            # If no emails at all, or last insert is stale → cache miss
            row = conn.execute("SELECT MAX(inserted_at) FROM emails").fetchone()
            if not row or row[0] is None or row[0] < cutoff:
                return None

            rows = conn.execute(
                "SELECT uid, subject, from_addr, date, body_preview, body_full, source "
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
            }
            for r in rows
        ]
    except sqlite3.Error as exc:
        logger.error("Email cache read failed: %s", exc)
        return None


# ─── Generic JSON Cache (grades, attendance, schedule) ───────────────────────

def get_json(cache_key: str, user_id: int) -> Any | None:
    """Return cached data if within TTL, else None."""
    _ensure_init()
    ttl = _TTL.get(cache_key, 300)
    cutoff = time.time() - ttl
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT json_data, updated_at FROM data_cache "
                "WHERE cache_key=? AND user_id=?",
                (cache_key, user_id),
            ).fetchone()
        if row is None or row[1] < cutoff:
            return None
        return json.loads(row[0])
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        logger.error("Cache read failed [%s/%s]: %s", cache_key, user_id, exc)
        return None


def set_json(cache_key: str, user_id: int, data: Any) -> None:
    """Store data in cache. Overwrites existing entry."""
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
