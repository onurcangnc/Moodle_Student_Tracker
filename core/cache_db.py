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

Cleanup: emails older than CLEANUP_DAYS are removed by a weekly job.
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
CLEANUP_DAYS = 90  # delete emails older than this

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
    """Upsert emails into persistent store. Returns number of rows written."""
    if not mails:
        return 0
    _ensure_init()
    now = time.time()
    rows = []
    for m in mails:
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
