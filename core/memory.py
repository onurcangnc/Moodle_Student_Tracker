"""
Hybrid Memory System
=====================
Optimal cost/accuracy/sustainability balance:

┌─────────────────────────────────────────────────────────────┐
│                      MEMORY ARCHITECTURE                     │
│                                                              │
│  ┌─────────────────────┐    ┌──────────────────────────────┐│
│  │   STATIC LAYER      │    │      DYNAMIC LAYER           ││
│  │   (profile.md)      │    │      (SQLite DB)             ││
│  │                     │    │                              ││
│  │ • Identity          │    │ • Semantic memories          ││
│  │ • Core preferences  │    │ • Learning progress          ││
│  │ • Course list       │    │ • Conversation history       ││
│  │ • Study schedule    │    │ • Struggle detection         ││
│  │ • Long-term goals   │    │ • Topic mastery tracking     ││
│  │                     │    │                              ││
│  │ Always in prompt    │    │ Query-time selective fetch   ││
│  │ ~300-500 tokens     │    │ ~300-800 tokens              ││
│  │ Updated rarely      │    │ Updated every turn           ││
│  │ User-editable       │    │ LLM-managed                 ││
│  └─────────────────────┘    └──────────────────────────────┘│
│                                                              │
│  Total per-turn cost: ~600-1300 tokens (vs 4000-8000 full md)│
└─────────────────────────────────────────────────────────────┘

Why hybrid?
- Static layer: guarantees critical info is NEVER missed (no query risk)
- Dynamic layer: scales infinitely without growing prompt size
- Cost: ~80% cheaper than full-markdown approach
- Accuracy: critical info always present + relevant dynamic info on demand
"""

import json
import logging
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from core import config

logger = logging.getLogger(__name__)

DB_PATH = config.data_dir / "memory.db"
PROFILE_PATH = config.data_dir / "profile.md"


# ─── Turkish Keyword Extraction (for deep recall) ─────────────────────────────

_TR_STOPWORDS = frozenset(
    {
        "bir",
        "bu",
        "şu",
        "ben",
        "sen",
        "biz",
        "siz",
        "ve",
        "ile",
        "de",
        "da",
        "mi",
        "mı",
        "ne",
        "nasıl",
        "neden",
        "ama",
        "fakat",
        "için",
        "gibi",
        "kadar",
        "daha",
        "çok",
        "az",
        "var",
        "yok",
        "olan",
        "olarak",
        "den",
        "dan",
        "nin",
        "nın",
        "hakkında",
        "geçen",
        "hafta",
        "bugün",
        "dün",
        "konuştuğumuz",
        "konuşmuştuk",
        "demiştin",
        "sormuştum",
        "hatırla",
        "hatırlıyor",
        "musun",
        "mısın",
        "neydi",
        "nedir",
        "bana",
        "sana",
        "beni",
        "seni",
        "bunu",
        "şunu",
        "şimdi",
        "sonra",
        "önce",
        "biraz",
    }
)


def _extract_keywords(text: str, max_kw: int = 5) -> list[str]:
    """Extract meaningful keywords from a Turkish message (stopword filter)."""
    words = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", text.lower())
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        if len(w) >= 3 and w not in _TR_STOPWORDS and w not in seen:
            seen.add(w)
            result.append(w)
    return result[:max_kw]


# ─── Static Profile (Markdown Layer) ────────────────────────────────────────

DEFAULT_PROFILE = """# Öğrenci Profili

## Kimlik
- İsim: 
- Üniversite: Bilkent
- Bölüm: 
- Dönem: 

## Aktif Kurslar
<!-- Sync sonrası otomatik güncellenir -->

## Tercihler
- Dil: Türkçe (teknik terimler İngilizce parantezde)
- Açıklama stili: Detaylı, örnekli
- Kod dili: Python tercih

## Çalışma Programı
<!-- Kendi programını buraya yaz -->

## Uzun Vadeli Hedefler
<!-- Kariyer hedefleri, sertifikalar, vb. -->
"""


class StaticProfile:
    """
    Markdown-based static profile.
    Always included in system prompt (~300-500 tokens).
    User-editable, rarely changes.
    """

    def __init__(self, path: Path = PROFILE_PATH):
        self.path = path
        self._ensure_exists()

    def _ensure_exists(self):
        if not self.path.exists():
            self.path.write_text(DEFAULT_PROFILE, encoding="utf-8")
            logger.info(f"Created default profile at {self.path}")

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def update(self, content: str):
        self.path.write_text(content, encoding="utf-8")

    def update_section(self, section_name: str, content: str):
        """Update a specific ## section in the profile."""
        profile = self.read()
        pattern = rf"(## {re.escape(section_name)}\n)(.*?)(\n## |\Z)"
        match = re.search(pattern, profile, re.DOTALL)

        if match:
            new_profile = profile[: match.start(2)] + content + "\n" + profile[match.start(3) :]
            self.update(new_profile)
        else:
            self.update(profile.rstrip() + f"\n\n## {section_name}\n{content}\n")

    def update_courses(self, courses: list[str]):
        """Auto-update the active courses section after sync."""
        course_list = "\n".join(f"- {c}" for c in courses)
        self.update_section("Aktif Kurslar", course_list)

    def auto_populate_from_moodle(self, site_info: dict, courses: list[str]):
        """
        Auto-populate profile from Moodle site_info API response.
        Only fills in empty/placeholder fields — doesn't overwrite user edits.

        site_info fields used:
          - fullname: user's display name
          - username: student ID
          - sitename: university/site name
        """
        profile = self.read()

        fullname = site_info.get("fullname", "")
        username = site_info.get("username", "")
        sitename = site_info.get("sitename", "")

        # Parse university name from sitename (e.g., "2025-2026-Spring" → keep as is)
        university = ""
        if sitename:
            # Common Bilkent patterns
            if "bilkent" in sitename.lower():
                university = "İhsan Doğramacı Bilkent Üniversitesi"
            else:
                university = sitename

        # Detect semester from MOODLE_URL
        import os

        moodle_url = os.getenv("MOODLE_URL", "")
        semester = ""
        if "spring" in moodle_url.lower():
            semester = "Bahar"
        elif "fall" in moodle_url.lower():
            semester = "Güz"
        elif "summer" in moodle_url.lower():
            semester = "Yaz"
        # Extract year: e.g., /2025-2026-spring → 2025-2026
        import re as _re

        year_match = _re.search(r"(\d{4}-\d{4})", moodle_url)
        if year_match and semester:
            semester = f"{year_match.group(1)} {semester}"

        # Only fill empty fields (don't overwrite user edits)
        if fullname and "- İsim: \n" in profile or "- İsim:\n" in profile:
            profile = _re.sub(r"- İsim:.*", f"- İsim: {fullname}", profile)

        if username and "- Öğrenci No:" not in profile:
            profile = _re.sub(r"(- İsim:.*\n)", rf"\1- Öğrenci No: {username}\n", profile)

        if university and (
            "- Üniversite: Bilkent" in profile or "- Üniversite: \n" in profile or "- Üniversite:\n" in profile
        ):
            profile = _re.sub(r"- Üniversite:.*", f"- Üniversite: {university}", profile)

        if semester and "- Dönem: \n" in profile or "- Dönem:\n" in profile:
            profile = _re.sub(r"- Dönem:.*", f"- Dönem: {semester}", profile)

        self.update(profile)

        # Update courses section
        if courses:
            self.update_courses(courses)

        logger.info(f"Profile auto-populated: {fullname} ({username}), {len(courses)} courses")

    def get_token_estimate(self) -> int:
        """Rough token count (1 token ≈ 3 chars for Turkish)."""
        return len(self.read()) // 3


# ─── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class MemoryEntry:
    id: int | None = None
    category: str = ""  # preference, fact, goal, struggle, insight, exam
    content: str = ""
    source_message_id: int | None = None
    course: str = ""
    confidence: float = 1.0
    created_at: str = ""
    last_accessed: str = ""
    access_count: int = 0
    ttl_days: int = -1  # -1 = permanent, >0 = auto-expire after N days
    is_active: int = 1

    def to_text(self) -> str:
        prefix = {
            "preference": "🔧",
            "fact": "📌",
            "goal": "🎯",
            "struggle": "⚠️",
            "insight": "💡",
            "exam": "📅",
        }.get(self.category, "•")
        return f"{prefix} {self.content}"


@dataclass
class LearningRecord:
    id: int | None = None
    course: str = ""
    topic: str = ""
    mastery_level: float = 0.0
    times_asked: int = 0
    times_correct: int = 0
    last_studied: str = ""
    notes: str = ""


# ─── Dynamic Memory (SQLite Layer) ──────────────────────────────────────────


class DynamicMemoryDB:
    """
    SQLite-based dynamic memory.
    Selectively queried each turn (~300-800 tokens).
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    active_course TEXT,
                    summary TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tokens_used INTEGER DEFAULT 0,
                    rag_sources TEXT DEFAULT '',
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_message_id INTEGER,
                    course TEXT DEFAULT '',
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    ttl_days INTEGER DEFAULT -1,
                    is_active INTEGER DEFAULT 1,
                    FOREIGN KEY (source_message_id) REFERENCES messages(id)
                );

                CREATE TABLE IF NOT EXISTS learning_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    course TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    mastery_level REAL DEFAULT 0.0,
                    times_asked INTEGER DEFAULT 0,
                    times_correct INTEGER DEFAULT 0,
                    last_studied TEXT,
                    notes TEXT DEFAULT '',
                    UNIQUE(course, topic)
                );

                CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_sem_cat ON semantic_memory(category);
                CREATE INDEX IF NOT EXISTS idx_sem_course ON semantic_memory(course);
                CREATE INDEX IF NOT EXISTS idx_sem_active ON semantic_memory(is_active);
                CREATE INDEX IF NOT EXISTS idx_lp_course ON learning_progress(course);
            """)

    # ─── Session ─────────────────────────────────────────────────────────

    def create_session(self, active_course: str = "") -> int:
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO sessions (started_at, active_course) VALUES (?, ?)",
                (now, active_course),
            )
            return cur.lastrowid

    def end_session(self, session_id: int, summary: str = ""):
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
                (now, summary, session_id),
            )

    # ─── Messages ────────────────────────────────────────────────────────

    def add_message(self, session_id: int, role: str, content: str, rag_sources: str = "") -> int:
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO messages (session_id, role, content, timestamp, rag_sources)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, role, content, now, rag_sources),
            )
            return cur.lastrowid

    def get_session_messages(self, session_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM messages ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in reversed(rows)]

    def search_messages(self, query: str, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def search_semantic_memories(self, query: str, limit: int = 10) -> list:
        """Search semantic_memory table by content keyword match."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM semantic_memory
                   WHERE is_active = 1 AND content LIKE ?
                   ORDER BY confidence DESC, last_accessed DESC LIMIT ?""",
                (f"%{query}%", limit),
            ).fetchall()
            return [MemoryEntry(**{k: r[k] for k in r.keys()}) for r in rows]

    # ─── Semantic Memory ─────────────────────────────────────────────────

    def add_memory(self, entry: MemoryEntry) -> int:
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            existing = conn.execute(
                """SELECT id FROM semantic_memory
                   WHERE content = ? AND category = ? AND is_active = 1""",
                (entry.content, entry.category),
            ).fetchone()

            if existing:
                conn.execute(
                    """UPDATE semantic_memory
                       SET access_count = access_count + 1, last_accessed = ?
                       WHERE id = ?""",
                    (now, existing["id"]),
                )
                return existing["id"]

            cur = conn.execute(
                """INSERT INTO semantic_memory
                   (category, content, source_message_id, course, confidence,
                    created_at, last_accessed, ttl_days)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.category,
                    entry.content,
                    entry.source_message_id,
                    entry.course,
                    entry.confidence,
                    now,
                    now,
                    entry.ttl_days,
                ),
            )
            return cur.lastrowid

    def get_memories(self, category: str = None, course: str = None, limit: int = 30) -> list[MemoryEntry]:
        query = "SELECT * FROM semantic_memory WHERE is_active = 1"
        params = []

        if category:
            query += " AND category = ?"
            params.append(category)
        if course:
            query += " AND (course = ? OR course = '')"
            params.append(course)

        query += " ORDER BY confidence * (1 + access_count * 0.1) DESC, last_accessed DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [MemoryEntry(**{k: r[k] for k in r.keys()}) for r in rows]

    def get_memories_for_context(self, course: str = None, max_tokens: int = 800) -> list[MemoryEntry]:
        """
        Smart fetch: get most relevant memories within token budget.
        THE key cost optimization method.
        """
        memories = self.get_memories(course=course, limit=50)

        selected = []
        token_count = 0

        for m in memories:
            mem_tokens = len(m.to_text()) // 3  # ~3 chars/token for TR
            if token_count + mem_tokens > max_tokens:
                break
            selected.append(m)
            token_count += mem_tokens
            self._touch_memory(m.id)

        return selected

    def _touch_memory(self, memory_id: int):
        now = datetime.now(UTC).isoformat()
        try:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE semantic_memory SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                    (now, memory_id),
                )
        except sqlite3.Error as exc:
            logger.debug("Failed to update memory access metadata for id=%s: %s", memory_id, exc)

    def deactivate_memory(self, memory_id: int):
        with self._conn() as conn:
            conn.execute("UPDATE semantic_memory SET is_active = 0 WHERE id = ?", (memory_id,))

    def cleanup_expired(self):
        with self._conn() as conn:
            conn.execute("""
                UPDATE semantic_memory SET is_active = 0
                WHERE ttl_days > 0
                AND julianday('now') - julianday(created_at) > ttl_days
            """)

    # ─── Learning Progress ───────────────────────────────────────────────

    def update_learning(
        self,
        course: str,
        topic: str,
        mastery_delta: float = 0.0,
        asked: bool = False,
        correct: bool = False,
        notes: str = "",
    ):
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT * FROM learning_progress WHERE course = ? AND topic = ?",
                (course, topic),
            ).fetchone()

            if existing:
                new_mastery = min(1.0, max(0.0, existing["mastery_level"] + mastery_delta))
                conn.execute(
                    """UPDATE learning_progress
                       SET mastery_level = ?, times_asked = times_asked + ?,
                           times_correct = times_correct + ?, last_studied = ?,
                           notes = CASE WHEN ? != '' THEN ? ELSE notes END
                       WHERE course = ? AND topic = ?""",
                    (new_mastery, int(asked), int(correct), now, notes, notes, course, topic),
                )
            else:
                conn.execute(
                    """INSERT INTO learning_progress
                       (course, topic, mastery_level, times_asked, times_correct,
                        last_studied, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (course, topic, max(0.0, mastery_delta), int(asked), int(correct), now, notes),
                )

    def get_learning_progress(self, course: str = None) -> list[LearningRecord]:
        query = "SELECT * FROM learning_progress"
        params = []
        if course:
            query += " WHERE course = ?"
            params.append(course)
        query += " ORDER BY last_studied DESC"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [LearningRecord(**{k: r[k] for k in r.keys()}) for r in rows]

    def get_weak_topics(self, course: str = None, threshold: float = 0.4) -> list[LearningRecord]:
        query = "SELECT * FROM learning_progress WHERE mastery_level < ?"
        params = [threshold]
        if course:
            query += " AND course = ?"
            params.append(course)
        query += " ORDER BY mastery_level ASC LIMIT 10"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [LearningRecord(**{k: r[k] for k in r.keys()}) for r in rows]

    def get_stats(self) -> dict:
        with self._conn() as conn:
            return {
                "total_sessions": conn.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"],
                "total_messages": conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"],
                "semantic_memories": conn.execute(
                    "SELECT COUNT(*) c FROM semantic_memory WHERE is_active = 1"
                ).fetchone()["c"],
                "tracked_topics": conn.execute("SELECT COUNT(*) c FROM learning_progress").fetchone()["c"],
            }


# ─── LLM Extraction Prompts ─────────────────────────────────────────────────

EXTRACTION_PROMPT = """Analyze this conversation turn and extract important information worth remembering.

USER: {user_msg}
ASSISTANT: {assistant_msg}
COURSE: {course}

Return a JSON list. Be VERY selective — only genuinely useful info. Return [] if nothing notable.

```json
[
  {{
    "category": "preference|fact|goal|struggle|insight|exam",
    "content": "concise memory (max 100 chars)",
    "course": "course name or empty",
    "confidence": 0.0-1.0,
    "ttl_days": -1
  }}
]
```

Categories: preference (study style), fact (personal info), goal (targets/deadlines),
struggle (difficult topics), insight (breakthroughs), exam (upcoming exams/deadlines).
Set ttl_days for temporary info (e.g. 30 for "exam next month"), -1 for permanent."""

TOPIC_DETECTION_PROMPT = """What specific academic topics are discussed here?
USER: {user_msg}
COURSE: {course}
Return JSON list of specific topic strings. [] if none.
```json
["topic1", "topic2"]
```"""


# ─── Memory Manager (Orchestrator) ──────────────────────────────────────────


class HybridMemoryManager:
    """
    Orchestrates both layers.
    Total cost per turn: ~600-1300 tokens (vs ~4000-8000 for full-md).
    ~80% savings.
    """

    def __init__(self):
        self.profile = StaticProfile()
        self.db = DynamicMemoryDB()
        self.current_session_id: int | None = None
        self._engine = None

    @property
    def engine(self):
        """Lazy-init MultiProviderEngine to avoid circular imports."""
        if self._engine is None:
            from core.llm_providers import MultiProviderEngine

            self._engine = MultiProviderEngine()
        return self._engine

    # ─── Session ─────────────────────────────────────────────────────────

    def start_session(self, active_course: str = "") -> int:
        self.db.cleanup_expired()
        self.current_session_id = self.db.create_session(active_course)
        return self.current_session_id

    def end_session(self, summary: str = ""):
        if self.current_session_id:
            self.db.end_session(self.current_session_id, summary)
            self.current_session_id = None

    # ─── Context Builder (THE critical method) ───────────────────────────

    def build_memory_context(self, course: str | None = None, query: str | None = None) -> str:
        """
        Build memory context for system prompt injection.

        Token budget allocation:
        - Static profile:    ~300-500 tokens (always)
        - Semantic memories:  ~200-400 tokens (selective)
        - Weak topics:        ~100-150 tokens (if any)
        - Recent messages:    ~100-200 tokens (continuity)
        - Deep recall:        ~100-300 tokens (if query provided)
        - TOTAL:              ~700-1550 tokens
        """
        parts = []

        # Layer 1: Static profile (always)
        profile_text = self.profile.read()
        if profile_text.strip() and profile_text.strip() != DEFAULT_PROFILE.strip():
            parts.append(f"[PROFİL]\n{profile_text}")

        # Layer 2: Dynamic memories (budget-aware selective fetch)
        memories = self.db.get_memories_for_context(course=course, max_tokens=400)
        if memories:
            parts.append("[HAFIZA]\n" + "\n".join(m.to_text() for m in memories))

        # Layer 3: Weak topics (proactive help)
        weak = self.db.get_weak_topics(course)
        if weak:
            lines = [f"⚠️ {w.topic} ({w.mastery_level:.0%})" for w in weak[:5]]
            parts.append("[ZAYIF KONULAR]\n" + "\n".join(lines))

        # Layer 4: Recent messages (cross-session continuity)
        recent = self.db.get_recent_messages(limit=4)
        recent_ids = {m.get("id") for m in recent} if recent else set()
        if recent:
            lines = []
            for msg in recent:
                role = "K" if msg["role"] == "user" else "A"
                lines.append(f"{role}: {msg['content'][:100]}")
            parts.append("[SON KONUŞMA]\n" + "\n".join(lines))

        # Layer 5: Deep recall — keyword-based past conversation search
        if query and len(query) > 10:
            keywords = _extract_keywords(query)
            if keywords:
                seen_contents: set[str] = set()
                recall_lines: list[str] = []
                for kw in keywords:
                    for m in self.db.search_messages(kw, limit=3):
                        if m.get("id") in recent_ids:
                            continue
                        preview = m["content"][:120].replace("\n", " ")
                        if preview not in seen_contents:
                            seen_contents.add(preview)
                            role = "K" if m["role"] == "user" else "A"
                            recall_lines.append(f"{role}: {preview}")
                    for mem in self.db.search_semantic_memories(kw, limit=3):
                        if mem.content not in seen_contents:
                            seen_contents.add(mem.content)
                            recall_lines.append(mem.to_text())
                if recall_lines:
                    combined = "\n".join(recall_lines[:8])[:900]
                    parts.append("[İLGİLİ GEÇMİŞ KONUŞMALAR]\n" + combined)

        return "\n\n".join(parts) if parts else ""

    # ─── Record & Extract ────────────────────────────────────────────────

    def record_exchange(self, user_message: str, assistant_response: str, course: str = "", rag_sources: str = ""):
        if not self.current_session_id:
            self.start_session(course)

        user_msg_id = self.db.add_message(
            self.current_session_id,
            "user",
            user_message,
            rag_sources,
        )
        self.db.add_message(
            self.current_session_id,
            "assistant",
            assistant_response,
        )

        if len(user_message) > 50 and not user_message.startswith("/"):
            self._extract_memories(user_message, assistant_response, course, user_msg_id)
            self._detect_topics(user_message, course)

    def _extract_memories(self, user_msg: str, assistant_msg: str, course: str, source_id: int):
        try:
            prompt = EXTRACTION_PROMPT.format(
                user_msg=user_msg,
                assistant_msg=assistant_msg[:500],
                course=course or "N/A",
            )
            text = self.engine.complete(
                task="extraction",
                system="You are a memory extraction assistant. Return only valid JSON.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
            )
            for mem in self._parse_json(text):
                entry = MemoryEntry(
                    category=mem.get("category", "fact"),
                    content=mem.get("content", "")[:200],
                    source_message_id=source_id,
                    course=mem.get("course", course),
                    confidence=mem.get("confidence", 0.8),
                    ttl_days=mem.get("ttl_days", -1),
                )
                if entry.content:
                    self.db.add_memory(entry)
        except Exception as e:
            logger.debug(f"Memory extraction skipped: {e}")

    def _detect_topics(self, user_msg: str, course: str):
        if not course or len(user_msg) < 15:
            return
        try:
            prompt = TOPIC_DETECTION_PROMPT.format(user_msg=user_msg, course=course)
            text = self.engine.complete(
                task="topic_detect",
                system="You are a topic detection assistant. Return only valid JSON.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            for topic in self._parse_json(text):
                if isinstance(topic, str) and topic:
                    self.db.update_learning(course, topic, mastery_delta=0.05, asked=True)
        except Exception as e:
            logger.debug(f"Topic detection skipped: {e}")

    # ─── Manual Controls ─────────────────────────────────────────────────

    def remember(self, content: str, category: str = "fact", course: str = ""):
        self.db.add_memory(MemoryEntry(category=category, content=content, course=course))

    def forget(self, memory_id: int):
        self.db.deactivate_memory(memory_id)

    def list_memories(self, course: str = None) -> list[MemoryEntry]:
        return self.db.get_memories(course=course)

    def get_learning_progress(self, course: str = None) -> list[LearningRecord]:
        return self.db.get_learning_progress(course)

    def get_stats(self) -> dict:
        stats = self.db.get_stats()
        stats["profile_tokens"] = self.profile.get_token_estimate()
        return stats

    def update_profile_courses(self, courses: list[str]):
        self.profile.update_courses(courses)

    def edit_profile_path(self) -> str:
        return str(self.profile.path)

    @staticmethod
    def _parse_json(text: str) -> list:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1)
        try:
            result = json.loads(text.strip())
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []


# Alias for backward compatibility
MemoryManager = HybridMemoryManager
