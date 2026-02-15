# Moodle Student Tracker

<p align="center">
  <img src="./images/1.png" alt="Bilkent Moodle" width="600"/>
</p>

A **fully-automated, RAG-based personal academic assistant** for Bilkent University students. Indexes Moodle course materials, auto-authenticates STARS (grades/attendance/exams) with email 2FA, monitors university emails — all through a single Telegram bot with zero manual intervention.

**Dual-mode UX:** 8-button persistent keyboard for one-tap access + inline button navigation for reading mode. Two explicit modes — 📖 **Okuma Modu** (file-scoped reading with chunk navigation) and 💬 **Normal Mod** (RAG chat + academic tools) — with seamless switching.

---

## Table of Contents

- [Architecture](#architecture)
- [Design Patterns](#design-patterns)
- [Features](#features)
- [Data Flow](#data-flow)
- [Memory System](#memory-system)
- [Setup](#setup)
- [Deployment](#deployment)
- [Recommended Usage](#recommended-usage)
- [Tech Stack](#tech-stack)
- [File Structure](#file-structure)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             TELEGRAM BOT                                    │
│                          (telegram_bot.py)                                  │
│  Dual Mode (Reading/Normal) · Keyword Router · Persistent Keyboard          │
│  8 Button Handlers · Callback Engine · 6 Background Jobs                    │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────────┘
       │          │          │          │          │          │
 ┌─────▼─────┐ ┌──▼──────┐ ┌▼───────┐ ┌▼────────┐│  ┌───────▼──────────┐
 │LLM Engine │ │  Sync   │ │ Vector │ │ Memory  ││  │   Notification   │
 │ (RAG +    │ │ Engine  │ │ Store  │ │(Hybrid) ││  │   Engine (Diff)  │
 │ Prompts)  │ │         │ │ FAISS+ │ │         ││  │                  │
 └──┬────┬───┘ └──┬──┬───┘ │ BM25  │ └─────────┘│  └──────────────────┘
    │    │        │  │      └────────┘            │
┌───▼┐ ┌─▼──────┐│ ┌▼──────────────┐              │
│LLM │ │ Vector ││ │   Document    │              │
│Prov│ │ Store  ││ │  Processor    │              │
│iders│ │       ││ │ PDF/DOCX/OCR  │              │
└─────┘ └───────┘│ └───────────────┘              │
                 │                                 │
         ┌───────▼───────┐                         │
         │ Moodle Client │                         │
         │ (Web Services)│                         │
         └───────────────┘                         │
                                                   │
┌──────────────────────────┐  ┌────────────────────▼─────┐
│     STARS Client          │  │     Webmail Client        │
│  OAuth + Email 2FA        │◄─│     IMAP (AIRS/DAIS)     │
│  Auto-login (10 min)      │  │     Email monitoring      │
│  Grades · Exams ·         │  │     2FA code extraction   │
│  Attendance · GPA         │  │                           │
└──────────────────────────┘  └──────────────────────────┘
```

### Hexagonal Architecture (Ports & Adapters)

| Layer | Files | Role |
|-------|-------|------|
| **UI Adapters** | `telegram_bot.py`, `main.py` | User interfaces (Telegram, CLI) |
| **Core Logic** | `llm_engine.py`, `sync_engine.py`, `vector_store.py`, `memory.py` | Business logic, RAG pipeline, memory management |
| **External Adapters** | `moodle_client.py`, `stars_client.py`, `webmail_client.py`, `llm_providers.py` | External service integrations |

---

## Design Patterns

### Strategy Pattern — Document Extraction & LLM Providers
Different extraction strategies per file type, common interface for LLM providers:
```
DocumentProcessor._extract_pdf()  / _extract_docx() / _extract_pptx() / _extract_html()
MultiProviderEngine → Gemini / OpenAI / GLM (all OpenAI-compatible)
```

### Factory Pattern — Task-Based Model Routing
Environment-variable-driven model selection per task via `TaskRouter`:
```python
MODEL_CHAT=gemini-2.5-flash        # Main chat (RAG) + reading mode
MODEL_STUDY=gemini-2.5-flash       # Study mode (strict grounding)
MODEL_EXTRACTION=gpt-4.1-nano      # Memory extraction
MODEL_TOPIC_DETECT=gpt-4.1-nano    # Topic detection
MODEL_SUMMARY=gemini-2.5-flash     # Weekly summary
MODEL_QUESTIONS=gemini-2.5-flash   # Practice questions + quiz eval
MODEL_OVERVIEW=gemini-2.5-flash    # Course overview + file summaries
```

### Repository Pattern — Data Abstraction
`VectorStore` and `DynamicMemoryDB` abstract storage. Chunk dedup, FAISS persistence, SQLite memory:
```
VectorStore.add_chunks()  → deduplicate → encode → FAISS index → persist
VectorStore.hybrid_search() → FAISS (semantic) + BM25 (keyword) → RRF fusion → filter
DynamicMemoryDB           → SQLite (WAL mode) → token-budget ranking
```

### State Machine — Dual Mode + STARS Sessions
```
Bot Mode:    Normal ←→ Reading (via rd|normal / rd|resume)
             Reading states: active (reading_mode=True) | paused (reading_paused=True)

STARS:       idle → awaiting_sms → ready
             auth_time > 3500s (~58 min) → re-authenticate
```

### Chain of Responsibility — Sync Pipeline
Sequential stages, each transforms and passes forward:
```
Moodle API → Download → Extract (PDF/DOCX/OCR) → Math Normalize → Chunk → Embed → FAISS+BM25 Index
```

### Observer Pattern — Background Job Queue
6 periodic jobs via python-telegram-bot's APScheduler:
```
auto_sync_job        → 10 min   → Moodle sync + new material notification
auto_stars_login_job → 10 min   → STARS re-auth + data refresh + diff notifications
assignment_check     → 10 min   → New assignment detection
mail_check           → 30 min   → AIRS/DAIS email check + LLM summary
moodle_keepalive     → 2 min    → Moodle session keep-alive
deadline_reminder    → Daily 9AM → 3-day advance deadline warning
```

### Template Method — Context Injection
Every LLM call follows the same enrichment template:
```
system_prompt += _build_student_context()  →  date + schedule + STARS + assignments + courses
```
Context is **TTL-cached (5 min)** with manual invalidation on data changes.

### Adapter Pattern — External API Normalization
```
MoodleClient  → Moodle Web Services REST API
StarsClient   → OAuth 1.0 + HTML scraping (BeautifulSoup)
WebmailClient → IMAP4_SSL (mail.bilkent.edu.tr)
```

---

## Features

### Dual-Mode UX

The bot operates in two explicit modes with seamless switching:

**📖 Okuma Modu (Reading Mode)**
- File-scoped chunk-by-chunk reading with inline navigation buttons
- `[◀️ Geri]` `[▶️ Devam Et]` — navigate chunks
- `[🧠 Quiz]` — comprehensive quiz over all read chunks
- `[✅ Bitir]` — finish and return to normal mode
- `[💬 Normal Mod]` — pause reading (resumable) and switch to normal mode
- Free-text questions answered from the current file's content only
- Strict no-question LLM prompt — bot teaches, never asks

**💬 Normal Mod**
- 8-button persistent keyboard for one-tap access to all features
- RAG-powered chat with hybrid search (FAISS + BM25)
- Slash commands: `/calis`, `/notlar`, `/bugun`, `/haftam`, `/mail`, `/odevler`
- Paused reading reminder on RAG responses + "devam et" to resume

**Mode Transitions:**
```
[💬 Normal Mod] → pauses reading (state preserved) → normal mode
[▶️ Okumaya Dön] or "devam et" → resumes from where you left off
[✅ Bitir] → full reset → normal mode
```

### Persistent Keyboard (8 Buttons)

```
┌─────────────────┬─────────────────┐
│  📚 Ders Çalış  │  📊 Notlarım    │
├─────────────────┼─────────────────┤
│  📅 Bugün       │  📅 Bu Hafta    │
├─────────────────┼─────────────────┤
│  📬 Mailler     │  📝 Ödevler     │
├─────────────────┼─────────────────┤
│  🔄 Sync        │  ⚙️ Ayarlar     │
└─────────────────┴─────────────────┘
```

| Button | Action |
|--------|--------|
| 📚 Ders Çalış | Course selection → file list → enter reading mode |
| 📊 Notlarım | CGPA, grades, attendance summary + drill-down buttons |
| 📅 Bugün | Today's schedule (+ tomorrow preview) |
| 📅 Bu Hafta | Full weekly schedule (Mon–Fri) |
| 📬 Mailler | Latest AIRS/DAIS emails with LLM summary |
| 📝 Ödevler | Assignment deadlines and submission status |
| 🔄 Sync | Manual Moodle sync |
| ⚙️ Ayarlar | Socratic mode toggle, clear history |

### Full Automation (Zero Manual Intervention)
- **Auto STARS login** — Re-authenticates every 10 min, reads email 2FA code from IMAP automatically
- **Auto Moodle sync** — Checks for new materials every 10 min, notifies when new content is indexed
- **Auto assignment tracking** — Detects new assignments every 10 min
- **Auto email monitoring** — AIRS/DAIS emails checked every 30 min with LLM-summarized notifications
- **Deadline reminders** — Daily 9 AM notifications for assignments due within 3 days
- **STARS diff notifications** — Real-time alerts for grade changes, new exam dates, attendance updates
- **12-hour STARS summary** — Periodic push with CGPA, upcoming exams, attendance status

### Hybrid RAG Search

<p align="center">
  <img src="./images/5.png" alt="Study Mode" width="500"/>
  <br/>
  <em>Progressive study mode — deep teaching with RAG-grounded content</em>
</p>

- **Hybrid search** — FAISS (semantic) + BM25 (keyword) fused via Reciprocal Rank Fusion (k=60)
- **BM25 stemming** — Snowball TR/EN stemmers via PyStemmer (C extension, 1.1s build for 3600+ chunks)
- **Multilingual embedding** — `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages, +8% Turkish retrieval)
- **Adaptive threshold** — `max(top_score * 0.60, 0.20)` instead of fixed cutoff
- **Strict course filter** — course-filtered search → cross-course fallback only on zero results
- **File summaries** — Per-file LLM-generated overviews for richer context
- **Source attribution** — inline 📖 [dosya.pdf] tags + programmatic footer
- **Hybrid PDF extraction** — pre-scans pages (text vs scanned), routes text→pymupdf4llm, scanned→OCR
- **OCR quality check** — probe first 3 scanned pages, majority vote: 2+ fail → skip remaining
- **Math-aware pipeline** — ~50 Unicode symbol normalization, formula-aware chunking

### Keyword-Based Routing (Zero LLM Intent)
Message routing uses keyword matching with zero LLM overhead:
- `_STARS_KEYWORDS` → STARS data (grades, exams, attendance, schedule, CGPA)
- `_SYNC_KEYWORDS` → Moodle sync
- `_MAIL_KEYWORDS` → Email check
- `BUTTON_ROUTES` → 8 persistent keyboard button handlers
- Rule-based course detection (exact code → number match → history)
- Fallback → hybrid RAG search + LLM response

### STARS Integration

<p align="center">
  <img src="./images/3.png" alt="STARS Exams" width="500"/>
  <br/>
  <em>Exam schedule with countdown + course awareness</em>
</p>

<p align="center">
  <img src="./images/6.jpeg" alt="Grades" width="350"/>
  <br/>
  <em>Grade overview — all courses at a glance</em>
</p>

<p align="center">
  <img src="./images/4.png" alt="Attendance" width="500"/>
  <br/>
  <em>Attendance tracking with per-course ratio and absence details</em>
</p>

- **Fully automated** — OAuth + Email 2FA (reads verification code from IMAP)
- **Session management** — Auto-refresh every 10 min when expired
- **STARS diff notifications** — Grade changes, new exam dates, attendance ratio changes → instant Telegram alert
- **Full academic awareness** — CGPA, grades, exams, attendance, schedule injected into all LLM calls
- **Drill-down buttons** — `srs|grades_detail`, `srs|attendance` for detailed breakdowns

### Email Monitoring

<p align="center">
  <img src="./images/2.jpeg" alt="Mail Summary" width="350"/>
  <br/>
  <em>LLM-summarized email notifications from AIRS/DAIS</em>
</p>

- AIRS (instructor) and DAIS (department) emails
- Background check every 30 min with LLM-summarized notifications
- Natural language: "maillerime bak" triggers on-demand check
- **2FA code extraction** — Reads STARS verification codes from starsmsg@bilkent.edu.tr

### Memory & Personalization
- **3-layer architecture**: RAM conversation history → SQLite semantic memories → deep recall keyword search
- **Conversation history persistence** — survives bot restart (JSON file)
- **Deep cross-session recall** — Turkish keyword extraction + SQLite search for messages beyond 20-turn window
- Learning progress tracking (topic mastery 0–1.0)
- Weak topic detection and review suggestions
- Semantic memory extraction (preferences, goals, challenges)

---

## Data Flow

### Message Router (Keyword-Based)

```
User Message
  │
  ├─→ Reading Mode active? → clean wall (all text stays in reading handler)
  │   ├─→ Quiz answer (quiz_active) → evaluate with ✅/🔶/❌
  │   ├─→ "devam et" → next chunk batch
  │   ├─→ "test et" → comprehensive quiz over all read chunks
  │   └─→ Free text → file-scoped RAG question
  │
  ├─→ BUTTON_ROUTES match? → 8 persistent keyboard handlers (zero LLM)
  │   ├─→ 📚 Ders Çalış → course list → file list → reading mode
  │   ├─→ 📊 Notlarım   → STARS cache → grades/attendance/CGPA
  │   ├─→ 📅 Bugün      → today's schedule from STARS
  │   ├─→ 📅 Bu Hafta   → weekly schedule (Mon–Fri)
  │   ├─→ 📬 Mailler    → IMAP fetch + LLM summary
  │   ├─→ 📝 Ödevler    → Moodle assignments + deadlines
  │   ├─→ 🔄 Sync       → Moodle sync pipeline
  │   └─→ ⚙️ Ayarlar    → socratic toggle, clear history
  │
  ├─→ "devam et" + reading_paused? → resume reading from paused state
  │
  ├─→ Keyword routing (zero LLM):
  │   ├─→ _STARS_KEYWORDS → multi-intent STARS data
  │   ├─→ _SYNC_KEYWORDS  → sync pipeline
  │   └─→ _MAIL_KEYWORDS  → email check
  │
  └─→ RAG Pipeline (fallback):
      ├─→ Course detection (rule-based: exact code → number → history)
      ├─→ hybrid_search() → FAISS + BM25 → RRF fusion (top 10)
      │   └─→ Course filter → fallback to all courses only if 0 results
      ├─→ _build_student_context() (cached 5 min)
      ├─→ LLMEngine.chat_with_history()
      ├─→ Paused reading reminder (if applicable)
      └─→ Memory update + source footer
```

### Callback Router

```
Callback Query (InlineKeyboard)
  │
  ├─→ rd|  → Reading mode navigation
  │   ├─→ rd|next    → next chunk batch + populate reading_chunks_read
  │   ├─→ rd|back    → previous chunk batch
  │   ├─→ rd|quiz    → comprehensive quiz (all read chunks)
  │   ├─→ rd|normal  → pause reading → switch to normal mode
  │   ├─→ rd|resume  → restore paused reading → continue
  │   └─→ rd|finish  → full reset → return to normal mode
  │
  ├─→ rf|  → File selection → enter reading mode
  ├─→ cs|  → Course selection (study menu / file navigation)
  ├─→ srs| → STARS drill-down (grades detail, attendance)
  └─→ set| → Settings (socratic toggle, clear history)
```

### STARS Authentication (Fully Automated)

```
auto_stars_login_job (every 10 min):
  │
  ├─→ Session valid? → skip
  │
  └─→ Session expired (>58 min):
      ├─→ GET /srs/ → 4 redirects → login page
      ├─→ POST credentials → detect verification type
      ├─→ Poll IMAP (6×5s) for starsmsg@bilkent.edu.tr → extract code
      ├─→ POST verification code → oauth/authorize → authenticated
      ├─→ Fetch all data: grades, exams, attendance, schedule, CGPA
      ├─→ Inject into LLM context
      ├─→ Diff snapshot → notify grade/exam/attendance changes
      └─→ Every 12h: send summary notification
```

### Sync Pipeline (Every 10 min)

```
auto_sync_job:
  ├─→ Moodle API → discover courses & files
  ├─→ Download new files to data/downloads/
  ├─→ DocumentProcessor (hybrid extraction):
  │   ├─→ Pre-scan: classify pages as text vs scanned
  │   ├─→ Scanned → OCR probe (3 pages) → majority vote → early exit if bad
  │   ├─→ Text → pymupdf4llm batch (BATCH_SIZE=50, structured Markdown)
  │   ├─→ Math normalization (~50 Unicode symbols)
  │   └─→ RecursiveCharacterTextSplitter (1000 char, 200 overlap)
  ├─→ sentence-transformers encode → FAISS add + BM25 rebuild → persist
  ├─→ Generate file summaries (GPT-4.1-mini, per-file overviews)
  └─→ Notify user: "🆕 {n} yeni chunk indexlendi"
```

### Startup Sequence

```
post_init()
  ├─→ Moodle: auto-login (username/password → token)
  ├─→ Webmail: IMAP connect + seed AIRS/DAIS UIDs
  ├─→ STARS: auto-login + email 2FA → fetch all → set diff baseline
  ├─→ Vector store: load FAISS index + metadata + build BM25 index
  ├─→ Study sessions: restore from data/study_sessions.json
  ├─→ Conversation history: restore from data/conversation_history.json
  └─→ Register 6 background jobs
```

---

## Memory System

Three-layer hybrid architecture:

```
┌──────────────────────────┐  ┌──────────────────────────────┐  ┌────────────────────────────┐
│     STATIC LAYER          │  │      DYNAMIC LAYER            │  │      DEEP RECALL            │
│     (profile.md)          │  │      (SQLite DB)              │  │      (Keyword Search)       │
│                           │  │                               │  │                             │
│ Identity, preferences     │  │ Semantic memories             │  │ Cross-session search        │
│ Course list               │  │ Learning progress             │  │ Turkish keyword extraction  │
│ Study schedule            │  │ Conversation history (20 msg) │  │ SQLite message + memory     │
│                           │  │ Weak topic detection          │  │ search on every query       │
│ Always in prompt          │  │ Query-time selective           │  │ Activated for >10 char      │
│ ~300-500 tokens           │  │ ~300-800 tokens               │  │ queries, max 8 results      │
│ Rarely updated            │  │ Updated every turn            │  │ ~100-900 tokens             │
└──────────────────────────┘  └──────────────────────────────┘  └────────────────────────────┘

Total per-turn memory cost: ~700-2200 tokens
```

**Conversation history** is persisted to JSON and survives bot restarts. The deep recall layer enables the bot to reference conversations from days or weeks ago through keyword-based SQLite search.

---

## Setup

> **Detayli adim adim kurulum icin: [SETUP.md](./SETUP.md)**

### Requirements
- Python 3.11+ (3.12 recommended)
- Moodle 3.9+ (Web Services enabled)
- Tesseract OCR (for scanned PDFs)
- PyStemmer (for fast BM25 stemming)

### Installation

```bash
# 1. Clone and install
git clone <repo-url>
cd Moodle_Student_Tracker
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your credentials (see below)

# 3. Run
python telegram_bot.py
```

### Environment Variables

```bash
# ─── Moodle ──────────────────────────────────────────────
MOODLE_URL=https://moodle.bilkent.edu.tr/2025-2026-spring
MOODLE_USERNAME=
MOODLE_PASSWORD=

# ─── LLM API Keys ───────────────────────────────────────
GEMINI_API_KEY=                    # Google AI Studio
OPENAI_API_KEY=                    # OpenAI (extraction + fallback)
GLM_API_KEY=                       # Z.ai (optional fallback)

# ─── Task → Model Routing ───────────────────────────────
MODEL_CHAT=gemini-2.5-flash
MODEL_STUDY=gemini-2.5-flash
MODEL_EXTRACTION=gpt-4.1-nano
MODEL_TOPIC_DETECT=gpt-4.1-nano
MODEL_SUMMARY=gemini-2.5-flash
MODEL_QUESTIONS=gemini-2.5-flash
MODEL_OVERVIEW=gemini-2.5-flash

# ─── Telegram Bot ────────────────────────────────────────
TELEGRAM_BOT_TOKEN=                # @BotFather → /newbot
TELEGRAM_OWNER_ID=                 # Your Telegram chat ID

# ─── STARS ───────────────────────────────────────────────
STARS_USERNAME=
STARS_PASSWORD=

# ─── Webmail IMAP ────────────────────────────────────────
WEBMAIL_EMAIL=
WEBMAIL_PASSWORD=

# ─── Tuning (optional) ──────────────────────────────────
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
AUTO_SYNC_INTERVAL=600
ASSIGNMENT_CHECK_INTERVAL=600
```

### LLM Providers

| Provider | Model | Usage | Cost |
|----------|-------|-------|------|
| Google | Gemini 2.5 Flash | Chat, study, reading mode, summary, questions, overview | Free tier (1500 req/day) |
| OpenAI | GPT-4.1-nano | Memory extraction, topic detection | ~$0.005/1K req |
| Z.ai (GLM) | glm-4.7 | Fallback (optional) | Free tier |

**No LLM intent classifier** — keyword-based routing eliminates per-message classification cost.
**Estimated monthly cost for active daily use: ~$0.50**

### CLI Interface (Alternative)

```bash
# Sync Moodle materials
python main.py sync

# Interactive chat
python main.py chat

# Course summary
python main.py summary

# Web interface (requires: pip install gradio)
python main.py web
```

---

## Deployment

### Production (systemd)

```bash
# Copy files to server
scp telegram_bot.py main.py root@server:/opt/moodle-bot/
scp -r core/ root@server:/opt/moodle-bot/core/

# Syntax check before deploy
python -c "import ast; ast.parse(open('telegram_bot.py').read()); print('OK')"

# Restart service
ssh root@server "systemctl restart moodle-bot"

# Verify
ssh root@server "systemctl status moodle-bot --no-pager"

# View logs
ssh root@server "journalctl -u moodle-bot -f"
```

### Re-index from scratch

```bash
ssh root@server "cd /opt/moodle-bot && rm -f data/faiss.index data/metadata.json data/sync_state.json"
ssh root@server "systemctl restart moodle-bot"
# Then send /sync in Telegram
```

---

## Recommended Usage

### First Time Setup
1. Fill `.env` with all credentials
2. Run `python telegram_bot.py`
3. Open Telegram → find your bot → send `/start`
4. The bot will auto-login to Moodle, STARS, and Webmail
5. First sync happens automatically — wait for "indexing complete" notification
6. The 8-button persistent keyboard appears automatically

### Daily Workflow
- **Tap buttons** — most actions are one tap away from the persistent keyboard
- **📚 Ders Çalış** → pick a course → pick a file → bot reads it to you chunk by chunk
- **📊 Notlarım** → CGPA, grades, attendance at a glance
- **📅 Bugün** → today's class schedule
- **📬 Mailler** → latest emails summarized
- **Or just type naturally** — "hegemonya nedir?", "sınavlarım ne zaman?"

### Reading Mode (Recommended for Exam Prep)
1. Tap **📚 Ders Çalış** → select course → select file
2. Bot enters **📖 Okuma Modu** and starts teaching chunk by chunk
3. Navigate with inline buttons:
   - `[▶️ Devam Et]` — next section
   - `[◀️ Geri]` — previous section
   - `[🧠 Quiz]` — quiz over everything you've read so far
   - `[✅ Bitir]` — finish and return to normal mode
4. Ask questions anytime — answered from the current file only
5. Tap `[💬 Normal Mod]` to pause and check grades/schedule/etc.
6. Say "devam et" or tap `[▶️ Okumaya Dön]` to resume where you left off

### Commands

| Command | Description |
|---------|-------------|
| `/start` | Show welcome message + persistent keyboard |
| `/help` | Dual mode info + current mode status |
| `/calis` | Course selection (= 📚 Ders Çalış) |
| `/notlar` | Grades summary (= 📊 Notlarım) |
| `/bugun` | Today's schedule (= 📅 Bugün) |
| `/haftam` | Weekly schedule (= 📅 Bu Hafta) |
| `/mail` | Check emails (= 📬 Mailler) |
| `/odevler` | Assignment status (= 📝 Ödevler) |
| `/menu` | Course list |
| `/login` | Manual STARS login |
| `/sync` | Manual Moodle sync |
| `/stars` | STARS data panel |
| `/temizle` | Clear study sessions + history |

### Pro Tips
- The bot **understands Turkish naturally** — no formal syntax needed
- Compound STARS queries work: "hem notlarım hem devamsızlığım?"
- In reading mode, **all text stays file-scoped** — no accidental course mixing
- Paused readings survive mode switches — resume anytime with "devam et"
- All notifications are automatic — grades, exams, assignments, emails arrive without asking

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Bot Framework | python-telegram-bot 21+ (APScheduler job queue) |
| Embedding | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2, 384 dim) |
| Vector DB | FAISS (IndexFlatIP, cosine similarity) |
| Keyword Search | BM25 with Snowball TR/EN stemmers (PyStemmer C extension) |
| Hybrid Fusion | Reciprocal Rank Fusion (k=60, 2× candidate pool) |
| LLM | Gemini 2.5 Flash + GPT-4.1-nano |
| Document Processing | pymupdf4llm (batch), PyMuPDF, PyPDF2, python-docx, BeautifulSoup |
| OCR | Tesseract DPI=200 (tur+eng+equ) with probe-based quality check |
| Text Splitting | langchain RecursiveCharacterTextSplitter (equation-aware) |
| Memory | SQLite (WAL mode) + Markdown profile + JSON persistence |
| Web Scraping | requests + BeautifulSoup (STARS OAuth + HTML parsing) |
| Email | imaplib IMAP4_SSL (on-demand connection) |
| Async | asyncio + asyncio.to_thread() (non-blocking I/O) |

---

## File Structure

```
.
├── telegram_bot.py            # Main bot (dual mode + 8 button handlers + callback engine + 6 jobs)
├── main.py                    # CLI interface (sync, chat, summary, web)
├── core/
│   ├── config.py              # Environment variable management
│   ├── moodle_client.py       # Moodle Web Services API client
│   ├── document_processor.py  # Hybrid PDF extraction (pymupdf4llm + OCR) + DOCX/PPTX/HTML
│   ├── vector_store.py        # FAISS + BM25 hybrid search + dedup + RRF fusion
│   ├── llm_engine.py          # RAG orchestration + dual prompts + student context cache
│   ├── llm_providers.py       # Multi-provider LLM routing (TaskRouter)
│   ├── sync_engine.py         # Moodle → index pipeline
│   ├── memory.py              # 3-layer memory (static + dynamic SQLite + deep recall)
│   ├── stars_client.py        # STARS scraper (OAuth + Email/SMS 2FA)
│   └── webmail_client.py      # IMAP email monitoring + 2FA code extraction
├── tests/
│   ├── test_rag_quality.py    # RAG quality suite (34 queries, precision/pass_rate metrics)
│   └── rag_baseline.json      # RAG baseline for regression comparison
├── data/
│   ├── downloads/             # Downloaded course files
│   ├── file_summaries.json    # Per-file LLM-generated overviews
│   ├── study_sessions.json    # Persistent study session state
│   ├── conversation_history.json # Persistent conversation history
│   ├── memory.db              # SQLite dynamic memory
│   ├── faiss.index            # FAISS vector index
│   ├── metadata.json          # Chunk metadata
│   ├── sync_state.json        # Sync state
│   └── .moodle_token          # Cached Moodle token
├── images/                    # Screenshots for README
├── .env                       # Environment variables (not committed)
├── .env.example               # Example configuration
└── requirements.txt           # Python dependencies
```

---

## Stats

| Metric | Value |
|--------|-------|
| Indexed chunks | ~3,660 |
| Courses | 5 |
| Files | 28 |
| File summaries | 28 |
| Background jobs | 6 |
| Persistent keyboard buttons | 8 |
| Callback prefixes | 6 (rd\|, rf\|, cs\|, srs\|, set\|, ozet\_) |
| Slash commands | 13 |
| Embedding dimensions | 384 |
| Supported languages | 50+ |
| Hybrid search (BM25+FAISS) | precision 94%, pass rate 97% |
| BM25 build time | ~1.1s (PyStemmer) |
| Estimated monthly cost | ~$0.50 |
