# Moodle Student Tracker

<p align="center">
  <img src="./images/1.png" alt="Bilkent Moodle" width="600"/>
</p>

A **fully-automated, RAG-based personal academic assistant** for Bilkent University students. Indexes Moodle course materials, auto-authenticates STARS (grades/attendance/exams) with email 2FA, monitors university emails — all through a single Telegram bot with zero manual intervention.

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
│    Commands · Intent Router · Callback Handler · 6 Background Jobs          │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────────┘
       │          │          │          │          │          │
 ┌─────▼─────┐ ┌──▼──────┐ ┌▼───────┐ ┌▼────────┐│  ┌───────▼──────────┐
 │LLM Engine │ │  Sync   │ │ Vector │ │ Memory  ││  │   Notification   │
 │ (RAG +    │ │ Engine  │ │ Store  │ │(Hybrid) ││  │   Engine (Diff)  │
 │ Prompts)  │ │         │ │ FAISS  │ │         ││  │                  │
 └──┬────┬───┘ └──┬──┬───┘ └────────┘ └─────────┘│  └──────────────────┘
    │    │        │  │                             │
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
MODEL_CHAT=gemini-2.5-flash        # Main chat (RAG)
MODEL_STUDY=gemini-2.5-flash       # Study mode (strict grounding)
MODEL_INTENT=gpt-4.1-mini          # Intent classification (~600ms, 97%)
MODEL_EXTRACTION=gpt-4.1-nano      # Memory extraction
MODEL_TOPIC_DETECT=gpt-4.1-nano    # Topic detection
MODEL_SUMMARY=gemini-2.5-flash     # Weekly summary
MODEL_QUESTIONS=gemini-2.5-flash   # Practice questions
MODEL_OVERVIEW=gemini-2.5-flash    # Course overview
```

### Repository Pattern — Data Abstraction
`VectorStore` and `DynamicMemoryDB` abstract storage. Chunk dedup, FAISS persistence, SQLite memory:
```
VectorStore.add_chunks()  → deduplicate → encode → FAISS index → persist
VectorStore.query()       → encode query → cosine similarity → filter → return
DynamicMemoryDB           → SQLite (WAL mode) → token-budget ranking
```

### State Machine — STARS Session Management
```
StarsSession._phase:  idle → awaiting_sms → ready
StarsSession.expired:  auth_time > 3500s (~58 min) → re-authenticate
```

### Chain of Responsibility — Sync Pipeline
Sequential stages, each transforms and passes forward:
```
Moodle API → Download → Extract (PDF/DOCX/OCR) → Math Normalize → Chunk → Embed → FAISS Index
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

### Full Automation (Zero Manual Intervention)
- **Auto STARS login** — Re-authenticates every 10 min, reads email 2FA code from IMAP automatically
- **Auto Moodle sync** — Checks for new materials every 10 min, notifies when new content is indexed
- **Auto assignment tracking** — Detects new assignments every 10 min
- **Auto email monitoring** — AIRS/DAIS emails checked every 30 min with LLM-summarized notifications
- **Deadline reminders** — Daily 9 AM notifications for assignments due within 3 days
- **STARS diff notifications** — Real-time alerts for grade changes, new exam dates, attendance updates
- **12-hour STARS summary** — Periodic push with CGPA, upcoming exams, attendance status

### Natural Language Interface
- **Zero-command UX** — 4 essential commands, everything else via natural conversation
- **12 intent classes** — STUDY, ASSIGNMENTS, MAIL, SYNC, SUMMARY, QUESTIONS, EXAM, GRADES, SCHEDULE, ATTENDANCE, CGPA, CHAT
- **Multi-intent STARS queries** — "sinavlarim ne zaman ve devamsizligim?" → EXAM + ATTENDANCE
- **3-tier course detection** — exact code → number match → LLM-based (cached)
- **Study continuation** — fuzzy "devam" matching resumes active study session

### Academic Assistant (RAG)

<p align="center">
  <img src="./images/5.png" alt="Study Mode" width="500"/>
  <br/>
  <em>Progressive study mode — deep teaching with RAG-grounded content</em>
</p>

- **Multilingual embedding** — `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages, +8% Turkish retrieval vs English-only)
- **Hybrid PDF extraction** — pre-scans pages (text vs scanned), routes text→pymupdf4llm, scanned→OCR with quality probe
- **OCR quality check** — probe first 3 scanned pages, majority vote: 2+ fail → skip remaining
- **Math-aware pipeline** — ~50 Unicode symbol normalization, formula-aware chunking
- **Smart RAG fallback** — course-filtered → cross-course fallback → skip RAG if no materials
- **Source attribution** — programmatic footer with dedup (📚 Kaynak: file.pdf)
- **Progressive study mode** — 6-step deep teaching per subtopic (teach → quiz → reteach → summary)
- Practice question generation, course overview, weekly summary

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
- Natural language: "notlarim nedir?", "sinav ne zaman?", "devamsizligim?"

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

### Intent Router (NLU)

```
User Message
  │
  ├─→ Study session active? → fuzzy "devam" match → continue study
  │
  ├─→ _classify_intent() → GPT-4.1-mini (~600ms, 12 intents)
  │   ├─→ STUDY       → progressive 6-step deep teaching
  │   ├─→ ASSIGNMENTS → Moodle API fetch + format
  │   ├─→ MAIL        → IMAP + LLM summary
  │   ├─→ SYNC        → sync stats + new chunk count
  │   ├─→ SUMMARY     → course content overview
  │   ├─→ QUESTIONS   → practice question generation
  │   ├─→ EXAM/GRADES/SCHEDULE/ATTENDANCE/CGPA
  │   │   └─→ multi-intent keyword detection → reply ALL detected
  │   └─→ CHAT        → RAG pipeline (below)
  │
  └─→ RAG Pipeline (CHAT intent):
      ├─→ Course detection (3-tier: exact code → number → LLM-based)
      ├─→ VectorStore.query() → FAISS cosine similarity (top 15)
      │   ├─→ Course filter + smart fallback:
      │   │   ├─→ Weak match → search all courses
      │   │   ├─→ Proper noun not found → force cross-course
      │   │   └─→ No materials → skip RAG, use general knowledge
      │   └─→ Source attribution: extract top source files
      ├─→ _build_student_context() (cached 5 min)
      ├─→ LLMEngine.chat_with_history() → Gemini 2.5 Flash
      └─→ Memory update + source footer (📚 Kaynak: file.pdf)
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
  ├─→ sentence-transformers encode → FAISS add → persist
  └─→ Notify user: "🆕 {n} yeni chunk indexlendi"
```

### Startup Sequence

```
post_init()
  ├─→ Moodle: auto-login (username/password → token)
  ├─→ Webmail: IMAP connect + seed AIRS/DAIS UIDs
  ├─→ STARS: auto-login + email 2FA → fetch all → set diff baseline
  ├─→ Vector store: load FAISS index + metadata
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
OPENAI_API_KEY=                    # OpenAI (intent + extraction)
GLM_API_KEY=                       # Z.ai (optional fallback)

# ─── Task → Model Routing ───────────────────────────────
MODEL_CHAT=gemini-2.5-flash
MODEL_STUDY=gemini-2.5-flash
MODEL_INTENT=gpt-4.1-mini
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
| Google | Gemini 2.5 Flash | Chat, study, summary, questions, overview | Free tier (1500 req/day) |
| OpenAI | GPT-4.1-mini | Intent classification (97% accuracy) | ~$0.016/1K req |
| OpenAI | GPT-4.1-nano | Memory extraction, topic detection | ~$0.005/1K req |
| Z.ai (GLM) | glm-4.7 | Fallback (optional) | Free tier |

**Estimated monthly cost for active daily use: ~$0.90**

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

### Daily Workflow
- **Ask anything naturally** — no need to memorize commands. Just type your question.
- "Edeb dersine calismak istiyorum" → starts progressive study session
- "Sinav tarihlerim?" → shows upcoming exams with countdown
- "Maillerime bak" → checks AIRS/DAIS emails
- "Odevlerim ne durumda?" → shows assignment deadlines
- "Hegemonya nedir?" → RAG search across all course materials

### Study Mode (Recommended for Exam Prep)
1. Say "X dersine calismak istiyorum"
2. Select source files (PDFs) from toggle buttons
3. Bot teaches topic-by-topic with **6-step deep method**:
   - Teach → Mini quiz → Re-teach weak areas → Summary card
4. Say "devam" to continue, "plan" to see/jump topics
5. Session persists across bot restarts

### Commands
| Command | Description |
|---------|-------------|
| `/start` | Show main menu |
| `/menu` | Course list |
| `/odevler` | Assignment status |
| `/login` | Manual STARS login |
| `/sync` | Manual Moodle sync |
| `/stars` | STARS data panel |
| `/temizle` | Clear study sessions + history |

### Pro Tips
- The bot **understands Turkish naturally** — no formal syntax needed
- Compound queries work: "hem notlarim hem devamsizligim?"
- Course prefixes work: "Edeb devam" resumes study for that course
- The bot remembers past conversations across sessions — reference old topics freely
- All notifications are automatic — grades, exams, assignments, emails arrive without asking

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Bot Framework | python-telegram-bot 21+ (APScheduler job queue) |
| Embedding | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2, 384 dim) |
| Vector DB | FAISS (IndexFlatIP, cosine similarity) |
| LLM | Gemini 2.5 Flash + GPT-4.1-mini + GPT-4.1-nano |
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
├── telegram_bot.py            # Main bot (handlers + 6 background jobs + intent router + notifications)
├── main.py                    # CLI interface (sync, chat, summary, web)
├── core/
│   ├── config.py              # Environment variable management
│   ├── moodle_client.py       # Moodle Web Services API client
│   ├── document_processor.py  # Hybrid PDF extraction (pymupdf4llm + OCR) + DOCX/PPTX/HTML
│   ├── vector_store.py        # FAISS vector store + dedup + filename filter
│   ├── llm_engine.py          # RAG orchestration + dual prompts + student context cache
│   ├── llm_providers.py       # Multi-provider LLM routing (TaskRouter)
│   ├── sync_engine.py         # Moodle → index pipeline
│   ├── memory.py              # 3-layer memory (static + dynamic SQLite + deep recall)
│   ├── stars_client.py        # STARS scraper (OAuth + Email/SMS 2FA)
│   └── webmail_client.py      # IMAP email monitoring + 2FA code extraction
├── data/
│   ├── downloads/             # Downloaded course files
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
| Indexed chunks | ~3,600 |
| Courses | 5 |
| Files | 28 |
| Intents | 12 |
| Background jobs | 6 |
| Embedding dimensions | 384 |
| Supported languages | 50+ |
| Intent accuracy | 97% |
| Estimated monthly cost | ~$0.90 |
