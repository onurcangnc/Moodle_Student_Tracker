# Moodle Student Tracker

![Logo](./images/1.png)

A **fully-automated, RAG-based personal academic assistant** for Bilkent University students. Indexes Moodle course materials, auto-authenticates STARS (grades/attendance/exams) with email 2FA, monitors university emails — all through a single Telegram bot with zero manual intervention.

![Logo](./images/2.png)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TELEGRAM BOT                                  │
│                        (telegram_bot.py)                                │
│  Commands · Intent Router · Callback Handler · 6 Background Jobs        │
└──────────┬──────────┬──────────┬──────────┬──────────┬─────────────────┘
           │          │          │          │          │
  ┌────────▼───┐ ┌────▼────┐ ┌──▼───┐ ┌───▼────────┐ │
  │ LLM Engine │ │  Sync   │ │Vector│ │   Memory   │ │
  │  (RAG +    │ │ Engine  │ │Store │ │  (Hybrid)  │ │
  │  Prompts)  │ │         │ │FAISS │ │            │ │
  └──┬────┬────┘ └──┬──┬──┘ └──────┘ └────────────┘ │
     │    │         │  │                              │
┌────▼┐ ┌─▼───────┐│ ┌▼──────────────┐               │
│ LLM │ │ Vector  ││ │   Document    │               │
│Provid│ │ Store   ││ │  Processor    │               │
│ ers  │ │         ││ │ PDF/DOCX/OCR  │               │
└──────┘ └─────────┘│ └───────────────┘               │
                    │                                  │
            ┌───────▼───────┐                          │
            │ Moodle Client │                          │
            │ (Web Services)│                          │
            └───────────────┘                          │
                                                       │
┌──────────────────────────┐  ┌────────────────────────▼───┐
│     STARS Client          │  │     Webmail Client          │
│  OAuth + Email 2FA        │◄─│     IMAP (AIRS/DAIS)       │
│  Auto-login (10 min)      │  │     Email monitoring        │
│  Grades · Exams ·         │  │     2FA code extraction     │
│  Attendance · GPA         │  │                             │
└──────────────────────────┘  └────────────────────────────┘
```

![alt text](./images/4.png)

![alt text](./images/5.png)

### Hexagonal Architecture (Ports & Adapters)

| Layer | Files | Role |
|-------|-------|------|
| **UI Adapters** | `telegram_bot.py`, `main.py` | User interfaces (Telegram, CLI) |
| **Core Logic** | `llm_engine.py`, `sync_engine.py`, `vector_store.py`, `memory.py` | Business logic, RAG, memory management |
| **External Adapters** | `moodle_client.py`, `stars_client.py`, `webmail_client.py`, `llm_providers.py` | External service integrations |

---

## Design Patterns

### Strategy Pattern
Different extraction strategies for different file types, common interface for different LLM providers:
```
DocumentProcessor._extract_pdf()  / _extract_docx() / _extract_pptx() / _extract_html()
MultiProviderEngine → Gemini / OpenAI / GLM (all OpenAI-compatible)
```

![alt text](./images/4.png)

### Repository Pattern
`VectorStore` and `DynamicMemoryDB` abstract data access. Chunk dedup, FAISS persistence, SQLite memory store:
```
VectorStore.add_chunks()  → deduplicate → encode → FAISS index → persist
VectorStore.query()       → encode query → cosine similarity → filter → return
DynamicMemoryDB           → SQLite (WAL mode) → token-budget ranking
```

### State Machine
STARS session management with explicitly defined states:
```
StarsSession._phase:  idle → awaiting_sms → ready
StarsSession.expired:  auth_time > 3500s (~58 min) → re-authenticate
```

### Factory Pattern
Task-based LLM model selection via environment variables:
```python
# .env routing:
MODEL_CHAT=gemini-2.5-flash        # Main chat (RAG)
MODEL_STUDY=gemini-2.5-flash       # Study mode (strict grounding)
MODEL_EXTRACTION=gpt-4.1-nano      # Memory extraction
MODEL_TOPIC_DETECT=gpt-4.1-nano    # Topic detection
MODEL_SUMMARY=gemini-2.5-flash     # Weekly summary
MODEL_QUESTIONS=gemini-2.5-flash   # Practice questions
MODEL_OVERVIEW=gemini-2.5-flash    # Course overview
MODEL_INTENT=gpt-4.1-mini          # Intent classification
```

### Chain of Responsibility
Sync pipeline in sequential stages:
```
Moodle API → Download → Extract (PDF/DOCX/OCR) → Math Normalize → Chunk → Embed → FAISS Index
```

### Intent Router (NLU)
Multi-intent classification via LLM. 12 intents, only 4 explicit commands:
```
User Message → _classify_intent() (GPT-4.1-mini, ~600ms, 97% accuracy)
  → STUDY        → Progressive study session (6-step deep teaching)
  → ASSIGNMENTS  → Fetch & format Moodle assignments
  → MAIL         → IMAP fetch + LLM summary
  → SYNC         → Moodle sync status / new material check
  → SUMMARY      → Course content overview generation
  → QUESTIONS    → Practice question generation
  → EXAM         → STARS exam schedule (cached)
  → GRADES       → STARS grades (cached)
  → SCHEDULE     → STARS weekly schedule (cached)
  → ATTENDANCE   → STARS attendance (cached)
  → CGPA         → STARS academic info (cached)
  → CHAT         → RAG conversational chat (default)

Multi-intent: STARS queries auto-detect compound intents
  "sınavlarım ne zaman ve devamsızlığım?" → EXAM + ATTENDANCE

Explicit commands: /start  /login  /sync  /temizle
Hidden admin:      /stats  /maliyet  /modeller
```

### Observer (Job Queue)
6 background jobs via python-telegram-bot's APScheduler:
```
auto_sync_job        → 10 min  → Moodle sync + new material notification
auto_stars_login_job → 10 min  → STARS auto-login (email 2FA) + data refresh
assignment_check     → 10 min  → New assignment detection
mail_check           → 30 min  → AIRS/DAIS email check + LLM summary
moodle_keepalive     → 2 min   → Moodle session keep-alive
deadline_reminder    → Daily 9AM → 3-day advance deadline warning
```

### Adapter Pattern
External APIs transformed into a common interface:
```
MoodleClient  → Moodle Web Services REST API
StarsClient   → OAuth 1.0 + HTML scraping (BeautifulSoup)
WebmailClient → IMAP4_SSL (mail.bilkent.edu.tr)
```

### Template Method
Every LLM call follows the same context injection template:
```
system_prompt += _build_student_context()  →  date + schedule + STARS + assignments + all courses (~600 tokens)
```
RAG chat flow:
```
query → intent classify → detect course → vector search (+ fallback) → LLM call → save history
```

---

## Data Flow

### Message Flow (Intent-Routed)

```
User Message
  │
  ├─→ Study session active? → fuzzy "devam" match → continue study
  │
  ├─→ _classify_intent() → GPT-4.1-mini (~600ms, 12 intents)
  │   ├─→ STUDY       → _start_study_session() → progressive 6-step teaching
  │   │                  (or resume existing session if same course)
  │   ├─→ ASSIGNMENTS  → _format_assignments() → Moodle API fetch
  │   ├─→ MAIL         → _handle_mail_intent() → IMAP + LLM summary
  │   ├─→ SYNC         → Show last sync stats + new chunk count
  │   ├─→ SUMMARY      → _handle_summary_intent() → course overview
  │   ├─→ QUESTIONS    → _handle_questions_intent() → practice questions
  │   ├─→ EXAM/GRADES/SCHEDULE/ATTENDANCE/CGPA
  │   │   └─→ _detect_stars_intents() → multi-intent keyword detection
  │   │       └─→ Reply ALL detected intents (not just primary)
  │   └─→ CHAT         → RAG pipeline (below)
  │
  ├─→ RAG Pipeline (CHAT intent):
  │   ├─→ Active course detection (3-tier: exact code → number match → LLM-based)
  │   ├─→ Course material check (has indexed materials?)
  │   ├─→ VectorStore.query() → FAISS cosine similarity (top 15)
  │   │   ├─→ Course filter + smart fallback:
  │   │   │   ├─→ Course HAS materials but weak match → search all courses
  │   │   │   ├─→ Proper noun not found in results → force cross-course search
  │   │   │   └─→ Course has NO materials → skip RAG, use LLM general knowledge
  │   │   └─→ Source attribution: extract top source files for footer
  │   ├─→ _build_student_context() → date, schedule, STARS, assignments, all courses (~600 tokens)
  │   ├─→ LLMEngine.chat_with_history() → Gemini 2.5 Flash
  │   ├─→ Footer dedup: strip LLM-generated footer → append programmatic footer
  │   └─→ Memory update + source footer (📚 Kaynak: file1.pdf, file2.pdf)
  │
  └─→ Send response to Telegram (auto-split for messages > 4096 chars)
```

### Startup Sequence

```
post_init()
  ├─→ Moodle: auto-login (username/password → token)
  ├─→ Webmail: IMAP connect + seed AIRS/DAIS UIDs
  ├─→ STARS: auto-login + email 2FA auto-verify → fetch all data → inject context
  ├─→ Vector store: load FAISS index + metadata
  ├─→ Study sessions: restore from data/study_sessions.json
  └─→ Register 6 background jobs
```

### STARS Authentication Flow (Fully Automated)

```
auto_stars_login_job (every 10 min):
  │
  ├─→ Session valid? → skip
  │
  └─→ Session expired (>58 min):
      ├─→ GET /srs/ → 4 redirects → login page
      ├─→ POST credentials → detect verification type:
      │   ├─→ verifyEmail → EmailVerifyForm[verifyCode]
      │   └─→ verifySms  → SmsVerifyForm[verifyCode]
      ├─→ Poll IMAP (6×5s) for starsmsg@bilkent.edu.tr → extract code
      ├─→ POST verification code → oauth/authorize → authenticated
      ├─→ Fetch all data: grades, exams, attendance, schedule, CGPA
      ├─→ Inject into LLM context (_build_student_context)
      └─→ Every 12h: send summary notification to user
          (📊 CGPA, upcoming exams, attendance warnings)
```

### Sync Pipeline (Background, Every 10 min)

```
auto_sync_job:
  ├─→ Moodle API → discover courses & files
  ├─→ Download new files to data/downloads/
  ├─→ DocumentProcessor (hybrid extraction):
  │   ├─→ Pre-scan: classify pages as text vs scanned
  │   ├─→ Scanned pages → OCR probe (3 pages) → majority vote:
  │   │   ├─→ 2+ fail quality check → skip remaining (early exit)
  │   │   └─→ quality OK → OCR all scanned pages (Tesseract, DPI=200)
  │   ├─→ Text pages → pymupdf4llm batch (BATCH_SIZE=50, structured Markdown)
  │   ├─→ Math normalization (~50 Unicode symbols → searchable text)
  │   ├─→ Equation block protection (sentinel markers)
  │   └─→ RecursiveCharacterTextSplitter (1000 char, 200 overlap)
  ├─→ sentence-transformers encode → FAISS add → persist
  └─→ new_chunks > 0 ? → notify user: "🆕 {n} yeni chunk indexlendi"
```

---

## Memory System

Two-layer hybrid architecture:

```
┌──────────────────────┐    ┌──────────────────────────┐
│   STATIC LAYER       │    │     DYNAMIC LAYER        │
│   (profile.md)       │    │     (SQLite DB)          │
│                      │    │                          │
│ Identity, prefs      │    │ Semantic memories        │
│ Course list          │    │ Learning progress        │
│ Study schedule       │    │ Conversation history     │
│                      │    │ Weak topic detection     │
│ Always in prompt     │    │ Query-time selective     │
│ ~300-500 tokens      │    │ ~300-800 tokens          │
│ Rarely updated       │    │ Updated every turn       │
└──────────────────────┘    └──────────────────────────┘

Total per-turn cost: ~600-1300 tokens (vs 4000-8000 full-context)
```

---

## Features

### Full Automation (Zero Manual Intervention)
- **Auto STARS login** — Re-authenticates every 10 min when session expires, reads email 2FA code from IMAP automatically
- **Auto Moodle sync** — Checks for new materials every 10 min, notifies user when new content is indexed
- **Auto assignment tracking** — Checks for new assignments every 10 min
- **Auto email monitoring** — AIRS/DAIS emails checked every 30 min with LLM-summarized notifications
- **Deadline reminders** — Daily 9 AM notifications for assignments due within 3 days
- **12-hour STARS summary** — Periodic notification with CGPA, upcoming exams, attendance status

### Natural Language Interface
- **Zero-command UX** — 4 essential commands, everything else via natural conversation
- **Multi-intent classification** — LLM-based intent routing (GPT-4.1-mini, ~600ms, 97% accuracy)
- **12 intent classes** — STUDY, ASSIGNMENTS, MAIL, SYNC, SUMMARY, QUESTIONS, EXAM, GRADES, SCHEDULE, ATTENDANCE, CGPA, CHAT
- **Multi-intent STARS queries** — "sınavlarım ne zaman ve devamsızlığım?" → both EXAM + ATTENDANCE
- **3-tier course detection** — exact code match → number match → LLM-based (cached, no network call per message)
- **Study continuation** — fuzzy "devam" matching resumes active study session even with course prefix ("Edebe devam")

### Academic Assistant (RAG)
- Automatically indexes Moodle course materials (PDF, DOCX, PPTX, HTML, RTF + OCR)
- **Multilingual embedding** — `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages, +8% better Turkish retrieval vs English-only model)
- **Hybrid PDF extraction** — pre-scans pages (text vs scanned), routes text→pymupdf4llm, scanned→OCR with quality probe and early exit
- **OCR quality check** — probe first 3 scanned pages, majority vote: if 2+ fail → skip remaining (avoids wasting time on unreadable manuscripts)
- **Math-aware pipeline** — Unicode symbol normalization (~50 symbols), formula-aware chunking with equation block protection
- **Dual-text embedding** — original text for LLM, normalized text for FAISS (e.g. `∫x²dx` → `integral x^2 dx`)
- **Smart RAG fallback** — course-filtered → cross-course fallback (proper noun detection) → skip RAG if no materials
- **Source attribution** — programmatic footer with dedup (strips LLM-generated footers before appending)
- **Progressive study mode** — 6-step deep teaching per subtopic (teach → quiz → reteach → summary card)
- **Unified student context** — every LLM call knows: date, schedule, grades, exams, assignments, all enrolled courses + material status
- Practice question generation, course overview, weekly summary

### STARS Integration
- **Fully automated** — Auto-login via OAuth + Email 2FA (reads verification code from IMAP)
- **Session management** — Auto-refresh every 10 min when expired (>58 min lifetime)
- **Full academic awareness** — CGPA, grades, exams, attendance, schedule injected into all LLM calls
- **12-hour summary notifications** — Periodic push with CGPA, upcoming exams, attendance warnings
- Exam schedule with countdown (days remaining)
- Attendance tracking (percentage + details)
- Natural language queries: "notlarım nedir?", "sınavım ne zaman?"

### Moodle Tracking
- **Automatic synchronization** — Every 10 minutes (configurable via `AUTO_SYNC_INTERVAL`)
- **New material notifications** — Telegram push when new chunks are indexed
- Assignment deadline tracking — injected into LLM context
- Deadline reminders (3 days in advance, daily 9 AM)
- File upload + indexing (user-submitted PDF/DOCX/PPTX)
- Semester reset detection (MOODLE_URL change → auto-clear + re-sync)

### Email Monitoring
- AIRS (instructor) and DAIS (department) emails
- Background check every 30 minutes with LLM-summarized notifications
- Natural language: "maillerime bak" triggers on-demand check
- **2FA code extraction** — Reads STARS verification codes from starsmsg@bilkent.edu.tr

### Memory & Personalization
- Learning progress tracking (topic mastery 0-1.0)
- Weak topic detection and review suggestions
- Conversation history (last 20 messages)
- Semantic memory (preferences, goals, challenges)

---

## File Structure

```
.
├── telegram_bot.py          # Main Telegram bot (handlers + 6 background jobs + intent router)
├── main.py                  # CLI interface (sync, chat, summary, web)
├── core/
│   ├── config.py            # Environment variable management
│   ├── moodle_client.py     # Moodle Web Services API client
│   ├── document_processor.py # Hybrid PDF extraction (pymupdf4llm + OCR) + DOCX/PPTX/HTML
│   ├── vector_store.py      # FAISS vector store + dedup + filename filter
│   ├── llm_engine.py        # RAG orchestration + dual system prompts (chat/study)
│   ├── llm_providers.py     # Multi-provider LLM routing (TaskRouter)
│   ├── sync_engine.py       # Moodle → index pipeline
│   ├── memory.py            # Hybrid memory (static profile + dynamic SQLite)
│   ├── stars_client.py      # Bilkent STARS scraper (OAuth + Email/SMS 2FA)
│   └── webmail_client.py    # IMAP email monitoring + 2FA code extraction
├── data/
│   ├── downloads/           # Downloaded course files
│   ├── study_sessions.json  # Persistent study session state
│   ├── memory.db            # SQLite dynamic memory
│   ├── faiss.index          # FAISS vector index
│   ├── metadata.json        # Chunk metadata
│   ├── sync_state.json      # Sync state
│   └── .moodle_token        # Cached Moodle token
├── .env                     # Environment variables (not committed)
├── .env.example             # Example configuration
└── requirements.txt         # Python dependencies
```

---

## Setup

### Requirements
- Python 3.11+
- Moodle 3.9+ (Web Services enabled)
- Tesseract OCR (for scanned PDFs)

### Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment variables
cp .env.example .env
# Edit the .env file (Moodle, LLM API keys, Telegram token, STARS, Webmail)

# 3. Run with Telegram bot
python telegram_bot.py

# --- or with CLI ---

# Synchronization
python main.py sync

# Interactive chat
python main.py chat

# Web interface (Gradio)
python main.py web
```

### LLM API Keys

| Provider | Model | Usage | Env Variable |
|----------|-------|-------|-------------|
| Google | Gemini 2.5 Flash | Chat, study, summary, questions, overview | `GEMINI_API_KEY` |
| OpenAI | GPT-4.1-mini | Intent classification (97% accuracy) | `OPENAI_API_KEY` |
| OpenAI | GPT-4.1-nano | Memory extraction, topic detection | `OPENAI_API_KEY` |
| Z.ai (GLM) | glm-4.7 | Fallback | `GLM_API_KEY` |

### Moodle Token

Obtained automatically (via `MOODLE_USERNAME` + `MOODLE_PASSWORD`) or manually:
```
https://MOODLE_URL/login/token.php?username=XXX&password=XXX&service=moodle_mobile_app
```

---

## Deployment

Production deployment with systemd:

```bash
# Copy files to server
scp telegram_bot.py root@server:/opt/moodle-bot/
scp -r core/ root@server:/opt/moodle-bot/core/

# Start the service
ssh root@server "systemctl restart moodle-bot"

# Check status
ssh root@server "systemctl status moodle-bot --no-pager"

# Syntax check before deploy
python3 -c "import ast; ast.parse(open('telegram_bot.py').read()); print('OK')"

# Re-index from scratch
ssh root@server "cd /opt/moodle-bot && rm -f data/faiss.index data/metadata.json data/sync_state.json"
ssh root@server "systemctl restart moodle-bot"
# Then send /sync in Telegram
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Bot Framework | python-telegram-bot 21+ (APScheduler job queue) |
| Embedding | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2, 384 dim, 50+ langs) |
| Vector DB | FAISS (IndexFlatIP, cosine similarity) |
| LLM | Gemini 2.5 Flash (chat/study) + GPT-4.1-mini (intent) + GPT-4.1-nano (extraction) |
| Document Processing | pymupdf4llm (batch), PyMuPDF, PyPDF2, python-docx, BeautifulSoup |
| OCR | Tesseract DPI=200 (tur+eng+equ) with probe-based quality check and early exit |
| Math Normalization | ~50 Unicode symbols → searchable text + equation block protection |
| Text Splitting | langchain RecursiveCharacterTextSplitter (equation-aware separators) |
| Memory | SQLite (WAL mode) + Markdown profile |
| Web Scraping | requests + BeautifulSoup (STARS OAuth + HTML parsing) |
| Email | imaplib IMAP4_SSL (on-demand connection, no persistent keepalive) |
| Async | asyncio + asyncio.to_thread() (non-blocking sync/IMAP/STARS) |

---

## Stats

| Metric | Value |
|--------|-------|
| Indexed chunks | ~3600 |
| Courses | 5 |
| Files | 28 |
| Intents | 12 |
| Background jobs | 6 |
| Embedding dimensions | 384 |
| Supported languages | 50+ |
| Intent accuracy | 97% (30-case benchmark) |
