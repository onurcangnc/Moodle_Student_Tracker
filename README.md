# Moodle Student Tracker

![Bilkent + Moodle](images/1.png)

A Telegram-based academic assistant that indexes Bilkent University Moodle course materials and delivers **chat-driven, pedagogy-first teaching** powered by hybrid RAG (Retrieval-Augmented Generation) and a multi-tool agentic loop.

Students pick a course, ask a question in natural language, and the bot retrieves relevant lecture materials through **hybrid search** (FAISS + BM25), then generates a grounded, pedagogical answer via an LLM. When material coverage is insufficient, the bot guides students toward available topics instead of hallucinating.

---

## Features

- **Agentic loop** — multi-step reasoning with parallel tool calls (read files, query assignments, check emails, cross-reference sources)
- **Hybrid RAG** — FAISS (semantic) + BM25 (keyword) fused via Reciprocal Rank Fusion
- **Teaching / Guidance mode** — teaches from material when coverage is sufficient; redirects to relevant topics otherwise
- **Fuzzy filename matching** — resolves partial or misspelled file names automatically
- **Source-level pagination** — browse large documents in 30-chunk pages with `offset` parameter
- **Multi-provider LLM** — Gemini, OpenAI, GLM with task-based model routing
- **STARS integration** — grades, attendance, exam schedule
- **Webmail (DAIS & AIRS)** — read, search, and cross-reference instructor emails
- **Assignment tracking** — upcoming deadlines with normalized date formatting
- **Cross-reference queries** — "Do I have homework?" checks both Moodle and email simultaneously
- **Background notifications** — new assignment, grade change, new mail alerts
- **Conversation memory** — per-user context window (15 messages, 60-min TTL)
- **Moodle sync** — auto-fetches and indexes materials from Moodle REST API
- **Admin document upload** — index PDF / DOCX / PPTX directly via Telegram
- **Rate limiting** — per-user request throttling
- **Health check** — HTTP `/health` endpoint (uptime, chunk count, active users)
- **Docker & systemd** — production-ready deployment options

---

## Architecture

The bot is built on a **layered architecture** where each layer communicates only with the layer immediately below it.

```
                        Telegram API
                             │
                        bot/main.py
                    (Application wiring)
                             │
              ┌──────────────┴──────────────┐
              │                             │
       bot/handlers/                 bot/middleware/
       commands.py                   auth.py
       messages.py                   error_handler.py
              │
       bot/services/
       agent_service.py ──────────────────────────────► Agentic loop + tool dispatch
       rag_service.py   ──────► core/vector_store.py   (FAISS + BM25)
       llm_service.py   ──────► core/llm_engine.py     (Multi-provider LLM)
       user_service.py          core/llm_providers.py  (Adapter + Strategy)
       document_service.py      core/moodle_client.py  (Moodle REST API)
       topic_cache.py           core/sync_engine.py    (Material pipeline)
       conversation_memory.py   core/document_processor.py (PDF/DOCX/PPTX)
                                core/stars_client.py   (Bilkent STARS)
                                core/webmail_client.py (IMAP webmail)
```

### Layers

| Layer | Directory | Responsibility |
|-------|-----------|----------------|
| **Handlers** | `bot/handlers/` | Telegram command and message routing |
| **Agent** | `bot/services/agent_service.py` | Agentic loop, tool definitions, system prompt |
| **Services** | `bot/services/` | Business logic — RAG retrieval, LLM calls, user state |
| **Middleware** | `bot/middleware/` | Authorization (admin gate), global error handling |
| **Config** | `bot/config.py` | Typed `AppConfig` dataclass, all `.env` values |
| **State** | `bot/state.py` | Shared `BotState` singleton (runtime container) |
| **Core** | `core/` | Domain logic — vector store, LLM engine, Moodle/STARS/webmail clients |

---

## Design Patterns

| Pattern | Usage | File |
|---------|-------|------|
| **Adapter** | Unified `LLMAdapter` ABC for every LLM provider | `core/llm_providers.py` |
| **Strategy** | `TaskRouter` selects model per task type at runtime | `core/llm_providers.py` |
| **Facade** | `LLMEngine` exposes a single interface hiding RAG + memory + prompt logic | `core/llm_engine.py` |
| **Singleton** | `CONFIG` and `STATE` — one instance across the entire application | `bot/config.py`, `bot/state.py` |
| **Service Layer** | Handler → Service → Core separation of concerns | `bot/services/` |
| **Chain of Responsibility** | Primary model → fallback chain for LLM provider failures | `core/llm_providers.py` |
| **Tool-Use / ReAct** | Agentic loop: LLM emits tool calls, executor runs them, result fed back | `bot/services/agent_service.py` |
| **Observer** | `post_init` hook updates Telegram command menu on startup | `bot/handlers/commands.py` |

---

## How It Works

### Agentic Message Flow

```
User message
    │
    ▼
[Rate limit check] ──✗──► "Too many requests"
    │
    ▼
[Active course check] ──✗──► "Select a course: /courses"
    │
    ▼
[Load conversation history]
    │
    ▼
[Agent loop — max 5 iterations]
    │
    ├─ LLM decides which tools to call
    ├─ Tools execute in parallel (asyncio.gather)
    │   ├─ read_source      → fetch file chunks (+ fuzzy match + pagination)
    │   ├─ study_topic      → cross-source semantic search
    │   ├─ get_assignments  → Moodle deadlines (normalized dates)
    │   ├─ get_emails       → DAIS/AIRS inbox
    │   ├─ get_email_detail → full email body (subject + body search)
    │   ├─ get_grades       → STARS grade records
    │   ├─ get_attendance   → STARS attendance
    │   ├─ get_schedule     → weekly timetable
    │   └─ get_source_map   → course file index
    ├─ Results appended to message history
    └─ LLM generates final response when confident
    │
    ▼
[Save to conversation memory]
    │
    ▼
[Send Markdown to Telegram]
```

### Teaching Mode

When sufficient material is found, the bot teaches from the source using the instructor's own terminology. Source file names are cited as `[file.pdf]`. Information absent from the material is explicitly flagged rather than fabricated.

![Teaching mode example](images/5.png)

### Guidance Mode

When material coverage is insufficient, the bot guides the student toward available topics and suggests more specific example questions — without exposing technical internals.

---

## Screenshots

| Teaching Mode | Material Selection |
|:---:|:---:|
| ![Step-by-step teaching](images/moodle1.png) | ![Source selection](images/moodle3.png) |

| RAG-grounded Answer | Upcoming Exams |
|:---:|:---:|
| ![RAG answer](images/moodle2.png) | ![Exam schedule](images/3.png) |

| Attendance | Grades |
|:---:|:---:|
| ![Attendance](images/4.png) | ![Grades](images/6.jpeg) |

| Email Summaries | Literature Course Teaching |
|:---:|:---:|
| ![Email summary](images/2.jpeg) | ![Literature](images/5.png) |

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/onurcangnc/Moodle_Student_Tracker.git
cd Moodle_Student_Tracker
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
# or
make install
```

### 4. Configure environment

```bash
cp .env.example .env
```

Minimum required fields:

| Variable | Description |
|----------|-------------|
| `MOODLE_URL` | Bilkent Moodle URL (semester-specific) |
| `MOODLE_USERNAME` | Moodle username |
| `MOODLE_PASSWORD` | Moodle password |
| `TELEGRAM_BOT_TOKEN` | Token from @BotFather |
| `TELEGRAM_OWNER_ID` | Your Telegram chat ID |
| `OPENAI_API_KEY` or `GEMINI_API_KEY` | At least one LLM API key |

See [SETUP.md](SETUP.md) for full configuration reference.

### 5. Run

```bash
python -m bot.main
# or
make run
```

Expected startup output:

```
INFO | Initializing bot components...
INFO | Vector store loaded. 3661 chunks.
INFO | BM25 index built: 3661 chunks in 1.22s
INFO | Moodle connection established (courses=5)
INFO | Healthcheck endpoint listening on 0.0.0.0:9090/health
INFO | Bot started
```

---

## Bot Commands

| Command | Description | Access |
|---------|-------------|--------|
| `/start` | Welcome message and usage guide | Everyone |
| `/help` | Step-by-step usage instructions | Everyone |
| `/courses` | List loaded courses | Everyone |
| `/courses <name>` | Set active course | Everyone |
| `/upload` | Open document upload mode (next file will be indexed) | Admin |
| `/stats` | Bot statistics (chunks, courses, files) | Admin |

**Typical workflow:** `/courses` → select course → type your question → get a material-grounded answer.

---

## Configuration

All configuration is read from `.env`. Full template: [.env.example](.env.example)

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `MOODLE_URL` | — | Bilkent Moodle URL (changes each semester) |
| `MOODLE_USERNAME` | — | Moodle username |
| `MOODLE_PASSWORD` | — | Moodle password |
| `TELEGRAM_BOT_TOKEN` | — | Token from @BotFather |
| `TELEGRAM_OWNER_ID` | — | Bot owner's Telegram chat ID |
| `TELEGRAM_ADMIN_IDS` | — | Additional admin IDs (comma-separated) |

### LLM Model Routing

| Variable | Default | Task |
|----------|---------|------|
| `MODEL_CHAT` | `gemini-2.5-flash` | Main chat (RAG + agent) |
| `MODEL_STUDY` | `gemini-2.5-flash` | Deep teaching mode |
| `MODEL_EXTRACTION` | `gpt-4.1-nano` | Memory extraction |
| `MODEL_SUMMARY` | `gemini-2.5-flash` | Weekly digest |

Supported models: Gemini 2.5 Flash/Pro, GPT-4.1 nano/mini, GPT-5 mini, GLM 4.5/4.7, Claude Haiku/Sonnet/Opus.

### RAG Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_SIMILARITY_THRESHOLD` | `0.65` | Minimum similarity score for teaching mode |
| `RAG_MIN_CHUNKS` | `2` | Minimum chunks required for teaching mode |
| `RAG_TOP_K` | `5` | Chunks returned per search |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | 384-dim, 50+ languages |
| `CHUNK_SIZE` | `1000` | Chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Chunk overlap in characters |

### Operational

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_MAX` | `30` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window (seconds) |
| `MEMORY_MAX_MESSAGES` | `15` | Conversation history size |
| `MEMORY_TTL_MINUTES` | `60` | Memory TTL (minutes) |
| `HEALTHCHECK_PORT` | `9090` | Health endpoint port |
| `LOG_LEVEL` | `INFO` | Log verbosity |

---

## Testing

```bash
# Install dev dependencies first
make dev
# or: pip install -r requirements-dev.txt

# Unit tests only (fast, no external deps)
make test
# or: python -m pytest tests/unit/ -v --tb=short

# All tests (unit + integration)
make test-all
# or: python -m pytest tests/ -v --tb=short

# With coverage report
make test-cov
# or: python -m pytest tests/ -v --cov=bot --cov-report=term-missing

# Run a single test file
python -m pytest tests/unit/test_conversation_memory.py -v

# Run a single test by name
python -m pytest tests/unit/test_rag_service.py::test_hybrid_search_returns_results -v

# Run only slow-marked tests
python -m pytest -m slow -v

# Skip slow tests
python -m pytest -m "not slow" -v

# Integration tests only
python -m pytest tests/integration/ -v --tb=short

# Generate HTML coverage report
python -m pytest tests/ --cov=bot --cov-report=html
# open htmlcov/index.html
```

---

## Deployment

### Systemd (Recommended)

```bash
sudo cp scripts/moodle-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable moodle-bot
sudo systemctl start moodle-bot

# Follow logs
journalctl -u moodle-bot -f
```

### Docker

```bash
cp .env.example .env
# fill in .env

docker compose up -d
docker compose logs -f
```

### Health Check

```bash
curl http://localhost:9090/health
```

```json
{
  "status": "ok",
  "uptime_seconds": 3600,
  "version": "abc1234",
  "chunks_loaded": 3661,
  "active_users_24h": 5
}
```

---

## Development

```bash
make dev        # install dev dependencies
make lint       # ruff check
make format     # ruff auto-format
make test       # unit tests
make test-cov   # coverage report
make clean      # remove __pycache__, .pyc, .pytest_cache
make deploy     # lint → test → push → SSH deploy
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Runtime | Python ≥ 3.10 |
| Telegram | python-telegram-bot 21.x (asyncio) |
| LLM | OpenAI, Google Gemini, GLM, Anthropic (task-based routing) |
| Embedding | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dim) |
| Semantic Search | FAISS-CPU |
| Keyword Search | rank-bm25 (Snowball TR/EN stemming via PyStemmer) |
| Search Fusion | Reciprocal Rank Fusion (k=60) |
| PDF Extraction | PyMuPDF + pymupdf4llm + Tesseract OCR (text + scanned pages) |
| DOCX / PPTX | python-docx, python-pptx |
| Chunking | langchain-text-splitters (`RecursiveCharacterTextSplitter`) |
| Config | python-dotenv, typed dataclass |
| Lint / Format | ruff |
| Test | pytest, pytest-asyncio, pytest-cov |
| Deploy | Docker, systemd, Makefile |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
