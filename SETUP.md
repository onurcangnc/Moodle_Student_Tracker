# Setup Guide

Step-by-step installation from zero. No prior experience required.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Python Installation](#2-python-installation)
3. [Clone the Repository](#3-clone-the-repository)
4. [Install Dependencies](#4-install-dependencies)
5. [Tesseract OCR (Optional)](#5-tesseract-ocr-optional)
6. [Create a Telegram Bot](#6-create-a-telegram-bot)
7. [API Keys](#7-api-keys)
8. [Bilkent Moodle Credentials](#8-bilkent-moodle-credentials)
9. [Configure .env](#9-configure-env)
10. [Run the Bot](#10-run-the-bot)
11. [Testing](#11-testing)
12. [Server Deployment (VPS)](#12-server-deployment-vps)
13. [Updating & Maintenance](#13-updating--maintenance)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.12 |
| RAM | 2 GB | 4 GB |
| Disk | 2 GB | 5 GB |
| OS | Ubuntu 22.04 / Windows 10 | Ubuntu 24.04 |

**Notes:**
- The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) downloads ~500 MB on first run.
- Tesseract OCR requires an additional ~100 MB (only needed for scanned PDFs).
- Server deployment requires at least 2 GB RAM — the embedding model stays resident in memory.

---

## 2. Python Installation

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
python3 --version   # must be 3.10+
```

### Windows

1. Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. During installation, check **"Add Python to PATH"**
3. Verify:

```powershell
python --version   # 3.10+
```

### macOS

```bash
brew install python@3.12
python3 --version   # 3.10+
```

---

## 3. Clone the Repository

```bash
git clone https://github.com/onurcangnc/Moodle_Student_Tracker.git
cd Moodle_Student_Tracker
```

---

## 4. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows (PowerShell)
# venv\Scripts\activate.bat     # Windows (CMD)

# Install production dependencies
pip install -r requirements.txt
```

Dev dependencies (tests, linter):

```bash
pip install -r requirements-dev.txt
# or
make dev
```

### PyStemmer (Performance)

The BM25 index uses Snowball stemming. The `snowballstemmer` package falls back to pure Python when the C extension is unavailable (~30s startup). Install PyStemmer for the C extension (~1.5s startup):

```bash
pip install PyStemmer
```

> PyStemmer is not in `requirements.txt` because it requires a C compiler. If it fails to install, the bot continues working — startup is just slower.

---

## 5. Tesseract OCR (Optional)

Required only for scanned (image-based) PDFs. Text-based PDFs work without it. The bot inspects each PDF page individually and sends only scanned pages to OCR.

### Ubuntu / Debian

```bash
sudo apt install -y tesseract-ocr tesseract-ocr-tur
tesseract --version
```

### Windows

1. Download the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Select Turkish language pack during installation
3. Add Tesseract to PATH, or specify `TESSERACT_CMD` in `.env`

### macOS

```bash
brew install tesseract tesseract-lang
```

---

## 6. Create a Telegram Bot

### 6.1. Get a Bot Token

1. Open a conversation with [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot`
3. Enter a display name (e.g., `Moodle Assistant`)
4. Enter a username (e.g., `moodle_asistan_bot`)
5. Save the token BotFather gives you:

   ```
   5123456789:ABCdefGHIjklMNOpqrs-TUVwxyz12345
   ```

### 6.2. Find Your Chat ID

1. Open a conversation with [@userinfobot](https://t.me/userinfobot)
2. Send `/start`
3. Save the **Id** value (e.g., `123456789`)

This is your `TELEGRAM_OWNER_ID`. The owner can use admin commands (`/upload`, `/stats`).

### 6.3. Add Extra Admins (Optional)

If multiple people need admin access, comma-separate their chat IDs in `.env`:

```
TELEGRAM_ADMIN_IDS=111222333,444555666
```

---

## 7. API Keys

The bot requires at least one LLM API key. Multiple providers can be configured; the bot selects the best model per task via `TaskRouter`.

### 7.1. Google Gemini (Recommended)

Gemini 2.5 Flash is used for the main chat and teaching tasks. The free tier is limited (5 RPM / 20 RPD).

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Add to `.env`:

   ```
   GEMINI_API_KEY=AIzaSy...
   ```

### 7.2. OpenAI

GPT-4.1 nano is used for memory extraction and topic detection.

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click "Create new secret key"
3. Add to `.env`:

   ```
   OPENAI_API_KEY=sk-proj-...
   ```

### 7.3. Model Routing (Optional)

Override the model for each task in `.env`:

```ini
MODEL_CHAT=gemini-2.5-flash        # Main chat (RAG + agentic)
MODEL_STUDY=gemini-2.5-flash       # Deep teaching mode
MODEL_EXTRACTION=gpt-4.1-nano      # Memory extraction
MODEL_TOPIC_DETECT=gpt-4.1-nano    # Topic detection
MODEL_SUMMARY=gemini-2.5-flash     # Weekly digest
MODEL_QUESTIONS=gemini-2.5-flash   # Practice questions
MODEL_OVERVIEW=gemini-2.5-flash    # Course overview
```

Supported models: Gemini 2.5 Flash/Pro, GPT-4.1 nano/mini, GPT-5 mini, GLM 4.5/4.7, Claude Haiku/Sonnet/Opus.

---

## 8. Bilkent Moodle Credentials

Bilkent uses a different Moodle URL every semester. Check the current URL on the Bilkent website.

```ini
# Example (2025-2026 Spring)
MOODLE_URL=https://moodle.bilkent.edu.tr/2025-2026-spring
MOODLE_USERNAME=22003467
MOODLE_PASSWORD=your_password
```

**Important notes:**
- On first connection, the bot fetches a Moodle token and saves it to `data/moodle_token.txt`. Subsequent restarts reuse the cached token.
- When the semester changes, update `MOODLE_URL`.
- If you change your Moodle password, delete `data/moodle_token.txt` and restart the bot.

---

## 9. Configure .env

```bash
cp .env.example .env
```

Open `.env` and fill in:

```ini
# ── REQUIRED ──────────────────────────────────────
MOODLE_URL=https://moodle.bilkent.edu.tr/2025-2026-spring
MOODLE_USERNAME=...
MOODLE_PASSWORD=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_OWNER_ID=...

# ── AT LEAST ONE LLM KEY ──────────────────────────
OPENAI_API_KEY=...
GEMINI_API_KEY=...

# ── OPTIONAL (defaults are usually fine) ──────────
# MODEL_CHAT=gemini-2.5-flash
# RAG_SIMILARITY_THRESHOLD=0.65
# HEALTHCHECK_PORT=9090
# LOG_LEVEL=INFO
```

### Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_MAX` | `30` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window (seconds) |
| `MEMORY_MAX_MESSAGES` | `15` | Conversation history size per user |
| `MEMORY_TTL_MINUTES` | `60` | Memory TTL (minutes) |
| `HEALTHCHECK_PORT` | `9090` | Health endpoint port |
| `HEALTHCHECK_ENABLED` | `true` | Enable/disable health endpoint |
| `CHUNK_SIZE` | `1000` | Text chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Chunk overlap (characters) |
| `RAG_TOP_K` | `5` | Max chunks returned per search |

---

## 10. Run the Bot

```bash
# Activate virtual environment
source venv/bin/activate

# Start
python -m bot.main
```

Or with Makefile:

```bash
make run
```

Expected startup output:

```
INFO | Initializing bot components...
INFO | Vector store loaded. 3661 chunks.
INFO | BM25 index built: 3661 chunks in 1.62s
INFO | Moodle connection established (courses=5)
INFO | Healthcheck endpoint listening on 0.0.0.0:9090/health
INFO | Bot started
```

### First Use

1. Open a conversation with your bot on Telegram
2. Send `/start` to see the welcome message
3. Send `/courses` to list loaded courses
4. Select a course: `/courses CTIS 363`
5. Ask a question: `What is ethics?`
6. The bot searches course materials and returns a grounded answer

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
  "active_users_24h": 1
}
```

---

## 11. Testing

```bash
# Install dev dependencies
make dev
# or: pip install -r requirements-dev.txt

# Unit tests only — fast, no external dependencies
make test
# or: python -m pytest tests/unit/ -v --tb=short

# All tests (unit + integration)
make test-all
# or: python -m pytest tests/ -v --tb=short

# With inline coverage report
make test-cov
# or: python -m pytest tests/ -v --cov=bot --cov-report=term-missing

# Run a single test file
python -m pytest tests/unit/test_conversation_memory.py -v

# Run a specific test by name
python -m pytest tests/unit/test_rag_service.py::test_hybrid_search_returns_results -v

# Run tests matching a keyword
python -m pytest -k "memory" -v

# Skip slow tests
python -m pytest -m "not slow" -v

# Run only slow-marked tests
python -m pytest -m slow -v

# Integration tests only
python -m pytest tests/integration/ -v --tb=short

# Generate HTML coverage report
python -m pytest tests/ --cov=bot --cov-report=html
# then open: htmlcov/index.html

# Run with more verbose output
python -m pytest tests/unit/ -v --tb=long

# Stop on first failure
python -m pytest tests/unit/ -x -v

# Run last failed tests
python -m pytest --lf -v
```

---

## 12. Server Deployment (VPS)

### Method A: Systemd (Recommended)

```bash
# 1. Copy project to server
scp -r . root@SERVER_IP:/opt/moodle-bot/

# 2. Connect to server
ssh root@SERVER_IP

# 3. Create a dedicated user (security best practice)
useradd -r -s /bin/false botuser
chown -R botuser:botuser /opt/moodle-bot

# 4. Install dependencies
cd /opt/moodle-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install PyStemmer   # recommended for BM25 performance

# 5. Create .env
cp .env.example .env
nano .env   # fill in required fields

# 6. Install systemd service
cp scripts/moodle-bot.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable moodle-bot
systemctl start moodle-bot

# 7. Verify
systemctl status moodle-bot
journalctl -u moodle-bot -f
curl http://localhost:9090/health
```

#### Systemd Security Hardening

`scripts/moodle-bot.service` includes these protections:

| Directive | Effect |
|-----------|--------|
| `NoNewPrivileges=true` | Prevents privilege escalation |
| `ProtectSystem=strict` | Makes system directories read-only |
| `ProtectHome=true` | Blocks access to `/home` |
| `ReadWritePaths=...` | Only `data/` and `logs/` are writable |
| `PrivateTmp=true` | Uses an isolated temporary directory |

> The service file uses `User=botuser`. To run as root, comment out or change those lines.

### Method B: Docker

```bash
# 1. Copy project to server and connect
scp -r . root@SERVER_IP:/opt/moodle-bot/
ssh root@SERVER_IP
cd /opt/moodle-bot

# 2. Create .env
cp .env.example .env
nano .env

# 3. Start with Docker
docker compose up -d

# 4. Verify
docker compose logs -f
docker compose ps
```

The Docker volume `bot_data` persists FAISS index, downloaded files, and conversation state across container restarts.

---

## 13. Updating & Maintenance

### Makefile Deploy

```bash
# Full pipeline: lint → test → push → SSH deploy
make deploy
```

This runs:
1. `ruff check .` — lint
2. `pytest tests/unit/` — unit tests
3. `git push origin main` — push to remote
4. SSH into server and runs `scripts/deploy-remote.sh`

### Manual Update

```bash
# Push local changes
git push origin main

# Pull on server
ssh root@SERVER_IP "cd /opt/moodle-bot && git pull && systemctl restart moodle-bot"
```

### Makefile Reference

| Command | Description |
|---------|-------------|
| `make install` | Install production dependencies |
| `make dev` | Install dev dependencies |
| `make run` | Start the bot |
| `make test` | Run unit tests |
| `make test-all` | Run all tests |
| `make test-cov` | Coverage report |
| `make lint` | ruff lint check |
| `make format` | ruff auto-format |
| `make deploy` | Deploy to server |
| `make logs` | Tail server logs |
| `make status` | Server service status |
| `make restart` | Restart server service |
| `make health` | Server health check |
| `make clean` | Remove `__pycache__`, `.pyc`, cache files |

### Rebuild the Index from Scratch

If the RAG index is corrupt or outdated:

```bash
# On the server
cd /opt/moodle-bot/data
rm -f faiss.index metadata.json sync_state.json
systemctl restart moodle-bot
# Bot re-downloads and re-indexes all Moodle materials on startup
```

### Offline Mode for Embedding Model

The embedding model is downloaded from Hugging Face on first run (~500 MB). After that, switch to offline mode:

```ini
# .env
HF_HUB_OFFLINE=1
```

---

## 14. Troubleshooting

### Bot won't start

```bash
journalctl -u moodle-bot --no-pager -n 50
# or with Docker:
docker compose logs --tail 50
```

| Error | Fix |
|-------|-----|
| `TELEGRAM_BOT_TOKEN is empty` | Check token in `.env` |
| `Moodle connection failed` | Is `MOODLE_URL` correct? It changes every semester |
| `Port 9090 already in use` | Change `HEALTHCHECK_PORT` or set `HEALTHCHECK_ENABLED=false` |
| `No module named 'bot'` | Run from the project root: `python -m bot.main` |
| `ModuleNotFoundError` | Activate virtual environment: `source venv/bin/activate` |

### LLM API errors

```bash
# Test OpenAI connection
python -c "from openai import OpenAI; c = OpenAI(); print(c.models.list().data[0].id)"

# Test Gemini connection
python -c "
from openai import OpenAI
c = OpenAI(api_key='YOUR_GEMINI_KEY', base_url='https://generativelanguage.googleapis.com/v1beta/openai/')
print(c.models.list().data[0].id)
"
```

| Error | Fix |
|-------|-----|
| `AuthenticationError` | API key is wrong or expired |
| `RateLimitError` | Gemini free tier: 5 RPM / 20 RPD — upgrade or switch to OpenAI |
| `proxies TypeError` | Upgrade openai: `pip install 'openai>=1.58.0'` |

### Moodle connection fails

- The URL is semester-specific. Check the current URL on the Bilkent website.
- Token may be expired:

```bash
rm data/moodle_token.txt
# Restart bot — a new token is fetched automatically
```

### Embedding model won't download

```bash
# Download manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Then switch to offline mode in .env
# HF_HUB_OFFLINE=1
```

### Telegram Conflict error

```
telegram.error.Conflict: terminated by other getUpdates request
```

Another bot instance is running or a previous polling session hasn't closed:

```bash
# Systemd
systemctl stop moodle-bot
sleep 15
systemctl start moodle-bot

# Docker
docker compose down
sleep 15
docker compose up -d
```

### BM25 index is slow to build

If BM25 indexing takes 20+ seconds:

```bash
pip install PyStemmer
# Restart bot — startup drops to ~1.5s
```

### Health check not responding

```bash
# Check if port is listening
ss -tlnp | grep 9090

# Manual test
curl -v http://localhost:9090/health

# Disable if not needed
# .env: HEALTHCHECK_ENABLED=false
```
