# Changelog

## [1.3.0] - 2026-02-17

### Added

- **Planner step** (`_plan_agent`): lightweight pre-loop call using gpt-4.1-nano that generates a 2–4 step JSON execution plan before the tool loop, injected into the system prompt to guide tool selection and reduce wasted iterations.
- **Critic step** (`_critic_agent`): post-loop grounding validation using gpt-4.1-nano that checks whether dates, filenames, and factual claims in the final response are supported by tool-provided data. Appends a ⚠️ disclaimer on detected hallucinations (does not regenerate).
- **Adaptive model escalation** (`_score_complexity`): heuristic 0–1 complexity score based on query length, multi-step indicators (Turkish + English), technical keywords, and multi-question patterns. Queries scoring > 0.65 are escalated to `MODEL_COMPLEXITY` (default: gpt-4.1-mini).
- **Prompt injection protection**: `_sanitize_user_input()` blocks system-block spoofing (`---SYSTEM---`, `[SYSTEM]`, `<system>`, `<<SYS>>`), instruction override phrases, and output extraction attempts.
- **Tool output sanitization**: `_sanitize_tool_output()` strips injection patterns and HTML tags from email tool results before they are appended to the LLM message history.
- **Turkish character normalization** (`_normalize_tr`): maps ç→c, ş→s, ğ→g, ü→u, ö→o, ı→i, İ→i for accent-insensitive matching. Uses `translate()` before `lower()` to avoid the Python U+0130 (İ) decomposition bug.
- **Email detail cache-first**: `get_email_detail` now checks the SQLite cache (50 emails) with normalized subject match before falling back to a live IMAP fetch.
- **Cache stale fallback for filtered email queries**: if a sender or subject filter returns an empty result from the cache, the bot automatically retries with a live IMAP fetch to cover the interval between background job runs.
- **`MODEL_COMPLEXITY` env var**: added to `TaskRouter` in `core/llm_providers.py`; configures the escalation target model independently from the main chat model.
- **Comprehensive test suite** (`tests/unit/test_agent_service.py`): 76 tests across 8 classes covering all safety, reliability, and email tool behaviors — including prompt injection, tool output filtering, complexity scoring, planner/critic agents, Turkish normalization, email caching, and stale fallback logic.

### Changed

- Critic prompt tuned to be lenient about reformatting (day name translation, time range collapsing) to eliminate false-positive disclaimers on schedule queries.
- `.env.example` translated to English; `MODEL_COMPLEXITY` and `WEBMAIL_*`/`STARS_*` fields added.
- `data/cache.db`, `data/file_summaries.json`, `data/source_summaries/`, `data/conversation_history.json`, `logs/` added to `.gitignore`.

---

## [1.2.0] - 2026-02-17

### Added

- **SQLite persistent cache** (`core/cache_db.py`): WAL-mode SQLite store that decouples data freshness from response latency. Background jobs are the only writers; tool handlers are read-only.
- **9 background jobs** via `python-telegram-bot` job queue: `email_check` (5 min), `assignment_check` (10 min), `grades_sync` (30 min), `attendance_sync` (60 min), `schedule_sync` (6 h), `deadline_reminder` (30 min), `session_refresh` (60 min), `summary_generation` (60 min), `cache_cleanup` (weekly).
- **Context switching**: agent correctly switches between courses mid-conversation without losing history.
- **Temporal awareness tests**: 88 tests including multi-turn scenarios, temporal context (today/tomorrow/this week), and tool edge cases.

### Changed

- Tool handlers rewritten to read from SQLite cache first; live API calls only on cache miss (first run before background jobs have populated the cache).

---

## [1.1.0] - 2026-02-17

### Changed

- Dead file cleanup: removed 20+ stale scripts, duplicate service files, legacy test files.
- All documentation rewritten to match current modular architecture.
- Python version requirement relaxed from 3.11 to 3.10 (server compatibility).

### Removed

- Root-level `deploy.sh`, `moodle-bot.service` (duplicates of `scripts/` versions).
- 17 development artifact shell scripts (`task1_*.sh`, `_remote_*.sh`, `_tmp_*.sh`).
- `SETUP_GUIDE.md` (consolidated into `SETUP.md`).
- `scripts/telegram-bot.service` (incorrect paths).

---

## [1.0.0] - 2026-02-16

### Changed

- Monolithic `telegram_bot.py` runtime moved behind modular `bot/` package entrypoint.
- Exception handling hardened; fail-silent `except ...: pass` patterns removed in active runtime paths.
- Test infrastructure migrated to pytest-based unit/integration/e2e structure.

### Added

- Docker support (`Dockerfile`, `docker-compose.yml`) with `/health` endpoint.
- Structured logging bootstrap (`bot/logging_config.py`) and runtime performance logs.
- Development tooling (`Makefile`, `requirements-dev.txt`, `pyproject.toml`).
- Contribution guide (`CONTRIBUTING.md`).
- Service layer (`bot/services/`) for RAG, LLM, user management, document indexing.
- Middleware layer (`bot/middleware/`) for auth and error handling.

---

## [0.9.0] - 2026-02-15

### Added

- Hybrid RAG search: FAISS (semantic) + BM25 (keyword) via Reciprocal Rank Fusion.
- Multi-provider LLM routing (OpenAI, Gemini, GLM) with task-based model selection.
- Moodle auto-sync with document processing (PDF, DOCX, PPTX).
- Conversation memory with per-user history and TTL.
- Bilkent STARS client and webmail IMAP integration.
- Rate limiting per user.
