# Changelog

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

## [1.0.0] - 2026-02-16

### Changed

- Monolithic `telegram_bot.py` runtime moved behind modular `bot/` package entrypoint.
- Exception handling hardened and fail-silent `except ...: pass` patterns removed in active runtime paths.
- Test infrastructure migrated to pytest-based unit/integration/e2e structure.

### Added

- Docker support (`Dockerfile`, `docker-compose.yml`) with `/health` endpoint.
- Structured logging bootstrap (`bot/logging_config.py`) and runtime performance logs.
- Development tooling (`Makefile`, `requirements-dev.txt`, `pyproject.toml`).
- Contribution guide (`CONTRIBUTING.md`).
- Service layer (`bot/services/`) for RAG, LLM, user management, document indexing.
- Middleware layer (`bot/middleware/`) for auth and error handling.

## [0.9.0] - 2026-02-15

### Added

- Hybrid RAG search: FAISS (semantic) + BM25 (keyword) via Reciprocal Rank Fusion.
- Multi-provider LLM routing (OpenAI, Gemini, GLM) with task-based model selection.
- Moodle auto-sync with document processing (PDF, DOCX, PPTX).
- Conversation memory with per-user history and TTL.
- Bilkent STARS client and webmail IMAP integration.
- Rate limiting per user.
