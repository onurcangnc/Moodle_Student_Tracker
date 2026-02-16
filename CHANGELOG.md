# Changelog

## [1.0.0] - 2026-02-16

### Changed

- Monolithic `telegram_bot.py` runtime moved behind modular `bot/` package entrypoint.
- Exception handling hardened and fail-silent `except ...: pass` patterns removed in active runtime paths.
- Test infrastructure migrated to pytest-based unit/integration/e2e structure.

### Added

- CI pipeline with GitHub Actions (`.github/workflows/ci.yml`).
- Docker support (`Dockerfile`, `docker-compose.yml`) with `/health` endpoint.
- Structured logging bootstrap (`bot/logging_config.py`) and runtime performance logs.
- Development tooling (`Makefile`, `requirements-dev.txt`, `pyproject.toml`).
- Contribution guide (`CONTRIBUTING.md`).
