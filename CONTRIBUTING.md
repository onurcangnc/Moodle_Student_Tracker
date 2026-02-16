# Contributing

## Gelistirme Ortami

```bash
# Tum bagimliliklari kur (prod + dev)
make dev

# Veya elle
pip install -r requirements-dev.txt
```

## Branch Naming

Use one of these prefixes:

- `feat/<short-description>`
- `fix/<short-description>`
- `refactor/<short-description>`
- `docs/<short-description>`
- `test/<short-description>`

## Gelistirme Akisi

```bash
# 1. Lint kontrolu
make lint

# 2. Unit testleri calistir
make test

# 3. Tum testleri calistir
make test-all

# 4. Coverage raporu
make test-cov

# 5. Format
make format
```

## Pull Request Checklist

Before opening a PR:

1. Rebase or merge latest `main`.
2. Run `make lint` — hata olmamali.
3. Run `make test-all` — tum testler gecmeli.
4. Update docs when behavior/config changes.
5. Add changelog entry when release-impacting.

## PR Template

Use this structure in PR descriptions:

1. Summary
2. Motivation
3. Scope of changes
4. Test evidence
5. Risks / rollback plan

## Code Style

- Python style is enforced with Ruff config in `pyproject.toml`.
- Target: Python 3.10+ (sunucu uyumlulugu).
- Keep functions typed and documented.
- Prefer service-layer delegation in handlers.
- Avoid fail-silent exception blocks.

## Proje Mimarisi

```
bot/handlers/    → Telegram komut/mesaj routing
bot/services/    → Is mantigi (RAG, LLM, kullanici yonetimi)
bot/middleware/   → Auth, error handling
bot/config.py    → Typed AppConfig
bot/state.py     → Paylasimli BotState container
core/            → Domain logic (vector store, LLM, Moodle, sync)
tests/unit/      → Birim testleri
tests/integration/ → Entegrasyon testleri
tests/e2e/       → Uctan uca testler
```

## Testing Rules

- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- E2E tests: `tests/e2e/`
- Yeni ozellik eklediginde ilgili test dosyasini da guncelle.
