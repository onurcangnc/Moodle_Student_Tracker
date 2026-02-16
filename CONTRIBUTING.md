# Contributing

## Branch Naming

Use one of these prefixes:

- `feat/<short-description>`
- `fix/<short-description>`
- `refactor/<short-description>`
- `docs/<short-description>`
- `test/<short-description>`

## Pull Request Checklist

Before opening a PR:

1. Rebase or merge latest `master`.
2. Run `make lint`.
3. Run `make test-all`.
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
- Keep functions typed and documented.
- Prefer service-layer delegation in handlers.
- Avoid fail-silent exception blocks.

## Testing Rules

- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- E2E tests: `tests/e2e/`
- Legacy evaluation scripts remain under `tests/test_*.py` and are not part of default pytest discovery.
