.PHONY: install dev test test-all test-cov lint format run clean deploy logs status restart

REMOTE_HOST ?= user@server-ip

# === Kurulum ===
install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

# === Gelistirme ===
test:
	python -m pytest tests/unit/ -v --tb=short

test-all:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --cov=bot --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .

run:
	python -m bot.main

# === Deploy ===
deploy:
	bash scripts/deploy.sh

# === Sunucu yonetimi (SSH uzerinden) ===
logs:
	ssh $(REMOTE_HOST) "sudo journalctl -u moodle-bot -f --no-pager -n 50"

status:
	ssh $(REMOTE_HOST) "sudo systemctl status moodle-bot"

restart:
	ssh $(REMOTE_HOST) "sudo systemctl restart moodle-bot"

# === Temizlik ===
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache
