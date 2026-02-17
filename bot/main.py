"""Application entrypoint and runtime wiring for the modular bot package."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from telegram import Update
from telegram.ext import Application

from bot.config import CONFIG
from bot.handlers.commands import post_init, register_command_handlers
from bot.handlers.messages import register_message_handlers
from bot.logging_config import setup_logging
from bot.middleware.error_handler import global_error_handler
from bot.services.notification_service import register_notification_jobs
from bot.state import STATE
from core import config as core_config
from core.document_processor import DocumentProcessor
from core.llm_engine import LLMEngine
from core.moodle_client import MoodleClient
from core.stars_client import StarsClient
from core.sync_engine import SyncEngine
from core.vector_store import VectorStore
from core.webmail_client import WebmailClient

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_git_version() -> str:
    """Read the short git commit hash for health endpoint metadata."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _build_health_payload(started_at_monotonic: float, version: str) -> dict[str, object]:
    """Build health endpoint payload with runtime and index metrics."""
    uptime_seconds = int(max(0.0, time.monotonic() - started_at_monotonic))
    chunks_loaded = 0
    store = STATE.vector_store
    if store is not None:
        try:
            stats = store.get_stats()
            chunks_loaded = int(stats.get("total_chunks", 0))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            logger.warning("Vector store stats unavailable for health payload", exc_info=True)

    cutoff = time.time() - 86400
    active_users_24h = sum(1 for ts in STATE.user_last_seen.values() if ts >= cutoff)
    return {
        "status": "ok",
        "uptime_seconds": uptime_seconds,
        "version": version,
        "chunks_loaded": chunks_loaded,
        "active_users_24h": active_users_24h,
    }


def _ensure_event_loop() -> None:
    """Ensure a current asyncio event loop exists before PTB polling starts."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def _start_health_server(started_at_monotonic: float, version: str) -> None:
    """Start lightweight HTTP server exposing /health endpoint."""
    if not CONFIG.healthcheck_enabled:
        logger.info("Healthcheck server disabled by configuration.")
        return

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return

            payload = json.dumps(_build_health_payload(started_at_monotonic, version)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            logger.debug("healthcheck_http: " + format, *args)

    def serve() -> None:
        server = HTTPServer((CONFIG.healthcheck_host, CONFIG.healthcheck_port), HealthHandler)
        logger.info("Healthcheck endpoint listening on %s:%s/health", CONFIG.healthcheck_host, CONFIG.healthcheck_port)
        server.serve_forever()

    thread = threading.Thread(target=serve, name="healthcheck-server", daemon=True)
    thread.start()


def _validate_startup_config() -> None:
    """Validate required env configuration before app startup."""
    if not CONFIG.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in environment. Add it to .env before startup.")
    if not CONFIG.owner_id:
        raise RuntimeError("TELEGRAM_OWNER_ID not set in environment. Owner check is required for secure startup.")


def _initialize_components() -> None:
    """Initialize core RAG components and cache Moodle course metadata."""
    errors = core_config.validate()
    if errors:
        joined = "\n".join(f"- {msg}" for msg in errors)
        raise RuntimeError(f"Core configuration errors:\n{joined}")

    moodle = MoodleClient()
    processor = DocumentProcessor()
    vector_store = VectorStore()
    vector_store.initialize()
    llm = LLMEngine(vector_store)
    sync_engine = SyncEngine(moodle, processor, vector_store)

    STATE.moodle = moodle
    STATE.processor = processor
    STATE.vector_store = vector_store
    STATE.llm = llm
    STATE.sync_engine = sync_engine
    STATE.stars_client = StarsClient()
    STATE.webmail_client = WebmailClient()
    logger.info("STARS and Webmail clients initialized (auth required per-user)")

    if moodle.connect():
        courses = moodle.get_courses()
        llm.moodle_courses = [{"shortname": c.shortname, "fullname": c.fullname} for c in courses]
        logger.info("Moodle connection established (courses=%s)", len(courses))
    else:
        logger.warning("Moodle connection failed, running with cached materials only.")


def create_application() -> Application:
    """Build and configure Telegram application with modular handlers."""
    app = Application.builder().token(CONFIG.telegram_bot_token).post_init(post_init).build()
    register_command_handlers(app)
    register_message_handlers(app)
    register_notification_jobs(app)
    app.add_error_handler(global_error_handler)
    return app


def main() -> None:
    """Start Telegram polling with modular runtime wiring."""
    setup_logging(CONFIG.log_level)
    STATE.started_at_monotonic = time.monotonic()
    STATE.startup_version = get_git_version()
    try:
        _validate_startup_config()
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("Initializing bot components...")
    try:
        _initialize_components()
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    _start_health_server(STATE.started_at_monotonic, STATE.startup_version)

    app = create_application()
    logger.info(
        "Bot started",
        extra={
            "owner_id": CONFIG.owner_id,
            "auto_sync_interval": CONFIG.auto_sync_interval,
            "assignment_check_interval": CONFIG.assignment_check_interval,
        },
    )
    _ensure_event_loop()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
