"""Application entrypoint and runtime wiring for the modular bot package."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import Update
from telegram.ext import Application

from bot import legacy
from bot.config import CONFIG
from bot.handlers.admin import register_admin_handlers
from bot.handlers.callbacks import register_callback_handlers
from bot.handlers.commands import post_init, register_command_handlers
from bot.handlers.messages import register_message_handlers
from bot.logging_config import setup_logging
from bot.middleware.error_handler import global_error_handler
from bot.state import sync_from_legacy

logger = logging.getLogger(__name__)


def _ensure_event_loop() -> None:
    """Ensure a current asyncio event loop exists before PTB polling starts."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def _start_health_server() -> None:
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

            payload = json.dumps({"status": "ok"}).encode("utf-8")
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


def create_application() -> Application:
    """Build and configure Telegram application with modular handlers."""
    app = Application.builder().token(CONFIG.telegram_bot_token).post_init(post_init).build()
    register_command_handlers(app)
    register_admin_handlers(app)
    register_callback_handlers(app)
    register_message_handlers(app)
    app.add_error_handler(global_error_handler)
    return app


def main() -> None:
    """Start Telegram polling with modular runtime wiring."""
    setup_logging(CONFIG.log_level)
    try:
        _validate_startup_config()
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("Initializing bot components...")
    legacy.init_components()
    sync_from_legacy(legacy)
    _start_health_server()

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
