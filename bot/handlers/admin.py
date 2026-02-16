"""Admin-only/debug command handlers and registration."""

from __future__ import annotations

from telegram.ext import Application, CommandHandler

from bot import legacy

cmd_stats = legacy.cmd_stats
cmd_cost = legacy.cmd_cost
cmd_models = legacy.cmd_models
cmd_memory = legacy.cmd_memory


def register_admin_handlers(app: Application) -> None:
    """Register hidden admin/debug command handlers."""
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("maliyet", cmd_cost))
    app.add_handler(CommandHandler("modeller", cmd_models))
    app.add_handler(CommandHandler("memory", cmd_memory))
