"""Command handlers and registration for user-facing bot commands."""

from __future__ import annotations

from telegram.ext import Application, CommandHandler

from bot import legacy

cmd_start = legacy.cmd_start
cmd_help = legacy.cmd_help
cmd_menu = legacy.cmd_menu
cmd_calis = legacy.cmd_calis
cmd_notlar = legacy.cmd_notlar
cmd_bugun = legacy.cmd_bugun
cmd_haftam = legacy.cmd_haftam
cmd_assignments = legacy.cmd_assignments
cmd_login = legacy.cmd_login
cmd_stars = legacy.cmd_stars
cmd_mail = legacy.cmd_mail
cmd_sync = legacy.cmd_sync
cmd_clear = legacy.cmd_clear


async def post_init(app):
    """Run post-init hooks for command metadata and background jobs."""
    await legacy.post_init(app)


def register_command_handlers(app: Application) -> None:
    """Register user-visible command handlers."""
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("menu", cmd_menu))
    app.add_handler(CommandHandler("calis", cmd_calis))
    app.add_handler(CommandHandler("notlar", cmd_notlar))
    app.add_handler(CommandHandler("bugun", cmd_bugun))
    app.add_handler(CommandHandler("haftam", cmd_haftam))
    app.add_handler(CommandHandler("odevler", cmd_assignments))
    app.add_handler(CommandHandler("login", cmd_login))
    app.add_handler(CommandHandler("stars", cmd_stars))
    app.add_handler(CommandHandler("mail", cmd_mail))
    app.add_handler(CommandHandler("sync", cmd_sync))
    app.add_handler(CommandHandler("temizle", cmd_clear))
