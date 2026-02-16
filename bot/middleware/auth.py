"""Authorization checks used by command and message handlers."""

from __future__ import annotations

from telegram import Update
from telegram.error import TelegramError

from bot.config import CONFIG


def is_admin_user(user_id: int) -> bool:
    """Return whether user is allowed to run admin-only operations."""
    if CONFIG.owner_id == 0:
        return True
    if user_id == CONFIG.owner_id:
        return True
    return user_id in CONFIG.admin_ids


async def admin_only(update: Update) -> bool:
    """Enforce admin access and send user-facing denial message when needed."""
    user = update.effective_user
    if user is None:
        return False

    if is_admin_user(user.id):
        return True

    message = update.effective_message
    if message is None:
        return False

    try:
        await message.reply_text("Bu komut sadece admin kullanıcılar için kullanılabilir.")
    except TelegramError:
        return False
    return False
