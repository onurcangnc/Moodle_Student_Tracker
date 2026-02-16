"""Backward-compatible entrypoint for the legacy Telegram bot runtime.

This module re-exports legacy symbols so existing imports continue to work,
while delegating process startup to the modular `bot` package.
"""

from bot import legacy as _legacy
from bot.main import main

for _name in dir(_legacy):
    if _name.startswith("__"):
        continue
    globals().setdefault(_name, getattr(_legacy, _name))


__all__ = [name for name in globals() if not name.startswith("__")]


if __name__ == "__main__":
    main()
