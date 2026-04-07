"""
LLM Router — LiteLLM wrapper with latency-based routing.
=========================================================
Encapsulates LiteLLM Router for multi-provider support.

Features:
- Lazy initialization
- Latency-based model selection
- Output sanitization (DeepSeek control tokens)
- Streaming support
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any

from litellm import Router
from telegram import Message
from telegram.error import TelegramError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["LLMRouter"]

# DeepSeek V3 sometimes leaks internal control tokens
_DEEPSEEK_LEAK_RE = re.compile(r"<｜[A-Z]+｜[^>]*>")
_CONTROL_TAG_RE = re.compile(r"<\|[a-z_]+\|>")

# Stream edit interval for Telegram
_STREAM_EDIT_INTERVAL = 1.0


class LLMRouter:
    """
    Wrapper around LiteLLM Router with lazy initialization.

    Provides:
    - Multi-provider support (OpenAI, Gemini, DeepSeek)
    - Latency-based routing
    - Output sanitization
    - Streaming to Telegram
    """

    def __init__(self) -> None:
        self._router: Router | None = None

    def _ensure_router(self) -> Router:
        """Lazy-init LiteLLM router with latency-based routing."""
        if self._router is not None:
            return self._router

        model_list = []

        # OpenAI GPT-4.1-mini (fast, reliable, best tool support)
        if os.getenv("OPENAI_API_KEY"):
            model_list.append({
                "model_name": "fast",
                "litellm_params": {
                    "model": "gpt-4.1-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            })

        # Google Gemini 2.5 Flash (free tier: 15 RPM, good tool support)
        if os.getenv("GEMINI_API_KEY"):
            model_list.append({
                "model_name": "fast",
                "litellm_params": {
                    "model": "gemini/gemini-2.5-flash",
                    "api_key": os.getenv("GEMINI_API_KEY"),
                },
            })

        # DeepSeek V3 (very cheap: $0.14/M input, good tool support)
        if os.getenv("DEEPSEEK_API_KEY"):
            model_list.append({
                "model_name": "fast",
                "litellm_params": {
                    "model": "deepseek/deepseek-chat",
                    "api_key": os.getenv("DEEPSEEK_API_KEY"),
                },
            })

        if not model_list:
            raise RuntimeError("No LLM API keys configured!")

        self._router = Router(
            model_list=model_list,
            routing_strategy="latency-based-routing",
            num_retries=2,
            timeout=30,
            enable_pre_call_checks=True,
        )
        logger.info("LiteLLM router initialized with %d models", len(model_list))
        return self._router

    def sanitize_output(self, text: str) -> str:
        """Strip internal model control tokens from LLM output."""
        if not text:
            return text
        text = _DEEPSEEK_LEAK_RE.sub("", text)
        text = _CONTROL_TAG_RE.sub("", text)
        return text.strip()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
    ) -> Any:
        """
        Call LLM with function calling via LiteLLM Router.

        Args:
            messages: Conversation messages
            system_prompt: System prompt
            tools: OpenAI function calling tools (optional)
            max_tokens: Max tokens for response

        Returns:
            LLM response message or None on failure
        """
        router = self._ensure_router()

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        kwargs: dict[str, Any] = {
            "model": "fast",
            "messages": full_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = True

        try:
            response = await router.acompletion(**kwargs)
            logger.debug("LiteLLM response from: %s", response.model)
            return response.choices[0].message
        except Exception as exc:
            logger.error("LiteLLM call failed: %s", exc)
            return None

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        message: Message,
    ) -> str:
        """
        Stream LLM response directly to Telegram via progressive message edits.

        Args:
            messages: Conversation messages
            system_prompt: System prompt
            message: Telegram message to reply to

        Returns:
            Accumulated response text
        """
        router = self._ensure_router()

        full_messages = [{"role": "system", "content": system_prompt}] + messages
        kwargs: dict[str, Any] = {
            "model": "fast",
            "messages": full_messages,
            "max_tokens": 4096,
            "stream": True,
        }

        try:
            stream = await router.acompletion(**kwargs)
        except Exception as exc:
            logger.error("LiteLLM streaming failed: %s", exc)
            return ""

        accumulated = ""
        sent_msg = None
        last_edit = 0.0

        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    accumulated += delta.content

                now = time.monotonic()
                if accumulated and (now - last_edit) >= _STREAM_EDIT_INTERVAL:
                    try:
                        if sent_msg is None:
                            sent_msg = await message.reply_text(accumulated, parse_mode=None)
                        else:
                            await sent_msg.edit_text(accumulated, parse_mode=None)
                        last_edit = now
                    except TelegramError:
                        pass

            # Final edit with Markdown formatting
            if accumulated and sent_msg is not None:
                try:
                    await sent_msg.edit_text(accumulated, parse_mode="Markdown")
                except TelegramError:
                    try:
                        await sent_msg.edit_text(accumulated, parse_mode=None)
                    except TelegramError:
                        pass
            elif accumulated and sent_msg is None:
                await message.reply_text(accumulated, parse_mode="Markdown")
        except Exception as exc:
            logger.warning("Streaming failed, falling back: %s", exc)
            if not accumulated:
                return ""

        return accumulated

    async def warmup(self) -> None:
        """
        Pre-warm LLM connections at startup.

        Makes a lightweight call to establish TCP/TLS connections
        and populate the router's latency measurements.
        """
        try:
            router = self._ensure_router()
            start = time.time()
            await router.acompletion(
                model="fast",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            elapsed = (time.time() - start) * 1000
            logger.info("LLM connections warmed up in %.0fms", elapsed)
        except Exception as exc:
            logger.warning("LLM warmup failed (non-critical): %s", exc)
