"""
Multi-Provider LLM Adapter
============================
Provider-agnostic LLM interface supporting Gemini, OpenAI, GLM, and Claude.

Cost optimization strategy:
┌────────────────────────────────────────────────────────────────┐
│  TASK             │ RECOMMENDED        │ WHY                   │
│──────────────────────────────────────────────────────────────│
│  Main Chat (RAG)  │ Gemini 2.5 Flash  │ Fast, cheap, multilingual │
│  Memory Extract   │ GPT-4.1 nano      │ $0.10/$0.40 - dirt cheap │
│  Intent Classify  │ GPT-4.1 mini      │ Turkish NLU, context     │
│  Topic Detection  │ GPT-4.1 nano      │ Simple classification    │
│  Weekly Summary   │ Gemini 2.5 Flash  │ Good quality, low cost   │
│  Practice Qs      │ Gemini 2.5 Flash  │ Strong reasoning         │
└────────────────────────────────────────────────────────────────┘

All providers use OpenAI-compatible chat completion format.
Gemini/GLM natively support this. Claude uses the Anthropic SDK.
"""

import os
import json
import logging
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from core import config

logger = logging.getLogger(__name__)


# ─── Provider Definitions ────────────────────────────────────────────────────

class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GLM = "glm"
    GEMINI = "gemini"
    OPENAI_COMPAT = "openai_compat"  # Any OpenAI-compatible endpoint


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: Provider
    model_id: str
    api_key: str = ""
    base_url: str = ""  # For OpenAI-compatible endpoints
    input_cost_per_mtok: float = 0.0   # $/1M input tokens
    output_cost_per_mtok: float = 0.0  # $/1M output tokens
    max_tokens: int = 4096
    description: str = ""


# ─── Preset Model Configs ───────────────────────────────────────────────────

def _get_presets() -> dict[str, ModelConfig]:
    """Build model presets from environment variables."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    glm_key = os.getenv("GLM_API_KEY", "")
    glm_base = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    return {
        # ─── Anthropic ──────────────────────────────────
        "claude-sonnet": ModelConfig(
            provider=Provider.ANTHROPIC,
            model_id="claude-sonnet-4-20250514",
            api_key=anthropic_key,
            input_cost_per_mtok=3.0,
            output_cost_per_mtok=15.0,
            description="Best quality for complex reasoning & Turkish text",
        ),
        "claude-haiku": ModelConfig(
            provider=Provider.ANTHROPIC,
            model_id="claude-haiku-4-5-20251001",
            api_key=anthropic_key,
            input_cost_per_mtok=1.0,
            output_cost_per_mtok=5.0,
            description="Fast, good for utility tasks",
        ),
        "claude-opus": ModelConfig(
            provider=Provider.ANTHROPIC,
            model_id="claude-opus-4-6",
            api_key=anthropic_key,
            input_cost_per_mtok=5.0,
            output_cost_per_mtok=25.0,
            description="Most capable, use sparingly",
        ),

        # ─── OpenAI ─────────────────────────────────────
        "gpt-5-mini": ModelConfig(
            provider=Provider.OPENAI,
            model_id="gpt-5-mini",
            api_key=openai_key,
            input_cost_per_mtok=0.25,
            output_cost_per_mtok=2.0,
            description="Great balance of cost/performance",
        ),
        "gpt-4.1-nano": ModelConfig(
            provider=Provider.OPENAI,
            model_id="gpt-4.1-nano",
            api_key=openai_key,
            input_cost_per_mtok=0.20,
            output_cost_per_mtok=0.80,
            description="Cheapest option for simple extraction tasks",
        ),
        "gpt-4.1-mini": ModelConfig(
            provider=Provider.OPENAI,
            model_id="gpt-4.1-mini",
            api_key=openai_key,
            input_cost_per_mtok=0.80,
            output_cost_per_mtok=3.20,
            description="Good mid-range option",
        ),
        "gpt-5.2": ModelConfig(
            provider=Provider.OPENAI,
            model_id="gpt-5.2",
            api_key=openai_key,
            input_cost_per_mtok=1.75,
            output_cost_per_mtok=14.0,
            description="Frontier OpenAI model",
        ),

        # ─── GLM (Z.ai - OpenAI compatible) ─────────────
        "glm-4.7": ModelConfig(
            provider=Provider.GLM,
            model_id="glm-4.7",
            api_key=glm_key,
            base_url=glm_base,
            input_cost_per_mtok=0.60,
            output_cost_per_mtok=2.20,
            description="Best price/performance, strong coding",
        ),
        "glm-4.5": ModelConfig(
            provider=Provider.GLM,
            model_id="glm-4.5",
            api_key=glm_key,
            base_url=glm_base,
            input_cost_per_mtok=0.35,
            output_cost_per_mtok=1.55,
            description="Cheaper GLM option",
        ),

        # ─── Google Gemini (OpenAI-compatible) ─────────────
        "gemini-2.5-flash": ModelConfig(
            provider=Provider.GEMINI,
            model_id="gemini-2.5-flash",
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            input_cost_per_mtok=0.15,
            output_cost_per_mtok=0.60,
            description="Fast, cheap, strong multilingual (4.4x faster than GLM)",
        ),
        "gemini-2.5-pro": ModelConfig(
            provider=Provider.GEMINI,
            model_id="gemini-2.5-pro",
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            input_cost_per_mtok=1.25,
            output_cost_per_mtok=10.00,
            description="Best quality, expensive",
        ),
    }


# ─── Task-Based Model Router ────────────────────────────────────────────────

@dataclass
class TaskRouter:
    """
    Routes tasks to optimal models based on cost/quality tradeoff.
    User configures which model handles which task.
    """
    # Task → model key mapping (defaults: Gemini Flash + OpenAI nano/mini)
    chat: str = "gemini-2.5-flash"              # Main conversation (RAG)
    study: str = "gemini-2.5-flash"             # Study mode (strict grounding, deep teaching)
    extraction: str = "gpt-4.1-nano"            # Memory extraction (cheapest)
    intent: str = "gpt-4.1-mini"               # Intent classification (needs Turkish understanding)
    topic_detect: str = "gpt-4.1-nano"          # Topic detection (cheapest)
    summary: str = "gemini-2.5-flash"           # Weekly summaries
    questions: str = "gemini-2.5-flash"         # Practice questions
    overview: str = "gemini-2.5-flash"          # Course overview

    @classmethod
    def from_env(cls) -> "TaskRouter":
        """Load task routing from environment variables."""
        return cls(
            chat=os.getenv("MODEL_CHAT", "gemini-2.5-flash"),
            study=os.getenv("MODEL_STUDY", "gemini-2.5-flash"),
            extraction=os.getenv("MODEL_EXTRACTION", "gpt-4.1-nano"),
            intent=os.getenv("MODEL_INTENT", "gpt-4.1-mini"),
            topic_detect=os.getenv("MODEL_TOPIC_DETECT", "gpt-4.1-nano"),
            summary=os.getenv("MODEL_SUMMARY", "gemini-2.5-flash"),
            questions=os.getenv("MODEL_QUESTIONS", "gemini-2.5-flash"),
            overview=os.getenv("MODEL_OVERVIEW", "gemini-2.5-flash"),
        )

    def estimate_monthly_cost(self, turns_per_day: int = 20) -> dict:
        """Estimate monthly cost based on average usage."""
        presets = _get_presets()
        estimates = {}

        # Average tokens per task type
        task_profiles = {
            "chat": {"input": 2000, "output": 1000, "calls_per_turn": 1},
            "extraction": {"input": 500, "output": 200, "calls_per_turn": 1},
            "topic_detect": {"input": 200, "output": 100, "calls_per_turn": 1},
            "summary": {"input": 3000, "output": 2000, "calls_per_turn": 0.05},  # ~once/day
            "questions": {"input": 2000, "output": 2000, "calls_per_turn": 0.1},
        }

        total = 0
        for task, model_key in [
            ("chat", self.chat),
            ("extraction", self.extraction),
            ("topic_detect", self.topic_detect),
            ("summary", self.summary),
            ("questions", self.questions),
        ]:
            model = presets.get(model_key)
            if not model:
                continue
            profile = task_profiles[task]
            daily_calls = turns_per_day * profile["calls_per_turn"]
            monthly_calls = daily_calls * 30

            input_cost = (profile["input"] * monthly_calls / 1_000_000) * model.input_cost_per_mtok
            output_cost = (profile["output"] * monthly_calls / 1_000_000) * model.output_cost_per_mtok
            task_cost = input_cost + output_cost
            total += task_cost

            estimates[task] = {
                "model": model_key,
                "monthly_cost": round(task_cost, 3),
            }

        estimates["total_monthly"] = round(total, 2)
        return estimates


# ─── Provider Adapters ───────────────────────────────────────────────────────

class LLMAdapter(ABC):
    """Abstract LLM adapter interface."""

    @abstractmethod
    def complete(self, system: str, messages: list[dict], max_tokens: int = 4096) -> str:
        """Send a chat completion request and return the response text."""
        pass


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Claude API."""

    def __init__(self, model_config: ModelConfig):
        import anthropic
        self.client = anthropic.Anthropic(api_key=model_config.api_key)
        self.model = model_config.model_id

    def complete(self, system: str, messages: list[dict], max_tokens: int = 4096) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API."""

    def __init__(self, model_config: ModelConfig):
        from openai import OpenAI
        kwargs = {"api_key": model_config.api_key}
        if model_config.base_url:
            kwargs["base_url"] = model_config.base_url
        self.client = OpenAI(**kwargs)
        self.model = model_config.model_id

    def complete(self, system: str, messages: list[dict], max_tokens: int = 4096) -> str:
        full_messages = [{"role": "system", "content": system}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class GLMAdapter(LLMAdapter):
    """
    Adapter for GLM (Z.ai) API.
    Uses OpenAI-compatible endpoint.
    """

    def __init__(self, model_config: ModelConfig):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=model_config.api_key,
            base_url=model_config.base_url,
        )
        self.model = model_config.model_id

    def complete(self, system: str, messages: list[dict], max_tokens: int = 4096) -> str:
        full_messages = [{"role": "system", "content": system}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


# ─── Adapter Factory ────────────────────────────────────────────────────────

def create_adapter(model_config: ModelConfig) -> LLMAdapter:
    """Create the appropriate adapter for a model config."""
    if model_config.provider == Provider.ANTHROPIC:
        return AnthropicAdapter(model_config)
    elif model_config.provider == Provider.OPENAI:
        return OpenAIAdapter(model_config)
    elif model_config.provider in (Provider.GLM, Provider.GEMINI, Provider.OPENAI_COMPAT):
        return GLMAdapter(model_config)
    else:
        raise ValueError(f"Unknown provider: {model_config.provider}")


# ─── Multi-Provider Engine ──────────────────────────────────────────────────

class MultiProviderEngine:
    """
    Manages multiple LLM providers and routes tasks to optimal models.
    This replaces the single-provider LLM engine.
    """

    def __init__(self):
        self.presets = _get_presets()
        self.router = TaskRouter.from_env()
        self._adapters: dict[str, LLMAdapter] = {}
        self._cost_tracker: dict[str, float] = {}

    def get_adapter(self, model_key: str) -> LLMAdapter:
        """Get or create an adapter for the given model key."""
        if model_key not in self._adapters:
            model_config = self.presets.get(model_key)
            if not model_config:
                raise ValueError(
                    f"Unknown model: '{model_key}'. "
                    f"Available: {list(self.presets.keys())}"
                )
            if not model_config.api_key:
                raise ValueError(
                    f"No API key configured for '{model_key}' "
                    f"(provider: {model_config.provider.value}). "
                    f"Set the appropriate env var."
                )
            self._adapters[model_key] = create_adapter(model_config)
            logger.info(f"Initialized adapter: {model_key} ({model_config.provider.value})")

        return self._adapters[model_key]

    def complete(self, task: str, system: str, messages: list[dict],
                 max_tokens: int = 4096) -> str:
        """
        Route a task to the appropriate model and get a completion.

        Args:
            task: Task type (chat, extraction, summary, etc.)
            system: System prompt
            messages: Chat messages
            max_tokens: Max output tokens
        """
        # Get model key for this task
        model_key = getattr(self.router, task, self.router.chat)
        adapter = self.get_adapter(model_key)

        try:
            result = adapter.complete(system, messages, max_tokens)
            logger.debug(f"[{task}] → {model_key}: OK")
            return result
        except Exception as e:
            logger.error(f"[{task}] {model_key} failed: {e}")
            # Fallback: try another model
            return self._fallback_complete(task, system, messages, max_tokens, failed=model_key)

    def _fallback_complete(self, task: str, system: str, messages: list[dict],
                           max_tokens: int, failed: str) -> str:
        """Try alternative models if the primary fails."""
        # Fallback priority: glm → openai → anthropic (if available)
        fallback_chain = ["gemini-2.5-flash", "glm-4.7", "gpt-4.1-mini", "gpt-5-mini", "claude-haiku", "claude-sonnet"]

        for model_key in fallback_chain:
            if model_key == failed:
                continue
            if model_key not in self.presets:
                continue
            if not self.presets[model_key].api_key:
                continue

            try:
                adapter = self.get_adapter(model_key)
                result = adapter.complete(system, messages, max_tokens)
                logger.info(f"[{task}] Fallback to {model_key}: OK")
                return result
            except Exception:
                continue

        return f"Tüm modeller başarısız oldu. API key'lerinizi kontrol edin."

    def get_available_models(self) -> list[dict]:
        """List all models with configured API keys."""
        available = []
        for key, mc in self.presets.items():
            available.append({
                "key": key,
                "provider": mc.provider.value,
                "model_id": mc.model_id,
                "has_key": bool(mc.api_key),
                "cost": f"${mc.input_cost_per_mtok}/${mc.output_cost_per_mtok} per MTok",
                "description": mc.description,
            })
        return available

    def estimate_costs(self, turns_per_day: int = 20) -> dict:
        """Estimate monthly costs for current routing config."""
        return self.router.estimate_monthly_cost(turns_per_day)
