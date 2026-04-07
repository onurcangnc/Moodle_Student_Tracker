"""
Tool Registry Pattern for Agentic LLM Service.
==============================================
Implements Strategy pattern for tools with OpenAI function calling support.

SOLID Compliance:
- SRP: Each tool has single responsibility
- OCP: Add new tools without modifying existing code
- LSP: All tools implement BaseTool contract
- DIP: Tools depend on abstractions (ServiceContainer)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bot.state import ServiceContainer

logger = logging.getLogger(__name__)

__all__ = ["BaseTool", "ToolRegistry", "create_default_registry"]


class BaseTool(ABC):
    """
    Abstract base class for all tools (Strategy pattern).

    Each tool implements this interface and is registered with ToolRegistry.
    Tools are auto-converted to OpenAI function calling format.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name for function calling."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM (explains when to use)."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        ...

    @abstractmethod
    async def execute(
        self,
        args: dict[str, Any],
        user_id: int,
        services: ServiceContainer,
    ) -> str:
        """
        Execute the tool with given arguments.

        Args:
            args: Parsed arguments from LLM
            user_id: Telegram user ID
            services: DI container with all dependencies

        Returns:
            Result string to feed back to LLM
        """
        ...

    def to_openai_schema(self) -> dict[str, Any]:
        """Generate OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """
    Registry pattern for tool management.

    Collects tools, generates OpenAI schemas, and dispatches execution.
    Replaces the old TOOLS list + TOOL_HANDLERS dict pattern.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._definitions_cache: list[dict[str, Any]] | None = None

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            logger.warning("Tool '%s' already registered, overwriting", tool.name)
        self._tools[tool.name] = tool
        self._definitions_cache = None  # Invalidate cache
        logger.debug("Tool registered: %s", tool.name)

    def register_all(self, tools: list[BaseTool]) -> None:
        """Register multiple tools at once."""
        for tool in tools:
            self._tools[tool.name] = tool
            logger.debug("Tool registered: %s", tool.name)
        self._definitions_cache = None  # Invalidate cache once

    def get_definitions(self) -> list[dict[str, Any]]:
        """
        Generate OpenAI function calling definitions (cached).

        Definitions are generated once and cached since tools don't
        change at runtime. Cache is invalidated on register().
        """
        if self._definitions_cache is None:
            self._definitions_cache = [tool.to_openai_schema() for tool in self._tools.values()]
        return self._definitions_cache

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        user_id: int,
        services: ServiceContainer,
    ) -> str:
        """
        Execute a tool by name.

        Replaces the old _execute_tool_call function.

        Args:
            name: Tool name from LLM
            args: Parsed arguments
            user_id: Telegram user ID
            services: DI container

        Returns:
            Tool result string
        """
        tool = self._tools.get(name)
        if tool is None:
            logger.warning("Unknown tool called: %s", name)
            return f"Bilinmeyen araç: {name}"

        try:
            result = await tool.execute(args, user_id, services)
            logger.info(
                "Tool executed: %s (result_len=%d)",
                name,
                len(result),
                extra={"tool": name, "user_id": user_id},
            )
            return result
        except Exception as exc:
            logger.error("Tool %s failed: %s", name, exc, exc_info=True)
            return f"[{name}] şu anda çalışmıyor ({type(exc).__name__}). Alternatif bilgi kaynağı dene veya kullanıcıya bildir."

    @property
    def tool_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)


def create_default_registry() -> ToolRegistry:
    """
    Factory function to create registry with all default tools.

    This is the composition root for tools.
    """
    from bot.services.tools.academic import get_academic_tools
    from bot.services.tools.communication import get_communication_tools
    from bot.services.tools.content import get_content_tools
    from bot.services.tools.moodle import get_moodle_tools
    from bot.services.tools.system import get_system_tools

    registry = ToolRegistry()
    registry.register_all(get_content_tools())
    registry.register_all(get_academic_tools())
    registry.register_all(get_moodle_tools())
    registry.register_all(get_communication_tools())
    registry.register_all(get_system_tools())

    logger.info("Tool registry initialized with %d tools", len(registry))
    return registry
