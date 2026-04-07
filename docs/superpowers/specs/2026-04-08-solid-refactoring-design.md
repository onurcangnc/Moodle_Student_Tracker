# SOLID Refactoring Design Spec

**Date:** 2026-04-08  
**Status:** Approved  
**Scope:** Medium — Tool Registry Pattern + Simple DI

## Problem Statement

Current `agent_service.py` (2147 lines) violates multiple SOLID principles:
- **SRP:** 18 tools + router + system prompt + error handling in one file
- **OCP:** Adding new tools requires modifying existing code
- **ISP:** `STATE` singleton exposes 16 fields to all consumers
- **DIP:** Manual instantiation in `main.py`, no DI container

## Solution Overview

Apply **Tool Registry Pattern** with **Strategy Pattern** for tools, plus simple **Dependency Injection** via `ServiceContainer`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      main.py                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           ServiceContainer (DI)                      │   │
│  │  - moodle, vector_store, llm, stars, webmail        │   │
│  │  - tool_registry, llm_router                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                                   ▼
┌───────────────────────┐           ┌───────────────────────┐
│     LLMRouter         │           │    ToolRegistry       │
│  (LiteLLM wrapper)    │           │  register(tool)       │
│  - latency routing    │           │  get_definitions()    │
│  - multi-provider     │           │  execute(name, args)  │
│  - fallback           │           └───────────────────────┘
└───────────────────────┘                     │
                                ┌─────────────┼─────────────┐
                                ▼             ▼             ▼
                          AcademicTools ContentTools  MoodleTools
```

## File Structure

```
bot/
├── state.py              # ServiceContainer (replaces STATE singleton)
├── services/
│   ├── agent_service.py  # Orchestration only (~300 lines)
│   ├── llm_router.py     # LiteLLM wrapper (~150 lines) [NEW]
│   └── tools/            # [NEW]
│       ├── __init__.py   # ToolRegistry + BaseTool ABC
│       ├── academic.py   # schedule, grades, attendance, exams, transcript, letter_grades
│       ├── content.py    # source_map, read_source, study_topic, rag_search
│       ├── moodle.py     # materials, assignments, list_courses, set_active_course
│       ├── communication.py  # emails, email_detail
│       └── system.py     # stats, query_db
```

## Core Abstractions

### BaseTool (ABC)

```python
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Strategy pattern - each tool implements this interface."""
    
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @property
    @abstractmethod
    def description(self) -> str: ...
    
    @property
    @abstractmethod
    def parameters(self) -> dict: ...
    
    @abstractmethod
    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str: ...
    
    def to_openai_schema(self) -> dict:
        """Generate OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
```

### ToolRegistry

```python
class ToolRegistry:
    """Registry pattern - collects tools and dispatches execution."""
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
    
    def get_definitions(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]
    
    async def execute(self, name: str, args: dict, user_id: int, services) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Unknown tool: {name}"
        return await tool.execute(args, user_id, services)
```

### ServiceContainer

```python
@dataclass
class ServiceContainer:
    """DI container - typed dependencies."""
    moodle: MoodleClient
    vector_store: VectorStore
    llm: LLMEngine
    stars: StarsClient
    webmail: WebmailClient
    sync_engine: SyncEngine
    tool_registry: ToolRegistry
    llm_router: LLMRouter
```

### LLMRouter

```python
class LLMRouter:
    """Wraps LiteLLM Router with lazy initialization."""
    
    def __init__(self):
        self._router: Router | None = None
    
    def _ensure_router(self) -> Router: ...
    
    async def complete(self, messages, tools=None, max_tokens=1024) -> Any: ...
    
    async def stream(self, messages, message: Message) -> str: ...
    
    async def warmup(self) -> None: ...
    
    def sanitize_output(self, text: str) -> str: ...
```

## Feature Mapping (Preserved Features)

| Current Feature | New Location | Notes |
|-----------------|--------------|-------|
| `_get_fast_router()` | `LLMRouter._ensure_router()` | Lazy init preserved |
| `warmup_llm_connections()` | `LLMRouter.warmup()` | |
| `_sanitize_llm_output()` | `LLMRouter.sanitize_output()` | |
| `_check_instant_response()` | `agent_service.py` (stays) | Orchestration logic |
| `_is_complex_query()` | `agent_service.py` (stays) | |
| `_smart_error()` | `agent_service.py` (stays) | |
| `_build_system_prompt()` | `agent_service.py` (stays) | |
| `TOOLS` definitions | `ToolRegistry.get_definitions()` | Auto-generated |
| `TOOL_HANDLERS` dict | `ToolRegistry.execute()` | |
| `_execute_tool_call()` | `ToolRegistry.execute()` | |
| `_detect_language()` | `agent_service.py` (stays) | |
| `_stream_final_response()` | `LLMRouter.stream()` | |
| `_send_progressive()` | `agent_service.py` (stays) | Telegram-specific |
| `handle_agent_message()` | `agent_service.py` (stays, slimmed) | ~300 lines |
| 18 `_tool_*` functions | `tools/*.py` | Strategy pattern |

## Tool Distribution

| Module | Tools |
|--------|-------|
| `academic.py` | get_schedule, get_grades, get_attendance, get_exams, get_transcript, get_letter_grades |
| `content.py` | get_source_map, read_source, study_topic, rag_search |
| `moodle.py` | get_moodle_materials, get_assignments, list_courses, set_active_course |
| `communication.py` | get_emails, get_email_detail |
| `system.py` | get_stats, query_db |

## SOLID Compliance

| Principle | Before | After |
|-----------|--------|-------|
| **S** (SRP) | 2147-line agent_service.py | ~300 lines orchestration, tools in separate modules |
| **O** (OCP) | Modify TOOLS + TOOL_HANDLERS to add tool | Create new BaseTool subclass, register() |
| **L** (LSP) | N/A | All tools implement BaseTool contract |
| **I** (ISP) | STATE exposes 16 fields | ServiceContainer with typed dependencies |
| **D** (DIP) | Manual instantiation in main.py | Constructor injection via ServiceContainer |

## Migration Strategy

1. Create `tools/` module with `BaseTool` and `ToolRegistry`
2. Create `llm_router.py` with `LLMRouter` class
3. Migrate tools one group at a time (academic → content → moodle → communication → system)
4. Update `state.py` to `ServiceContainer`
5. Update `agent_service.py` to use new abstractions
6. Update `main.py` composition root
7. Run tests, verify functionality

## Testing

- Existing tests should pass without modification
- New unit tests for `ToolRegistry` and `LLMRouter`
- Integration test: `handle_agent_message()` behavior unchanged

## Rollback Plan

Git revert to previous commit if issues arise in production.
