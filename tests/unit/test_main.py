"""Unit tests for bot.main startup and health utilities."""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

import bot.main as main


def _find_free_port() -> int:
    """Find an available localhost TCP port for test HTTP server."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_get_git_version_success(monkeypatch):
    """Git hash helper should return stripped hash when command succeeds."""
    monkeypatch.setattr(main.subprocess, "check_output", lambda *args, **kwargs: "abc123\n")
    assert main.get_git_version() == "abc123"


def test_get_git_version_fallback_on_error(monkeypatch):
    """Git hash helper should return 'unknown' on subprocess failure."""
    def _raise(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["git"])

    monkeypatch.setattr(main.subprocess, "check_output", _raise)
    assert main.get_git_version() == "unknown"


def test_build_health_payload_contains_runtime_metrics(monkeypatch):
    """Health payload should include uptime, version, chunks and active user counts."""
    monkeypatch.setattr(main.time, "monotonic", lambda: 120.0)
    monkeypatch.setattr(main.time, "time", lambda: 1_000_000.0)
    monkeypatch.setattr(main.STATE, "vector_store", SimpleNamespace(get_stats=lambda: {"total_chunks": 3661}))
    monkeypatch.setattr(
        main.STATE,
        "user_last_seen",
        {1: 1_000_000.0, 2: 999_000.0, 3: 900_000.0},
    )

    payload = main._build_health_payload(started_at_monotonic=100.0, version="abb38f7")
    assert payload["status"] == "ok"
    assert payload["uptime_seconds"] == 20
    assert payload["version"] == "abb38f7"
    assert payload["chunks_loaded"] == 3661
    assert payload["active_users_24h"] == 2


def test_build_health_payload_handles_stats_failure(monkeypatch):
    """Health payload should degrade gracefully if vector stats fail."""
    def _boom():
        raise ValueError("stats unavailable")

    monkeypatch.setattr(main.STATE, "vector_store", SimpleNamespace(get_stats=_boom))
    monkeypatch.setattr(main.STATE, "user_last_seen", {})
    payload = main._build_health_payload(started_at_monotonic=time.monotonic(), version="v1")
    assert payload["chunks_loaded"] == 0


def test_ensure_event_loop_creates_loop_on_runtime_error(monkeypatch):
    """Event-loop helper should create and set loop when none exists."""
    loop_obj = object()
    monkeypatch.setattr(main.asyncio, "get_event_loop", lambda: (_ for _ in ()).throw(RuntimeError("no loop")))
    monkeypatch.setattr(main.asyncio, "new_event_loop", lambda: loop_obj)
    called: dict[str, object] = {}
    monkeypatch.setattr(main.asyncio, "set_event_loop", lambda loop: called.setdefault("loop", loop))
    main._ensure_event_loop()
    assert called["loop"] is loop_obj


def test_validate_startup_config_requires_token(monkeypatch):
    """Startup config validation should fail when token is missing."""
    monkeypatch.setattr(
        main,
        "CONFIG",
        SimpleNamespace(telegram_bot_token="", owner_id=1),
    )
    with pytest.raises(RuntimeError):
        main._validate_startup_config()


def test_validate_startup_config_requires_owner(monkeypatch):
    """Startup config validation should fail when owner id is missing."""
    monkeypatch.setattr(
        main,
        "CONFIG",
        SimpleNamespace(telegram_bot_token="token", owner_id=0),
    )
    with pytest.raises(RuntimeError):
        main._validate_startup_config()


def test_initialize_components_validation_error(monkeypatch):
    """Component init should raise when core config validation fails."""
    monkeypatch.setattr(main.core_config, "validate", lambda: ["broken env"])
    with pytest.raises(RuntimeError):
        main._initialize_components()


def test_initialize_components_success(monkeypatch):
    """Component init should wire state and cache Moodle course metadata."""
    class FakeMoodle:
        def connect(self):
            return True

        def get_courses(self):
            return [SimpleNamespace(shortname="CTIS 363", fullname="CTIS 363 Ethics")]

    class FakeProcessor:
        pass

    class FakeVectorStore:
        def initialize(self):
            self.initialized = True

    class FakeLLM:
        def __init__(self, vector_store):
            self.vector_store = vector_store
            self.moodle_courses = []

    class FakeSync:
        def __init__(self, moodle, processor, vector_store):
            self.moodle = moodle
            self.processor = processor
            self.vector_store = vector_store

    monkeypatch.setattr(main.core_config, "validate", lambda: [])
    monkeypatch.setattr(main, "MoodleClient", FakeMoodle)
    monkeypatch.setattr(main, "DocumentProcessor", FakeProcessor)
    monkeypatch.setattr(main, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(main, "LLMEngine", FakeLLM)
    monkeypatch.setattr(main, "SyncEngine", FakeSync)

    main._initialize_components()
    assert main.STATE.moodle is not None
    assert main.STATE.vector_store is not None
    assert main.STATE.llm is not None
    assert main.STATE.sync_engine is not None
    assert main.STATE.llm.moodle_courses[0]["shortname"] == "CTIS 363"


def test_create_application_registers_handlers(monkeypatch):
    """Application factory should register command/message and error handlers."""
    class FakeApp:
        def __init__(self):
            self.error_handler = None

        def add_error_handler(self, handler):
            self.error_handler = handler

    class FakeBuilder:
        def __init__(self, app):
            self.app = app
            self.token_value = None
            self.post_init_cb = None

        def token(self, token):
            self.token_value = token
            return self

        def post_init(self, callback):
            self.post_init_cb = callback
            return self

        def build(self):
            return self.app

    fake_app = FakeApp()
    builder = FakeBuilder(fake_app)
    monkeypatch.setattr(main, "Application", SimpleNamespace(builder=lambda: builder))
    monkeypatch.setattr(main, "CONFIG", SimpleNamespace(telegram_bot_token="bot-token"))

    marks: dict[str, int] = {}
    monkeypatch.setattr(main, "register_command_handlers", lambda app: marks.setdefault("commands", 1))
    monkeypatch.setattr(main, "register_message_handlers", lambda app: marks.setdefault("messages", 1))

    app = main.create_application()
    assert app is fake_app
    assert builder.token_value == "bot-token"
    assert builder.post_init_cb is main.post_init
    assert marks == {"commands": 1, "messages": 1}
    assert fake_app.error_handler is main.global_error_handler


def test_start_health_server_serves_health_json(monkeypatch):
    """Health server should expose /health with expected payload keys."""
    port = _find_free_port()
    monkeypatch.setattr(
        main,
        "CONFIG",
        SimpleNamespace(healthcheck_enabled=True, healthcheck_host="127.0.0.1", healthcheck_port=port),
    )
    monkeypatch.setattr(main.STATE, "user_last_seen", {1: time.time()})
    monkeypatch.setattr(main.STATE, "vector_store", SimpleNamespace(get_stats=lambda: {"total_chunks": 10}))

    main._start_health_server(time.monotonic() - 1, "abc123")

    response = None
    for _ in range(10):
        try:
            response = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            break
        except OSError:
            time.sleep(0.1)

    assert response is not None
    assert response.status == 200
    payload = json.loads(response.read().decode("utf-8"))
    assert payload["status"] == "ok"
    assert payload["version"] == "abc123"
    assert "uptime_seconds" in payload
    assert "chunks_loaded" in payload
    assert "active_users_24h" in payload

    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/x", timeout=2)
    assert exc.value.code == 404


def test_main_happy_path(monkeypatch):
    """main should wire startup helpers and start polling once."""
    calls: dict[str, object] = {}

    class FakeApp:
        def run_polling(self, allowed_updates):
            calls["allowed_updates"] = allowed_updates

    monkeypatch.setattr(main, "setup_logging", lambda level: calls.setdefault("log_level", level))
    monkeypatch.setattr(main, "get_git_version", lambda: "ver123")
    monkeypatch.setattr(main, "_validate_startup_config", lambda: calls.setdefault("validated", True))
    monkeypatch.setattr(main, "_initialize_components", lambda: calls.setdefault("initialized", True))
    monkeypatch.setattr(
        main,
        "_start_health_server",
        lambda started, version: calls.setdefault("health", (started > 0, version)),
    )
    monkeypatch.setattr(main, "create_application", lambda: FakeApp())
    monkeypatch.setattr(main, "_ensure_event_loop", lambda: calls.setdefault("loop", True))
    monkeypatch.setattr(
        main,
        "CONFIG",
        SimpleNamespace(log_level="INFO", owner_id=1, auto_sync_interval=600, assignment_check_interval=600),
    )

    main.main()
    assert calls["log_level"] == "INFO"
    assert calls["validated"] is True
    assert calls["initialized"] is True
    assert calls["health"][0] is True
    assert calls["health"][1] == "ver123"
    assert calls["loop"] is True
    assert calls["allowed_updates"] == main.Update.ALL_TYPES


def test_main_exits_when_validation_fails(monkeypatch):
    """main should exit with code 1 on startup validation failure."""
    monkeypatch.setattr(main, "setup_logging", lambda level: None)
    monkeypatch.setattr(main, "_validate_startup_config", lambda: (_ for _ in ()).throw(RuntimeError("bad cfg")))
    monkeypatch.setattr(main.sys, "exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    monkeypatch.setattr(main, "CONFIG", SimpleNamespace(log_level="INFO"))
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 1


def test_main_exits_when_component_init_fails(monkeypatch):
    """main should exit with code 1 when component initialization fails."""
    monkeypatch.setattr(main, "setup_logging", lambda level: None)
    monkeypatch.setattr(main, "_validate_startup_config", lambda: None)
    monkeypatch.setattr(main, "_initialize_components", lambda: (_ for _ in ()).throw(RuntimeError("init fail")))
    monkeypatch.setattr(main.sys, "exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    monkeypatch.setattr(main, "CONFIG", SimpleNamespace(log_level="INFO"))
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 1
