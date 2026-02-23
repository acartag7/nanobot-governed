"""Tests for edictum governance integration."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import InboundMessage

# --- Helpers ---


class EchoTool(Tool):
    """Simple tool that echoes input for testing."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo input"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return kwargs.get("text", "")


def _make_registry() -> ToolRegistry:
    """Create a ToolRegistry with an echo tool registered."""
    reg = ToolRegistry()
    reg.register(EchoTool())
    return reg


def _make_mock_guard(*, result: str = "ok", deny: bool = False, error: bool = False) -> MagicMock:
    """Create a mock Edictum guard."""
    guard = MagicMock()
    if deny:
        exc = type("EdictumDenied", (Exception,), {"reason": "not allowed"})()
        guard.run = AsyncMock(side_effect=exc)
    elif error:
        guard.run = AsyncMock(side_effect=RuntimeError("guard broke"))
    else:
        guard.run = AsyncMock(return_value=result)
    return guard


# --- Test: Delegation ---


def test_governed_registry_delegates():
    """GovernedToolRegistry delegates register/has/get/tool_names to inner."""
    from nanobot.agent.governance import GovernedToolRegistry

    inner = _make_registry()
    guard = _make_mock_guard()
    governed = GovernedToolRegistry(inner=inner, guard=guard)

    assert governed.has("echo")
    assert not governed.has("missing")
    assert governed.get("echo") is not None
    assert governed.get("missing") is None
    assert "echo" in governed.tool_names
    assert len(governed) == 1
    assert "echo" in governed

    defs = governed.get_definitions()
    assert len(defs) == 1
    assert defs[0]["function"]["name"] == "echo"


# --- Test: Execute allowed ---


async def test_governed_execute_allowed():
    """Tool executes normally when governance allows."""
    from nanobot.agent.governance import GovernedToolRegistry

    inner = _make_registry()
    guard = _make_mock_guard(result="hello world")
    governed = GovernedToolRegistry(inner=inner, guard=guard)

    result = await governed.execute("echo", {"text": "hello world"})
    assert result == "hello world"
    guard.run.assert_awaited_once()

    call_kwargs = guard.run.call_args.kwargs
    assert call_kwargs["tool_name"] == "echo"
    assert call_kwargs["args"] == {"text": "hello world"}


# --- Test: Execute denied ---


async def test_governed_execute_denied():
    """Returns [DENIED] string when governance denies."""
    from nanobot.agent.governance import GovernedToolRegistry

    inner = _make_registry()
    mock_denied = type("EdictumDenied", (Exception,), {"reason": "dangerous operation"})

    guard = MagicMock()
    guard.run = AsyncMock(side_effect=mock_denied("denied"))

    with patch("nanobot.agent.governance.EdictumDenied", mock_denied):
        governed = GovernedToolRegistry(inner=inner, guard=guard)
        result = await governed.execute("echo", {"text": "hello"})

    assert result.startswith("[DENIED]")
    assert "dangerous operation" in result


# --- Test: Execute error (graceful degradation) ---


async def test_graceful_degradation():
    """Governance error returns [ERROR] message."""
    from nanobot.agent.governance import GovernedToolRegistry

    inner = _make_registry()
    guard = _make_mock_guard(error=True)
    governed = GovernedToolRegistry(inner=inner, guard=guard)

    result = await governed.execute("echo", {"text": "hello"})
    assert result.startswith("[ERROR]")
    assert "echo" in result


# --- Test: principal_from_message ---


def test_principal_from_message():
    """Maps InboundMessage fields to edictum Principal correctly."""
    mock_principal = MagicMock()
    with patch("nanobot.agent.governance.Principal", return_value=mock_principal) as mock_cls:
        from nanobot.agent.governance import principal_from_message

        msg = InboundMessage(
            channel="telegram",
            sender_id="user123",
            chat_id="chat456",
            content="hello",
        )
        principal_from_message(msg)

        mock_cls.assert_called_once_with(
            user_id="telegram:user123",
            role="user",
            claims={"channel": "telegram", "chat_id": "chat456"},
        )


# --- Test: for_subagent ---


def test_for_subagent():
    """for_subagent creates child with own session but shared guard."""
    from nanobot.agent.governance import GovernedToolRegistry

    inner = _make_registry()
    guard = _make_mock_guard()
    parent = GovernedToolRegistry(inner=inner, guard=guard, session_id="parent-session")

    child = parent.for_subagent()
    assert child._guard is guard
    assert child._session_id != "parent-session"
    assert child._inner is inner


def test_for_subagent_with_custom_inner():
    """for_subagent can wrap a different inner registry."""
    from nanobot.agent.governance import GovernedToolRegistry

    inner1 = _make_registry()
    inner2 = ToolRegistry()
    guard = _make_mock_guard()

    parent = GovernedToolRegistry(inner=inner1, guard=guard)
    child = parent.for_subagent(inner=inner2)

    assert child._inner is inner2
    assert child._guard is guard


# --- Test: set_principal ---


def test_set_principal():
    """set_principal updates the principal for subsequent calls."""
    from nanobot.agent.governance import GovernedToolRegistry

    inner = _make_registry()
    guard = _make_mock_guard()
    governed = GovernedToolRegistry(inner=inner, guard=guard)

    assert governed._principal is None
    mock_principal = MagicMock()
    governed.set_principal(mock_principal)
    assert governed._principal is mock_principal


# --- Test: create_governed_registry (local) ---


def test_create_governed_registry_local():
    """Factory creates GovernedToolRegistry with template (no server)."""
    from nanobot.agent.governance import GovernedToolRegistry

    mock_edictum = MagicMock()
    mock_guard = MagicMock()
    mock_edictum.from_template.return_value = mock_guard

    with patch("nanobot.agent.governance.HAS_EDICTUM", True), patch(
        "nanobot.agent.governance.Edictum", mock_edictum
    ):
        from nanobot.agent.governance import create_governed_registry

        inner = _make_registry()
        result = create_governed_registry(inner, template="nanobot-agent", mode="enforce")

        assert isinstance(result, GovernedToolRegistry)
        mock_edictum.from_template.assert_called_once_with(
            "nanobot-agent",
            mode="enforce",
            approval_backend=None,
            audit_sink=None,
        )


# --- Test: create_governed_registry raises without edictum ---


def test_create_governed_registry_no_edictum():
    """Factory raises ImportError when edictum is not installed."""
    with patch("nanobot.agent.governance.HAS_EDICTUM", False):
        from nanobot.agent.governance import create_governed_registry

        inner = _make_registry()
        with pytest.raises(ImportError, match="edictum is not installed"):
            create_governed_registry(inner)


# --- Test: EdictumConfig schema ---


def test_config_schema():
    """EdictumConfig validates correctly."""
    from nanobot.config.schema import EdictumConfig

    cfg = EdictumConfig(enabled=True, mode="observe")
    assert cfg.enabled is True
    assert cfg.mode == "observe"
    assert cfg.template == "nanobot-agent"
    assert cfg.contract_path is None
    assert cfg.server_url is None
    assert cfg.api_key is None
    assert cfg.agent_id == "nanobot"


def test_config_schema_defaults():
    """EdictumConfig has sane defaults."""
    from nanobot.config.schema import EdictumConfig

    cfg = EdictumConfig()
    assert cfg.enabled is False
    assert cfg.mode == "enforce"


def test_disabled_config():
    """edictum.enabled=false means default Config has governance disabled."""
    from nanobot.config.schema import Config

    c = Config()
    assert c.edictum.enabled is False


def test_config_with_edictum_enabled():
    """Config accepts edictum section."""
    from nanobot.config.schema import Config

    c = Config(edictum={"enabled": True, "mode": "observe"})
    assert c.edictum.enabled is True
    assert c.edictum.mode == "observe"


# --- Test: register/unregister through governed ---


def test_governed_register_unregister():
    """register and unregister pass through to inner."""
    from nanobot.agent.governance import GovernedToolRegistry

    inner = ToolRegistry()
    guard = _make_mock_guard()
    governed = GovernedToolRegistry(inner=inner, guard=guard)

    assert not governed.has("echo")
    governed.register(EchoTool())
    assert governed.has("echo")
    governed.unregister("echo")
    assert not governed.has("echo")
