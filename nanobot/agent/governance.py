"""Edictum governance integration for nanobot agents."""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

try:
    from edictum import Edictum, EdictumDenied, Principal

    HAS_EDICTUM = True
except ImportError:
    HAS_EDICTUM = False


class GovernedToolRegistry:
    """Drop-in wrapper for nanobot's ToolRegistry with edictum governance.

    Intercepts execute() calls, runs them through the Edictum pipeline,
    and handles deny/approve/allow paths. All other ToolRegistry methods
    delegate directly to the inner registry.
    """

    def __init__(
        self,
        inner: "ToolRegistry",
        guard: "Edictum",
        *,
        session_id: str | None = None,
        principal: "Principal | None" = None,
        principal_resolver: Callable | None = None,
    ) -> None:
        self._inner = inner
        self._guard = guard
        self._session_id = session_id or str(uuid.uuid4())
        self._principal = principal
        self._principal_resolver = principal_resolver

    # --- Delegate all ToolRegistry methods to inner ---

    def register(self, tool: Any) -> None:
        return self._inner.register(tool)

    def unregister(self, name: str) -> None:
        return self._inner.unregister(name)

    def get(self, name: str) -> Any:
        return self._inner.get(name)

    def has(self, name: str) -> bool:
        return self._inner.has(name)

    def get_definitions(self) -> list[dict[str, Any]]:
        return self._inner.get_definitions()

    @property
    def tool_names(self) -> list[str]:
        return self._inner.tool_names

    def __len__(self) -> int:
        return len(self._inner)

    def __contains__(self, name: str) -> bool:
        return name in self._inner

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool with governance checks.

        Uses guard.run() which handles the full governance pipeline:
        pre-checks, approval flow, tool execution, post-checks, audit.

        On deny: returns "[DENIED] reason" string (nanobot expects string results).
        On approval timeout/deny: returns "[DENIED] Approval denied: reason".
        On error: returns "[ERROR] message" and logs the exception.
        """

        async def tool_callable(**kwargs: Any) -> str:
            return await self._inner.execute(name, kwargs)

        try:
            result = await self._guard.run(
                tool_name=name,
                args=params,
                tool_callable=tool_callable,
                session_id=self._session_id,
                principal=self._principal,
            )
            return str(result)
        except EdictumDenied as e:
            logger.info("Tool call denied: %s — %s", name, e.reason)
            return f"[DENIED] {e.reason}"
        except Exception:
            logger.exception("Governance error for tool %s", name)
            return f"[ERROR] Governance check failed for {name}"

    def set_principal(self, principal: Any) -> None:
        """Update principal for subsequent calls."""
        self._principal = principal

    def for_subagent(
        self,
        *,
        inner: "ToolRegistry | None" = None,
        session_id: str | None = None,
    ) -> GovernedToolRegistry:
        """Create a child GovernedToolRegistry for a sub-agent.

        Shares the same guard but can wrap a different inner registry
        (e.g. the subagent's own ToolRegistry) and gets its own session.
        """
        return GovernedToolRegistry(
            inner=inner if inner is not None else self._inner,
            guard=self._guard,
            session_id=session_id or str(uuid.uuid4()),
            principal=self._principal,
            principal_resolver=self._principal_resolver,
        )


def principal_from_message(message: Any) -> "Principal":
    """Map a nanobot InboundMessage to an edictum Principal."""
    return Principal(
        user_id=f"{message.channel}:{message.sender_id}",
        role="user",
        claims={
            "channel": message.channel,
            "chat_id": message.chat_id,
        },
    )


def create_governed_registry(
    inner: "ToolRegistry",
    *,
    contract_path: str | None = None,
    template: str = "nanobot-agent",
    server_url: str | None = None,
    api_key: str | None = None,
    agent_id: str = "nanobot",
    mode: str = "enforce",
) -> GovernedToolRegistry:
    """Factory for local-only governance (no server connection).

    Use create_governed_registry_from_server() for server-connected mode.
    """
    if not HAS_EDICTUM:
        raise ImportError(
            "edictum is not installed. Install with: pip install 'nanobot-ai[governance]'"
        )

    if contract_path:
        guard = Edictum.from_yaml(contract_path, mode=mode)
    else:
        guard = Edictum.from_template(template, mode=mode)

    return GovernedToolRegistry(inner=inner, guard=guard)


async def create_governed_registry_from_server(
    inner: "ToolRegistry",
    *,
    server_url: str,
    api_key: str,
    agent_id: str = "nanobot",
    mode: str = "enforce",
    contract_path: str | None = None,
    template: str = "nanobot-agent",
) -> GovernedToolRegistry:
    """Factory for server-connected governance using Edictum.from_server().

    Connects to edictum-console, fetches contracts, sets up SSE auto-reload,
    and wires audit + approval backends.

    Falls back to local contracts if server connection fails.
    """
    if not HAS_EDICTUM:
        raise ImportError(
            "edictum is not installed. Install with: pip install 'nanobot-ai[governance]'"
        )

    try:
        guard = await Edictum.from_server(
            url=server_url,
            api_key=api_key,
            agent_id=agent_id,
            mode=mode,
            auto_watch=True,
        )
        logger.info(
            "Edictum connected to server %s (agent=%s, mode=%s)",
            server_url, agent_id, mode,
        )
        return GovernedToolRegistry(inner=inner, guard=guard)
    except Exception:
        logger.warning(
            "Failed to connect to edictum server %s, falling back to local contracts",
            server_url,
            exc_info=True,
        )
        return create_governed_registry(
            inner,
            contract_path=contract_path,
            template=template,
            mode=mode,
        )
