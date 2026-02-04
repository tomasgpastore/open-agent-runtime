from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


LOGGER = logging.getLogger(__name__)
ENV_PLACEHOLDER = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


@dataclass(slots=True)
class MCPServerConfig:
    name: str
    transport: str
    enabled: bool
    command: str | None = None
    args: list[str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    env: dict[str, str] | None = None

    def to_connection_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "transport": self.transport,
        }
        if self.command:
            payload["command"] = self.command
        if self.args:
            payload["args"] = self.args
        if self.url:
            payload["url"] = self.url
        if self.headers:
            payload["headers"] = self.headers
        if self.env:
            payload["env"] = self.env
        return payload


@dataclass(slots=True)
class MCPServerStatus:
    name: str
    enabled: bool
    connected: bool
    tools: list[str]


class MCPManager:
    """Loads MCP server config and exposes currently-enabled tools."""

    def __init__(self, config_path: Path, fallback_config_path: Path) -> None:
        self._config_path = config_path
        self._fallback_config_path = fallback_config_path
        self._servers: dict[str, MCPServerConfig] = self._load_server_configs()
        self._connected: dict[str, bool] = {name: False for name in self._servers}
        self._tools_by_server: dict[str, list[BaseTool]] = {name: [] for name in self._servers}
        self._tool_registry: dict[str, BaseTool] = {}

    def _ensure_config_file(self) -> None:
        if self._config_path.exists():
            return
        if not self._fallback_config_path.exists():
            raise FileNotFoundError(
                f"MCP config not found at {self._config_path} and fallback missing at {self._fallback_config_path}"
            )
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._fallback_config_path, self._config_path)

    def _expand_env_placeholders(self, value: Any) -> Any:
        if isinstance(value, str):
            match = ENV_PLACEHOLDER.match(value)
            if not match:
                return value
            return os.getenv(match.group(1), "")
        if isinstance(value, list):
            return [self._expand_env_placeholders(item) for item in value]
        if isinstance(value, dict):
            return {key: self._expand_env_placeholders(item) for key, item in value.items()}
        return value

    def _load_server_configs(self) -> dict[str, MCPServerConfig]:
        self._ensure_config_file()

        with self._config_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        parsed: dict[str, MCPServerConfig] = {}
        for name, payload in raw.items():
            expanded = self._expand_env_placeholders(payload)
            parsed[name] = MCPServerConfig(
                name=name,
                transport=expanded["transport"],
                enabled=bool(expanded.get("enabled", True)),
                command=expanded.get("command"),
                args=expanded.get("args"),
                url=expanded.get("url"),
                headers=expanded.get("headers"),
                env=expanded.get("env"),
            )
        return parsed

    def _persist_config(self) -> None:
        serializable = {
            name: {
                "transport": cfg.transport,
                "enabled": cfg.enabled,
                **({"command": cfg.command} if cfg.command else {}),
                **({"args": cfg.args} if cfg.args else {}),
                **({"url": cfg.url} if cfg.url else {}),
                **({"headers": cfg.headers} if cfg.headers else {}),
                **({"env": cfg.env} if cfg.env else {}),
            }
            for name, cfg in self._servers.items()
        }
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with self._config_path.open("w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2)
            fh.write("\n")

    async def refresh_connections(self) -> None:
        self._connected = {name: False for name in self._servers}
        self._tools_by_server = {name: [] for name in self._servers}
        self._tool_registry = {}

        for name, cfg in self._servers.items():
            if not cfg.enabled:
                continue

            client = MultiServerMCPClient({name: cfg.to_connection_dict()})
            try:
                tools = await client.get_tools(tool_name_prefix=True)
                self._connected[name] = True
                self._tools_by_server[name] = tools
                for tool in tools:
                    self._tool_registry[tool.name] = tool
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to connect MCP server '%s': %s", name, exc)
                self._connected[name] = False
                self._tools_by_server[name] = []

    def list_statuses(self) -> list[MCPServerStatus]:
        statuses: list[MCPServerStatus] = []
        for name in sorted(self._servers.keys()):
            statuses.append(
                MCPServerStatus(
                    name=name,
                    enabled=self._servers[name].enabled,
                    connected=self._connected.get(name, False),
                    tools=[tool.name for tool in self._tools_by_server.get(name, [])],
                )
            )
        return statuses

    def active_tools(self) -> list[BaseTool]:
        return list(self._tool_registry.values())

    def get_tool(self, tool_name: str) -> BaseTool | None:
        return self._tool_registry.get(tool_name)

    def tool_names(self) -> list[str]:
        return sorted(self._tool_registry)

    def is_server_known(self, server_name: str) -> bool:
        return server_name in self._servers

    def set_server_enabled(self, server_name: str, enabled: bool) -> None:
        if server_name not in self._servers:
            raise KeyError(f"Unknown MCP server: {server_name}")
        self._servers[server_name].enabled = enabled
        self._persist_config()

    def server_enabled(self, server_name: str) -> bool:
        if server_name not in self._servers:
            raise KeyError(f"Unknown MCP server: {server_name}")
        return self._servers[server_name].enabled

    def server_names(self) -> list[str]:
        return sorted(self._servers)

    def tool_to_server_name(self, tool_name: str) -> str | None:
        if "_" not in tool_name:
            return None
        prefix, _ = tool_name.split("_", 1)
        return prefix if prefix in self._servers else None
