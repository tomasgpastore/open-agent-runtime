from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from assistant_cli.graph.schema import validate_graph_definition
from assistant_cli.llm_client import LLMClient


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", re.IGNORECASE)


@dataclass(slots=True)
class GraphBuildResult:
    graph: dict[str, Any]
    attempts: int
    source: str
    warnings: list[str]


class GraphBuilderAnton:
    """Intent-to-graph builder constrained to Anton graph schema."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def build_from_intent(
        self,
        *,
        intent: str,
        available_tools: list[str],
        graph_id: str | None = None,
        graph_name: str | None = None,
    ) -> GraphBuildResult:
        warnings: list[str] = []
        max_attempts = 3
        last_errors: list[str] = []

        target_id = graph_id or f"graph-{uuid.uuid4().hex[:8]}"
        target_name = graph_name or self._title_from_intent(intent)

        for attempt in range(1, max_attempts + 1):
            candidate = await self._ask_llm_for_graph(
                intent=intent,
                available_tools=available_tools,
                graph_id=target_id,
                graph_name=target_name,
                previous_errors=last_errors,
            )
            if candidate is None:
                last_errors = ["Model did not return parseable JSON graph."]
                continue

            candidate.setdefault("id", target_id)
            candidate.setdefault("name", target_name)
            candidate.setdefault("state_access", {"default_enabled": False})

            errors = validate_graph_definition(candidate)
            if not errors:
                return GraphBuildResult(
                    graph=candidate,
                    attempts=attempt,
                    source="llm",
                    warnings=warnings,
                )

            last_errors = errors

        warnings.extend(last_errors)
        fallback = self._fallback_graph(
            intent=intent,
            available_tools=available_tools,
            graph_id=target_id,
            graph_name=target_name,
        )
        return GraphBuildResult(
            graph=fallback,
            attempts=max_attempts,
            source="fallback",
            warnings=warnings,
        )

    async def _ask_llm_for_graph(
        self,
        *,
        intent: str,
        available_tools: list[str],
        graph_id: str,
        graph_name: str,
        previous_errors: list[str],
    ) -> dict[str, Any] | None:
        tool_list = ", ".join(available_tools) if available_tools else "(none)"
        error_block = "\n".join(f"- {item}" for item in previous_errors)
        system = SystemMessage(
            content=(
                "You are Graph Builder Anton. Output ONLY valid JSON for Anton graph schema. "
                "No markdown, no prose. Use explicit node_id values and valid next edges. "
                "Use tool nodes only with available tool names. "
                "For tool/ai_template nodes include timeout_seconds, max_retries, idempotency_key."
            )
        )
        human = HumanMessage(
            content=(
                f"Build a graph for this intent: {intent}\n"
                f"Required id: {graph_id}\n"
                f"Required name: {graph_name}\n"
                f"Available tools: {tool_list}\n"
                "Guarantee-compatible defaults:\n"
                "- execution_defaults.guarantee_mode = bounded\n"
                "- state_access.default_enabled = false\n"
                "- include start and end nodes\n"
                "- condition nodes require strategy, if_true, if_false\n"
                "- state nodes require state_access_enabled=true per node if used\n"
                + (f"Previous validation errors to fix:\n{error_block}\n" if error_block else "")
            )
        )

        response = await self._llm_client.invoke([system, human], tools=None, on_token=None)
        raw = response.content if isinstance(response.content, str) else str(response.content)
        return self._extract_json(raw)

    def patch_node_field(
        self,
        *,
        graph: dict[str, Any],
        node_id: str,
        field: str,
        value: object,
    ) -> dict[str, Any]:
        cloned = json.loads(json.dumps(graph))
        nodes = cloned.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("Graph has no nodes list.")
        for node in nodes:
            if isinstance(node, dict) and str(node.get("node_id")) == node_id:
                node[field] = value
                return cloned
        raise ValueError(f"Node '{node_id}' was not found.")

    def _extract_json(self, raw: str) -> dict[str, Any] | None:
        text = raw.strip()
        if not text:
            return None

        match = JSON_BLOCK_RE.search(text)
        if match:
            text = match.group(1).strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None

        if not isinstance(payload, dict):
            return None
        return payload

    def _fallback_graph(
        self,
        *,
        intent: str,
        available_tools: list[str],
        graph_id: str,
        graph_name: str,
    ) -> dict[str, Any]:
        tool_name = available_tools[0] if available_tools else "noop_tool"
        summary = intent.strip() or "automation"
        return {
            "id": graph_id,
            "name": graph_name,
            "description": summary,
            "start": "start",
            "execution_defaults": {"guarantee_mode": "bounded"},
            "state_access": {"default_enabled": False},
            "nodes": [
                {
                    "node_id": "start",
                    "type": "start",
                    "next": "run_primary_tool",
                },
                {
                    "node_id": "run_primary_tool",
                    "type": "tool",
                    "tool": tool_name,
                    "args": {"query": summary},
                    "timeout_seconds": 45,
                    "max_retries": 2,
                    "idempotency_key": f"{graph_id}:run_primary_tool",
                    "next": "end",
                },
                {
                    "node_id": "end",
                    "type": "end",
                },
            ],
        }

    def _title_from_intent(self, intent: str) -> str:
        collapsed = " ".join(intent.strip().split())
        if not collapsed:
            return "untitled_graph"
        words = collapsed[:64]
        return words.replace(" ", "_").lower()
