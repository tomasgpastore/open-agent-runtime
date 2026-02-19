from __future__ import annotations

import json
from pathlib import Path
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
                "For every node include input_schema/output_schema contract objects "
                "(except start can omit input_schema and end can omit output_schema). "
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
                "- execution_defaults.ai_edge_policy = always\n"
                "- state_access.default_enabled = false\n"
                "- include start and end nodes\n"
                "- condition nodes require strategy, if_true, if_false\n"
                "- state nodes require state_access_enabled=true per node if used\n"
                + (f"Previous validation errors to fix:\n{error_block}\n" if error_block else "")
            )
        )

        response = await self._llm_client.invoke([system, human], tools=None, on_token=None)
        raw = self._message_content_to_text(response.content)
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
        search_tool = self._pick_tool(
            available_tools,
            [
                "search_emails",
                "gmail_autoauth_search_emails",
                "read_email",
                "gmail_autoauth_read_email",
            ],
        )
        write_tool = self._pick_tool(
            available_tools,
            [
                "write_file",
                "filesystem_write_file",
            ],
        )
        output_path = self._normalize_output_path(
            self._extract_markdown_path(intent) or "./reports/automation_summary.md"
        )
        create_dir_tool = self._pick_tool(
            available_tools,
            [
                "create_directory",
                "filesystem_create_directory",
            ],
        )
        reports_dir = str(Path(output_path).parent)
        if not reports_dir:
            reports_dir = "."
        next_after_classification = (
            "ensure_reports_dir" if create_dir_tool and reports_dir not in {"", "."} else "write_markdown_summary"
        )

        if search_tool and write_tool:
            nodes: list[dict[str, Any]] = [
                {
                    "node_id": "start",
                    "type": "start",
                    "output_schema": {"type": "any"},
                    "next": "fetch_unread_gmail",
                },
                {
                    "node_id": "fetch_unread_gmail",
                    "type": "tool",
                    "tool": search_tool,
                    "args": {
                        "query": "is:unread newer_than:1d",
                        "max_results": 50,
                    },
                    "input_schema": {"type": "any"},
                    "output_schema": {"type": "any"},
                    "timeout_seconds": 45,
                    "max_retries": 2,
                    "idempotency_key": f"{graph_id}:fetch_unread_gmail",
                    "next": "classify_urgency",
                },
                {
                    "node_id": "classify_urgency",
                    "type": "ai_template",
                    "prompt_template": (
                        "You are classifying unread Gmail messages by urgency.\n"
                        "Input messages: {{fetch_unread_gmail}}\n"
                        "Return JSON with keys: total_count, urgent_count, medium_count, "
                        "low_count, urgent_items (array), summary_markdown (string). "
                        "summary_markdown must be a readable markdown report."
                    ),
                    "input_schema": {"type": "any"},
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "total_count": {"type": "integer"},
                            "urgent_count": {"type": "integer"},
                            "medium_count": {"type": "integer"},
                            "low_count": {"type": "integer"},
                            "urgent_items": {
                                "type": "array",
                                "items": {"type": "object"},
                            },
                            "summary_markdown": {"type": "string"},
                        },
                        "required": ["summary_markdown"],
                        "additionalProperties": True,
                    },
                    "output_format": "json",
                    "timeout_seconds": 45,
                    "max_retries": 2,
                    "idempotency_key": f"{graph_id}:classify_urgency",
                    "next": next_after_classification,
                },
            ]

            if create_dir_tool and reports_dir not in {"", "."}:
                nodes.append(
                    {
                        "node_id": "ensure_reports_dir",
                        "type": "tool",
                        "tool": create_dir_tool,
                        "args": {"path": reports_dir},
                        "input_schema": {"type": "any"},
                        "output_schema": {"type": "any"},
                        "timeout_seconds": 45,
                        "max_retries": 1,
                        "idempotency_key": f"{graph_id}:ensure_reports_dir",
                        "next": "write_markdown_summary",
                    }
                )

            nodes.extend(
                [
                    {
                        "node_id": "write_markdown_summary",
                        "type": "tool",
                        "tool": write_tool,
                        "args": {
                            "path": output_path,
                            "content": "{{classify_urgency.summary_markdown}}",
                        },
                        "input_schema": {"type": "any"},
                        "output_schema": {"type": "any"},
                        "timeout_seconds": 45,
                        "max_retries": 2,
                        "idempotency_key": f"{graph_id}:write_markdown_summary",
                        "next": "end",
                    },
                    {
                        "node_id": "end",
                        "type": "end",
                        "input_schema": {"type": "any"},
                    },
                ]
            )

            return {
                "id": graph_id,
                "name": graph_name,
                "description": summary,
                "start": "start",
                "execution_defaults": {
                    "guarantee_mode": "bounded",
                    "ai_edge_policy": "always",
                },
                "state_access": {"default_enabled": False},
                "nodes": nodes,
            }

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
                    "output_schema": {"type": "any"},
                    "next": "run_primary_tool",
                },
                {
                    "node_id": "run_primary_tool",
                    "type": "tool",
                    "tool": tool_name,
                    "args": {"query": summary},
                    "input_schema": {"type": "any"},
                    "output_schema": {"type": "any"},
                    "timeout_seconds": 45,
                    "max_retries": 2,
                    "idempotency_key": f"{graph_id}:run_primary_tool",
                    "next": "end",
                },
                {
                    "node_id": "end",
                    "type": "end",
                    "input_schema": {"type": "any"},
                },
            ],
        }

    def _title_from_intent(self, intent: str) -> str:
        collapsed = " ".join(intent.strip().split())
        if not collapsed:
            return "untitled_graph"
        words = collapsed[:64].lower().replace(" ", "_")
        words = re.sub(r"[^a-z0-9_]+", "", words)
        return words or "untitled_graph"

    def _message_content_to_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)
        return str(content)

    def _pick_tool(self, available_tools: list[str], preferred: list[str]) -> str | None:
        if not available_tools:
            return None
        available_set = set(available_tools)
        for candidate in preferred:
            if candidate in available_set:
                return candidate
            for tool_name in available_tools:
                if tool_name.endswith(f"_{candidate}"):
                    return tool_name
        return None

    def _extract_markdown_path(self, intent: str) -> str | None:
        match = re.search(r"([./~A-Za-z0-9_\-{}]+\.md)\b", intent)
        if not match:
            return None
        return match.group(1)

    def _normalize_output_path(self, output_path: str) -> str:
        normalized = output_path.strip()
        if not normalized:
            return "./reports/automation_summary.md"
        normalized = normalized.replace("{{date}}", "today")
        normalized = normalized.replace("{{ current_date }}", "today")
        normalized = normalized.replace("{{current_date}}", "today")
        return normalized
