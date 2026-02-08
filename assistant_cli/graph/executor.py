from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from assistant_cli.graph.schema import (
    ALLOWED_GUARANTEE_MODES,
    GraphValidationError,
    validate_graph_or_raise,
)
from assistant_cli.graph.state_store import GraphStateStore


TEMPLATE_VALUE_RE = re.compile(r"^\{\{([^{}]+)\}\}$")


class GraphExecutionError(RuntimeError):
    """Raised when graph execution fails."""


@dataclass(slots=True)
class GraphExecutionResult:
    run_id: str
    graph_id: str
    guarantee_mode: str
    status: str
    output: object
    visited_nodes: list[str]


class GraphExecutor:
    """Deterministic graph executor with explicit state nodes."""

    def __init__(self, state_store: GraphStateStore) -> None:
        self._state_store = state_store

    def run(
        self,
        graph: dict[str, Any],
        input_payload: object | None = None,
        guarantee_mode: str = "bounded",
    ) -> GraphExecutionResult:
        if guarantee_mode not in ALLOWED_GUARANTEE_MODES:
            raise GraphExecutionError(
                f"Unsupported guarantee_mode '{guarantee_mode}'. Use strict, bounded, or flex."
            )

        validate_graph_or_raise(graph)
        self._validate_mode_policy(graph, guarantee_mode)

        graph_id = str(graph["id"])
        nodes = graph["nodes"]
        node_map = {str(node["node_id"]): node for node in nodes}
        max_steps = int(graph.get("max_steps", max(20, len(nodes) * 10)))

        run_id = self._state_store.start_run(
            graph_id=graph_id,
            guarantee_mode=guarantee_mode,
            input_payload=input_payload,
        )

        current_node_id = str(graph["start"])
        context: dict[str, object] = {
            "input": input_payload,
            "run_id": run_id,
            "graph_id": graph_id,
        }
        visited_nodes: list[str] = []
        steps = 0
        last_output: object = input_payload

        try:
            while True:
                if steps >= max_steps:
                    raise GraphExecutionError(
                        f"Execution exceeded max_steps={max_steps}; possible graph loop."
                    )
                steps += 1

                node = node_map.get(current_node_id)
                if node is None:
                    raise GraphExecutionError(f"Node '{current_node_id}' was not found during execution.")

                node_type = str(node["type"])
                node_input = self._resolve_value(node.get("input", last_output), context)
                output: object = None
                next_node: str | None = None

                if node_type == "start":
                    output = node_input
                    next_node = self._first_next(node)
                elif node_type == "end":
                    output = node_input
                    self._state_store.add_checkpoint(
                        run_id=run_id,
                        graph_id=graph_id,
                        node_id=current_node_id,
                        status="ok",
                        input_payload=node_input,
                        output_payload=output,
                    )
                    visited_nodes.append(current_node_id)
                    self._state_store.finish_run(run_id=run_id, status="succeeded", output_payload=output)
                    return GraphExecutionResult(
                        run_id=run_id,
                        graph_id=graph_id,
                        guarantee_mode=guarantee_mode,
                        status="succeeded",
                        output=output,
                        visited_nodes=visited_nodes,
                    )
                elif node_type == "transform":
                    output = self._resolve_value(node.get("value", node_input), context)
                    next_node = self._first_next(node)
                elif node_type == "read_state":
                    key = str(node["key"])
                    value = self._state_store.read_state(graph_id=graph_id, key=key)
                    output = {"key": key, "value": value}
                    next_node = self._first_next(node)
                elif node_type == "write_state":
                    key = str(node["key"])
                    value_source = node.get("value", node_input)
                    value = self._resolve_value(value_source, context)
                    self._state_store.write_state(
                        graph_id=graph_id,
                        key=key,
                        value=value,
                        source_run_id=run_id,
                        source_node_id=current_node_id,
                    )
                    output = {"key": key, "value": value, "written": True}
                    next_node = self._first_next(node)
                elif node_type == "read_prior_runs":
                    limit = int(node.get("limit", 5))
                    runs = self._state_store.read_prior_runs(graph_id=graph_id, limit=limit)
                    output = {
                        "count": len(runs),
                        "runs": [
                            {
                                "run_id": run.run_id,
                                "status": run.status,
                                "guarantee_mode": run.guarantee_mode,
                                "started_at": run.started_at,
                                "finished_at": run.finished_at,
                                "error_text": run.error_text,
                            }
                            for run in runs
                            if run.run_id != run_id
                        ],
                    }
                    next_node = self._first_next(node)
                elif node_type == "condition":
                    strategy = str(node.get("strategy"))
                    if strategy == "typed_condition":
                        result = self._evaluate_typed_condition(node=node, context=context)
                        next_node = str(node["if_true"] if result else node["if_false"])
                        output = {
                            "strategy": strategy,
                            "result": result,
                            "next": next_node,
                        }
                    elif strategy == "llm_condition":
                        decision = self._resolve_value(node.get("decision"), context)
                        options = list(node.get("branch_options") or [])
                        if not isinstance(decision, str) or decision not in options:
                            if guarantee_mode == "strict":
                                raise GraphExecutionError(
                                    f"Strict mode requires llm_condition decision in allowed branch_options for node '{current_node_id}'."
                                )
                            decision = options[0]
                        next_node = str(node["if_true"] if decision == options[0] else node["if_false"])
                        output = {
                            "strategy": strategy,
                            "decision": decision,
                            "next": next_node,
                        }
                    else:
                        raise GraphExecutionError(
                            f"Unsupported condition strategy '{strategy}' in node '{current_node_id}'."
                        )
                elif node_type in {"tool", "ai_template"}:
                    if "mock_output" not in node:
                        raise GraphExecutionError(
                            f"Node '{current_node_id}' type '{node_type}' is not executable yet without 'mock_output'."
                        )
                    output = self._resolve_value(node.get("mock_output"), context)
                    next_node = self._first_next(node)
                else:
                    raise GraphExecutionError(
                        f"Unsupported node type '{node_type}' in node '{current_node_id}'."
                    )

                self._state_store.add_checkpoint(
                    run_id=run_id,
                    graph_id=graph_id,
                    node_id=current_node_id,
                    status="ok",
                    input_payload=node_input,
                    output_payload=output,
                )

                visited_nodes.append(current_node_id)
                context[current_node_id] = output
                context["last"] = output
                last_output = output

                if next_node is None:
                    raise GraphExecutionError(f"Node '{current_node_id}' did not resolve a next node.")
                current_node_id = next_node

        except Exception as exc:
            self._state_store.add_checkpoint(
                run_id=run_id,
                graph_id=graph_id,
                node_id=current_node_id,
                status="error",
                input_payload=last_output,
                output_payload=None,
                error_text=str(exc),
            )
            self._state_store.finish_run(
                run_id=run_id,
                status="failed",
                output_payload=None,
                error_text=str(exc),
            )
            if isinstance(exc, GraphValidationError | GraphExecutionError):
                raise
            raise GraphExecutionError(str(exc)) from exc

    def _validate_mode_policy(self, graph: dict[str, Any], guarantee_mode: str) -> None:
        violations: list[str] = []

        for node in graph["nodes"]:
            node_id = str(node.get("node_id"))
            node_type = str(node.get("type"))

            if guarantee_mode in {"strict", "bounded"} and node_type in {"tool", "ai_template"}:
                if "mock_output" not in node:
                    violations.append(
                        f"{guarantee_mode} mode requires 'mock_output' for node '{node_id}' "
                        f"until live tool execution is integrated into graph runtime."
                    )

            if guarantee_mode == "strict" and node_type == "condition":
                strategy = node.get("strategy")
                if strategy == "llm_condition":
                    options = node.get("branch_options")
                    if not isinstance(options, list) or len(options) != 2:
                        violations.append(
                            f"Strict mode requires llm_condition node '{node_id}' to provide exactly two branch_options."
                        )

        if violations:
            rendered = "\n".join(f"- {item}" for item in violations)
            raise GraphExecutionError(f"Guarantee mode policy violation:\n{rendered}")

    def _first_next(self, node: dict[str, Any]) -> str:
        raw_next = node.get("next")
        if isinstance(raw_next, str):
            return raw_next
        if isinstance(raw_next, list) and raw_next and isinstance(raw_next[0], str):
            return raw_next[0]
        raise GraphExecutionError(f"Node '{node.get('node_id')}' has invalid 'next'.")

    def _evaluate_typed_condition(self, node: dict[str, Any], context: dict[str, object]) -> bool:
        left = self._resolve_value(node.get("left"), context)
        right = self._resolve_value(node.get("right"), context)
        operator = str(node.get("operator"))

        if operator == "eq":
            return left == right
        if operator == "ne":
            return left != right
        if operator == "gt":
            return bool(left > right)
        if operator == "gte":
            return bool(left >= right)
        if operator == "lt":
            return bool(left < right)
        if operator == "lte":
            return bool(left <= right)
        if operator == "in":
            try:
                return bool(left in right)
            except TypeError:
                return False
        if operator == "contains":
            try:
                return bool(right in left)
            except TypeError:
                return False

        raise GraphExecutionError(f"Unsupported condition operator '{operator}'.")

    def _resolve_value(self, value: object, context: dict[str, object]) -> object:
        if isinstance(value, str):
            match = TEMPLATE_VALUE_RE.fullmatch(value.strip())
            if not match:
                return value
            path = match.group(1).strip()
            return self._resolve_path(path, context)

        if isinstance(value, dict):
            return {key: self._resolve_value(item, context) for key, item in value.items()}

        if isinstance(value, list):
            return [self._resolve_value(item, context) for item in value]

        return value

    def _resolve_path(self, path: str, context: dict[str, object]) -> object:
        current: object = context
        for part in [piece for piece in path.split(".") if piece]:
            if isinstance(current, dict):
                current = current.get(part)
                continue
            if isinstance(current, list) and part.isdigit():
                idx = int(part)
                if idx < 0 or idx >= len(current):
                    return None
                current = current[idx]
                continue
            return None
        return current
