from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from assistant_cli.graph.contracts import validate_contract_schema


ALLOWED_GUARANTEE_MODES = {"strict", "bounded", "flex"}
ALLOWED_AI_EDGE_POLICIES = {"always", "auto", "never"}
ALLOWED_NODE_TYPES = {
    "start",
    "end",
    "condition",
    "transform",
    "tool",
    "ai_template",
    "read_state",
    "write_state",
    "read_prior_runs",
}

SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "graph.schema.json"


class GraphValidationError(ValueError):
    """Raised when a graph definition fails validation."""



def load_graph_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))



def validate_graph_definition(graph: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    graph_id = graph.get("id")
    if not isinstance(graph_id, str) or not graph_id.strip():
        errors.append("Top-level field 'id' must be a non-empty string.")

    name = graph.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("Top-level field 'name' must be a non-empty string.")

    start = graph.get("start")
    if not isinstance(start, str) or not start.strip():
        errors.append("Top-level field 'start' must be a non-empty string.")

    guarantee_mode = (
        (graph.get("execution_defaults") or {}).get("guarantee_mode")
        if isinstance(graph.get("execution_defaults"), dict)
        else None
    )
    if guarantee_mode is not None and guarantee_mode not in ALLOWED_GUARANTEE_MODES:
        errors.append(
            "execution_defaults.guarantee_mode must be one of: strict, bounded, flex."
        )
    ai_edge_policy = (
        (graph.get("execution_defaults") or {}).get("ai_edge_policy")
        if isinstance(graph.get("execution_defaults"), dict)
        else None
    )
    if ai_edge_policy is not None and ai_edge_policy not in ALLOWED_AI_EDGE_POLICIES:
        errors.append(
            "execution_defaults.ai_edge_policy must be one of: always, auto, never."
        )

    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        errors.append("Top-level field 'nodes' must be a non-empty list.")
        return errors

    node_ids: set[str] = set()
    node_map: dict[str, dict[str, Any]] = {}

    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"nodes[{index}] must be an object.")
            continue

        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip():
            errors.append(f"nodes[{index}].node_id must be a non-empty string.")
            continue

        if node_id in node_ids:
            errors.append(f"Duplicate node_id '{node_id}'.")
            continue

        node_type = node.get("type")
        if node_type not in ALLOWED_NODE_TYPES:
            errors.append(
                f"nodes[{index}] '{node_id}' has invalid type '{node_type}'."
            )
            continue

        node_ids.add(node_id)
        node_map[node_id] = node

        _validate_node_contracts(node=node, errors=errors, index=index)

        if node_type == "condition":
            strategy = node.get("strategy")
            if strategy not in {"typed_condition", "llm_condition"}:
                errors.append(
                    f"Condition node '{node_id}' must define strategy as 'typed_condition' or 'llm_condition'."
                )
            if not isinstance(node.get("if_true"), str) or not node.get("if_true"):
                errors.append(f"Condition node '{node_id}' is missing non-empty 'if_true'.")
            if not isinstance(node.get("if_false"), str) or not node.get("if_false"):
                errors.append(f"Condition node '{node_id}' is missing non-empty 'if_false'.")

            if strategy == "typed_condition":
                if not isinstance(node.get("operator"), str):
                    errors.append(f"Condition node '{node_id}' requires string 'operator'.")
                if "left" not in node:
                    errors.append(f"Condition node '{node_id}' requires 'left'.")
                if "right" not in node:
                    errors.append(f"Condition node '{node_id}' requires 'right'.")

            if strategy == "llm_condition":
                branch_options = node.get("branch_options")
                if (
                    not isinstance(branch_options, list)
                    or not branch_options
                    or not all(isinstance(value, str) and value for value in branch_options)
                ):
                    errors.append(
                        f"Condition node '{node_id}' with llm_condition requires non-empty string list 'branch_options'."
                    )

        if node_type in {"read_state", "write_state", "read_prior_runs"}:
            if not _state_access_enabled(graph, node):
                errors.append(
                    f"State node '{node_id}' requires state access to be explicitly enabled. "
                    "Set state_access.default_enabled=true or node.state_access_enabled=true."
                )

        if node_type in {"read_state", "write_state"}:
            key = node.get("key")
            if not isinstance(key, str) or not key.strip():
                errors.append(f"State node '{node_id}' requires non-empty string 'key'.")

    if isinstance(start, str) and start not in node_ids:
        errors.append(f"Start node '{start}' does not exist in nodes list.")

    for node_id, node in node_map.items():
        node_type = node.get("type")
        if node_type == "end":
            continue

        if node_type == "condition":
            for key in ("if_true", "if_false"):
                target = node.get(key)
                if isinstance(target, str) and target not in node_ids:
                    errors.append(
                        f"Condition node '{node_id}' references missing target '{target}' in '{key}'."
                    )
            continue

        next_field = node.get("next")
        if next_field is None:
            errors.append(f"Node '{node_id}' is missing 'next'.")
            continue

        targets = [next_field] if isinstance(next_field, str) else next_field
        if not isinstance(targets, list) or not targets:
            errors.append(f"Node '{node_id}' has invalid 'next' field.")
            continue
        for target in targets:
            if not isinstance(target, str) or target not in node_ids:
                errors.append(f"Node '{node_id}' references missing next node '{target}'.")

    return errors



def validate_graph_or_raise(graph: dict[str, Any]) -> None:
    errors = validate_graph_definition(graph)
    if errors:
        rendered = "\n".join(f"- {error}" for error in errors)
        raise GraphValidationError(f"Graph validation failed:\n{rendered}")



def _state_access_enabled(graph: dict[str, Any], node: dict[str, Any]) -> bool:
    graph_default = False
    state_access = graph.get("state_access")
    if isinstance(state_access, dict):
        graph_default = bool(state_access.get("default_enabled", False))

    if "state_access_enabled" in node:
        return bool(node.get("state_access_enabled"))
    return graph_default


def _validate_node_contracts(*, node: dict[str, Any], errors: list[str], index: int) -> None:
    node_id = str(node.get("node_id"))
    node_type = str(node.get("type"))

    if _requires_input_schema(node_type):
        input_schema = node.get("input_schema")
        if not isinstance(input_schema, dict):
            errors.append(f"nodes[{index}] '{node_id}' requires object field 'input_schema'.")
        else:
            errors.extend(
                validate_contract_schema(
                    input_schema,
                    path=f"nodes[{index}].input_schema",
                )
            )

    if _requires_output_schema(node_type):
        output_schema = node.get("output_schema")
        if not isinstance(output_schema, dict):
            errors.append(f"nodes[{index}] '{node_id}' requires object field 'output_schema'.")
        else:
            errors.extend(
                validate_contract_schema(
                    output_schema,
                    path=f"nodes[{index}].output_schema",
                )
            )


def _requires_input_schema(node_type: str) -> bool:
    return node_type != "start"


def _requires_output_schema(node_type: str) -> bool:
    return node_type != "end"
