from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.tools import BaseTool

from assistant_cli.graph.compiler.models import CompileDiagnostic


def run_tool_contract_pass(
    graph: dict[str, Any],
    tool_map: dict[str, BaseTool],
) -> list[CompileDiagnostic]:
    diagnostics: list[CompileDiagnostic] = []

    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return diagnostics

    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("type") != "tool":
            continue

        node_id = str(node.get("node_id") or "") or None
        tool_name = node.get("tool")
        if not isinstance(tool_name, str) or not tool_name.strip():
            diagnostics.append(
                CompileDiagnostic(
                    code="TOOL_NAME_MISSING",
                    severity="error",
                    message="Tool node requires a non-empty 'tool' field.",
                    node_id=node_id,
                )
            )
            continue

        tool = tool_map.get(tool_name)
        if tool is None:
            diagnostics.append(
                CompileDiagnostic(
                    code="TOOL_NOT_AVAILABLE",
                    severity="error",
                    message=f"Tool '{tool_name}' is not available in current active tools.",
                    node_id=node_id,
                    path="tool",
                )
            )
            continue

        diagnostics.extend(_validate_tool_args(node=node, tool=tool))

    return diagnostics


def _validate_tool_args(node: dict[str, Any], tool: BaseTool) -> list[CompileDiagnostic]:
    diagnostics: list[CompileDiagnostic] = []
    node_id = str(node.get("node_id") or "") or None

    payload = node.get("args")
    if payload is None:
        payload = node.get("parameters")

    props, required = _tool_schema_fields(tool)
    if payload is None:
        if required:
            diagnostics.append(
                CompileDiagnostic(
                    code="TOOL_ARGS_MISSING",
                    severity="warning",
                    message=(
                        f"Tool '{tool.name}' declares required fields {sorted(required)} but node has no args."
                    ),
                    node_id=node_id,
                    hint="Provide explicit args or ensure node input shape matches tool schema.",
                )
            )
        return diagnostics

    if not isinstance(payload, Mapping):
        if len(props) > 1:
            diagnostics.append(
                CompileDiagnostic(
                    code="TOOL_ARGS_SCALAR_WITH_OBJECT_SCHEMA",
                    severity="warning",
                    message=(
                        f"Tool '{tool.name}' expects object-style args, but node args are scalar/dynamic."
                    ),
                    node_id=node_id,
                    hint="Use an args object for static validation and safer execution.",
                )
            )
        return diagnostics

    arg_keys = set(payload.keys())
    unknown = sorted([key for key in arg_keys if props and key not in props])
    if unknown:
        diagnostics.append(
            CompileDiagnostic(
                code="TOOL_ARGS_UNKNOWN_KEYS",
                severity="warning",
                message=f"Args contain keys not present in tool schema: {unknown}.",
                node_id=node_id,
                hint="Verify the tool schema or rename argument keys.",
            )
        )

    missing = sorted([key for key in required if key not in arg_keys])
    if missing:
        diagnostics.append(
            CompileDiagnostic(
                code="TOOL_ARGS_MISSING_REQUIRED_KEYS",
                severity="warning",
                message=f"Args missing required tool schema keys: {missing}.",
                node_id=node_id,
                hint="Add required fields or ensure they are provided by upstream transform nodes.",
            )
        )

    for key, value in payload.items():
        schema = props.get(key)
        if not isinstance(schema, dict):
            continue
        if _schema_accepts_type(schema, "array") and isinstance(value, str) and "{{" not in value:
            diagnostics.append(
                CompileDiagnostic(
                    code="TOOL_ARGS_ARRAY_EXPECTED",
                    severity="warning",
                    message=(
                        f"Arg '{key}' expects array but literal string was provided. Runtime may coerce it."
                    ),
                    node_id=node_id,
                    path=f"args.{key}",
                )
            )

    return diagnostics


def _tool_schema_fields(tool: BaseTool) -> tuple[dict[str, dict[str, Any]], set[str]]:
    try:
        schema = tool.get_input_schema().model_json_schema()
    except Exception:  # noqa: BLE001
        return {}, set()

    raw_props = schema.get("properties")
    props: dict[str, dict[str, Any]] = {}
    if isinstance(raw_props, dict):
        for key, value in raw_props.items():
            if key == "kwargs":
                continue
            if isinstance(value, dict):
                props[key] = value

    required_raw = schema.get("required")
    required: set[str] = set()
    if isinstance(required_raw, list):
        required = {str(item) for item in required_raw if isinstance(item, str)}

    return props, required


def _schema_accepts_type(schema: dict[str, Any], expected_type: str) -> bool:
    type_field = schema.get("type")
    if isinstance(type_field, str) and type_field == expected_type:
        return True
    if isinstance(type_field, list) and expected_type in type_field:
        return True

    for variant_key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(variant_key)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if isinstance(variant, dict) and _schema_accepts_type(variant, expected_type):
                return True
    return False
