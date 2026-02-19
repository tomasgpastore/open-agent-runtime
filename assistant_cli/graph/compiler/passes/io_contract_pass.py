from __future__ import annotations

from typing import Any

from assistant_cli.graph.compiler.models import CompileDiagnostic
from assistant_cli.graph.contracts import validate_contract_schema


def run_io_contract_pass(graph: dict[str, Any]) -> list[CompileDiagnostic]:
    diagnostics: list[CompileDiagnostic] = []
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return diagnostics

    for node in nodes:
        if not isinstance(node, dict):
            continue

        node_id = str(node.get("node_id") or "") or None
        node_type = str(node.get("type") or "")

        input_schema = node.get("input_schema")
        output_schema = node.get("output_schema")

        if _requires_input_schema(node_type):
            if not isinstance(input_schema, dict):
                diagnostics.append(
                    CompileDiagnostic(
                        code="NODE_INPUT_SCHEMA_MISSING",
                        severity="error",
                        message=f"Node type '{node_type}' requires object field 'input_schema'.",
                        node_id=node_id,
                        path="input_schema",
                    )
                )
            else:
                diagnostics.extend(
                    _schema_diagnostics(
                        schema=input_schema,
                        node_id=node_id,
                        path="input_schema",
                        code="NODE_INPUT_SCHEMA_INVALID",
                    )
                )

        if _requires_output_schema(node_type):
            if not isinstance(output_schema, dict):
                diagnostics.append(
                    CompileDiagnostic(
                        code="NODE_OUTPUT_SCHEMA_MISSING",
                        severity="error",
                        message=f"Node type '{node_type}' requires object field 'output_schema'.",
                        node_id=node_id,
                        path="output_schema",
                    )
                )
            else:
                diagnostics.extend(
                    _schema_diagnostics(
                        schema=output_schema,
                        node_id=node_id,
                        path="output_schema",
                        code="NODE_OUTPUT_SCHEMA_INVALID",
                    )
                )

    return diagnostics


def _schema_diagnostics(
    *,
    schema: dict[str, Any],
    node_id: str | None,
    path: str,
    code: str,
) -> list[CompileDiagnostic]:
    return [
        CompileDiagnostic(
            code=code,
            severity="error",
            message=message,
            node_id=node_id,
            path=path,
            hint="Use a valid contract schema object (type/properties/required/items).",
        )
        for message in validate_contract_schema(schema, path=path)
    ]


def _requires_input_schema(node_type: str) -> bool:
    return node_type != "start"


def _requires_output_schema(node_type: str) -> bool:
    return node_type != "end"
