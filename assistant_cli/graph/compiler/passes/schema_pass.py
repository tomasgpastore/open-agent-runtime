from __future__ import annotations

import re
from typing import Any

from assistant_cli.graph.compiler.models import CompileDiagnostic
from assistant_cli.graph.schema import validate_graph_definition


NODE_ID_PATTERNS = [
    re.compile(r"Node '([^']+)"),
    re.compile(r"Condition node '([^']+)"),
    re.compile(r"State node '([^']+)"),
    re.compile(r"nodes\[\d+\] '([^']+)"),
    re.compile(r"Duplicate node_id '([^']+)"),
]


def run_schema_pass(graph: dict[str, Any]) -> list[CompileDiagnostic]:
    diagnostics: list[CompileDiagnostic] = []
    errors = validate_graph_definition(graph)
    for error in errors:
        diagnostics.append(
            CompileDiagnostic(
                code="SCHEMA_VALIDATION_FAILED",
                severity="error",
                message=error,
                node_id=_extract_node_id(error),
                hint="Fix schema issues before execution.",
            )
        )
    return diagnostics


def _extract_node_id(message: str) -> str | None:
    for pattern in NODE_ID_PATTERNS:
        match = pattern.search(message)
        if match:
            return match.group(1)
    return None
