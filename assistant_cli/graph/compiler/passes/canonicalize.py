from __future__ import annotations

import copy
from typing import Any

from assistant_cli.graph.compiler.models import CompileDiagnostic


def run_canonicalize_pass(graph: dict[str, Any]) -> tuple[dict[str, Any], list[CompileDiagnostic]]:
    cloned = copy.deepcopy(graph)
    diagnostics: list[CompileDiagnostic] = []

    nodes = cloned.get("nodes")
    if not isinstance(nodes, list):
        return cloned, diagnostics

    normalized_nodes: list[dict[str, Any] | Any] = []
    for node in nodes:
        if not isinstance(node, dict):
            normalized_nodes.append(node)
            continue

        normalized = dict(node)
        node_id = str(normalized.get("node_id") or "") or None

        if "tool" not in normalized and isinstance(normalized.get("tool_name"), str):
            normalized["tool"] = normalized.pop("tool_name")
            diagnostics.append(
                CompileDiagnostic(
                    code="CANONICAL_TOOL_ALIAS",
                    severity="info",
                    message="Canonicalized 'tool_name' to 'tool'.",
                    node_id=node_id,
                    path="tool_name",
                )
            )

        if "args" not in normalized and "parameters" in normalized:
            normalized["args"] = normalized.pop("parameters")
            diagnostics.append(
                CompileDiagnostic(
                    code="CANONICAL_PARAMETERS_ALIAS",
                    severity="info",
                    message="Canonicalized 'parameters' to 'args'.",
                    node_id=node_id,
                    path="parameters",
                )
            )

        if "prompt_template" not in normalized and isinstance(normalized.get("prompt"), str):
            normalized["prompt_template"] = normalized.pop("prompt")
            diagnostics.append(
                CompileDiagnostic(
                    code="CANONICAL_PROMPT_ALIAS",
                    severity="info",
                    message="Canonicalized 'prompt' to 'prompt_template'.",
                    node_id=node_id,
                    path="prompt",
                )
            )

        next_field = normalized.get("next")
        if isinstance(next_field, list) and len(next_field) == 1 and isinstance(next_field[0], str):
            normalized["next"] = next_field[0]
            diagnostics.append(
                CompileDiagnostic(
                    code="CANONICAL_NEXT_SINGLETON",
                    severity="info",
                    message="Canonicalized single-item 'next' list to string.",
                    node_id=node_id,
                    path="next",
                )
            )

        normalized_nodes.append(normalized)

    cloned["nodes"] = normalized_nodes

    state_access = cloned.get("state_access")
    if not isinstance(state_access, dict):
        cloned["state_access"] = {"default_enabled": False}
        diagnostics.append(
            CompileDiagnostic(
                code="CANONICAL_STATE_ACCESS_DEFAULT",
                severity="info",
                message="Added default state_access block.",
                path="state_access",
            )
        )
    else:
        state_access.setdefault("default_enabled", False)

    return cloned, diagnostics
