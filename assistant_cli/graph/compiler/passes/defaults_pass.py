from __future__ import annotations

from typing import Any

from assistant_cli.graph.compiler.models import CompileDiagnostic, CompileTargetMode


DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_MAX_RETRIES = 1


def run_defaults_pass(graph: dict[str, Any], mode: CompileTargetMode) -> tuple[dict[str, Any], list[CompileDiagnostic]]:
    diagnostics: list[CompileDiagnostic] = []

    execution_defaults = graph.get("execution_defaults")
    if not isinstance(execution_defaults, dict):
        execution_defaults = {}
        graph["execution_defaults"] = execution_defaults
    execution_defaults["guarantee_mode"] = mode

    state_access = graph.get("state_access")
    if not isinstance(state_access, dict):
        state_access = {"default_enabled": False}
        graph["state_access"] = state_access
    state_access.setdefault("default_enabled", False)

    graph_id = str(graph.get("id") or "graph")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return graph, diagnostics

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = node.get("type")
        if node_type not in {"tool", "ai_template"}:
            continue

        node_id = str(node.get("node_id") or "") or None
        if mode in {"strict", "bounded"} and "timeout_seconds" not in node:
            node["timeout_seconds"] = DEFAULT_TIMEOUT_SECONDS
            diagnostics.append(
                CompileDiagnostic(
                    code="DEFAULT_TIMEOUT_INJECTED",
                    severity="warning",
                    message=f"Injected timeout_seconds={DEFAULT_TIMEOUT_SECONDS}.",
                    node_id=node_id,
                    path="timeout_seconds",
                )
            )

        if mode in {"strict", "bounded"} and "max_retries" not in node:
            node["max_retries"] = DEFAULT_MAX_RETRIES
            diagnostics.append(
                CompileDiagnostic(
                    code="DEFAULT_RETRIES_INJECTED",
                    severity="warning",
                    message=f"Injected max_retries={DEFAULT_MAX_RETRIES}.",
                    node_id=node_id,
                    path="max_retries",
                )
            )

        if mode in {"strict", "bounded"} and "idempotency_key" not in node:
            generated = f"{graph_id}:{node.get('node_id')}"
            node["idempotency_key"] = generated
            diagnostics.append(
                CompileDiagnostic(
                    code="DEFAULT_IDEMPOTENCY_INJECTED",
                    severity="warning",
                    message="Injected idempotency_key for deterministic retries.",
                    node_id=node_id,
                    path="idempotency_key",
                )
            )

    return graph, diagnostics
