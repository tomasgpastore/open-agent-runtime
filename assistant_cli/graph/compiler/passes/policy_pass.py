from __future__ import annotations

from typing import Any

from assistant_cli.graph.compiler.models import CompileDiagnostic, CompileTargetMode


def run_policy_pass(graph: dict[str, Any], mode: CompileTargetMode) -> list[CompileDiagnostic]:
    diagnostics: list[CompileDiagnostic] = []

    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return diagnostics

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or "") or None
        node_type = node.get("type")

        if node_type in {"tool", "ai_template"}:
            if mode in {"strict", "bounded"} and "timeout_seconds" not in node:
                diagnostics.append(
                    CompileDiagnostic(
                        code="POLICY_TIMEOUT_REQUIRED",
                        severity="error",
                        message=f"{mode} mode requires timeout_seconds on {node_type} node.",
                        node_id=node_id,
                    )
                )
            if mode in {"strict", "bounded"} and "max_retries" not in node:
                diagnostics.append(
                    CompileDiagnostic(
                        code="POLICY_RETRIES_REQUIRED",
                        severity="error",
                        message=f"{mode} mode requires max_retries on {node_type} node.",
                        node_id=node_id,
                    )
                )
            if mode == "strict" and "idempotency_key" not in node:
                diagnostics.append(
                    CompileDiagnostic(
                        code="POLICY_IDEMPOTENCY_REQUIRED",
                        severity="error",
                        message="strict mode requires idempotency_key on tool/ai_template nodes.",
                        node_id=node_id,
                    )
                )

        if node_type == "condition" and mode == "strict":
            strategy = node.get("strategy")
            if strategy == "llm_condition":
                options = node.get("branch_options")
                if not isinstance(options, list) or len(options) < 2:
                    diagnostics.append(
                        CompileDiagnostic(
                            code="POLICY_BRANCH_OPTIONS_REQUIRED",
                            severity="error",
                            message="strict mode requires llm_condition with at least two branch_options.",
                            node_id=node_id,
                        )
                    )

    return diagnostics
