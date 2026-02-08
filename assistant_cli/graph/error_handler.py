from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ErrorHandlingDecision:
    action: str
    reason: str
    fallback_output: object | None = None
    fallback_next_node: str | None = None


class ErrorHandlerAnton:
    """Policy-bounded graph runtime remediation."""

    TRANSIENT_MARKERS = (
        "timed out",
        "timeout",
        "temporary",
        "connection reset",
        "503",
        "rate limit",
    )

    def decide(
        self,
        *,
        error: Exception,
        node: dict[str, Any],
        guarantee_mode: str,
        retry_count: int,
        max_retries: int,
        context: dict[str, Any],
    ) -> ErrorHandlingDecision:
        _ = context
        text = str(error).lower()
        node_type = str(node.get("type") or "")

        if retry_count < max_retries and self._looks_transient(text):
            return ErrorHandlingDecision(
                action="retry",
                reason=f"Transient error detected, retry {retry_count + 1}/{max_retries}.",
            )

        if node_type == "condition" and str(node.get("strategy")) == "llm_condition":
            options = node.get("branch_options")
            if isinstance(options, list) and options:
                if guarantee_mode in {"bounded", "flex"}:
                    selected = str(options[0])
                    return ErrorHandlingDecision(
                        action="fallback",
                        reason="LLM branch decision invalid; falling back to first allowed branch option.",
                        fallback_output={
                            "strategy": "llm_condition",
                            "decision": selected,
                            "next": node.get("if_true"),
                            "fallback": True,
                        },
                        fallback_next_node=str(node.get("if_true")),
                    )

        if guarantee_mode == "flex" and "fallback_output" in node and "next" in node:
            return ErrorHandlingDecision(
                action="fallback",
                reason="Flex mode fallback_output configured on node.",
                fallback_output=node.get("fallback_output"),
                fallback_next_node=self._resolve_next(node),
            )

        return ErrorHandlingDecision(
            action="fail",
            reason="No allowed remediation path matched the current error policy.",
        )

    def _looks_transient(self, text: str) -> bool:
        return any(marker in text for marker in self.TRANSIENT_MARKERS)

    def _resolve_next(self, node: dict[str, Any]) -> str | None:
        next_field = node.get("next")
        if isinstance(next_field, str):
            return next_field
        if isinstance(next_field, list) and next_field and isinstance(next_field[0], str):
            return next_field[0]
        return None
