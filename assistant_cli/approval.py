from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence


@dataclass(slots=True)
class ToolApprovalStatus:
    tool_name: str
    approval_required: bool


class ApprovalManager:
    """Manages global and per-tool approval requirements."""

    def __init__(self) -> None:
        self._global_enabled = False
        self._tool_overrides: dict[str, bool] = {}

    def set_global(self, enabled: bool) -> None:
        self._global_enabled = enabled

    def global_enabled(self) -> bool:
        return self._global_enabled

    def set_tool(self, tool_name: str, enabled: bool) -> None:
        self._tool_overrides[tool_name] = enabled

    def tool_enabled(self, tool_name: str) -> bool:
        return self._tool_overrides.get(tool_name, self._global_enabled)

    def list_statuses(self, tool_names: Sequence[str]) -> list[ToolApprovalStatus]:
        return [
            ToolApprovalStatus(tool_name=name, approval_required=self.tool_enabled(name))
            for name in sorted(tool_names)
        ]

    async def request_approval(
        self,
        tool_name: str,
        payload: dict,
        input_fn: Callable[[str], str] = input,
        approval_prompt: Callable[[str, dict], Awaitable[bool]] | None = None,
    ) -> bool:
        if not self.tool_enabled(tool_name):
            return True

        if approval_prompt is not None:
            return bool(await approval_prompt(tool_name, payload))

        print(f"\nTool: {tool_name}")
        print("Arguments:")
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))

        while True:
            answer = await asyncio.to_thread(
                input_fn,
                "Allow or Reject? [allow/reject]: ",
            )
            normalized = answer.strip().lower()
            if normalized in {"allow", "a", "yes", "y"}:
                return True
            if normalized in {"reject", "r", "no", "n"}:
                print("Tool call rejected, stopping")
                return False
            print("Please respond with 'allow' or 'reject'.")
