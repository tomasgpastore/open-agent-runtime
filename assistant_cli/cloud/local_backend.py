from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from assistant_cli.cloud.contracts import ExecutionBackend
from assistant_cli.graph import GraphExecutionResult, GraphExecutor


class LocalExecutionBackend(ExecutionBackend):
    """Local implementation of execution contracts; cloud backend can implement the same interface."""

    def __init__(self, *, executor: GraphExecutor, tool_map: dict[str, BaseTool] | None = None) -> None:
        self._executor = executor
        self._tool_map = tool_map or {}

    def set_tool_map(self, tool_map: dict[str, BaseTool]) -> None:
        self._tool_map = tool_map

    async def run_graph(
        self,
        *,
        graph: dict[str, Any],
        input_payload: object,
        guarantee_mode: str,
        trigger_source: str,
    ) -> GraphExecutionResult:
        return await self._executor.arun(
            graph=graph,
            input_payload=input_payload,
            guarantee_mode=guarantee_mode,
            tool_map=self._tool_map,
            metadata={"trigger_source": trigger_source},
        )

    async def replay_run(self, *, run_id: str) -> GraphExecutionResult:
        return await self._executor.replay(run_id=run_id, tool_map=self._tool_map)

    async def resume_run(self, *, run_id: str) -> GraphExecutionResult:
        return await self._executor.resume(run_id=run_id, tool_map=self._tool_map)
