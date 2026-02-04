from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from assistant_cli.approval import ApprovalManager
from assistant_cli.llm_client import LLMCallError, OllamaLLMClient


LOGGER = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a pragmatic personal assistant. "
    "Use tools when needed, including multiple tools in sequence for complex tasks. "
    "Avoid unnecessary tool calls. "
    "If a task is multi-step or may exceed context limits, use memory tools to write durable checkpoints "
    "(plans, constraints, intermediate results) and retrieve them before continuing. "
    "Do not assume long-term memory is in context unless explicitly retrieved. "
    "Return concise final answers and never expose hidden reasoning."
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int
    stop_reason: str | None


@dataclass(slots=True)
class AgentRunResult:
    messages: list[BaseMessage]
    final_answer: str
    stop_reason: str | None


class LangGraphAgent:
    def __init__(
        self,
        db_path: Path,
        llm_client: OllamaLLMClient,
        max_iterations: int,
        request_timeout_seconds: int,
        tool_timeout_seconds: int,
    ) -> None:
        self._llm_client = llm_client
        self._max_iterations = max_iterations
        self._request_timeout_seconds = request_timeout_seconds
        self._tool_timeout_seconds = tool_timeout_seconds

        self._checkpointer_cm = SqliteSaver.from_conn_string(str(db_path))
        self._checkpointer = self._checkpointer_cm.__enter__()

        self._graph = self._build_graph()

    def close(self) -> None:
        self._checkpointer_cm.__exit__(None, None, None)

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("router", self._router_node)
        workflow.add_node("tool_executor", self._tool_executor_node)

        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "tool_executor": "tool_executor",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "tool_executor",
            self._route_after_tools,
            {
                "router": "router",
                "end": END,
            },
        )

        return workflow.compile(checkpointer=self._checkpointer)

    async def _router_node(self, state: AgentState, config: RunnableConfig) -> dict:
        iteration = state.get("iteration", 0)
        max_iterations = int(config["configurable"]["max_iterations"])
        if iteration >= max_iterations:
            return {
                "stop_reason": "max_iterations",
                "messages": [
                    AIMessage(
                        content=(
                            "I reached the maximum tool iterations for this request. "
                            "Please refine the request or run again."
                        )
                    )
                ],
            }

        tools = list(config["configurable"]["tools"].values())
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]

        try:
            response = await self._llm_client.invoke(messages, tools=tools)
            return {
                "messages": [response],
                "iteration": iteration + 1,
            }
        except LLMCallError as exc:
            LOGGER.exception("LLM call failed")
            return {
                "stop_reason": "llm_error",
                "messages": [
                    AIMessage(
                        content=f"I could not complete the request due to an LLM error: {exc}"
                    )
                ],
            }

    async def _tool_executor_node(self, state: AgentState, config: RunnableConfig) -> dict:
        tools: dict[str, BaseTool] = config["configurable"]["tools"]
        approval_manager: ApprovalManager = config["configurable"]["approval_manager"]
        input_fn: Callable[[str], str] = config["configurable"].get("input_fn", input)

        last_ai_message = self._last_ai_message(state["messages"])
        if last_ai_message is None:
            return {}

        tool_calls = last_ai_message.tool_calls or []
        if not tool_calls:
            return {}

        tool_messages: list[ToolMessage] = []

        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call.get("args") or {}
            tool_call_id = call.get("id") or str(uuid.uuid4())

            tool = tools.get(tool_name)
            if tool is None:
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=f"Tool '{tool_name}' is unavailable or disabled.",
                    )
                )
                continue

            approved = await approval_manager.request_approval(
                tool_name=tool_name,
                payload=tool_args,
                input_fn=input_fn,
            )
            if not approved:
                return {
                    "stop_reason": "tool_rejected",
                }

            try:
                result = await asyncio.wait_for(
                    tool.ainvoke(tool_args),
                    timeout=float(config["configurable"]["tool_timeout_seconds"]),
                )
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=self._stringify_tool_result(result),
                    )
                )
            except TimeoutError:
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=f"Tool '{tool_name}' timed out.",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Tool call failed: %s", tool_name)
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=f"Tool '{tool_name}' failed: {exc}",
                    )
                )

        return {"messages": tool_messages}

    def _route_after_router(self, state: AgentState) -> str:
        if state.get("stop_reason"):
            return "end"

        last_ai_message = self._last_ai_message(state["messages"])
        if last_ai_message and last_ai_message.tool_calls:
            return "tool_executor"

        return "end"

    def _route_after_tools(self, state: AgentState) -> str:
        return "end" if state.get("stop_reason") else "router"

    def _last_ai_message(self, messages: list[BaseMessage]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    def _stringify_tool_result(self, result: object) -> str:
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, indent=2)
        except TypeError:
            return str(result)

    async def run(
        self,
        messages: list[BaseMessage],
        tools: dict[str, BaseTool],
        approval_manager: ApprovalManager,
        input_fn: Callable[[str], str] = input,
    ) -> AgentRunResult:
        thread_id = f"request-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "tools": tools,
                "approval_manager": approval_manager,
                "input_fn": input_fn,
                "max_iterations": self._max_iterations,
                "tool_timeout_seconds": self._tool_timeout_seconds,
            }
        }

        initial_state: AgentState = {
            "messages": messages,
            "iteration": 0,
            "stop_reason": None,
        }

        final_state: AgentState = await asyncio.wait_for(
            self._graph.ainvoke(initial_state, config=config),
            timeout=self._request_timeout_seconds,
        )

        final_messages = final_state["messages"]
        final_answer = self._extract_final_answer(final_messages)

        return AgentRunResult(
            messages=final_messages,
            final_answer=final_answer,
            stop_reason=final_state.get("stop_reason"),
        )

    def _extract_final_answer(self, messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                content = message.content
                if isinstance(content, str):
                    return content
                return str(content)
        return "I could not produce a final answer."
