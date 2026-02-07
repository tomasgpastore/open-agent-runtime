from __future__ import annotations

import asyncio
import json
import logging
import platform
import uuid
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Awaitable, Callable, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from assistant_cli.approval import ApprovalManager
from assistant_cli.llm_client import LLMCallError, LLMClient, LLMToolUnsupportedError
from assistant_cli.skills_manager import SkillManager


LOGGER = logging.getLogger(__name__)
MAX_IDENTICAL_TOOL_CALLS_PER_TURN = 2
MAX_TOTAL_TOOL_CALLS_PER_TURN = 8
MAX_TOOL_RESULT_CHARS = 12000
INTERNAL_TOOL_ARG_KEYS = {"run_manager", "callbacks", "config"}
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "anton-0.1.md"


def _build_system_prompt(tool_names: list[str], skills_prompt: str) -> str:
    now = datetime.now().astimezone()
    now_timestamp = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    current_date = now.strftime("%A, %B %d, %Y")
    os_info = platform.platform()
    current_working_directory = str(Path.cwd())
    tool_list = ", ".join(sorted(tool_names)) if tool_names else "none"
    template = _load_system_prompt_template()
    rendered = (
        template.replace("{{currentDateTime}}", now_timestamp)
        .replace("{{current_date}}", current_date)
        .replace("{{osInfo}}", os_info)
        .replace("{{currentWorkingDirectory}}", current_working_directory)
        .replace("{{toolList}}", tool_list)
    )
    if "{{skillsList}}" in rendered:
        return rendered.replace("{{skillsList}}", skills_prompt)
    if skills_prompt:
        return f"{rendered}\n\n{skills_prompt}"
    return rendered


def _load_system_prompt_template() -> str:
    try:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        return (
            "You are Anton, a pragmatic personal assistant. "
            "Use tools when needed, including multiple tools in sequence for complex tasks. "
            "Avoid unnecessary tool calls. "
            "Do not repeat an identical tool call with the same arguments unless a transient error occurred. "
            "If a task is multi-step or may exceed context limits, use memory tools to write durable checkpoints "
            "(plans, constraints, intermediate results) and retrieve them before continuing. "
            "Do not assume long-term memory is in context unless explicitly retrieved. "
            "Return concise final answers and never expose hidden reasoning. "
            "Runtime context: current_time={{currentDateTime}}; os={{osInfo}}; cwd={{currentWorkingDirectory}}. "
            "Available tools: {{toolList}}."
        )


TOOL_RETRY_SYSTEM_PROMPT = (
    "Policy correction: the previous draft did not satisfy grounding rules. "
    "For requests asking for web search, live/current data, weather-now checks, or explicit rechecks, "
    "you MUST call at least one relevant external tool before answering. "
    "If no relevant tool is available, explicitly say you cannot verify live data."
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
        llm_client: LLMClient,
        max_iterations: int,
        request_timeout_seconds: int,
        tool_timeout_seconds: int,
        skill_manager: SkillManager | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._max_iterations = max_iterations
        self._request_timeout_seconds = request_timeout_seconds
        self._tool_timeout_seconds = tool_timeout_seconds
        self._db_path = db_path
        self._skill_manager = skill_manager

        self._graph = None

    def set_llm_client(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def _ensure_graph(self) -> None:
        if self._graph is not None:
            return
        self._graph = self._build_graph()

    async def aclose(self) -> None:
        self._graph = None

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

        return workflow.compile()

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
        tool_names = [tool.name for tool in tools]
        skill_manager: SkillManager | None = config["configurable"].get("skill_manager")
        skills_prompt = skill_manager.available_skills_prompt() if skill_manager else "(none)"
        latest_user_text = self._latest_user_message_text(state["messages"])
        activation_prompt = None
        if skill_manager and latest_user_text:
            activation_prompt = skill_manager.activation_prompt(latest_user_text)
        requires_tool = self._requires_external_tool(state["messages"])
        needs_fresh_tool_call = requires_tool and not self._has_tool_message_since_latest_user(
            state["messages"]
        )
        stream_callback: Callable[[str], None] | None = config["configurable"].get(
            "stream_callback"
        )
        # Do not stream drafts for turns that must be grounded via tool use.
        if needs_fresh_tool_call:
            stream_callback = None
        messages = [SystemMessage(content=_build_system_prompt(tool_names, skills_prompt))]
        if activation_prompt:
            messages.append(SystemMessage(content=activation_prompt))
        messages.extend(state["messages"])

        if needs_fresh_tool_call and not self._has_relevant_external_tool(tool_names):
            return {
                "stop_reason": "tool_unavailable",
                "messages": [
                    AIMessage(
                        content=(
                            "I can't verify that reliably because no web/fetch tool is connected. "
                            "Use /mcp to enable a web-capable MCP server, then ask again."
                        )
                    )
                ],
            }

        try:
            response = await self._llm_client.invoke(
                messages,
                tools=tools,
                on_token=stream_callback,
            )
            if needs_fresh_tool_call and not response.tool_calls:
                retry_messages = [
                    SystemMessage(content=TOOL_RETRY_SYSTEM_PROMPT),
                    *messages,
                    response,
                    HumanMessage(content="Policy check: call a relevant tool now."),
                ]
                response = await self._llm_client.invoke(
                    retry_messages,
                    tools=tools,
                    on_token=None,
                )
                if not response.tool_calls:
                    return {
                        "stop_reason": "tool_required_no_call",
                        "messages": [
                            AIMessage(
                                content=(
                                    "I need to call a web/fetch tool to answer that reliably, "
                                    "and I couldn't produce a valid grounded tool call. "
                                    "Please retry or check /mcp tool availability."
                                )
                            )
                        ],
                    }
            return {
                "messages": [response],
                "iteration": iteration + 1,
            }
        except LLMToolUnsupportedError:
            if needs_fresh_tool_call:
                return {
                    "stop_reason": "llm_tools_unsupported",
                    "messages": [
                        AIMessage(
                            content=(
                                f"The configured model '{self._llm_client.model_name}' does not support "
                                "tool calling, so I can't verify live data with MCP tools. "
                                "Please switch OLLAMA_MODEL to a tool-capable model and retry."
                            )
                        )
                    ],
                }
            # Fall back to plain chat for non-tool turns.
            response = await self._llm_client.invoke(
                messages,
                tools=None,
                on_token=stream_callback,
            )
            return {
                "messages": [response],
                "iteration": iteration + 1,
            }
        except LLMCallError as exc:
            LOGGER.error("LLM call failed: %s", exc)
            return {
                "stop_reason": "llm_error",
                "messages": [
                    AIMessage(
                        content=self._format_llm_error_message(exc)
                    )
                ],
            }

    async def _tool_executor_node(self, state: AgentState, config: RunnableConfig) -> dict:
        tools: dict[str, BaseTool] = config["configurable"]["tools"]
        approval_manager: ApprovalManager = config["configurable"]["approval_manager"]
        input_fn: Callable[[str], str] = config["configurable"].get("input_fn", input)
        tool_event_callback: Callable[[object], None] | None = config["configurable"].get(
            "tool_event_callback"
        )
        approval_prompt: Callable[[str, dict], Awaitable[bool]] | None = config["configurable"].get(
            "approval_prompt"
        )

        last_ai_message = self._last_ai_message(state["messages"])
        if last_ai_message is None:
            return {}

        tool_calls = last_ai_message.tool_calls or []
        if not tool_calls:
            return {}

        total_calls_current_turn = self._count_total_tool_calls_current_turn(state["messages"])
        if total_calls_current_turn + len(tool_calls) > MAX_TOTAL_TOOL_CALLS_PER_TURN:
            return {
                "stop_reason": "tool_loop_detected",
                "messages": [
                    AIMessage(
                        content=(
                            "I stopped because this turn triggered too many tool calls without converging. "
                            "Please narrow the request or provide a specific allowed path and try again."
                        )
                    )
                ],
            }

        tool_messages: list[ToolMessage] = []

        for call in tool_calls:
            tool_name = call["name"]
            raw_tool_args = call.get("args") or {}
            tool_args = deepcopy(raw_tool_args)
            event_args = self._sanitize_tool_args(tool_args)
            tool_call_id = call.get("id") or str(uuid.uuid4())
            repeated_count = self._count_tool_call_occurrences_current_turn(
                state["messages"], tool_name, event_args
            )
            if repeated_count > MAX_IDENTICAL_TOOL_CALLS_PER_TURN:
                return {
                    "stop_reason": "tool_loop_detected",
                    "messages": [
                        AIMessage(
                            content=(
                                f"I detected a tool loop: `{tool_name}` was called with the same arguments "
                                f"{repeated_count} times in this turn. I stopped to avoid wasting cycles. "
                                "Please refine the request or run it again."
                            )
                        )
                    ],
                }
            tool = tools.get(tool_name)
            if tool is None:
                self._emit_tool_event(tool_event_callback, tool_name, event_args, status="ERROR")
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=f"Tool '{tool_name}' is unavailable or disabled.",
                    )
                )
                continue

            approved = await approval_manager.request_approval(
                tool_name=tool_name,
                payload=event_args if isinstance(event_args, dict) else {"args": event_args},
                input_fn=input_fn,
                approval_prompt=approval_prompt,
            )
            if not approved:
                self._emit_tool_event(tool_event_callback, tool_name, event_args, status="ERROR")
                return {
                    "stop_reason": "tool_rejected",
                }

            try:
                self._emit_tool_event(tool_event_callback, tool_name, event_args, status="RUNNING")
                result = await asyncio.wait_for(
                    tool.ainvoke(tool_args),
                    timeout=float(config["configurable"]["tool_timeout_seconds"]),
                )
                self._emit_tool_event(tool_event_callback, tool_name, event_args, status="OK")
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=self._stringify_tool_result(result),
                    )
                )
            except TimeoutError:
                self._emit_tool_event(tool_event_callback, tool_name, event_args, status="ERROR")
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=f"Tool '{tool_name}' timed out.",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Tool call failed: %s (%s)", tool_name, exc)
                error_text = str(exc)
                self._emit_tool_event(tool_event_callback, tool_name, event_args, status="ERROR")
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=f"Tool '{tool_name}' failed: {error_text}",
                    )
                )
                if self._is_non_retryable_tool_error(tool_name, error_text):
                    return {
                        "stop_reason": "tool_error_non_retryable",
                        "messages": [
                            *tool_messages,
                            AIMessage(
                                content=self._format_non_retryable_tool_error(tool_name, error_text)
                            ),
                        ],
                    }

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
            text = result
        else:
            try:
                text = json.dumps(result, ensure_ascii=False, indent=2)
            except TypeError:
                text = str(result)

        if len(text) > MAX_TOOL_RESULT_CHARS:
            clipped = text[:MAX_TOOL_RESULT_CHARS]
            omitted = len(text) - MAX_TOOL_RESULT_CHARS
            return f"{clipped}\n\n[tool output truncated: {omitted} chars omitted]"
        return text

    async def run(
        self,
        messages: list[BaseMessage],
        tools: dict[str, BaseTool],
        approval_manager: ApprovalManager,
        input_fn: Callable[[str], str] = input,
        stream_callback: Callable[[str], None] | None = None,
        tool_event_callback: Callable[[object], None] | None = None,
        approval_prompt: Callable[[str, dict], Awaitable[bool]] | None = None,
    ) -> AgentRunResult:
        await self._ensure_graph()
        thread_id = f"request-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "tools": tools,
                "approval_manager": approval_manager,
                "input_fn": input_fn,
                "stream_callback": stream_callback,
                "tool_event_callback": tool_event_callback,
                "approval_prompt": approval_prompt,
                "max_iterations": self._max_iterations,
                "tool_timeout_seconds": self._tool_timeout_seconds,
                "skill_manager": self._skill_manager,
            }
        }

        initial_state: AgentState = {
            "messages": messages,
            "iteration": 0,
            "stop_reason": None,
        }

        if self._graph is None:
            raise RuntimeError("Agent graph was not initialized.")

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

    def _requires_external_tool(self, messages: list[BaseMessage]) -> bool:
        latest_user = self._latest_user_message_text(messages)
        if not latest_user:
            return False

        text = latest_user.lower()
        explicit_web = any(
            phrase in text
            for phrase in (
                "search the web",
                "search web",
                "look it up online",
                "browse the web",
                "web search",
                "please recheck",
                "recheck that",
            )
        )
        live_weather = (
            any(word in text for word in ("weather", "temperature", "forecast"))
            and any(word in text for word in ("now", "current", "right now", "today", "latest"))
        )
        return explicit_web or live_weather

    def _format_llm_error_message(self, exc: Exception) -> str:
        text = str(exc)
        lowered = text.lower()
        if "openrouter" in lowered or "key limit exceeded" in lowered or "insufficient" in lowered:
            return (
                "OpenRouter rejected the request due to key limits or insufficient credits. "
                "Add credits or raise the key limit at OpenRouter settings, or lower "
                "OPENROUTER_MAX_COMPLETION_TOKENS. You can also switch providers with `/llm`."
            )
        if "quota" in lowered or "billing" in lowered:
            return (
                "The LLM provider rejected this request due to quota or billing limits. "
                "Check your provider account, or switch providers with `/llm`."
            )
        return (
            "I hit an LLM backend error and couldn't finish this request. "
            "Try `/new` to reset the session, or switch model/provider with `/llm`."
        )

    def _latest_user_message_text(self, messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                if isinstance(message.content, str):
                    return message.content
                return str(message.content)
        return ""

    def _has_relevant_external_tool(self, tool_names: list[str]) -> bool:
        return any(
            any(marker in name.lower() for marker in ("web", "search", "fetch", "http"))
            for name in tool_names
        )

    def _has_tool_message_since_latest_user(self, messages: list[BaseMessage]) -> bool:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, ToolMessage):
                return True
        return False

    def _count_tool_call_occurrences_current_turn(
        self,
        messages: list[BaseMessage],
        tool_name: str,
        tool_args: object,
    ) -> int:
        signature = self._tool_call_signature(tool_name, tool_args)
        count = 0
        for message in self._messages_in_current_turn(messages):
            if not isinstance(message, AIMessage) or not message.tool_calls:
                continue
            for call in message.tool_calls:
                if not isinstance(call, dict):
                    continue
                if call.get("name") != tool_name:
                    continue
                call_args = call.get("args") or {}
                if self._tool_call_signature(tool_name, call_args) == signature:
                    count += 1
        return count

    def _messages_in_current_turn(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        recent: list[BaseMessage] = []
        seen_latest_user = False
        for message in reversed(messages):
            recent.append(message)
            if isinstance(message, HumanMessage):
                seen_latest_user = True
                break
        if not seen_latest_user:
            return list(messages)
        recent.reverse()
        return recent

    def _count_total_tool_calls_current_turn(self, messages: list[BaseMessage]) -> int:
        count = 0
        for message in self._messages_in_current_turn(messages):
            if isinstance(message, AIMessage) and message.tool_calls:
                count += len(message.tool_calls)
        return count

    def _tool_call_signature(self, tool_name: str, tool_args: object) -> str:
        sanitized_args = self._sanitize_tool_args(tool_args)
        try:
            payload = json.dumps(sanitized_args, sort_keys=True, ensure_ascii=False)
        except TypeError:
            payload = str(sanitized_args)
        return f"{tool_name}:{payload}"

    def _is_non_retryable_tool_error(self, tool_name: str, error_text: str) -> bool:
        lowered_tool = tool_name.lower()
        lowered_error = error_text.lower()
        is_filesystem_tool = any(
            marker in lowered_tool
            for marker in (
                "list_directory",
                "read_file",
                "write_file",
                "filesystem",
            )
        )
        has_permission_denied = any(
            marker in lowered_error
            for marker in (
                "access denied",
                "permission denied",
                "outside allowed directories",
            )
        )
        return is_filesystem_tool and has_permission_denied

    def _format_non_retryable_tool_error(self, tool_name: str, error_text: str) -> str:
        lowered_error = error_text.lower()
        if "outside allowed directories" in lowered_error:
            return (
                f"`{tool_name}` was blocked by filesystem path restrictions. "
                "Use `/paths` to view allowed directories, then `/paths add <directory>` "
                "or run the request against a currently allowed path and retry."
            )
        return (
            f"`{tool_name}` failed due to a non-retryable permission error. "
            "Adjust filesystem permissions/allowed paths, then retry."
        )

    def _emit_tool_event(
        self,
        callback: Callable[[object], None] | None,
        tool_name: str,
        tool_args: object,
        status: str,
    ) -> None:
        if callback is None:
            return
        payload = {
            "tool_name": tool_name,
            "args": tool_args,
            "status": status,
        }
        try:
            callback(payload)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Tool event callback failed", exc_info=True)

    def _sanitize_tool_args(self, value: object) -> object:
        if isinstance(value, dict):
            sanitized: dict[str, object] = {}
            for key, item in value.items():
                if key in INTERNAL_TOOL_ARG_KEYS:
                    continue
                sanitized[key] = self._sanitize_tool_args(item)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize_tool_args(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_tool_args(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return f"<{type(value).__name__}>"
