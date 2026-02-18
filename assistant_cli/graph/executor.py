from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime
import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from assistant_cli.graph.compiler import CompileOptions, CompiledGraph, GraphCompiler, render_diagnostics
from assistant_cli.graph.error_handler import ErrorHandlerAnton
from assistant_cli.graph.hooks import GraphHookRegistry
from assistant_cli.graph.schema import (
    ALLOWED_GUARANTEE_MODES,
    GraphValidationError,
    validate_graph_or_raise,
)
from assistant_cli.graph.state_store import GraphStateStore
from assistant_cli.llm_client import LLMClient


TEMPLATE_VALUE_RE = re.compile(r"^\{\{([^{}]+)\}\}$")
TEMPLATE_INLINE_RE = re.compile(r"\{\{([^{}]+)\}\}")


class GraphExecutionError(RuntimeError):
    """Raised when graph execution fails."""


@dataclass(slots=True)
class GraphExecutionResult:
    run_id: str
    graph_id: str
    guarantee_mode: str
    status: str
    output: object
    visited_nodes: list[str]


class GraphExecutor:
    """State-aware graph executor with hooks, retries, replay, and resume support."""

    def __init__(
        self,
        state_store: GraphStateStore,
        *,
        llm_client: LLMClient | None = None,
        hook_registry: GraphHookRegistry | None = None,
        error_handler: ErrorHandlerAnton | None = None,
        compiler: GraphCompiler | None = None,
        tool_timeout_seconds: float = 45.0,
    ) -> None:
        self._state_store = state_store
        self._llm_client = llm_client
        self._hooks = hook_registry or GraphHookRegistry()
        self._error_handler = error_handler or ErrorHandlerAnton()
        self._compiler = compiler or GraphCompiler()
        self._tool_timeout_seconds = max(1.0, float(tool_timeout_seconds))

    def set_llm_client(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    @property
    def hooks(self) -> GraphHookRegistry:
        return self._hooks

    async def arun(
        self,
        *,
        graph: dict[str, Any],
        input_payload: object | None = None,
        guarantee_mode: str = "bounded",
        tool_map: dict[str, BaseTool] | None = None,
        llm_client: LLMClient | None = None,
        compiled_graph: CompiledGraph | None = None,
        start_node_id: str | None = None,
        context_override: dict[str, object] | None = None,
        parent_run_id: str | None = None,
        resume_from_run_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> GraphExecutionResult:
        if guarantee_mode not in ALLOWED_GUARANTEE_MODES:
            raise GraphExecutionError(
                f"Unsupported guarantee_mode '{guarantee_mode}'. Use strict, bounded, or flex."
            )

        active_llm_client = llm_client or self._llm_client
        active_tools = tool_map or {}
        active_compiled = compiled_graph

        if active_compiled is None:
            compile_result = self._compiler.compile(
                graph=graph,
                tool_map=active_tools,
                options=CompileOptions(mode=guarantee_mode),
            )
            if not compile_result.ok or compile_result.compiled is None:
                rendered = render_diagnostics(compile_result.diagnostics)
                raise GraphExecutionError(f"Graph compilation failed:\n{rendered}")
            active_compiled = compile_result.compiled
            if compile_result.rewritten_graph is not None:
                graph = compile_result.rewritten_graph
        else:
            if active_compiled.mode != guarantee_mode:
                raise GraphExecutionError(
                    "Provided compiled_graph mode does not match requested guarantee_mode."
                )
            graph = active_compiled.graph

        validate_graph_or_raise(graph)
        self._validate_mode_policy(graph, guarantee_mode)

        graph_id = str(graph["id"])
        nodes = graph["nodes"]
        node_map = {str(node["node_id"]): node for node in nodes}
        max_steps = int(graph.get("max_steps", max(20, len(nodes) * 12)))

        run_metadata = dict(metadata or {})
        if active_compiled is not None:
            run_metadata["compile"] = {
                "compiler_version": active_compiled.compiler_version,
                "compile_hash": active_compiled.compile_hash,
                "mode": active_compiled.mode,
                "warning_count": len(active_compiled.warnings),
            }

        run_id = self._state_store.start_run(
            graph_id=graph_id,
            guarantee_mode=guarantee_mode,
            input_payload=input_payload,
            parent_run_id=parent_run_id,
            resume_from_run_id=resume_from_run_id,
            metadata=run_metadata,
        )

        current_node_id = start_node_id or str(graph["start"])
        context: dict[str, object] = dict(context_override or {})
        now = datetime.now().astimezone()
        context.update(
            {
                "input": input_payload,
                "run_id": run_id,
                "graph_id": graph_id,
            }
        )
        context.setdefault("date", now.date().isoformat())
        context.setdefault("datetime", now.isoformat())
        context.setdefault("now", now.isoformat())

        visited_nodes: list[str] = []
        node_retries: dict[str, int] = {}
        steps = 0
        last_output: object = input_payload

        await self._emit_hook(
            "before_run",
            {
                "run_id": run_id,
                "graph_id": graph_id,
                "guarantee_mode": guarantee_mode,
                "input": input_payload,
            },
        )

        try:
            while True:
                if steps >= max_steps:
                    raise GraphExecutionError(
                        f"Execution exceeded max_steps={max_steps}; possible graph loop."
                    )
                steps += 1

                node = node_map.get(current_node_id)
                if node is None:
                    raise GraphExecutionError(f"Node '{current_node_id}' was not found during execution.")

                node_type = str(node["type"])
                node_input = self._resolve_value(node.get("input", last_output), context)

                await self._emit_hook(
                    "before_node",
                    {
                        "run_id": run_id,
                        "graph_id": graph_id,
                        "node_id": current_node_id,
                        "node_type": node_type,
                        "step": steps,
                        "input": node_input,
                    },
                )

                retry_count = node_retries.get(current_node_id, 0)
                max_retries = int(node.get("max_retries", 0))
                status = "ok"
                output: object = None
                next_node: str | None = None

                while True:
                    try:
                        output, next_node = await self._execute_node(
                            node=node,
                            node_input=node_input,
                            context=context,
                            guarantee_mode=guarantee_mode,
                            tools=active_tools,
                            llm_client=active_llm_client,
                        )
                        break
                    except Exception as exc:  # noqa: BLE001
                        decision = self._error_handler.decide(
                            error=exc,
                            node=node,
                            guarantee_mode=guarantee_mode,
                            retry_count=retry_count,
                            max_retries=max_retries,
                            context=context,
                        )
                        await self._emit_hook(
                            "on_error",
                            {
                                "run_id": run_id,
                                "graph_id": graph_id,
                                "node_id": current_node_id,
                                "step": steps,
                                "error": str(exc),
                                "decision": decision.action,
                                "reason": decision.reason,
                            },
                        )

                        if decision.action == "retry":
                            retry_count += 1
                            node_retries[current_node_id] = retry_count
                            continue

                        if decision.action == "fallback":
                            status = "fallback"
                            output = decision.fallback_output
                            next_node = decision.fallback_next_node or self._first_next(node)
                            break

                        raise GraphExecutionError(
                            f"Node '{current_node_id}' failed: {exc}. Error handler: {decision.reason}"
                        ) from exc

                if node_type != "end" and next_node is None:
                    raise GraphExecutionError(f"Node '{current_node_id}' did not resolve a next node.")

                context[current_node_id] = output
                context["last"] = output
                last_output = output

                self._state_store.add_checkpoint(
                    run_id=run_id,
                    graph_id=graph_id,
                    node_id=current_node_id,
                    status=status,
                    input_payload=node_input,
                    output_payload=output,
                    context_payload=context,
                    next_node_id=next_node,
                    step=steps,
                    retry_count=retry_count,
                )

                await self._emit_hook(
                    "after_node",
                    {
                        "run_id": run_id,
                        "graph_id": graph_id,
                        "node_id": current_node_id,
                        "node_type": node_type,
                        "step": steps,
                        "status": status,
                        "output": output,
                        "next_node": next_node,
                    },
                )

                visited_nodes.append(current_node_id)

                if node_type == "end":
                    self._state_store.finish_run(
                        run_id=run_id,
                        status="succeeded",
                        output_payload=output,
                    )
                    result = GraphExecutionResult(
                        run_id=run_id,
                        graph_id=graph_id,
                        guarantee_mode=guarantee_mode,
                        status="succeeded",
                        output=output,
                        visited_nodes=visited_nodes,
                    )
                    await self._emit_hook(
                        "after_run",
                        {
                            "run_id": run_id,
                            "graph_id": graph_id,
                            "status": "succeeded",
                            "output": output,
                            "visited_nodes": visited_nodes,
                        },
                    )
                    return result

                current_node_id = next_node

        except Exception as exc:
            self._state_store.add_checkpoint(
                run_id=run_id,
                graph_id=graph_id,
                node_id=current_node_id,
                status="error",
                input_payload=last_output,
                output_payload=None,
                error_text=str(exc),
                context_payload=context,
                next_node_id=current_node_id,
                step=steps,
            )
            self._state_store.finish_run(
                run_id=run_id,
                status="failed",
                output_payload=None,
                error_text=str(exc),
            )
            await self._emit_hook(
                "after_run",
                {
                    "run_id": run_id,
                    "graph_id": graph_id,
                    "status": "failed",
                    "error": str(exc),
                    "visited_nodes": visited_nodes,
                },
            )
            if isinstance(exc, (GraphExecutionError, GraphValidationError)):
                raise
            raise GraphExecutionError(str(exc)) from exc

    async def replay(
        self,
        *,
        run_id: str,
        tool_map: dict[str, BaseTool] | None = None,
        llm_client: LLMClient | None = None,
    ) -> GraphExecutionResult:
        run = self._state_store.get_run(run_id)
        if run is None:
            raise GraphExecutionError(f"Run '{run_id}' was not found.")

        graph = self._state_store.get_graph_definition(run.graph_id)
        if graph is None:
            raise GraphExecutionError(
                f"Graph '{run.graph_id}' for run '{run_id}' is not stored in graph_definitions."
            )

        return await self.arun(
            graph=graph,
            input_payload=run.input_payload,
            guarantee_mode=run.guarantee_mode,
            tool_map=tool_map,
            llm_client=llm_client,
            parent_run_id=run.run_id,
            metadata={"replay_of": run.run_id},
        )

    async def resume(
        self,
        *,
        run_id: str,
        tool_map: dict[str, BaseTool] | None = None,
        llm_client: LLMClient | None = None,
    ) -> GraphExecutionResult:
        run = self._state_store.get_run(run_id)
        if run is None:
            raise GraphExecutionError(f"Run '{run_id}' was not found.")

        checkpoint = self._state_store.latest_error_checkpoint(run_id)
        if checkpoint is None:
            raise GraphExecutionError(f"Run '{run_id}' has no error checkpoint to resume from.")

        graph = self._state_store.get_graph_definition(run.graph_id)
        if graph is None:
            raise GraphExecutionError(
                f"Graph '{run.graph_id}' for run '{run_id}' is not stored in graph_definitions."
            )

        resume_context = checkpoint.context_payload
        if not isinstance(resume_context, dict):
            resume_context = {}

        return await self.arun(
            graph=graph,
            input_payload=run.input_payload,
            guarantee_mode=run.guarantee_mode,
            tool_map=tool_map,
            llm_client=llm_client,
            start_node_id=checkpoint.node_id,
            context_override=dict(resume_context),
            parent_run_id=run.run_id,
            resume_from_run_id=run.run_id,
            metadata={"resume_of": run.run_id, "checkpoint_id": checkpoint.id},
        )

    def run(self, **kwargs: Any) -> GraphExecutionResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError("GraphExecutor.run() cannot be called inside an active event loop. Use await arun().")
        return asyncio.run(self.arun(**kwargs))

    async def _execute_node(
        self,
        *,
        node: dict[str, Any],
        node_input: object,
        context: dict[str, object],
        guarantee_mode: str,
        tools: dict[str, BaseTool],
        llm_client: LLMClient | None,
    ) -> tuple[object, str | None]:
        node_type = str(node["type"])

        if node_type == "start":
            return node_input, self._first_next(node)

        if node_type == "end":
            return node_input, self._first_next(node, allow_missing=True)

        if node_type == "transform":
            return self._resolve_value(node.get("value", node_input), context), self._first_next(node)

        if node_type == "read_state":
            key = str(node["key"])
            return {"key": key, "value": self._state_store.read_state(graph_id=str(context["graph_id"]), key=key)}, self._first_next(node)

        if node_type == "write_state":
            key = str(node["key"])
            value_source = node.get("value", node_input)
            value = self._resolve_value(value_source, context)
            self._state_store.write_state(
                graph_id=str(context["graph_id"]),
                key=key,
                value=value,
                source_run_id=str(context["run_id"]),
                source_node_id=str(node["node_id"]),
            )
            return {"key": key, "value": value, "written": True}, self._first_next(node)

        if node_type == "read_prior_runs":
            limit = int(node.get("limit", 5))
            runs = self._state_store.read_prior_runs(graph_id=str(context["graph_id"]), limit=limit)
            return {
                "count": len(runs),
                "runs": [
                    {
                        "run_id": run.run_id,
                        "status": run.status,
                        "guarantee_mode": run.guarantee_mode,
                        "started_at": run.started_at,
                        "finished_at": run.finished_at,
                        "error_text": run.error_text,
                    }
                    for run in runs
                    if run.run_id != str(context["run_id"])
                ],
            }, self._first_next(node)

        if node_type == "condition":
            return await self._execute_condition_node(
                node=node,
                context=context,
                guarantee_mode=guarantee_mode,
                llm_client=llm_client,
            )

        if node_type == "tool":
            return await self._execute_tool_node(node=node, node_input=node_input, context=context, tools=tools)

        if node_type == "ai_template":
            return await self._execute_ai_template_node(
                node=node,
                node_input=node_input,
                context=context,
                llm_client=llm_client,
            )

        raise GraphExecutionError(f"Unsupported node type '{node_type}' in node '{node.get('node_id')}'.")

    async def _execute_tool_node(
        self,
        *,
        node: dict[str, Any],
        node_input: object,
        context: dict[str, object],
        tools: dict[str, BaseTool],
    ) -> tuple[object, str | None]:
        tool_name = node.get("tool") or node.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            raise GraphExecutionError(f"Tool node '{node.get('node_id')}' is missing non-empty 'tool'.")

        tool = tools.get(tool_name)
        if tool is None:
            raise GraphExecutionError(
                f"Tool node '{node.get('node_id')}' references unavailable tool '{tool_name}'."
            )

        args_source = node.get("args")
        if args_source is None:
            args_source = node.get("parameters")
        if args_source is None:
            args_source = node_input
        args_payload = self._resolve_value(args_source, context)
        timeout = float(node.get("timeout_seconds", self._tool_timeout_seconds))
        invocation_payload = self._coerce_tool_input(tool=tool, payload=args_payload)
        output = await self._invoke_tool_with_retry(
            tool=tool,
            payload=invocation_payload,
            timeout=timeout,
        )

        return output, self._first_next(node)

    async def _invoke_tool_with_retry(self, *, tool: BaseTool, payload: object, timeout: float) -> object:
        try:
            return await asyncio.wait_for(tool.ainvoke(payload), timeout=timeout)
        except TimeoutError as exc:
            raise GraphExecutionError(
                f"Tool '{tool.name}' timed out after {timeout:.1f}s."
            ) from exc
        except TypeError as exc:
            if not self._looks_like_signature_mismatch(exc):
                raise

            fallback_payload = self._coerce_tool_input(tool=tool, payload=payload, force_single_input=True)
            if fallback_payload == payload:
                raise
            try:
                return await asyncio.wait_for(tool.ainvoke(fallback_payload), timeout=timeout)
            except TimeoutError as timeout_exc:
                raise GraphExecutionError(
                    f"Tool '{tool.name}' timed out after {timeout:.1f}s."
                ) from timeout_exc

    def _coerce_tool_input(
        self,
        *,
        tool: BaseTool,
        payload: object,
        force_single_input: bool = False,
    ) -> object:
        arg_names = self._tool_arg_names(tool)
        if not arg_names:
            if isinstance(payload, Mapping):
                return self._coerce_mapping_payload(tool=tool, payload=payload)
            return payload

        if len(arg_names) == 1:
            target_name = arg_names[0]
            if isinstance(payload, Mapping) and target_name in payload and not force_single_input:
                return self._coerce_mapping_payload(tool=tool, payload=payload)
            if isinstance(payload, Mapping) and set(payload.keys()).issubset(set(arg_names)) and not force_single_input:
                return self._coerce_mapping_payload(tool=tool, payload=payload)
            return self._coerce_mapping_payload(tool=tool, payload={target_name: payload})

        if isinstance(payload, Mapping):
            if set(payload.keys()).issubset(set(arg_names)):
                return self._coerce_mapping_payload(tool=tool, payload=payload)
            if force_single_input:
                first_name = arg_names[0]
                return self._coerce_mapping_payload(tool=tool, payload={first_name: payload})
        return payload

    def _tool_arg_names(self, tool: BaseTool) -> list[str]:
        try:
            schema = tool.get_input_schema().model_json_schema()
        except Exception:  # noqa: BLE001
            return []

        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return []
        return [name for name in properties.keys() if name != "kwargs"]

    def _tool_schema_properties(self, tool: BaseTool) -> dict[str, object]:
        try:
            schema = tool.get_input_schema().model_json_schema()
        except Exception:  # noqa: BLE001
            return {}
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return {}
        return properties

    def _coerce_mapping_payload(self, *, tool: BaseTool, payload: Mapping[str, object]) -> dict[str, object]:
        properties = self._tool_schema_properties(tool)
        coerced = dict(payload)

        if properties:
            for key, value in list(coerced.items()):
                schema = properties.get(key)
                if isinstance(schema, dict):
                    coerced[key] = self._coerce_value_for_schema(value=value, schema=schema)

        if self._looks_like_email_send_tool(tool=tool):
            to_value = coerced.get("to")
            if isinstance(to_value, str):
                parts = [item.strip() for item in to_value.split(",") if item.strip()]
                coerced["to"] = parts if parts else [to_value]

        return coerced

    def _coerce_value_for_schema(self, *, value: object, schema: dict[str, object]) -> object:
        if self._schema_accepts_type(schema=schema, expected_type="array"):
            if isinstance(value, str):
                return [value]
        if self._schema_accepts_type(schema=schema, expected_type="string"):
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                return value[0]
        return value

    def _schema_accepts_type(self, *, schema: dict[str, object], expected_type: str) -> bool:
        type_field = schema.get("type")
        if isinstance(type_field, str) and type_field == expected_type:
            return True
        if isinstance(type_field, list) and expected_type in type_field:
            return True

        for variant_key in ("anyOf", "oneOf", "allOf"):
            variants = schema.get(variant_key)
            if not isinstance(variants, list):
                continue
            for variant in variants:
                if isinstance(variant, dict) and self._schema_accepts_type(
                    schema=variant, expected_type=expected_type
                ):
                    return True
        return False

    def _looks_like_email_send_tool(self, *, tool: BaseTool) -> bool:
        name = str(getattr(tool, "name", "")).strip().lower()
        if name in {"send_email", "draft_email"}:
            return True
        return name.endswith(".send_email") or name.endswith(".draft_email")

    def _looks_like_signature_mismatch(self, error: TypeError) -> bool:
        text = str(error)
        return "missing" in text and "required positional argument" in text

    async def _execute_ai_template_node(
        self,
        *,
        node: dict[str, Any],
        node_input: object,
        context: dict[str, object],
        llm_client: LLMClient | None,
    ) -> tuple[object, str | None]:
        if llm_client is None:
            raise GraphExecutionError(
                f"AI template node '{node.get('node_id')}' cannot execute because no LLM client is configured."
            )

        system_prompt = node.get("system_prompt")
        prompt_template = node.get("prompt_template") or node.get("prompt")
        if not isinstance(prompt_template, str) or not prompt_template.strip():
            raise GraphExecutionError(
                f"AI template node '{node.get('node_id')}' requires 'prompt_template' or 'prompt'."
            )

        rendered_prompt = self._render_template_string(prompt_template, context)
        messages = []
        if isinstance(system_prompt, str) and system_prompt.strip():
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=rendered_prompt))

        response = await llm_client.invoke(messages, tools=None, on_token=None)
        content = response.content
        if isinstance(content, str):
            text = content
        else:
            text = str(content)

        output_format = str(node.get("output_format", "text"))
        output: object
        if output_format == "json":
            output = self._extract_json_payload(text)
        else:
            output = text

        return output, self._first_next(node)

    async def _execute_condition_node(
        self,
        *,
        node: dict[str, Any],
        context: dict[str, object],
        guarantee_mode: str,
        llm_client: LLMClient | None,
    ) -> tuple[object, str | None]:
        strategy = str(node.get("strategy"))

        if strategy == "typed_condition":
            result = self._evaluate_typed_condition(node=node, context=context)
            next_node = str(node["if_true"] if result else node["if_false"])
            return {
                "strategy": strategy,
                "result": result,
                "next": next_node,
            }, next_node

        if strategy == "llm_condition":
            options = node.get("branch_options")
            if not isinstance(options, list) or not options:
                raise GraphExecutionError(
                    f"LLM condition node '{node.get('node_id')}' requires non-empty branch_options."
                )

            decision = self._resolve_value(node.get("decision"), context)
            if not isinstance(decision, str) or decision not in options:
                llm_prompt = node.get("llm_prompt") or node.get("prompt")
                if isinstance(llm_prompt, str) and llm_prompt.strip():
                    if llm_client is None:
                        raise GraphExecutionError(
                            f"LLM condition node '{node.get('node_id')}' requires an LLM client."
                        )
                    prompt = self._render_template_string(llm_prompt, context)
                    choice_messages = [
                        SystemMessage(
                            content=(
                                "Choose exactly one branch option and return strict JSON: "
                                '{"branch":"<option>","reason":"..."}. '
                                f"Allowed options: {options}."
                            )
                        ),
                        HumanMessage(content=prompt),
                    ]
                    response = await llm_client.invoke(choice_messages, tools=None, on_token=None)
                    response_text = response.content if isinstance(response.content, str) else str(response.content)
                    payload = self._extract_json_payload(response_text)
                    decision = payload.get("branch") if isinstance(payload, dict) else None

            if not isinstance(decision, str) or decision not in options:
                if guarantee_mode == "strict":
                    raise GraphExecutionError(
                        f"Strict mode requires llm_condition decision in branch_options for node '{node.get('node_id')}'."
                    )
                decision = str(options[0])

            branch_targets = node.get("branch_targets")
            if isinstance(branch_targets, dict) and decision in branch_targets:
                next_node = str(branch_targets[decision])
            else:
                if len(options) == 1:
                    next_node = str(node["if_true"])
                elif decision == str(options[0]):
                    next_node = str(node["if_true"])
                else:
                    next_node = str(node["if_false"])

            return {
                "strategy": strategy,
                "decision": decision,
                "next": next_node,
            }, next_node

        raise GraphExecutionError(
            f"Unsupported condition strategy '{strategy}' in node '{node.get('node_id')}'."
        )

    async def _emit_hook(self, event: str, context: dict[str, Any]) -> None:
        try:
            await self._hooks.emit(event, context)
        except Exception:  # noqa: BLE001
            # Hooks should never break graph execution.
            return

    def _validate_mode_policy(self, graph: dict[str, Any], guarantee_mode: str) -> None:
        violations: list[str] = []

        for node in graph["nodes"]:
            node_id = str(node.get("node_id"))
            node_type = str(node.get("type"))

            if node_type in {"tool", "ai_template"}:
                if "idempotency_key" not in node and guarantee_mode == "strict":
                    violations.append(
                        f"Strict mode requires idempotency_key on node '{node_id}'."
                    )
                if "timeout_seconds" not in node and guarantee_mode in {"strict", "bounded"}:
                    violations.append(
                        f"{guarantee_mode} mode requires timeout_seconds on node '{node_id}'."
                    )
                if "max_retries" not in node and guarantee_mode in {"strict", "bounded"}:
                    violations.append(
                        f"{guarantee_mode} mode requires max_retries on node '{node_id}'."
                    )

            if guarantee_mode == "strict" and node_type == "condition":
                strategy = node.get("strategy")
                if strategy == "llm_condition":
                    options = node.get("branch_options")
                    if not isinstance(options, list) or len(options) < 2:
                        violations.append(
                            f"Strict mode requires llm_condition node '{node_id}' to provide at least two branch_options."
                        )

        if violations:
            rendered = "\n".join(f"- {item}" for item in violations)
            raise GraphExecutionError(f"Guarantee mode policy violation:\n{rendered}")

    def _first_next(self, node: dict[str, Any], allow_missing: bool = False) -> str | None:
        raw_next = node.get("next")
        if raw_next is None and allow_missing:
            return None
        if isinstance(raw_next, str):
            return raw_next
        if isinstance(raw_next, list) and raw_next and isinstance(raw_next[0], str):
            return raw_next[0]
        if allow_missing:
            return None
        raise GraphExecutionError(f"Node '{node.get('node_id')}' has invalid 'next'.")

    def _evaluate_typed_condition(self, node: dict[str, Any], context: dict[str, object]) -> bool:
        left = self._resolve_value(node.get("left"), context)
        right = self._resolve_value(node.get("right"), context)
        operator = str(node.get("operator"))

        if operator == "eq":
            return left == right
        if operator == "ne":
            return left != right
        if operator == "gt":
            return bool(left > right)
        if operator == "gte":
            return bool(left >= right)
        if operator == "lt":
            return bool(left < right)
        if operator == "lte":
            return bool(left <= right)
        if operator == "in":
            try:
                return bool(left in right)
            except TypeError:
                return False
        if operator == "contains":
            try:
                return bool(right in left)
            except TypeError:
                return False

        raise GraphExecutionError(f"Unsupported condition operator '{operator}'.")

    def _resolve_value(self, value: object, context: dict[str, object]) -> object:
        if isinstance(value, str):
            match = TEMPLATE_VALUE_RE.fullmatch(value.strip())
            if match:
                path = match.group(1).strip()
                return self._resolve_path(path, context)
            if "{{" in value and "}}" in value:
                return self._render_template_string(value, context)
            return value

        if isinstance(value, dict):
            return {key: self._resolve_value(item, context) for key, item in value.items()}

        if isinstance(value, list):
            return [self._resolve_value(item, context) for item in value]

        return value

    def _render_template_string(self, template: str, context: dict[str, object]) -> str:
        def _replace(match: re.Match[str]) -> str:
            path = match.group(1).strip()
            value = self._resolve_path(path, context)
            if value is None:
                return ""
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)

        return TEMPLATE_INLINE_RE.sub(_replace, template)

    def _resolve_path(self, path: str, context: dict[str, object]) -> object:
        current: object = context
        for part in [piece for piece in path.split(".") if piece]:
            if isinstance(current, dict):
                if part in current:
                    current = current.get(part)
                    continue
                # Compat shim: treat "{{node.output}}" as "{{node}}" when node output
                # is stored directly in context.
                if part == "output":
                    continue
                return None
            if isinstance(current, list) and part.isdigit():
                idx = int(part)
                if idx < 0 or idx >= len(current):
                    return None
                current = current[idx]
                continue
            if part == "output":
                continue
            return None
        return current

    def _extract_json_payload(self, text: str) -> object:
        raw = text.strip()
        if not raw:
            return {}
        if raw.startswith("```"):
            raw = self._strip_fenced_block(raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
            raise GraphExecutionError("Expected JSON output but could not parse model response.")

    def _strip_fenced_block(self, raw: str) -> str:
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            return "\n".join(lines[1:-1]).strip()
        return raw
