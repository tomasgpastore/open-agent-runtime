from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from assistant_cli.daily_memory import DailyMemoryArchive
from assistant_cli.graph import GraphExecutionError, GraphExecutor, GraphHookRegistry, validate_graph_definition
from assistant_cli.graph.builder import GraphBuilderAnton
from assistant_cli.graph.scheduler import GraphScheduler
from assistant_cli.graph.state_store import GraphStateStore
from assistant_cli.long_term_memory import LongTermMemoryStore, MemoryRetrievalPlanner


def with_contracts(graph: dict) -> dict:
    cloned = copy.deepcopy(graph)
    nodes = cloned.get("nodes")
    if not isinstance(nodes, list):
        return cloned
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type") or "")
        if node_type != "start":
            node.setdefault("input_schema", {"type": "any"})
        if node_type != "end":
            node.setdefault("output_schema", {"type": "any"})
    return cloned


class _EchoTool(BaseTool):
    name: str = "echo_tool"
    description: str = "Echoes the payload"

    def _run(self, tool_input, **kwargs):
        return {"echo": tool_input}

    async def _arun(self, tool_input, **kwargs):
        return {"echo": tool_input}


class _FlakyTool(BaseTool):
    name: str = "flaky_tool"
    description: str = "Fails once then succeeds"
    calls: int = 0

    def _run(self, tool_input, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("first failure")
        return {"ok": True, "call": self.calls, "payload": tool_input}

    async def _arun(self, tool_input, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("first failure")
        return {"ok": True, "call": self.calls, "payload": tool_input}


class _SendEmailArgs(BaseModel):
    to: list[str]
    subject: str
    body: str


class _EmptyArgs(BaseModel):
    pass


class _SendEmailTool(BaseTool):
    name: str = "send_email"
    description: str = "Sends email"
    args_schema: type[BaseModel] = _SendEmailArgs

    def _run(self, to: list[str], subject: str, body: str, **kwargs):
        return {"to": to, "subject": subject, "body": body}

    async def _arun(self, to: list[str], subject: str, body: str, **kwargs):
        return {"to": to, "subject": subject, "body": body}


class _OpaqueSendEmailTool(BaseTool):
    name: str = "send_email"
    description: str = "Sends email with opaque schema"
    args_schema: type[BaseModel] = _EmptyArgs

    def _run(self, **kwargs):
        return kwargs

    async def _arun(self, **kwargs):
        return kwargs


@dataclass(slots=True)
class _DummyLLM:
    text: object

    @property
    def model_name(self) -> str:
        return "dummy"

    async def invoke(self, messages, tools=None, on_token=None):
        return AIMessage(content=self.text)


class GraphV2Tests(unittest.TestCase):
    def test_executor_fails_compile_preflight_before_run_record(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = GraphStateStore(db_path=Path(tmp_dir) / "graph.db")
            executor = GraphExecutor(state_store=store)

            graph = with_contracts({
                "id": "compile_fail_graph",
                "name": "compile fail graph",
                "start": "start",
                "nodes": [
                    {"node_id": "start", "type": "start", "next": "tool"},
                    {
                        "node_id": "tool",
                        "type": "tool",
                        "tool": "missing_tool",
                        "args": {"x": 1},
                        "next": "end",
                    },
                    {"node_id": "end", "type": "end"},
                ],
            })

            with self.assertRaises(GraphExecutionError):
                asyncio.run(executor.arun(graph=graph, input_payload={}, guarantee_mode="bounded"))

            self.assertEqual(store.read_prior_runs("compile_fail_graph", limit=5), [])

    def test_executor_runs_real_tool_node(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = GraphStateStore(db_path=Path(tmp_dir) / "graph.db")
            executor = GraphExecutor(state_store=store)
            tool = _EchoTool()

            graph = with_contracts({
                "id": "tool_graph",
                "name": "tool graph",
                "start": "start",
                "nodes": [
                    {"node_id": "start", "type": "start", "next": "tool"},
                    {
                        "node_id": "tool",
                        "type": "tool",
                        "tool": "echo_tool",
                        "args": {"value": "{{input.value}}"},
                        "timeout_seconds": 10,
                        "max_retries": 1,
                        "idempotency_key": "tool_graph:tool",
                        "next": "end",
                    },
                    {"node_id": "end", "type": "end"},
                ],
            })

            result = asyncio.run(
                executor.arun(
                    graph=graph,
                    input_payload={"value": 123},
                    guarantee_mode="strict",
                    tool_map={"echo_tool": tool},
                )
            )
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.output, {"echo": {"value": 123}})

    def test_executor_ai_template_node(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = GraphStateStore(db_path=Path(tmp_dir) / "graph.db")
            llm = _DummyLLM('{"summary":"done","count":2}')
            executor = GraphExecutor(state_store=store, llm_client=llm)

            graph = with_contracts({
                "id": "ai_graph",
                "name": "ai graph",
                "start": "start",
                "nodes": [
                    {"node_id": "start", "type": "start", "next": "ai"},
                    {
                        "node_id": "ai",
                        "type": "ai_template",
                        "prompt_template": "Summarize {{input.topic}}",
                        "output_format": "json",
                        "timeout_seconds": 10,
                        "max_retries": 1,
                        "idempotency_key": "ai_graph:ai",
                        "next": "end",
                    },
                    {"node_id": "end", "type": "end"},
                ],
            })

            result = asyncio.run(
                executor.arun(graph=graph, input_payload={"topic": "testing"}, guarantee_mode="strict")
            )
            self.assertEqual(result.output, {"summary": "done", "count": 2})

    def test_executor_supports_builder_tool_aliases_and_output_templates(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = GraphStateStore(db_path=Path(tmp_dir) / "graph.db")
            llm = _DummyLLM("# Summary\n- item")
            executor = GraphExecutor(state_store=store, llm_client=llm)

            graph = with_contracts({
                "id": "alias_graph",
                "name": "alias graph",
                "start": "start",
                "nodes": [
                    {"node_id": "start", "type": "start", "next": "classify"},
                    {
                        "node_id": "classify",
                        "type": "ai_template",
                        "prompt": "Summarize emails",
                        "timeout_seconds": 10,
                        "max_retries": 1,
                        "idempotency_key": "alias_graph:classify",
                        "next": "send",
                    },
                    {
                        "node_id": "send",
                        "type": "tool",
                        "tool_name": "send_email",
                        "parameters": {
                            "to": "tomasgpastore@gmail.com",
                            "subject": "Daily Inbox Summary {{date}}",
                            "body": "{{classify.output}}",
                        },
                        "timeout_seconds": 10,
                        "max_retries": 1,
                        "idempotency_key": "alias_graph:send",
                        "next": "end",
                    },
                    {"node_id": "end", "type": "end"},
                ],
            })

            result = asyncio.run(
                executor.arun(
                    graph=graph,
                    input_payload={},
                    guarantee_mode="bounded",
                    tool_map={"send_email": _SendEmailTool()},
                )
            )

            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.output["to"], ["tomasgpastore@gmail.com"])
            self.assertEqual(result.output["body"], "# Summary\n- item")
            self.assertNotIn("{{", result.output["subject"])
            self.assertRegex(result.output["subject"], r"Daily Inbox Summary \d{4}-\d{2}-\d{2}")

    def test_executor_coerces_email_to_list_for_opaque_schema(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = GraphStateStore(db_path=Path(tmp_dir) / "graph.db")
            executor = GraphExecutor(state_store=store)
            payload = {
                "to": "alpha@example.com, beta@example.com",
                "subject": "Test",
                "body": "Hello",
            }
            coerced = executor._coerce_tool_input(tool=_OpaqueSendEmailTool(), payload=payload)
            self.assertEqual(
                coerced["to"],
                ["alpha@example.com", "beta@example.com"],
            )

    def test_resume_failed_run(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = GraphStateStore(db_path=Path(tmp_dir) / "graph.db")
            tool = _FlakyTool()
            executor = GraphExecutor(state_store=store)

            graph = with_contracts({
                "id": "resume_graph",
                "name": "resume graph",
                "start": "start",
                "nodes": [
                    {"node_id": "start", "type": "start", "next": "tool"},
                    {
                        "node_id": "tool",
                        "type": "tool",
                        "tool": "flaky_tool",
                        "args": {"msg": "{{input.msg}}"},
                        "timeout_seconds": 10,
                        "max_retries": 0,
                        "idempotency_key": "resume_graph:tool",
                        "next": "end",
                    },
                    {"node_id": "end", "type": "end"},
                ],
            })
            store.save_graph_definition(graph)

            with self.assertRaises(Exception):
                asyncio.run(
                    executor.arun(
                        graph=graph,
                        input_payload={"msg": "hi"},
                        guarantee_mode="bounded",
                        tool_map={"flaky_tool": tool},
                    )
                )

            runs = store.read_prior_runs("resume_graph", limit=5)
            failed_run = runs[0]
            self.assertEqual(failed_run.status, "failed")

            resumed = asyncio.run(
                executor.resume(run_id=failed_run.run_id, tool_map={"flaky_tool": tool})
            )
            self.assertEqual(resumed.status, "succeeded")

    def test_hooks_fire_for_run_and_nodes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = GraphStateStore(db_path=Path(tmp_dir) / "graph.db")
            hooks = GraphHookRegistry()
            events: list[str] = []

            async def _before_run(context):
                events.append(f"before_run:{context['graph_id']}")

            async def _after_run(context):
                events.append(f"after_run:{context['status']}")

            async def _before_node(context):
                events.append(f"before_node:{context['node_id']}")

            async def _after_node(context):
                events.append(f"after_node:{context['node_id']}")

            hooks.register("before_run", _before_run)
            hooks.register("after_run", _after_run)
            hooks.register("before_node", _before_node)
            hooks.register("after_node", _after_node)

            executor = GraphExecutor(state_store=store, hook_registry=hooks)
            graph = with_contracts({
                "id": "hook_graph",
                "name": "hook graph",
                "start": "start",
                "nodes": [
                    {"node_id": "start", "type": "start", "next": "end"},
                    {"node_id": "end", "type": "end"},
                ],
            })
            asyncio.run(executor.arun(graph=graph, input_payload={}, guarantee_mode="bounded"))

            self.assertIn("before_run:hook_graph", events)
            self.assertIn("after_run:succeeded", events)
            self.assertIn("before_node:start", events)
            self.assertIn("after_node:end", events)

    def test_scheduler_triggers_due_graph(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = GraphStateStore(db_path=Path(tmp_dir) / "graph.db")

            calls: list[tuple[str, str, object, str]] = []

            async def _execute(graph_id: str, mode: str, payload: object, schedule_id: str):
                calls.append((graph_id, mode, payload, schedule_id))
                return {"ok": True}

            scheduler = GraphScheduler(state_store=store, execute_callback=_execute)
            now = datetime(2026, 2, 8, 10, 0, 0).astimezone()
            sched_id = scheduler.add_schedule(
                graph_id="g1",
                cron_expr="* * * * *",
                guarantee_mode="bounded",
                input_payload={"hello": "world"},
                now=now,
            )

            tick_time = now + timedelta(minutes=1)
            results = asyncio.run(scheduler.trigger_due(now=tick_time))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].schedule_id, sched_id)
            self.assertEqual(len(calls), 1)

    def test_graph_builder_fallback_still_valid(self) -> None:
        builder = GraphBuilderAnton(llm_client=_DummyLLM("not json"))
        built = asyncio.run(
            builder.build_from_intent(
                intent="Check inbox and summarize",
                available_tools=["web_search_search", "filesystem_write_file"],
                graph_id="builder_graph",
            )
        )
        errors = validate_graph_definition(built.graph)
        self.assertEqual(errors, [])
        self.assertIn(built.source, {"llm", "fallback"})
        self.assertTrue(any(node.get("type") == "tool" for node in built.graph.get("nodes", [])))

    def test_graph_builder_gmail_fallback_creates_reports_dir(self) -> None:
        builder = GraphBuilderAnton(llm_client=_DummyLLM("not json"))
        built = asyncio.run(
            builder.build_from_intent(
                intent=(
                    "Build a bounded graph that reads unread Gmail and writes a markdown summary "
                    "to ./reports/gmail_{{date}}.md."
                ),
                available_tools=["search_emails", "write_file", "create_directory"],
                graph_id="gmail_fb",
            )
        )
        nodes = built.graph["nodes"]
        node_ids = {str(node["node_id"]): node for node in nodes}
        self.assertIn("ensure_reports_dir", node_ids)
        self.assertEqual(node_ids["ensure_reports_dir"]["tool"], "create_directory")
        self.assertEqual(node_ids["write_markdown_summary"]["args"]["path"], "./reports/gmail_today.md")

    def test_graph_builder_parses_text_block_content(self) -> None:
        payload = with_contracts({
            "id": "graph_from_blocks",
            "name": "graph_from_blocks",
            "start": "start",
            "nodes": [
                {"node_id": "start", "type": "start", "next": "end"},
                {"node_id": "end", "type": "end"},
            ],
        })
        builder = GraphBuilderAnton(
            llm_client=_DummyLLM(
                [
                    {
                        "type": "text",
                        "text": json.dumps(payload),
                    }
                ]
            )
        )
        built = asyncio.run(
            builder.build_from_intent(
                intent="Build a minimal graph",
                available_tools=[],
                graph_id="graph_from_blocks",
            )
        )
        self.assertEqual(built.source, "llm")
        self.assertEqual(built.graph["id"], "graph_from_blocks")


class LongTermMemoryPlannerTests(unittest.TestCase):
    def test_retrieval_planner_combines_daily_and_long_term(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "daily"
            db_path = Path(tmp_dir) / "assistant.db"
            daily = DailyMemoryArchive(root_dir=root, db_path=db_path)
            long_term = LongTermMemoryStore(db_path=db_path)
            planner = MemoryRetrievalPlanner(daily_archive=daily, long_term_store=long_term)

            daily.append_exchange(
                user_text="remember automation for canvas",
                assistant_text="okay",
                timestamp=datetime(2026, 2, 8, 8, 0, 0),
            )
            long_term.add_fact(namespace="work", content="Canvas automation should run at 11 PM", importance=5)

            items = planner.retrieve(query="canvas automation", limit=5)
            sources = {item.source for item in items}
            self.assertIn("daily", sources)
            self.assertIn("long_term", sources)


if __name__ == "__main__":
    unittest.main()
