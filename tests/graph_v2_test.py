from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from assistant_cli.daily_memory import DailyMemoryArchive
from assistant_cli.graph import GraphExecutor, GraphHookRegistry, validate_graph_definition
from assistant_cli.graph.builder import GraphBuilderAnton
from assistant_cli.graph.scheduler import GraphScheduler
from assistant_cli.graph.state_store import GraphStateStore
from assistant_cli.long_term_memory import LongTermMemoryStore, MemoryRetrievalPlanner


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


@dataclass(slots=True)
class _DummyLLM:
    text: str

    @property
    def model_name(self) -> str:
        return "dummy"

    async def invoke(self, messages, tools=None, on_token=None):
        return AIMessage(content=self.text)


class GraphV2Tests(unittest.TestCase):
    def test_executor_runs_real_tool_node(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = GraphStateStore(db_path=Path(tmp_dir) / "graph.db")
            executor = GraphExecutor(state_store=store)
            tool = _EchoTool()

            graph = {
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
            }

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

            graph = {
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
            }

            result = asyncio.run(
                executor.arun(graph=graph, input_payload={"topic": "testing"}, guarantee_mode="strict")
            )
            self.assertEqual(result.output, {"summary": "done", "count": 2})

    def test_resume_failed_run(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = GraphStateStore(db_path=Path(tmp_dir) / "graph.db")
            tool = _FlakyTool()
            executor = GraphExecutor(state_store=store)

            graph = {
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
            }
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
            graph = {
                "id": "hook_graph",
                "name": "hook graph",
                "start": "start",
                "nodes": [
                    {"node_id": "start", "type": "start", "next": "end"},
                    {"node_id": "end", "type": "end"},
                ],
            }
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
