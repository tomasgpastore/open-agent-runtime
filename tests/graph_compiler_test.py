from __future__ import annotations

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from assistant_cli.graph import GraphExecutionError, GraphExecutor
from assistant_cli.graph.compiler import CompileOptions, GraphCompiler
from assistant_cli.graph.state_store import GraphStateStore


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


class _WriteFileArgs(BaseModel):
    path: str
    content: str


class _WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = "writes file"
    args_schema: type[BaseModel] = _WriteFileArgs

    def _run(self, path: str, content: str, **kwargs):
        return {"path": path, "content": content}

    async def _arun(self, path: str, content: str, **kwargs):
        return {"path": path, "content": content}


class _SearchArgs(BaseModel):
    query: str
    max_results: int | None = None


class _SearchTool(BaseTool):
    name: str = "search_emails"
    description: str = "search emails"
    args_schema: type[BaseModel] = _SearchArgs

    def _run(self, query: str, max_results: int | None = None, **kwargs):
        return {"query": query, "max_results": max_results}

    async def _arun(self, query: str, max_results: int | None = None, **kwargs):
        return {"query": query, "max_results": max_results}


class GraphCompilerTests(unittest.TestCase):
    def test_compile_canonicalizes_aliases_and_injects_defaults(self) -> None:
        compiler = GraphCompiler()
        graph = with_contracts({
            "id": "g1",
            "name": "alias graph",
            "start": "start",
            "nodes": [
                {"node_id": "start", "type": "start", "next": "send"},
                {
                    "node_id": "send",
                    "type": "tool",
                    "tool_name": "write_file",
                    "parameters": {
                        "path": "./reports/test.md",
                        "content": "hello",
                    },
                    "next": ["end"],
                },
                {"node_id": "end", "type": "end"},
            ],
        })

        result = compiler.compile(
            graph=graph,
            tool_map={"write_file": _WriteFileTool()},
            options=CompileOptions(mode="bounded"),
        )

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.compiled)
        self.assertIsNotNone(result.rewritten_graph)
        rewritten = result.rewritten_graph or {}
        send = next(node for node in rewritten["nodes"] if node["node_id"] == "send")
        self.assertEqual(send["tool"], "write_file")
        self.assertIn("args", send)
        self.assertNotIn("parameters", send)
        self.assertIsInstance(send["next"], str)
        self.assertIn("timeout_seconds", send)
        self.assertIn("max_retries", send)
        self.assertIn("idempotency_key", send)

    def test_compile_fails_when_tool_missing(self) -> None:
        compiler = GraphCompiler()
        graph = with_contracts({
            "id": "g2",
            "name": "missing tool",
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

        result = compiler.compile(graph=graph, tool_map={}, options=CompileOptions(mode="bounded"))
        self.assertFalse(result.ok)
        self.assertTrue(any(item.code == "TOOL_NOT_AVAILABLE" for item in result.diagnostics))

    def test_compile_fails_when_node_contract_is_missing(self) -> None:
        compiler = GraphCompiler()
        graph = {
            "id": "missing_contracts",
            "name": "missing_contracts",
            "start": "start",
            "nodes": [
                {"node_id": "start", "type": "start", "next": "tool"},
                {
                    "node_id": "tool",
                    "type": "tool",
                    "tool": "write_file",
                    "args": {"path": "./reports/a.md", "content": "hi"},
                    "next": "end",
                },
                {"node_id": "end", "type": "end"},
            ],
        }

        result = compiler.compile(
            graph=graph,
            tool_map={"write_file": _WriteFileTool()},
            options=CompileOptions(mode="bounded"),
        )

        self.assertFalse(result.ok)
        self.assertTrue(any(item.code == "NODE_INPUT_SCHEMA_MISSING" for item in result.diagnostics))
        self.assertTrue(any(item.code == "NODE_OUTPUT_SCHEMA_MISSING" for item in result.diagnostics))

    def test_compile_fails_on_unknown_template_reference(self) -> None:
        compiler = GraphCompiler()
        graph = with_contracts({
            "id": "g3",
            "name": "bad template",
            "start": "start",
            "nodes": [
                {"node_id": "start", "type": "start", "next": "tool"},
                {
                    "node_id": "tool",
                    "type": "tool",
                    "tool": "search_emails",
                    "args": {"query": "{{does_not_exist.value}}"},
                    "next": "end",
                },
                {"node_id": "end", "type": "end"},
            ],
        })

        result = compiler.compile(
            graph=graph,
            tool_map={"search_emails": _SearchTool()},
            options=CompileOptions(mode="bounded"),
        )
        self.assertFalse(result.ok)
        self.assertTrue(any(item.code == "TEMPLATE_UNKNOWN_REFERENCE" for item in result.diagnostics))

    def test_compile_hash_is_deterministic(self) -> None:
        compiler = GraphCompiler()
        graph = with_contracts({
            "id": "g4",
            "name": "deterministic",
            "start": "start",
            "nodes": [
                {"node_id": "start", "type": "start", "next": "search"},
                {
                    "node_id": "search",
                    "type": "tool",
                    "tool": "search_emails",
                    "args": {"query": "is:unread"},
                    "timeout_seconds": 10,
                    "max_retries": 1,
                    "idempotency_key": "g4:search",
                    "next": "end",
                },
                {"node_id": "end", "type": "end"},
            ],
        })

        first = compiler.compile(
            graph=graph,
            tool_map={"search_emails": _SearchTool()},
            options=CompileOptions(mode="strict"),
        )
        second = compiler.compile(
            graph=graph,
            tool_map={"search_emails": _SearchTool()},
            options=CompileOptions(mode="strict"),
        )

        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        self.assertIsNotNone(first.compiled)
        self.assertIsNotNone(second.compiled)
        self.assertEqual(first.compiled.compile_hash, second.compiled.compile_hash)

    def test_compile_inserts_ai_normalizer_when_policy_always(self) -> None:
        compiler = GraphCompiler()
        graph = {
            "id": "g5",
            "name": "always_normalize",
            "start": "start",
            "nodes": [
                {
                    "node_id": "start",
                    "type": "start",
                    "output_schema": {
                        "type": "object",
                        "properties": {"foo": {"type": "string"}},
                        "required": ["foo"],
                        "additionalProperties": False,
                    },
                    "next": "t1",
                },
                {
                    "node_id": "t1",
                    "type": "transform",
                    "input_schema": {
                        "type": "object",
                        "properties": {"bar": {"type": "string"}},
                        "required": ["bar"],
                        "additionalProperties": False,
                    },
                    "output_schema": {"type": "any"},
                    "value": "{{input}}",
                    "next": "end",
                },
                {"node_id": "end", "type": "end", "input_schema": {"type": "any"}},
            ],
        }

        result = compiler.compile(
            graph=graph,
            tool_map={},
            options=CompileOptions(mode="bounded", ai_edge_policy="always"),
        )

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.rewritten_graph)
        rewritten = result.rewritten_graph or {}
        node_ids = {str(node.get("node_id")): node for node in rewritten.get("nodes", [])}
        self.assertIn("normalize__start__to__t1", node_ids)
        self.assertEqual(node_ids["start"]["next"], "normalize__start__to__t1")

    def test_executor_rejects_compile_errors_before_starting_run(self) -> None:
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
                executor.run(graph=graph, input_payload={}, guarantee_mode="bounded")

            self.assertEqual(store.read_prior_runs("compile_fail_graph", limit=5), [])


if __name__ == "__main__":
    unittest.main()
