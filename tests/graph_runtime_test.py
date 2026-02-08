from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assistant_cli.graph import GraphExecutionError, GraphExecutor, validate_graph_definition
from assistant_cli.graph.state_store import GraphStateStore


class GraphSchemaTests(unittest.TestCase):
    def test_state_nodes_require_explicit_state_access(self) -> None:
        graph = {
            "id": "g1",
            "name": "missing_state_access",
            "start": "start",
            "nodes": [
                {"node_id": "start", "type": "start", "next": "write"},
                {"node_id": "write", "type": "write_state", "key": "k", "value": 1, "next": "end"},
                {"node_id": "end", "type": "end"},
            ],
        }
        errors = validate_graph_definition(graph)
        self.assertTrue(any("requires state access" in err for err in errors))

    def test_valid_graph_with_state_nodes(self) -> None:
        graph = {
            "id": "g2",
            "name": "valid_state_graph",
            "start": "start",
            "state_access": {"default_enabled": True},
            "nodes": [
                {"node_id": "start", "type": "start", "next": "write"},
                {
                    "node_id": "write",
                    "type": "write_state",
                    "key": "counter",
                    "value": 1,
                    "next": "read",
                },
                {
                    "node_id": "read",
                    "type": "read_state",
                    "key": "counter",
                    "next": "end",
                },
                {"node_id": "end", "type": "end"},
            ],
        }
        self.assertEqual(validate_graph_definition(graph), [])


class GraphExecutionTests(unittest.TestCase):
    def test_write_state_then_read_state(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = GraphStateStore(db_path=Path(tmp_dir) / "graph.db")
            executor = GraphExecutor(state_store=store)

            graph = {
                "id": "state_graph",
                "name": "state_graph",
                "start": "start",
                "state_access": {"default_enabled": True},
                "nodes": [
                    {"node_id": "start", "type": "start", "next": "write"},
                    {
                        "node_id": "write",
                        "type": "write_state",
                        "key": "last_user",
                        "value": "{{input.user}}",
                        "next": "read",
                    },
                    {
                        "node_id": "read",
                        "type": "read_state",
                        "key": "last_user",
                        "next": "end",
                    },
                    {"node_id": "end", "type": "end"},
                ],
            }

            result = executor.run(graph=graph, input_payload={"user": "tomas"}, guarantee_mode="bounded")
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.output, {"key": "last_user", "value": "tomas"})
            self.assertEqual(store.read_state("state_graph", "last_user"), "tomas")

    def test_typed_condition_branching(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = GraphStateStore(db_path=Path(tmp_dir) / "graph.db")
            executor = GraphExecutor(state_store=store)

            graph = {
                "id": "cond_graph",
                "name": "typed_cond",
                "start": "start",
                "nodes": [
                    {"node_id": "start", "type": "start", "next": "cond"},
                    {
                        "node_id": "cond",
                        "type": "condition",
                        "strategy": "typed_condition",
                        "operator": "eq",
                        "left": "{{input.mode}}",
                        "right": "fast",
                        "if_true": "fast_end",
                        "if_false": "slow_end",
                    },
                    {"node_id": "fast_end", "type": "end"},
                    {"node_id": "slow_end", "type": "end"},
                ],
            }

            result = executor.run(graph=graph, input_payload={"mode": "fast"}, guarantee_mode="strict")
            self.assertEqual(result.status, "succeeded")
            self.assertIn("fast_end", result.visited_nodes)

    def test_strict_mode_rejects_unbounded_tool_nodes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = GraphStateStore(db_path=Path(tmp_dir) / "graph.db")
            executor = GraphExecutor(state_store=store)

            graph = {
                "id": "tool_graph",
                "name": "tool_graph",
                "start": "start",
                "nodes": [
                    {"node_id": "start", "type": "start", "next": "tool1"},
                    {
                        "node_id": "tool1",
                        "type": "tool",
                        "tool": "read_file",
                        "next": "end",
                    },
                    {"node_id": "end", "type": "end"},
                ],
            }

            with self.assertRaises(GraphExecutionError):
                executor.run(graph=graph, input_payload={}, guarantee_mode="strict")

    def test_state_persists_across_runs_same_graph(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = GraphStateStore(db_path=Path(tmp_dir) / "graph.db")
            executor = GraphExecutor(state_store=store)

            graph = {
                "id": "persist_graph",
                "name": "persist_graph",
                "start": "start",
                "state_access": {"default_enabled": True},
                "nodes": [
                    {"node_id": "start", "type": "start", "next": "write"},
                    {
                        "node_id": "write",
                        "type": "write_state",
                        "key": "last_number",
                        "value": "{{input.value}}",
                        "next": "end",
                    },
                    {"node_id": "end", "type": "end"},
                ],
            }

            executor.run(graph=graph, input_payload={"value": 42}, guarantee_mode="bounded")
            runs = store.read_prior_runs("persist_graph", limit=10)
            self.assertGreaterEqual(len(runs), 1)
            self.assertEqual(store.read_state("persist_graph", "last_number"), 42)


if __name__ == "__main__":
    unittest.main()
