from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from assistant_cli.agent_graph import LangGraphAgent
from assistant_cli.approval import ApprovalManager
from assistant_cli.cli import AssistantCLI
from assistant_cli.llm_client import OpenAILLMClient, OpenAILLMConfig
from assistant_cli.mcp_manager import MCPManager
from assistant_cli.memory_store import SQLiteMemoryStore


class MarkdownRenderTests(unittest.TestCase):
    def test_render_markdown_ansi_has_no_stdout_side_effect(self) -> None:
        cli = AssistantCLI.__new__(AssistantCLI)
        captured_stdout = io.StringIO()
        with redirect_stdout(captured_stdout):
            rendered = AssistantCLI._render_markdown_ansi(cli, "# Nebula")

        self.assertEqual(captured_stdout.getvalue(), "")
        self.assertIn("Nebula", rendered)
        self.assertIn("\x1b[", rendered)

    def test_markdown_detection_distinguishes_plain_text(self) -> None:
        cli = AssistantCLI.__new__(AssistantCLI)
        self.assertTrue(AssistantCLI._looks_like_markdown(cli, "## Header\n- item"))
        self.assertFalse(AssistantCLI._looks_like_markdown(cli, "Hey! 👋\n\nWhat's up?"))

    def test_plain_response_collapses_paragraph_breaks_and_leaves_prompt_gap(self) -> None:
        cli = AssistantCLI.__new__(AssistantCLI)
        captured_stdout = io.StringIO()
        with redirect_stdout(captured_stdout):
            AssistantCLI._print_plain_response(cli, "Hey! 👋\n\nWhat's up?")

        self.assertEqual(captured_stdout.getvalue(), "> Hey! 👋\nWhat's up?\n\n")


class MemoryStoreTests(unittest.TestCase):
    def test_enforce_token_limit_drops_oldest_messages(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = SQLiteMemoryStore(
                db_path=Path(tmp_dir) / "test.db",
                session_id="s1",
                token_limit=20,
                context_window=8000,
            )

            messages = [
                HumanMessage(content="first turn"),
                AIMessage(content="second turn"),
                HumanMessage(content="third turn"),
            ]

            trimmed, truncated = store.enforce_token_limit(messages)

            self.assertTrue(truncated)
            self.assertGreaterEqual(len(trimmed), 1)
            self.assertEqual(trimmed[-1].content, "third turn")

    def test_normalize_removes_orphan_tool_messages(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = SQLiteMemoryStore(
                db_path=Path(tmp_dir) / "test.db",
                session_id="s1",
                token_limit=3000,
                context_window=8000,
            )

            messages = [
                HumanMessage(content="hello"),
                ToolMessage(content="orphan", tool_call_id="missing"),
                HumanMessage(content="next"),
            ]

            trimmed, truncated = store.enforce_token_limit(messages)
            self.assertTrue(truncated)
            self.assertEqual(len(trimmed), 2)
            self.assertIsInstance(trimmed[0], HumanMessage)
            self.assertIsInstance(trimmed[1], HumanMessage)


class ApprovalManagerTests(unittest.TestCase):
    def test_tool_override_beats_global(self) -> None:
        approval = ApprovalManager()
        approval.set_global(False)
        approval.set_tool("memory_read_graph", True)

        self.assertTrue(approval.tool_enabled("memory_read_graph"))
        self.assertFalse(approval.tool_enabled("filesystem_read_file"))


class MCPManagerTests(unittest.TestCase):
    def test_update_filesystem_allowed_paths(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            config_path = tmp / "mcp.json"
            config_payload = {
                "filesystem": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", str(tmp)],
                    "enabled": True,
                }
            }
            config_path.write_text(json.dumps(config_payload), encoding="utf-8")

            manager = MCPManager(config_path=config_path, fallback_config_path=config_path)
            original_paths = manager.list_filesystem_allowed_directories()
            self.assertEqual(len(original_paths), 1)

            extra = tmp / "extra"
            extra.mkdir(parents=True, exist_ok=True)
            added = manager.add_filesystem_allowed_directory(str(extra))
            self.assertIn(added, manager.list_filesystem_allowed_directories())

            removed = manager.remove_filesystem_allowed_directory(str(extra))
            self.assertEqual(removed, added)
            self.assertEqual(len(manager.list_filesystem_allowed_directories()), 1)


class OpenAIToolSchemaTests(unittest.TestCase):
    def test_migrate_legacy_json_schema_bounds(self) -> None:
        client = OpenAILLMClient(OpenAILLMConfig(api_key="test", model="gpt-5.2"))
        schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "exclusiveMinimum": True,
                },
                "limit": {
                    "type": "number",
                    "maximum": 10,
                    "exclusiveMaximum": True,
                },
                "items": {
                    "type": "array",
                    "minItems": True,
                },
            },
        }

        migrated = client._migrate_json_schema(schema)
        self.assertEqual(migrated["properties"]["score"]["exclusiveMinimum"], 0)
        self.assertNotIn("minimum", migrated["properties"]["score"])
        self.assertEqual(migrated["properties"]["limit"]["exclusiveMaximum"], 10)
        self.assertNotIn("maximum", migrated["properties"]["limit"])
        self.assertNotIn("minItems", migrated["properties"]["items"])

    def test_drop_incompatible_tool_by_name_or_index(self) -> None:
        client = OpenAILLMClient(OpenAILLMConfig(api_key="test", model="gpt-5.2"))
        definitions = [
            {"type": "function", "function": {"name": "alpha", "parameters": {}}},
            {"type": "function", "function": {"name": "read_pdf", "parameters": {}}},
            {"type": "function", "function": {"name": "gamma", "parameters": {}}},
        ]

        by_name = client._drop_incompatible_tool(
            definitions,
            RuntimeError('Invalid schema for function "read_pdf": True is not of type "number"'),
        )
        self.assertEqual([item["function"]["name"] for item in by_name or []], ["alpha", "gamma"])

        many = [{"type": "function", "function": {"name": f"t{i}", "parameters": {}}} for i in range(30)]
        by_index = client._drop_incompatible_tool(
            many,
            RuntimeError("Invalid schema at tools[26].function.parameters"),
        )
        self.assertIsNotNone(by_index)
        self.assertEqual(len(by_index or []), 29)
        self.assertNotIn("t26", [item["function"]["name"] for item in by_index or []])


class _DummyLLMClient:
    @property
    def model_name(self) -> str:
        return "dummy"

    async def invoke(  # pragma: no cover - helper for construction only
        self,
        messages: list[BaseMessage],
        tools: list[BaseTool] | None = None,
        on_token=None,
    ) -> AIMessage:
        return AIMessage(content="ok")


class ToolLoopGuardTests(unittest.TestCase):
    def test_tool_call_count_scoped_to_latest_turn(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            agent = LangGraphAgent(
                db_path=Path(tmp_dir) / "graph.db",
                llm_client=_DummyLLMClient(),
                max_iterations=10,
                request_timeout_seconds=30,
                tool_timeout_seconds=5,
            )

            messages = [
                HumanMessage(content="old turn"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "old-1", "name": "read_pdf", "args": {"path": "a.pdf", "page": 1}}
                    ],
                ),
                ToolMessage(content="old result", tool_call_id="old-1"),
                HumanMessage(content="new turn"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "new-1", "name": "read_pdf", "args": {"path": "a.pdf", "page": 1}}
                    ],
                ),
            ]

            count = agent._count_tool_call_occurrences_current_turn(
                messages,
                "read_pdf",
                {"path": "a.pdf", "page": 1},
            )
            self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
