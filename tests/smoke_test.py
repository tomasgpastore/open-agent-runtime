from __future__ import annotations

from contextlib import redirect_stdout
import asyncio
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from assistant_cli.agent_graph import AgentRunResult, LangGraphAgent, MAX_TOOL_RESULT_CHARS
from assistant_cli.approval import ApprovalManager
from assistant_cli.cli import AssistantCLI, ResponseStreamPrinter, _run_with_cleanup
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

    def test_markdown_mode_keeps_real_streaming(self) -> None:
        class _DummyMemoryStore:
            def __init__(self) -> None:
                self.saved_messages = None
                self.saved_truncation = None

            def load_messages(self):
                return []

            def enforce_token_limit(self, messages):
                return messages, False

            def save_messages(self, messages, truncation_occurred: bool) -> None:
                self.saved_messages = messages
                self.saved_truncation = truncation_occurred

        class _DummyMCPManager:
            def active_tools(self):
                return []

        class _DummyAgent:
            def __init__(self) -> None:
                self.last_stream_callback = None

            async def run(self, **kwargs):
                self.last_stream_callback = kwargs.get("stream_callback")
                stream_cb = kwargs.get("stream_callback")
                if stream_cb is not None:
                    stream_cb("Hello")
                return AgentRunResult(
                    messages=kwargs["messages"] + [AIMessage(content="Hello")],
                    final_answer="Hello",
                    stop_reason=None,
                )

        cli = AssistantCLI.__new__(AssistantCLI)
        cli.memory_store = _DummyMemoryStore()
        cli.mcp_manager = _DummyMCPManager()
        cli.approval_manager = ApprovalManager()
        cli.agent = _DummyAgent()
        cli._markdown_enabled = True
        cli.console = None

        captured_stdout = io.StringIO()
        with redirect_stdout(captured_stdout):
            asyncio.run(
                AssistantCLI._handle_user_message_with_content(
                    cli,
                    HumanMessage(content="Hi"),
                )
            )

        self.assertIsNotNone(cli.agent.last_stream_callback)
        self.assertIn("> Hello", captured_stdout.getvalue())

    def test_tool_event_includes_args_and_output_status(self) -> None:
        streamer = ResponseStreamPrinter()
        captured_stdout = io.StringIO()
        with redirect_stdout(captured_stdout):
            streamer.on_tool(
                {
                    "tool_name": "read_file",
                    "args": {"path": "notes.txt"},
                    "status": "OK",
                }
            )

        text = captured_stdout.getvalue()
        self.assertIn("tool> read_file", text)
        self.assertIn('args={"path": "notes.txt"}', text)
        self.assertIn("output=OK", text)

    def test_streaming_ignores_leading_whitespace_only_chunks(self) -> None:
        streamer = ResponseStreamPrinter()
        captured_stdout = io.StringIO()
        with redirect_stdout(captured_stdout):
            streamer.on_token("   ")
            streamer.on_token("\n")
            streamer.on_token("Hello")
            streamer.finish()

        self.assertEqual(captured_stdout.getvalue(), "> Hello\n\n")

    def test_run_with_cleanup_closes_on_cancellation(self) -> None:
        class _DummyApp:
            def __init__(self) -> None:
                self.closed = False

            async def run(self) -> None:
                raise asyncio.CancelledError()

            async def aclose(self) -> None:
                self.closed = True

        app = _DummyApp()
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(_run_with_cleanup(app))
        self.assertTrue(app.closed)


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

    def test_total_tool_call_count_scoped_to_latest_turn(self) -> None:
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
                        {"id": "old-1", "name": "read_pdf", "args": {"path": "a.pdf", "page": 1}},
                        {"id": "old-2", "name": "read_pdf", "args": {"path": "a.pdf", "page": 2}},
                    ],
                ),
                ToolMessage(content="old result", tool_call_id="old-1"),
                HumanMessage(content="new turn"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "new-1", "name": "read_pdf", "args": {"path": "a.pdf", "page": 1}},
                        {"id": "new-2", "name": "read_pdf", "args": {"path": "a.pdf", "page": 2}},
                    ],
                ),
            ]

            count = agent._count_total_tool_calls_current_turn(messages)
            self.assertEqual(count, 2)

    def test_filesystem_access_denied_is_non_retryable(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            agent = LangGraphAgent(
                db_path=Path(tmp_dir) / "graph.db",
                llm_client=_DummyLLMClient(),
                max_iterations=10,
                request_timeout_seconds=30,
                tool_timeout_seconds=5,
            )

            self.assertTrue(
                agent._is_non_retryable_tool_error(
                    "list_directory",
                    "Access denied - path outside allowed directories: /a not in /b",
                )
            )
            self.assertFalse(
                agent._is_non_retryable_tool_error(
                    "web_search",
                    "Request timed out",
                )
            )

    def test_emit_tool_event_payload_shape(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            agent = LangGraphAgent(
                db_path=Path(tmp_dir) / "graph.db",
                llm_client=_DummyLLMClient(),
                max_iterations=10,
                request_timeout_seconds=30,
                tool_timeout_seconds=5,
            )

            events: list[object] = []
            agent._emit_tool_event(
                events.append,
                "list_directory",
                {"path": "/tmp"},
                "ERROR",
            )

            self.assertEqual(len(events), 1)
            self.assertIsInstance(events[0], dict)
            payload = events[0]
            self.assertEqual(payload.get("tool_name"), "list_directory")
            self.assertEqual(payload.get("status"), "ERROR")

    def test_tool_signature_ignores_runtime_arg_injection(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            agent = LangGraphAgent(
                db_path=Path(tmp_dir) / "graph.db",
                llm_client=_DummyLLMClient(),
                max_iterations=10,
                request_timeout_seconds=30,
                tool_timeout_seconds=5,
            )

            sig_a = agent._tool_call_signature(
                "browser_navigate",
                {
                    "url": "https://mail.google.com",
                    "run_manager": object(),
                    "config": {"thread_id": "request-a"},
                },
            )
            sig_b = agent._tool_call_signature(
                "browser_navigate",
                {
                    "url": "https://mail.google.com",
                    "run_manager": object(),
                    "config": {"thread_id": "request-b"},
                },
            )
            self.assertEqual(sig_a, sig_b)

    def test_stringify_tool_result_truncates_large_payloads(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            agent = LangGraphAgent(
                db_path=Path(tmp_dir) / "graph.db",
                llm_client=_DummyLLMClient(),
                max_iterations=10,
                request_timeout_seconds=30,
                tool_timeout_seconds=5,
            )

            raw = "x" * (MAX_TOOL_RESULT_CHARS + 123)
            rendered = agent._stringify_tool_result(raw)
            self.assertIn("[tool output truncated:", rendered)
            self.assertLess(len(rendered), len(raw))

    def test_extract_final_answer_skips_empty_ai_messages(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            agent = LangGraphAgent(
                db_path=Path(tmp_dir) / "graph.db",
                llm_client=_DummyLLMClient(),
                max_iterations=10,
                request_timeout_seconds=30,
                tool_timeout_seconds=5,
            )

            messages: list[BaseMessage] = [
                HumanMessage(content="do it"),
                AIMessage(content=""),
                AIMessage(content="   "),
                AIMessage(content="Done."),
            ]
            self.assertEqual(agent._extract_final_answer(messages), "Done.")


if __name__ == "__main__":
    unittest.main()
