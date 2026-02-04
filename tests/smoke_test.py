from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from assistant_cli.approval import ApprovalManager
from assistant_cli.mcp_manager import MCPManager
from assistant_cli.memory_store import SQLiteMemoryStore


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


if __name__ == "__main__":
    unittest.main()
