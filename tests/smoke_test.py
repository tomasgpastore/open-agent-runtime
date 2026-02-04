from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from langchain_core.messages import AIMessage, HumanMessage

from assistant_cli.approval import ApprovalManager
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


class ApprovalManagerTests(unittest.TestCase):
    def test_tool_override_beats_global(self) -> None:
        approval = ApprovalManager()
        approval.set_global(False)
        approval.set_tool("memory_read_graph", True)

        self.assertTrue(approval.tool_enabled("memory_read_graph"))
        self.assertFalse(approval.tool_enabled("filesystem_read_file"))


if __name__ == "__main__":
    unittest.main()
