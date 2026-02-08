from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assistant_cli.daily_memory import DailyMemoryArchive


class DailyMemoryArchiveTests(unittest.TestCase):
    def test_append_creates_daily_markdown_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "daily"
            db_path = Path(tmp_dir) / "assistant.db"
            archive = DailyMemoryArchive(root_dir=root, db_path=db_path)

            ts = datetime(2026, 2, 8, 10, 30, 0)
            path = archive.append_exchange(
                user_text="Hi Anton",
                assistant_text="Hello Tomas",
                timestamp=ts,
            )

            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("### User", text)
            self.assertIn("Hi Anton", text)
            self.assertIn("### Anton", text)
            self.assertIn("Hello Tomas", text)

    def test_search_returns_indexed_content(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "daily"
            db_path = Path(tmp_dir) / "assistant.db"
            archive = DailyMemoryArchive(root_dir=root, db_path=db_path)

            ts = datetime(2026, 2, 8, 11, 0, 0)
            archive.append_exchange(
                user_text="Please remind me about Canvas homework automation",
                assistant_text="I will check Canvas and report due assignments.",
                timestamp=ts,
            )

            results = archive.search(query="Canvas", day="2026-02-08", limit=5)
            self.assertGreaterEqual(len(results), 1)
            joined = "\n".join(result.content for result in results)
            self.assertIn("Canvas", joined)

    def test_chunking_with_overlap_produces_multiple_chunks(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "daily"
            db_path = Path(tmp_dir) / "assistant.db"
            archive = DailyMemoryArchive(
                root_dir=root,
                db_path=db_path,
                chunk_tokens=12,
                overlap_tokens=4,
            )

            long_user_text = " ".join(f"token{i}" for i in range(40))
            archive.append_exchange(
                user_text=long_user_text,
                assistant_text="ack",
                timestamp=datetime(2026, 2, 8, 12, 0, 0),
            )

            results = archive.search(query="token25", day="2026-02-08", limit=20)
            self.assertGreaterEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
