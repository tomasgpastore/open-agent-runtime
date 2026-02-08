from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sqlite3
from typing import Iterable


WORD_RE = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "your",
    "you",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "but",
    "can",
    "could",
    "would",
    "should",
    "about",
    "into",
    "also",
    "just",
    "then",
    "what",
    "when",
    "where",
    "which",
}


@dataclass(slots=True)
class DailyMemorySearchResult:
    day: str
    chunk_index: int
    content: str
    keywords: list[str]


class DailyMemoryArchive:
    """Append-only daily markdown archive with chunk index for retrieval."""

    def __init__(
        self,
        *,
        root_dir: Path,
        db_path: Path,
        chunk_tokens: int = 400,
        overlap_tokens: int = 80,
    ) -> None:
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
            raise ValueError("overlap_tokens must be >=0 and < chunk_tokens")

        self._root_dir = root_dir
        self._db_path = db_path
        self._chunk_tokens = chunk_tokens
        self._overlap_tokens = overlap_tokens

        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def append_exchange(
        self,
        *,
        user_text: str,
        assistant_text: str,
        timestamp: datetime | None = None,
    ) -> Path:
        ts = timestamp or datetime.now().astimezone()
        day = ts.strftime("%Y-%m-%d")
        time_label = ts.strftime("%H:%M:%S")
        path = self._root_dir / f"{day}.md"

        entry = (
            f"## {time_label}\n\n"
            f"### User\n"
            f"{user_text.strip() or '(empty)'}\n\n"
            f"### Anton\n"
            f"{assistant_text.strip() or '(empty)'}\n\n"
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry)

        self._index_entry(day=day, text=entry)
        return path

    def recent_days(self, limit: int = 7) -> list[str]:
        files = sorted(self._root_dir.glob("*.md"), reverse=True)
        return [item.stem for item in files[: max(1, limit)]]

    def search(
        self,
        *,
        query: str,
        day: str | None = None,
        limit: int = 5,
    ) -> list[DailyMemorySearchResult]:
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        sql = (
            "SELECT day, chunk_index, content, keywords "
            "FROM daily_memory_chunks "
            "WHERE (content LIKE ? OR keywords LIKE ?)"
        )
        params: list[object] = [f"%{cleaned_query}%", f"%{cleaned_query}%"]
        if day:
            sql += " AND day = ?"
            params.append(day)
        sql += " ORDER BY day DESC, chunk_index ASC LIMIT ?"
        params.append(max(1, limit))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()

        results: list[DailyMemorySearchResult] = []
        for row in rows:
            keywords = [part for part in (row["keywords"] or "").split(",") if part]
            results.append(
                DailyMemorySearchResult(
                    day=row["day"],
                    chunk_index=int(row["chunk_index"]),
                    content=row["content"],
                    keywords=keywords,
                )
            )
        return results

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_memory_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    day TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_daily_memory_day_chunk
                ON daily_memory_chunks(day, chunk_index)
                """
            )
            conn.commit()

    def _index_entry(self, *, day: str, text: str) -> None:
        tokens = self._tokenize(text)
        chunks = self._chunk_tokens_with_overlap(tokens)
        if not chunks:
            return

        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(chunk_index), -1) AS max_chunk FROM daily_memory_chunks WHERE day = ?",
                (day,),
            ).fetchone()
            start_index = int(row["max_chunk"]) + 1 if row else 0

            for offset, chunk_tokens in enumerate(chunks):
                chunk_text = " ".join(chunk_tokens).strip()
                keywords = ",".join(self._extract_keywords(chunk_tokens))
                conn.execute(
                    """
                    INSERT INTO daily_memory_chunks (day, chunk_index, content, keywords)
                    VALUES (?, ?, ?, ?)
                    """,
                    (day, start_index + offset, chunk_text, keywords),
                )
            conn.commit()

    def _tokenize(self, text: str) -> list[str]:
        return WORD_RE.findall(text)

    def _chunk_tokens_with_overlap(self, tokens: list[str]) -> list[list[str]]:
        if not tokens:
            return []

        chunks: list[list[str]] = []
        step = self._chunk_tokens - self._overlap_tokens
        index = 0
        while index < len(tokens):
            chunk = tokens[index : index + self._chunk_tokens]
            if not chunk:
                break
            chunks.append(chunk)
            if len(chunk) < self._chunk_tokens:
                break
            index += step
        return chunks

    def _extract_keywords(self, tokens: Iterable[str], limit: int = 12) -> list[str]:
        lowered = [token.lower() for token in tokens]
        filtered = [token for token in lowered if len(token) >= 4 and token not in STOPWORDS]
        if not filtered:
            return []
        counts = Counter(filtered)
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [item[0] for item in ordered[:limit]]
