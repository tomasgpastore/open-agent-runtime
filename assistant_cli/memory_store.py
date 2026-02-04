from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, messages_from_dict, messages_to_dict


@dataclass(slots=True)
class MemoryStats:
    estimated_tokens: int
    token_limit: int
    context_window_target: int
    recent_turns_kept: int
    truncation_occurred_last_turn: bool


class SQLiteMemoryStore:
    """Persists rolling short-term memory in SQLite."""

    def __init__(self, db_path: Path, session_id: str, token_limit: int, context_window: int) -> None:
        self._db_path = db_path
        self._session_id = session_id
        self._token_limit = token_limit
        self._context_window = context_window
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_state (
                    session_id TEXT PRIMARY KEY,
                    messages_json TEXT NOT NULL,
                    truncation_occurred_last_turn INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def load_messages(self) -> list[BaseMessage]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT messages_json FROM conversation_state WHERE session_id = ?",
                (self._session_id,),
            ).fetchone()

        if not row:
            return []

        data = json.loads(row["messages_json"])
        return list(messages_from_dict(data))

    def save_messages(self, messages: Sequence[BaseMessage], truncation_occurred: bool) -> None:
        serialized = json.dumps(messages_to_dict(list(messages)))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_state (session_id, messages_json, truncation_occurred_last_turn)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    messages_json = excluded.messages_json,
                    truncation_occurred_last_turn = excluded.truncation_occurred_last_turn,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (self._session_id, serialized, int(truncation_occurred)),
            )
            conn.commit()

    def clear_session(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversation_state WHERE session_id = ?", (self._session_id,))
            conn.commit()

    def enforce_token_limit(self, messages: Sequence[BaseMessage]) -> tuple[list[BaseMessage], bool]:
        working = list(messages)
        truncated = False

        # Keep at least the most recent message so the current user turn is never dropped.
        while len(working) > 1 and self.estimate_tokens(working) > self._token_limit:
            working.pop(0)
            truncated = True

        return working, truncated

    def estimate_tokens(self, messages: Sequence[BaseMessage] | None = None) -> int:
        target = list(messages) if messages is not None else self.load_messages()
        if not target:
            return 0

        total_chars = 0
        for message in messages_to_dict(target):
            total_chars += len(json.dumps(message, ensure_ascii=False))

        # Fast, lightweight heuristic: ~4 chars per token.
        return max(1, math.ceil(total_chars / 4))

    def stats(self) -> MemoryStats:
        messages = self.load_messages()
        turns = sum(1 for message in messages if isinstance(message, HumanMessage))

        with self._connect() as conn:
            row = conn.execute(
                "SELECT truncation_occurred_last_turn FROM conversation_state WHERE session_id = ?",
                (self._session_id,),
            ).fetchone()

        truncation_flag = bool(row["truncation_occurred_last_turn"]) if row else False
        return MemoryStats(
            estimated_tokens=self.estimate_tokens(messages),
            token_limit=self._token_limit,
            context_window_target=self._context_window,
            recent_turns_kept=turns,
            truncation_occurred_last_turn=truncation_flag,
        )
