from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GraphRunRecord:
    run_id: str
    graph_id: str
    guarantee_mode: str
    status: str
    input_payload: object
    output_payload: object
    error_text: str | None
    started_at: str
    finished_at: str | None


class GraphStateStore:
    """SQLite-backed persistence for graph definitions, runs, checkpoints, and state."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_definitions (
                    graph_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    definition_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_runs (
                    run_id TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    guarantee_mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_json TEXT,
                    output_json TEXT,
                    error_text TEXT,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    finished_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_json TEXT,
                    output_json TEXT,
                    error_text TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_state_kv (
                    graph_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    source_run_id TEXT,
                    source_node_id TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (graph_id, key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    graph_id TEXT NOT NULL,
                    run_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def save_graph_definition(self, graph: dict[str, Any]) -> None:
        graph_id = str(graph.get("id") or "").strip()
        name = str(graph.get("name") or "").strip() or graph_id
        serialized = json.dumps(graph, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_definitions (graph_id, name, definition_json)
                VALUES (?, ?, ?)
                ON CONFLICT(graph_id) DO UPDATE SET
                    name = excluded.name,
                    definition_json = excluded.definition_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (graph_id, name, serialized),
            )
            conn.commit()

    def get_graph_definition(self, graph_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT definition_json FROM graph_definitions WHERE graph_id = ?",
                (graph_id,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["definition_json"])

    def list_graph_definitions(self, limit: int = 50) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT graph_id, name, updated_at
                FROM graph_definitions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [
            {
                "graph_id": row["graph_id"],
                "name": row["name"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def start_run(
        self,
        graph_id: str,
        guarantee_mode: str,
        input_payload: object,
    ) -> str:
        run_id = f"run-{uuid.uuid4()}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_runs (run_id, graph_id, guarantee_mode, status, input_json)
                VALUES (?, ?, ?, 'running', ?)
                """,
                (run_id, graph_id, guarantee_mode, self._dump(input_payload)),
            )
            conn.execute(
                """
                INSERT INTO graph_events (graph_id, run_id, event_type, payload_json)
                VALUES (?, ?, 'run_started', ?)
                """,
                (
                    graph_id,
                    run_id,
                    self._dump({"guarantee_mode": guarantee_mode}),
                ),
            )
            conn.commit()
        return run_id

    def finish_run(
        self,
        run_id: str,
        status: str,
        output_payload: object = None,
        error_text: str | None = None,
    ) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT graph_id FROM graph_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if not row:
                return
            graph_id = row["graph_id"]
            conn.execute(
                """
                UPDATE graph_runs
                SET status = ?, output_json = ?, error_text = ?, finished_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
                """,
                (
                    status,
                    self._dump(output_payload),
                    error_text,
                    run_id,
                ),
            )
            conn.execute(
                """
                INSERT INTO graph_events (graph_id, run_id, event_type, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    graph_id,
                    run_id,
                    "run_finished",
                    self._dump({"status": status, "error": error_text}),
                ),
            )
            conn.commit()

    def add_checkpoint(
        self,
        *,
        run_id: str,
        graph_id: str,
        node_id: str,
        status: str,
        input_payload: object,
        output_payload: object,
        error_text: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_checkpoints (
                    run_id, graph_id, node_id, status, input_json, output_json, error_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    graph_id,
                    node_id,
                    status,
                    self._dump(input_payload),
                    self._dump(output_payload),
                    error_text,
                ),
            )
            conn.commit()

    def write_state(
        self,
        *,
        graph_id: str,
        key: str,
        value: object,
        source_run_id: str | None,
        source_node_id: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_state_kv (graph_id, key, value_json, source_run_id, source_node_id)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(graph_id, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    source_run_id = excluded.source_run_id,
                    source_node_id = excluded.source_node_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    graph_id,
                    key,
                    self._dump(value),
                    source_run_id,
                    source_node_id,
                ),
            )
            conn.execute(
                """
                INSERT INTO graph_events (graph_id, run_id, event_type, payload_json)
                VALUES (?, ?, 'state_written', ?)
                """,
                (
                    graph_id,
                    source_run_id,
                    self._dump({"key": key, "source_node_id": source_node_id}),
                ),
            )
            conn.commit()

    def read_state(self, graph_id: str, key: str) -> object | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT value_json
                FROM graph_state_kv
                WHERE graph_id = ? AND key = ?
                """,
                (graph_id, key),
            ).fetchone()
        if not row:
            return None
        return self._load(row["value_json"])

    def list_state(self, graph_id: str) -> dict[str, object]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key, value_json
                FROM graph_state_kv
                WHERE graph_id = ?
                ORDER BY key ASC
                """,
                (graph_id,),
            ).fetchall()
        return {row["key"]: self._load(row["value_json"]) for row in rows}

    def read_prior_runs(self, graph_id: str, limit: int = 5) -> list[GraphRunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, graph_id, guarantee_mode, status, input_json, output_json, error_text,
                       started_at, finished_at
                FROM graph_runs
                WHERE graph_id = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (graph_id, max(1, limit)),
            ).fetchall()
        records: list[GraphRunRecord] = []
        for row in rows:
            records.append(
                GraphRunRecord(
                    run_id=row["run_id"],
                    graph_id=row["graph_id"],
                    guarantee_mode=row["guarantee_mode"],
                    status=row["status"],
                    input_payload=self._load(row["input_json"]),
                    output_payload=self._load(row["output_json"]),
                    error_text=row["error_text"],
                    started_at=row["started_at"],
                    finished_at=row["finished_at"],
                )
            )
        return records

    def list_events(self, graph_id: str, limit: int = 50) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, event_type, payload_json, created_at
                FROM graph_events
                WHERE graph_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (graph_id, max(1, limit)),
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "event_type": row["event_type"],
                "payload": self._load(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _dump(self, value: object) -> str:
        return json.dumps(value, ensure_ascii=False)

    def _load(self, value: str | None) -> object:
        if value is None:
            return None
        return json.loads(value)
