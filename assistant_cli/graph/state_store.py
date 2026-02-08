from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
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
    parent_run_id: str | None
    resume_from_run_id: str | None
    metadata: dict[str, object] | None


@dataclass(slots=True)
class GraphCheckpointRecord:
    id: int
    run_id: str
    graph_id: str
    node_id: str
    status: str
    input_payload: object
    output_payload: object
    error_text: str | None
    context_payload: object
    next_node_id: str | None
    step: int
    retry_count: int
    created_at: str


@dataclass(slots=True)
class GraphScheduleRecord:
    schedule_id: str
    graph_id: str
    cron_expr: str
    guarantee_mode: str
    input_payload: object
    enabled: bool
    next_run_at: str
    last_run_at: str | None
    last_error: str | None
    created_at: str
    updated_at: str


class GraphStateStore:
    """SQLite persistence for graph definitions, runs, checkpoints, schedules, and state."""

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
                    finished_at TEXT,
                    parent_run_id TEXT,
                    resume_from_run_id TEXT,
                    metadata_json TEXT
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
                    context_json TEXT,
                    next_node_id TEXT,
                    step INTEGER NOT NULL DEFAULT 0,
                    retry_count INTEGER NOT NULL DEFAULT 0,
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

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_schedules (
                    schedule_id TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    cron_expr TEXT NOT NULL,
                    guarantee_mode TEXT NOT NULL,
                    input_json TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    next_run_at TEXT NOT NULL,
                    last_run_at TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_runs_graph_started
                ON graph_runs(graph_id, started_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_checkpoints_run
                ON graph_checkpoints(run_id, id DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_events_graph_created
                ON graph_events(graph_id, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_schedules_next_run
                ON graph_schedules(enabled, next_run_at)
                """
            )

            # Backward-compatible migrations for preexisting local databases.
            self._ensure_column(conn, "graph_runs", "parent_run_id TEXT")
            self._ensure_column(conn, "graph_runs", "resume_from_run_id TEXT")
            self._ensure_column(conn, "graph_runs", "metadata_json TEXT")
            self._ensure_column(conn, "graph_checkpoints", "context_json TEXT")
            self._ensure_column(conn, "graph_checkpoints", "next_node_id TEXT")
            self._ensure_column(conn, "graph_checkpoints", "step INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "graph_checkpoints", "retry_count INTEGER NOT NULL DEFAULT 0")

            conn.commit()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column_def: str) -> None:
        column_name = column_def.split()[0]
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row["name"] for row in rows}
        if column_name in existing:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")

    def save_graph_definition(self, graph: dict[str, Any]) -> None:
        graph_id = str(graph.get("id") or "").strip()
        if not graph_id:
            raise ValueError("Graph definition must contain a non-empty 'id'.")
        name = str(graph.get("name") or "").strip() or graph_id
        serialized = self._dump(graph)
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
        return self._load(row["definition_json"])

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
        *,
        graph_id: str,
        guarantee_mode: str,
        input_payload: object,
        parent_run_id: str | None = None,
        resume_from_run_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        run_id = f"run-{uuid.uuid4()}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_runs (
                    run_id, graph_id, guarantee_mode, status, input_json,
                    parent_run_id, resume_from_run_id, metadata_json
                )
                VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
                """,
                (
                    run_id,
                    graph_id,
                    guarantee_mode,
                    self._dump(input_payload),
                    parent_run_id,
                    resume_from_run_id,
                    self._dump(metadata),
                ),
            )
            conn.execute(
                """
                INSERT INTO graph_events (graph_id, run_id, event_type, payload_json)
                VALUES (?, ?, 'run_started', ?)
                """,
                (
                    graph_id,
                    run_id,
                    self._dump(
                        {
                            "guarantee_mode": guarantee_mode,
                            "parent_run_id": parent_run_id,
                            "resume_from_run_id": resume_from_run_id,
                        }
                    ),
                ),
            )
            conn.commit()
        return run_id

    def finish_run(
        self,
        *,
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
                (status, self._dump(output_payload), error_text, run_id),
            )
            conn.execute(
                """
                INSERT INTO graph_events (graph_id, run_id, event_type, payload_json)
                VALUES (?, ?, 'run_finished', ?)
                """,
                (
                    graph_id,
                    run_id,
                    self._dump({"status": status, "error": error_text}),
                ),
            )
            conn.commit()

    def get_run(self, run_id: str) -> GraphRunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, graph_id, guarantee_mode, status, input_json, output_json, error_text,
                       started_at, finished_at, parent_run_id, resume_from_run_id, metadata_json
                FROM graph_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_run(row)

    def read_prior_runs(self, graph_id: str, limit: int = 5) -> list[GraphRunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, graph_id, guarantee_mode, status, input_json, output_json, error_text,
                       started_at, finished_at, parent_run_id, resume_from_run_id, metadata_json
                FROM graph_runs
                WHERE graph_id = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (graph_id, max(1, limit)),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

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
        context_payload: object = None,
        next_node_id: str | None = None,
        step: int = 0,
        retry_count: int = 0,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_checkpoints (
                    run_id, graph_id, node_id, status, input_json, output_json,
                    error_text, context_json, next_node_id, step, retry_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    graph_id,
                    node_id,
                    status,
                    self._dump(input_payload),
                    self._dump(output_payload),
                    error_text,
                    self._dump(context_payload),
                    next_node_id,
                    int(step),
                    int(retry_count),
                ),
            )
            conn.commit()

    def list_checkpoints(self, run_id: str, limit: int = 200) -> list[GraphCheckpointRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, run_id, graph_id, node_id, status, input_json, output_json,
                       error_text, context_json, next_node_id, step, retry_count, created_at
                FROM graph_checkpoints
                WHERE run_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (run_id, max(1, limit)),
            ).fetchall()
        return [self._row_to_checkpoint(row) for row in rows]

    def latest_checkpoint(self, run_id: str) -> GraphCheckpointRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, run_id, graph_id, node_id, status, input_json, output_json,
                       error_text, context_json, next_node_id, step, retry_count, created_at
                FROM graph_checkpoints
                WHERE run_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_checkpoint(row)

    def latest_error_checkpoint(self, run_id: str) -> GraphCheckpointRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, run_id, graph_id, node_id, status, input_json, output_json,
                       error_text, context_json, next_node_id, step, retry_count, created_at
                FROM graph_checkpoints
                WHERE run_id = ? AND status = 'error'
                ORDER BY id DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_checkpoint(row)

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
                "SELECT value_json FROM graph_state_kv WHERE graph_id = ? AND key = ?",
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

    def create_schedule(
        self,
        *,
        graph_id: str,
        cron_expr: str,
        guarantee_mode: str,
        input_payload: object,
        next_run_at: datetime,
        enabled: bool = True,
    ) -> str:
        schedule_id = f"sched-{uuid.uuid4()}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_schedules (
                    schedule_id, graph_id, cron_expr, guarantee_mode, input_json, enabled, next_run_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    schedule_id,
                    graph_id,
                    cron_expr,
                    guarantee_mode,
                    self._dump(input_payload),
                    int(enabled),
                    next_run_at.isoformat(),
                ),
            )
            conn.commit()
        return schedule_id

    def get_schedule(self, schedule_id: str) -> GraphScheduleRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT schedule_id, graph_id, cron_expr, guarantee_mode, input_json, enabled,
                       next_run_at, last_run_at, last_error, created_at, updated_at
                FROM graph_schedules
                WHERE schedule_id = ?
                """,
                (schedule_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_schedule(row)

    def list_schedules(self, enabled_only: bool = False, limit: int = 200) -> list[GraphScheduleRecord]:
        with self._connect() as conn:
            if enabled_only:
                rows = conn.execute(
                    """
                    SELECT schedule_id, graph_id, cron_expr, guarantee_mode, input_json, enabled,
                           next_run_at, last_run_at, last_error, created_at, updated_at
                    FROM graph_schedules
                    WHERE enabled = 1
                    ORDER BY next_run_at ASC
                    LIMIT ?
                    """,
                    (max(1, limit),),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT schedule_id, graph_id, cron_expr, guarantee_mode, input_json, enabled,
                           next_run_at, last_run_at, last_error, created_at, updated_at
                    FROM graph_schedules
                    ORDER BY next_run_at ASC
                    LIMIT ?
                    """,
                    (max(1, limit),),
                ).fetchall()
        return [self._row_to_schedule(row) for row in rows]

    def due_schedules(self, now: datetime, limit: int = 20) -> list[GraphScheduleRecord]:
        now_iso = now.isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT schedule_id, graph_id, cron_expr, guarantee_mode, input_json, enabled,
                       next_run_at, last_run_at, last_error, created_at, updated_at
                FROM graph_schedules
                WHERE enabled = 1 AND next_run_at <= ?
                ORDER BY next_run_at ASC
                LIMIT ?
                """,
                (now_iso, max(1, limit)),
            ).fetchall()
        return [self._row_to_schedule(row) for row in rows]

    def update_schedule_after_run(
        self,
        *,
        schedule_id: str,
        next_run_at: datetime,
        last_error: str | None,
        last_run_at: datetime,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE graph_schedules
                SET next_run_at = ?,
                    last_run_at = ?,
                    last_error = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE schedule_id = ?
                """,
                (
                    next_run_at.isoformat(),
                    last_run_at.isoformat(),
                    last_error,
                    schedule_id,
                ),
            )
            conn.commit()

    def set_schedule_enabled(self, schedule_id: str, enabled: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE graph_schedules
                SET enabled = ?, updated_at = CURRENT_TIMESTAMP
                WHERE schedule_id = ?
                """,
                (int(enabled), schedule_id),
            )
            conn.commit()

    def delete_schedule(self, schedule_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM graph_schedules WHERE schedule_id = ?", (schedule_id,))
            conn.commit()

    def _row_to_run(self, row: sqlite3.Row) -> GraphRunRecord:
        metadata = self._load(row["metadata_json"])
        return GraphRunRecord(
            run_id=row["run_id"],
            graph_id=row["graph_id"],
            guarantee_mode=row["guarantee_mode"],
            status=row["status"],
            input_payload=self._load(row["input_json"]),
            output_payload=self._load(row["output_json"]),
            error_text=row["error_text"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            parent_run_id=row["parent_run_id"],
            resume_from_run_id=row["resume_from_run_id"],
            metadata=metadata if isinstance(metadata, dict) else None,
        )

    def _row_to_checkpoint(self, row: sqlite3.Row) -> GraphCheckpointRecord:
        return GraphCheckpointRecord(
            id=int(row["id"]),
            run_id=row["run_id"],
            graph_id=row["graph_id"],
            node_id=row["node_id"],
            status=row["status"],
            input_payload=self._load(row["input_json"]),
            output_payload=self._load(row["output_json"]),
            error_text=row["error_text"],
            context_payload=self._load(row["context_json"]),
            next_node_id=row["next_node_id"],
            step=int(row["step"] or 0),
            retry_count=int(row["retry_count"] or 0),
            created_at=row["created_at"],
        )

    def _row_to_schedule(self, row: sqlite3.Row) -> GraphScheduleRecord:
        return GraphScheduleRecord(
            schedule_id=row["schedule_id"],
            graph_id=row["graph_id"],
            cron_expr=row["cron_expr"],
            guarantee_mode=row["guarantee_mode"],
            input_payload=self._load(row["input_json"]),
            enabled=bool(row["enabled"]),
            next_run_at=row["next_run_at"],
            last_run_at=row["last_run_at"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _dump(self, value: object) -> str:
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            sanitized = self._sanitize_for_json(value)
            return json.dumps(sanitized, ensure_ascii=False)

    def _load(self, value: str | None) -> object:
        if value is None:
            return None
        return json.loads(value)

    def _sanitize_for_json(self, value: object) -> object:
        if isinstance(value, dict):
            return {str(key): self._sanitize_for_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize_for_json(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_for_json(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
