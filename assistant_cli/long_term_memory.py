from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
import uuid
from typing import Iterable, Protocol

from assistant_cli.daily_memory import DailyMemoryArchive, DailyMemorySearchResult


@dataclass(slots=True)
class LongTermFact:
    fact_id: str
    namespace: str
    content: str
    keywords: list[str]
    importance: int
    ttl_days: int | None
    created_at: str
    updated_at: str
    expires_at: str | None


@dataclass(slots=True)
class RetrievedMemoryItem:
    source: str
    score: float
    content: str
    metadata: dict[str, object]


class EmbeddingIndex(Protocol):
    def index_fact(self, fact_id: str, content: str) -> None: ...

    def delete_fact(self, fact_id: str) -> None: ...

    def search(self, query: str, limit: int = 5) -> list[tuple[str, float]]: ...


class LongTermMemoryStore:
    """Persistent long-term fact memory with lifecycle operations."""

    def __init__(self, db_path: Path, embedding_index: EmbeddingIndex | None = None) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._embedding_index = embedding_index
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_facts (
                    fact_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    content TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    importance INTEGER NOT NULL DEFAULT 3,
                    ttl_days INTEGER,
                    expires_at TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_long_term_namespace_updated
                ON long_term_facts(namespace, updated_at DESC)
                """
            )
            conn.commit()

    def add_fact(
        self,
        *,
        namespace: str,
        content: str,
        importance: int = 3,
        ttl_days: int | None = None,
        fact_id: str | None = None,
    ) -> str:
        normalized_content = " ".join(content.strip().split())
        if not namespace.strip() or not normalized_content:
            raise ValueError("namespace and content are required")

        actual_fact_id = fact_id or f"fact-{uuid.uuid4()}"
        keywords = self._extract_keywords(normalized_content)
        expires_at: str | None = None
        if ttl_days is not None:
            if ttl_days <= 0:
                raise ValueError("ttl_days must be positive")
            expires_at = (datetime.now().astimezone() + timedelta(days=ttl_days)).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO long_term_facts (
                    fact_id, namespace, content, keywords_json, importance, ttl_days, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_id) DO UPDATE SET
                    namespace = excluded.namespace,
                    content = excluded.content,
                    keywords_json = excluded.keywords_json,
                    importance = excluded.importance,
                    ttl_days = excluded.ttl_days,
                    expires_at = excluded.expires_at,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    actual_fact_id,
                    namespace.strip(),
                    normalized_content,
                    json.dumps(keywords, ensure_ascii=False),
                    int(max(1, min(5, importance))),
                    ttl_days,
                    expires_at,
                ),
            )
            conn.commit()

        if self._embedding_index is not None:
            self._embedding_index.index_fact(actual_fact_id, normalized_content)

        return actual_fact_id

    def delete_fact(self, fact_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM long_term_facts WHERE fact_id = ?", (fact_id,))
            conn.commit()
        removed = cursor.rowcount > 0
        if removed and self._embedding_index is not None:
            self._embedding_index.delete_fact(fact_id)
        return removed

    def get_fact(self, fact_id: str) -> LongTermFact | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT fact_id, namespace, content, keywords_json, importance, ttl_days,
                       created_at, updated_at, expires_at
                FROM long_term_facts
                WHERE fact_id = ?
                """,
                (fact_id,),
            ).fetchone()
        return self._row_to_fact(row) if row else None

    def list_facts(self, namespace: str | None = None, limit: int = 100) -> list[LongTermFact]:
        with self._connect() as conn:
            if namespace:
                rows = conn.execute(
                    """
                    SELECT fact_id, namespace, content, keywords_json, importance, ttl_days,
                           created_at, updated_at, expires_at
                    FROM long_term_facts
                    WHERE namespace = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (namespace, max(1, limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT fact_id, namespace, content, keywords_json, importance, ttl_days,
                           created_at, updated_at, expires_at
                    FROM long_term_facts
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (max(1, limit),),
                ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def search(self, query: str, namespace: str | None = None, limit: int = 10) -> list[LongTermFact]:
        cleaned = query.strip()
        if not cleaned:
            return []

        sql = (
            "SELECT fact_id, namespace, content, keywords_json, importance, ttl_days, "
            "created_at, updated_at, expires_at "
            "FROM long_term_facts "
            "WHERE (content LIKE ? OR keywords_json LIKE ?)"
        )
        params: list[object] = [f"%{cleaned}%", f"%{cleaned}%"]
        if namespace:
            sql += " AND namespace = ?"
            params.append(namespace)
        sql += " ORDER BY importance DESC, updated_at DESC LIMIT ?"
        params.append(max(1, limit))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()

        facts = [self._row_to_fact(row) for row in rows]

        if self._embedding_index is not None and len(facts) < limit:
            existing_ids = {fact.fact_id for fact in facts}
            for fact_id, _score in self._embedding_index.search(cleaned, limit=limit):
                if fact_id in existing_ids:
                    continue
                fact = self.get_fact(fact_id)
                if fact is not None:
                    facts.append(fact)
                if len(facts) >= limit:
                    break

        return facts[:limit]

    def prune_expired(self, now: datetime | None = None) -> int:
        ts = (now or datetime.now().astimezone()).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT fact_id FROM long_term_facts WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (ts,),
            ).fetchall()
            fact_ids = [row["fact_id"] for row in rows]
            if fact_ids:
                conn.executemany("DELETE FROM long_term_facts WHERE fact_id = ?", [(fid,) for fid in fact_ids])
            conn.commit()

        if self._embedding_index is not None:
            for fact_id in fact_ids:
                self._embedding_index.delete_fact(fact_id)
        return len(fact_ids)

    def clear_all(self) -> int:
        with self._connect() as conn:
            rows = conn.execute("SELECT fact_id FROM long_term_facts").fetchall()
            fact_ids = [row["fact_id"] for row in rows]
            conn.execute("DELETE FROM long_term_facts")
            conn.commit()
        if self._embedding_index is not None:
            for fact_id in fact_ids:
                self._embedding_index.delete_fact(fact_id)
        return len(fact_ids)

    def _row_to_fact(self, row: sqlite3.Row) -> LongTermFact:
        keywords_raw = row["keywords_json"]
        keywords = json.loads(keywords_raw) if isinstance(keywords_raw, str) else []
        return LongTermFact(
            fact_id=row["fact_id"],
            namespace=row["namespace"],
            content=row["content"],
            keywords=keywords if isinstance(keywords, list) else [],
            importance=int(row["importance"]),
            ttl_days=int(row["ttl_days"]) if row["ttl_days"] is not None else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
        )

    def _extract_keywords(self, content: str, limit: int = 12) -> list[str]:
        tokens = [token.strip(".,!?;:\"'()[]{}") for token in content.lower().split()]
        tokens = [token for token in tokens if len(token) >= 4]
        if not tokens:
            return []
        counts = Counter(tokens)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [item[0] for item in ranked[:limit]]


class MemoryRetrievalPlanner:
    """Combines hot/cold memory retrieval paths for user queries."""

    def __init__(self, *, daily_archive: DailyMemoryArchive, long_term_store: LongTermMemoryStore) -> None:
        self._daily_archive = daily_archive
        self._long_term_store = long_term_store

    def retrieve(self, *, query: str, limit: int = 8, day: str | None = None) -> list[RetrievedMemoryItem]:
        cleaned_query = " ".join(query.split())
        if not cleaned_query:
            return []

        combined: list[RetrievedMemoryItem] = []
        seen_daily: set[tuple[str, int]] = set()
        query_terms = self._query_terms(cleaned_query)
        daily_queries = [cleaned_query] + [term for term in query_terms if term != cleaned_query]

        daily_rank = 0
        for q in daily_queries:
            daily_hits: list[DailyMemorySearchResult] = self._daily_archive.search(
                query=q,
                day=day,
                limit=limit,
            )
            for item in daily_hits:
                marker = (item.day, item.chunk_index)
                if marker in seen_daily:
                    continue
                seen_daily.add(marker)
                score = 1.0 - (daily_rank * 0.05)
                combined.append(
                    RetrievedMemoryItem(
                        source="daily",
                        score=max(0.01, score),
                        content=item.content,
                        metadata={
                            "day": item.day,
                            "chunk_index": item.chunk_index,
                            "keywords": item.keywords,
                        },
                    )
                )
                daily_rank += 1
                if daily_rank >= limit:
                    break
            if daily_rank >= limit:
                break

        fact_hits = self._long_term_store.search(query=cleaned_query, limit=limit)
        for index, fact in enumerate(fact_hits):
            score = 1.2 + (fact.importance * 0.1) - (index * 0.03)
            combined.append(
                RetrievedMemoryItem(
                    source="long_term",
                    score=score,
                    content=fact.content,
                    metadata={
                        "fact_id": fact.fact_id,
                        "namespace": fact.namespace,
                        "importance": fact.importance,
                        "keywords": fact.keywords,
                        "expires_at": fact.expires_at,
                    },
                )
            )

        combined.sort(key=lambda item: item.score, reverse=True)
        return combined[:limit]

    def _query_terms(self, query: str) -> list[str]:
        terms = []
        for raw in query.lower().split():
            token = raw.strip(".,!?;:\"'()[]{}")
            if len(token) < 3:
                continue
            if token not in terms:
                terms.append(token)
        return terms
