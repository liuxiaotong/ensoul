"""Hybrid memory/search索引 — 结合持久记忆与会话记录.

支持 PostgreSQL（pg_trgm GIN 索引）和 SQLite（FTS5）双后端。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ensoul.database import is_pg
from ensoul.memory import get_memory_store
from ensoul.paths import resolve_project_dir

logger = logging.getLogger(__name__)


def _prepare_index_text(text: str) -> str:
    """为 FTS 预处理文本，拆分 CJK 字符."""
    pieces: list[str] = []
    for ch in text:
        if ord(ch) > 127:
            pieces.extend([" ", ch, " "])
        else:
            pieces.append(ch)
    return "".join(pieces)


@dataclass
class IndexStats:
    memory_entries: int = 0
    session_messages: int = 0

    def total(self) -> int:
        return self.memory_entries + self.session_messages


class MemorySearchIndex:
    """基于 PG pg_trgm / SQLite FTS 的检索索引."""

    def __init__(
        self,
        db_path: Path | None = None,
        memory_dir: Path | None = None,
        session_dir: Path | None = None,
        *,
        project_dir: Path | None = None,
    ) -> None:
        root = resolve_project_dir(project_dir)
        self._use_pg = is_pg()
        self.db_path = db_path or root / ".crew" / "memory-index.db"
        # TODO: memory_dir 参数仅为兼容保留，rebuild 已改用 get_memory_store()
        self.memory_dir = memory_dir or root / ".crew" / "memory"
        self.session_dir = session_dir or root / ".crew" / "sessions"

    # ── 连接管理 ──

    def _connect_sqlite(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _setup_tables_sqlite(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                employee TEXT,
                kind TEXT,
                source TEXT,
                title TEXT,
                content TEXT,
                metadata TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
            USING fts5(content, title, employee, tokenize='unicode61')
            """
        )

    # ── PG 辅助 ──

    def _setup_tables_pg(self, conn) -> None:
        """PG 模式下 entries 表由 database.init_db() 统一创建，此处为空操作.

        DDL 定义集中在 database._PG_CREATE_ENTRIES，避免重复维护。
        """
        pass

    # ── rebuild ──

    def rebuild(self) -> IndexStats:
        """重建索引，返回计数."""
        if self._use_pg:
            return self._rebuild_pg()
        return self._rebuild_sqlite()

    def _rebuild_sqlite(self) -> IndexStats:
        conn = self._connect_sqlite()
        try:
            self._setup_tables_sqlite(conn)
            conn.execute("DELETE FROM entries")
            conn.execute("DELETE FROM entries_fts")
            stats = IndexStats()
            stats.memory_entries = self._index_memory_sqlite(conn)
            stats.session_messages = self._index_sessions_sqlite(conn)
            conn.commit()
            return stats
        finally:
            conn.close()

    def _rebuild_pg(self) -> IndexStats:
        from ensoul.database import get_pg_connection

        with get_pg_connection() as conn:
            self._setup_tables_pg(conn)
            cur = conn.cursor()
            cur.execute("DELETE FROM entries")
            stats = IndexStats()
            stats.memory_entries = self._index_memory_pg(conn)
            stats.session_messages = self._index_sessions_pg(conn)
        return stats

    # ── insert entry ──

    def _insert_entry_sqlite(
        self,
        conn: sqlite3.Connection,
        *,
        entry_id: str,
        employee: str,
        kind: str,
        source: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None,
        created_at: str,
    ) -> None:
        payload = (
            entry_id,
            employee,
            kind,
            source,
            title,
            content,
            json.dumps(metadata or {}, ensure_ascii=False),
            created_at,
        )
        conn.execute("INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)", payload)
        conn.execute(
            "INSERT INTO entries_fts(rowid, content, title, employee)"
            " VALUES ((SELECT rowid FROM entries WHERE id=?), ?, ?, ?)",
            (
                entry_id,
                _prepare_index_text(content),
                _prepare_index_text(title),
                employee,
            ),
        )

    def _insert_entry_pg(
        self,
        conn,
        *,
        entry_id: str,
        employee: str,
        kind: str,
        source: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None,
        created_at: str,
    ) -> None:
        cur = conn.cursor()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        cur.execute(
            """
            INSERT INTO entries (id, employee, kind, source, title, content, metadata, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                employee = EXCLUDED.employee,
                kind = EXCLUDED.kind,
                source = EXCLUDED.source,
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                created_at = EXCLUDED.created_at
            """,
            (entry_id, employee, kind, source, title, content, meta_json, created_at),
        )

    # 兼容旧调用签名（保留给 _index_memory / _index_sessions 内部使用）
    def _insert_entry(
        self,
        conn,
        *,
        entry_id: str,
        employee: str,
        kind: str,
        source: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None,
        created_at: str,
    ) -> None:
        if self._use_pg:
            self._insert_entry_pg(
                conn,
                entry_id=entry_id,
                employee=employee,
                kind=kind,
                source=source,
                title=title,
                content=content,
                metadata=metadata,
                created_at=created_at,
            )
        else:
            self._insert_entry_sqlite(
                conn,
                entry_id=entry_id,
                employee=employee,
                kind=kind,
                source=source,
                title=title,
                content=content,
                metadata=metadata,
                created_at=created_at,
            )

    # ── index memory ──

    def _index_memory_sqlite(self, conn: sqlite3.Connection) -> int:
        return self._index_memory_common(conn)

    def _index_memory_pg(self, conn) -> int:
        return self._index_memory_common(conn)

    def _index_memory_common(self, conn) -> int:
        # 通过 get_memory_store() 获取 store（DB 优先，兼容文件版）
        store = get_memory_store()
        total = 0
        for employee in store.list_employees():
            entries = store.query(employee, limit=10_000)
            for entry in entries:
                title = f"[{entry.category}] {entry.content[:40]}"
                self._insert_entry(
                    conn,
                    entry_id=f"memory:{entry.id}",
                    employee=employee,
                    kind="memory",
                    source=entry.source_session,
                    title=title,
                    content=entry.content,
                    metadata={"category": entry.category, "source_session": entry.source_session},
                    created_at=entry.created_at,
                )
                total += 1
        return total

    # ── index sessions ──

    def _index_sessions_sqlite(self, conn: sqlite3.Connection) -> int:
        return self._index_sessions_common(conn)

    def _index_sessions_pg(self, conn) -> int:
        return self._index_sessions_common(conn)

    def _index_sessions_common(self, conn) -> int:
        if not self.session_dir.is_dir():
            return 0
        total = 0
        for path in sorted(self.session_dir.glob("*.jsonl")):
            session_id = path.stem
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as e:
                logger.debug("读取 session 文件失败 %s: %s", path.name, e)
                continue
            session_subject = ""
            start_metadata: dict[str, Any] = {}
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("event") == "start" and not session_subject:
                    session_subject = data.get("subject", "")
                    start_metadata = data.get("metadata", {}) or {}
                    continue
                if data.get("event") != "message":
                    continue
                role = data.get("role", "")
                content = data.get("content", "")
                if not content:
                    continue
                metadata = data.get("metadata", {}) or {}
                employee = (
                    metadata.get("employee") or start_metadata.get("employee") or session_subject
                )
                title = f"{session_id} ({role})"
                entry_id = f"session:{session_id}:{idx}"
                self._insert_entry(
                    conn,
                    entry_id=entry_id,
                    employee=employee or "",
                    kind="session",
                    source=session_id,
                    title=title,
                    content=content,
                    metadata=metadata,
                    created_at=data.get("timestamp", ""),
                )
                total += 1
        return total

    # ── search ──

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        employee: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._use_pg:
            return self._search_pg(query, limit=limit, employee=employee, kind=kind)
        return self._search_sqlite(query, limit=limit, employee=employee, kind=kind)

    def _search_sqlite(
        self,
        query: str,
        *,
        limit: int,
        employee: str | None,
        kind: str | None,
    ) -> list[dict[str, Any]]:
        conn = self._connect_sqlite()
        try:
            self._setup_tables_sqlite(conn)
            fts_query = _prepare_index_text(query).strip()
            if not fts_query:
                return []
            sql = (
                "SELECT e.id, e.employee, e.kind, e.source, e.title, e.content, "
                "e.metadata, e.created_at, snippet(entries_fts, 0, '[', ']', '...', 10) as snippet "
                "FROM entries_fts JOIN entries e ON e.rowid = entries_fts.rowid "
                "WHERE entries_fts MATCH ?"
            )
            params: list[Any] = [fts_query]
            if employee:
                sql += " AND e.employee = ?"
                params.append(employee)
            if kind:
                sql += " AND e.kind = ?"
                params.append(kind)
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return self._rows_to_results(rows, has_snippet=True)
        finally:
            conn.close()

    def _search_pg(
        self,
        query: str,
        *,
        limit: int,
        employee: str | None,
        kind: str | None,
    ) -> list[dict[str, Any]]:
        from ensoul.database import get_pg_connection

        if not query.strip():
            return []

        # 用 ILIKE 做模糊匹配（对中文友好，不依赖分词器）
        # 按字符拆分搜索词，对 CJK 字符逐个匹配
        like_pattern = f"%{query.strip()}%"

        with get_pg_connection() as conn:
            import psycopg2.extras

            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # TODO: 性能优化 - ILIKE 全表扫描对大数据集性能差
            # 建议方案：
            # 1. 使用 pg_trgm 扩展 + GIN 索引：
            #    CREATE EXTENSION pg_trgm;
            #    CREATE INDEX idx_entries_content_trgm ON entries USING gin(content gin_trgm_ops);
            #    CREATE INDEX idx_entries_title_trgm ON entries USING gin(title gin_trgm_ops);
            # 2. 或使用 PostgreSQL 全文搜索（tsvector + GIN）
            sql = (
                "SELECT id, employee, kind, source, title, content, "
                "metadata, created_at "
                "FROM entries "
                "WHERE (content ILIKE %s OR title ILIKE %s)"
            )
            params: list[Any] = [like_pattern, like_pattern]

            if employee:
                sql += " AND employee = %s"
                params.append(employee)
            if kind:
                sql += " AND kind = %s"
                params.append(kind)

            sql += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

            cur.execute(sql, params)
            rows = cur.fetchall()

        return self._rows_to_results(rows, has_snippet=False)

    def _rows_to_results(self, rows: list, *, has_snippet: bool) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for row in rows:
            row = dict(row)
            metadata = row.get("metadata", "")
            try:
                metadata_obj = json.loads(metadata) if metadata else {}
            except json.JSONDecodeError:
                metadata_obj = {}
            result = {
                "id": row["id"],
                "employee": row["employee"],
                "kind": row["kind"],
                "source": row["source"],
                "title": row["title"],
                "content": row["content"],
                "metadata": metadata_obj,
                "created_at": row["created_at"],
            }
            if has_snippet:
                result["snippet"] = row.get("snippet", "")
            else:
                # PG 模式下生成简易 snippet
                content = row.get("content", "")
                result["snippet"] = content[:200] if content else ""
            results.append(result)
        return results


__all__ = ["MemorySearchIndex", "IndexStats"]
