"""统一埋点事件收集器 — 支持 PostgreSQL 和 SQLite 双后端.

收集 tool_call / employee_run / pipeline_run / discussion_run 等各类事件，
提供 record / query / aggregate 三个核心方法。

用法::

    from ensoul.event_collector import get_event_collector

    ec = get_event_collector()
    ec.record(event_type="tool_call", event_name="list_employees",
              duration_ms=12.3, success=True, source="mcp")
    rows = ec.query(event_type="tool_call", limit=10)
    agg  = ec.aggregate(event_type="tool_call")
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ensoul.database import is_pg

logger = logging.getLogger(__name__)

# ── SQLite DDL（回退模式） ──

_SQLITE_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    event_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    duration_ms REAL,
    success INTEGER NOT NULL DEFAULT 1,
    error_type TEXT,
    source TEXT,
    metadata TEXT
)
"""

_SQLITE_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_events_name ON events(event_name)",
]


def _ph(n: int = 1) -> str:
    """返回占位符: PG 用 %s，SQLite 用 ?"""
    p = "%s" if is_pg() else "?"
    return ", ".join([p] * n)


class EventCollector:
    """事件收集器 — 线程安全，支持 PG / SQLite 双后端."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        # 显式传入 db_path → 强制 SQLite（兼容现有测试）
        self._use_pg = is_pg() if db_path is None else False

        if not self._use_pg:
            # SQLite 模式：保留原有路径逻辑
            if db_path is None:
                env = os.environ.get("ENSOUL_EVENTS_DB")
                if env:
                    db_path = Path(env)
                else:
                    db_path = Path(".ensoul") / "events.db"
            self._db_path = Path(db_path)
            self._lock = threading.Lock()
            self._conn: sqlite3.Connection | None = None
        else:
            # PG 模式：连接池由 database.py 管理，无需本地锁
            self._db_path = None  # type: ignore[assignment]
            self._lock = threading.Lock()  # 保留以兼容 close()
            self._conn = None

    # ── 连接管理 ──

    def _get_conn(self) -> sqlite3.Connection:
        """懒初始化 SQLite 连接（带表创建）. PG 模式下不应调用此方法."""
        if self._conn is not None:
            return self._conn
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=10.0,
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        conn.execute(_SQLITE_CREATE_TABLE)
        for idx_sql in _SQLITE_CREATE_INDEXES:
            conn.execute(idx_sql)
        conn.commit()
        self._conn = conn
        return conn

    def close(self) -> None:
        """关闭连接."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ── 写入 ──

    def record(
        self,
        *,
        event_type: str,
        event_name: str,
        duration_ms: float | None = None,
        success: bool = True,
        error_type: str = "",
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """记录一条事件.

        设计为不抛异常——写入失败仅记日志，不影响主流程。
        """
        ts = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        values = (
            event_type,
            event_name,
            ts,
            duration_ms,
            1 if success else 0,
            error_type or None,
            source or None,
            meta_json,
        )

        try:
            if self._use_pg:
                self._record_pg(values)
            else:
                self._record_sqlite(values)
        except Exception:
            logger.debug("event record failed", exc_info=True)

    def _record_pg(self, values: tuple) -> None:
        from ensoul.database import get_pg_connection

        with get_pg_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO events "
                "(event_type, event_name, timestamp, duration_ms, success, error_type, source, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                values,
            )

    def _record_sqlite(self, values: tuple) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO events "
                "(event_type, event_name, timestamp, duration_ms, success, error_type, source, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            conn.commit()

    # ── 查询 ──

    def query(
        self,
        *,
        event_type: str | None = None,
        event_name: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """按条件查询事件，返回 dict 列表（按时间倒序）."""
        if self._use_pg:
            return self._query_pg(
                event_type=event_type,
                event_name=event_name,
                since=since,
                until=until,
                limit=limit,
            )
        return self._query_sqlite(
            event_type=event_type,
            event_name=event_name,
            since=since,
            until=until,
            limit=limit,
        )

    def _build_query_clauses(
        self,
        *,
        event_type: str | None,
        event_name: str | None,
        since: str | None,
        until: str | None,
    ) -> tuple[str, list[Any]]:
        """构建 WHERE 子句和参数列表."""
        ph = "%s" if self._use_pg else "?"
        clauses: list[str] = []
        params: list[Any] = []
        if event_type:
            clauses.append(f"event_type = {ph}")
            params.append(event_type)
        if event_name:
            clauses.append(f"event_name = {ph}")
            params.append(event_name)
        if since:
            clauses.append(f"timestamp >= {ph}")
            params.append(since)
        if until:
            clauses.append(f"timestamp <= {ph}")
            params.append(until)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    def _query_pg(
        self,
        *,
        event_type: str | None,
        event_name: str | None,
        since: str | None,
        until: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        from ensoul.database import get_pg_connection

        where, params = self._build_query_clauses(
            event_type=event_type, event_name=event_name, since=since, until=until
        )
        sql = f"SELECT * FROM events{where} ORDER BY id DESC LIMIT %s"
        params.append(limit)

        with get_pg_connection() as conn:
            import psycopg2.extras

            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql, params)
            rows = cur.fetchall()

        return self._process_query_rows(rows)

    def _query_sqlite(
        self,
        *,
        event_type: str | None,
        event_name: str | None,
        since: str | None,
        until: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        where, params = self._build_query_clauses(
            event_type=event_type, event_name=event_name, since=since, until=until
        )
        sql = f"SELECT * FROM events{where} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(sql, params).fetchall()

        return self._process_query_rows([dict(r) for r in rows])

    def _process_query_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """统一处理查询结果行."""
        results: list[dict[str, Any]] = []
        for d in rows:
            d = dict(d)  # 确保可变
            if d.get("metadata"):
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except (json.JSONDecodeError, TypeError):
                    pass
            d["success"] = bool(d.get("success", 1))
            results.append(d)
        return results

    # ── 聚合 ──

    def aggregate(
        self,
        *,
        event_type: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """按 event_name 聚合统计.

        返回::

            [
              {
                "event_name": "list_employees",
                "count": 42,
                "success_count": 40,
                "fail_count": 2,
                "avg_duration_ms": 15.3,
                "last_seen": "2026-02-23T..."
              },
              ...
            ]
        """
        if self._use_pg:
            return self._aggregate_pg(event_type=event_type, since=since)
        return self._aggregate_sqlite(event_type=event_type, since=since)

    def _build_aggregate_clauses(
        self,
        *,
        event_type: str | None,
        since: str | None,
    ) -> tuple[str, list[Any]]:
        ph = "%s" if self._use_pg else "?"
        clauses: list[str] = []
        params: list[Any] = []
        if event_type:
            clauses.append(f"event_type = {ph}")
            params.append(event_type)
        if since:
            clauses.append(f"timestamp >= {ph}")
            params.append(since)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    def _aggregate_pg(self, *, event_type: str | None, since: str | None) -> list[dict[str, Any]]:
        from ensoul.database import get_pg_connection

        where, params = self._build_aggregate_clauses(event_type=event_type, since=since)
        sql = (
            "SELECT event_name, "
            "COUNT(*) as count, "
            "SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count, "
            "SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as fail_count, "
            "AVG(duration_ms) as avg_duration_ms, "
            "MAX(timestamp) as last_seen "
            f"FROM events{where} GROUP BY event_name ORDER BY count DESC"
        )
        with get_pg_connection() as conn:
            import psycopg2.extras

            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql, params)
            rows = cur.fetchall()

        return self._process_aggregate_rows(rows)

    def _aggregate_sqlite(
        self, *, event_type: str | None, since: str | None
    ) -> list[dict[str, Any]]:
        where, params = self._build_aggregate_clauses(event_type=event_type, since=since)
        sql = (
            "SELECT event_name, "
            "COUNT(*) as count, "
            "SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count, "
            "SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as fail_count, "
            "AVG(duration_ms) as avg_duration_ms, "
            "MAX(timestamp) as last_seen "
            f"FROM events{where} GROUP BY event_name ORDER BY count DESC"
        )
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(sql, params).fetchall()

        return self._process_aggregate_rows([dict(r) for r in rows])

    def _process_aggregate_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for row in rows:
            row = dict(row)
            results.append(
                {
                    "event_name": row["event_name"],
                    "count": row["count"],
                    "success_count": row["success_count"],
                    "fail_count": row["fail_count"],
                    "avg_duration_ms": round(row["avg_duration_ms"], 1)
                    if row["avg_duration_ms"] is not None
                    else None,
                    "last_seen": row["last_seen"],
                }
            )
        return results


# ── 全局单例 ──

_collector: EventCollector | None = None
_singleton_lock = threading.Lock()


def get_event_collector(db_path: str | Path | None = None) -> EventCollector:
    """获取全局 EventCollector 单例（懒初始化）."""
    global _collector
    if _collector is not None:
        return _collector
    with _singleton_lock:
        if _collector is None:
            _collector = EventCollector(db_path=db_path)
    return _collector


def _reset_singleton() -> None:
    """仅供测试使用 — 重置全局单例."""
    global _collector
    with _singleton_lock:
        if _collector is not None:
            _collector.close()
            _collector = None


__all__ = ["EventCollector", "get_event_collector"]
