"""统一数据库连接管理 — 支持 PostgreSQL 和 SQLite 双后端.

生产环境使用 PostgreSQL（连接池），开发/测试环境可回退到 SQLite。

后端选择优先级:
  1. 环境变量 ``ENSOUL_USE_SQLITE=1`` → SQLite
  2. 环境变量 ``ENSOUL_DATABASE_URL`` 以 ``sqlite`` 开头 → SQLite
  3. 其他 → PostgreSQL

用法::

    from ensoul.database import get_connection, is_pg, init_db

    # 服务启动时
    init_db()

    # 业务代码
    with get_connection() as conn:
        conn.execute(...)
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 配置 ──

_DEFAULT_PG_URL = "postgresql://ensoul:ensoul@localhost:5432/ensoul"


def _database_url() -> str:
    return os.environ.get("ENSOUL_DATABASE_URL", _DEFAULT_PG_URL)


def _force_sqlite() -> bool:
    return os.environ.get("ENSOUL_USE_SQLITE", "").strip() in ("1", "true", "yes")


def is_pg() -> bool:
    """当前是否使用 PostgreSQL 后端."""
    if _force_sqlite():
        return False
    url = _database_url()
    return not url.startswith("sqlite")


# ── PostgreSQL 连接池 ──

_pg_pool: Any = None  # psycopg2.pool.ThreadedConnectionPool | None
_pg_pool_lock = threading.Lock()


def _get_pg_pool():
    """获取或创建 PG 连接池（懒初始化、线程安全）."""
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool
    with _pg_pool_lock:
        if _pg_pool is None:
            import psycopg2.pool

            url = _database_url()
            _pg_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                dsn=url,
            )
            logger.info("PG 连接池已创建: %s", url.split("@")[-1])  # 不泄露密码
    return _pg_pool


@contextlib.contextmanager
def get_pg_connection():
    """从连接池借出一个 PG 连接，自动归还.

    用法::

        with get_pg_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")

    异常时自动回滚，正常时自动提交。
    """
    pool = _get_pg_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ── SQLite 辅助 ──

_sqlite_connections: dict[str, sqlite3.Connection] = {}
_sqlite_lock = threading.Lock()


def get_sqlite_connection(db_path: str | Path) -> sqlite3.Connection:
    """获取或创建 SQLite 连接（单例，线程安全）.

    注意：调用者自行管理连接生命周期；此函数提供的是懒初始化单例，
    不会每次打开新连接。
    """
    key = str(db_path)
    if key in _sqlite_connections:
        return _sqlite_connections[key]
    with _sqlite_lock:
        if key not in _sqlite_connections:
            p = Path(db_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(p), timeout=10.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            _sqlite_connections[key] = conn
            logger.info("SQLite 连接已创建: %s", p)
    return _sqlite_connections[key]


# ── 通用 get_connection ──


@contextlib.contextmanager
def get_connection(sqlite_path: str | Path | None = None) -> Generator:
    """统一连接获取入口.

    - PG 模式: 从连接池借出连接，自动 commit/rollback/归还
    - SQLite 模式: 返回 sqlite3.Connection（调用者需要知道是 SQLite）

    sqlite_path 只在 SQLite 模式下使用，PG 模式忽略。
    """
    if is_pg():
        with get_pg_connection() as conn:
            yield conn
    else:
        if sqlite_path is None:
            sqlite_path = Path(".crew") / "default.db"
        conn = get_sqlite_connection(sqlite_path)
        yield conn


# ── 建表（PG） ──

# events 表
_PG_CREATE_EVENTS = """\
CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    event_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    duration_ms DOUBLE PRECISION,
    success INTEGER NOT NULL DEFAULT 1,
    error_type TEXT,
    source TEXT,
    metadata TEXT
)
"""

_PG_CREATE_EVENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_events_name ON events(event_name)",
]

# entries 表（memory_index）
_PG_CREATE_ENTRIES = """\
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

_PG_CREATE_ENTRIES_TRGM = [
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    "CREATE INDEX IF NOT EXISTS idx_entries_content_trgm ON entries USING gin (content gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS idx_entries_title_trgm ON entries USING gin (title gin_trgm_ops)",
]

# memory_vectors 表（已废弃，保留以兼容旧数据库 schema）
_PG_CREATE_MEMORY_VECTORS = """\
CREATE TABLE IF NOT EXISTS memory_vectors (
    id TEXT PRIMARY KEY,
    employee TEXT NOT NULL,
    embedding BYTEA NOT NULL,
    content TEXT NOT NULL,
    tags TEXT DEFAULT ''
)
"""

_PG_CREATE_MEMORY_VECTORS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_mv_employee ON memory_vectors(employee)",
]


def init_db() -> None:
    """在 PG 模式下创建所有表和索引.

    SQLite 模式下各模块自行建表，此函数为空操作。
    应在服务启动时调用。
    """
    if not is_pg():
        logger.debug("SQLite 模式，跳过 PG init_db()")
        return

    with get_pg_connection() as conn:
        cur = conn.cursor()

        # events
        cur.execute(_PG_CREATE_EVENTS)
        for sql in _PG_CREATE_EVENTS_INDEXES:
            cur.execute(sql)

        # entries (memory_index)
        cur.execute(_PG_CREATE_ENTRIES)

        # memory_vectors
        cur.execute(_PG_CREATE_MEMORY_VECTORS)
        for sql in _PG_CREATE_MEMORY_VECTORS_INDEXES:
            cur.execute(sql)

    # pg_trgm 索引：用 SAVEPOINT 隔离每条 DDL，失败不回滚前面已成功的 DDL
    with get_pg_connection() as conn:
        cur = conn.cursor()
        for sql in _PG_CREATE_ENTRIES_TRGM:
            cur.execute("SAVEPOINT sp_trgm")
            try:
                cur.execute(sql)
            except Exception as e:
                cur.execute("ROLLBACK TO SAVEPOINT sp_trgm")
                logger.warning("pg_trgm 索引创建警告（非致命）: %s", e)
                break
            else:
                cur.execute("RELEASE SAVEPOINT sp_trgm")

    # 初始化配置存储表
    try:
        from ensoul.config_store import init_config_tables

        init_config_tables()
    except Exception as e:
        logger.warning("配置存储表初始化失败（非致命）: %s", e)

    # 多租户表初始化已移除（tenant 模块未迁入）

    logger.info("PG init_db() 完成")


# ── 清理 ──


def close_all() -> None:
    """关闭所有连接（池 + SQLite 单例）. 测试 teardown 用."""
    global _pg_pool
    with _pg_pool_lock:
        if _pg_pool is not None:
            _pg_pool.closeall()
            _pg_pool = None
    with _sqlite_lock:
        for conn in _sqlite_connections.values():
            try:
                # WAL checkpoint 防止 -wal/-shm 文件无限增长
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.close()
            except Exception:
                pass
        _sqlite_connections.clear()


def _reset_pg_pool() -> None:
    """仅供测试 — 重置 PG 连接池."""
    global _pg_pool
    with _pg_pool_lock:
        if _pg_pool is not None:
            try:
                _pg_pool.closeall()
            except Exception:
                pass
            _pg_pool = None


__all__ = [
    "get_connection",
    "get_pg_connection",
    "get_sqlite_connection",
    "init_db",
    "is_pg",
    "close_all",
]
