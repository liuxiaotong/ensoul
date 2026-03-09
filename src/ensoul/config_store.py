"""配置存储 — 员工灵魂、讨论会、流水线配置的数据库持久化.

将配置从文件系统迁移到数据库，支持版本管理和远程访问。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ensoul.database import get_connection, is_pg
DEFAULT_ADMIN_TENANT_ID = "admin"

logger = logging.getLogger(__name__)


# ── Schema 定义 ──

_PG_CREATE_EMPLOYEE_SOULS = """\
CREATE TABLE IF NOT EXISTS employee_souls (
    employee_name VARCHAR(255) PRIMARY KEY,
    content TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(255),
    version INTEGER NOT NULL DEFAULT 1,
    metadata TEXT
)
"""

_PG_CREATE_DISCUSSIONS = """\
CREATE TABLE IF NOT EXISTS discussions (
    name VARCHAR(255) PRIMARY KEY,
    yaml_content TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
)
"""

_PG_CREATE_PIPELINES = """\
CREATE TABLE IF NOT EXISTS pipelines (
    name VARCHAR(255) PRIMARY KEY,
    yaml_content TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
)
"""

_PG_CREATE_SOUL_HISTORY = """\
CREATE TABLE IF NOT EXISTS employee_soul_history (
    id SERIAL PRIMARY KEY,
    employee_name VARCHAR(255) NOT NULL,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    updated_by VARCHAR(255),
    UNIQUE(employee_name, version)
)
"""

_PG_CREATE_MEMORIES = """\
CREATE TABLE IF NOT EXISTS memories (
    id VARCHAR(12) PRIMARY KEY,
    employee VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    category VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    source_session VARCHAR(255) DEFAULT '',
    confidence FLOAT DEFAULT 1.0,
    superseded_by VARCHAR(12) DEFAULT '',
    ttl_days INTEGER DEFAULT 0,
    importance INTEGER DEFAULT 3,
    last_accessed TIMESTAMP,
    tags TEXT[],
    shared BOOLEAN DEFAULT FALSE,
    visibility VARCHAR(20) DEFAULT 'open',
    trigger_condition TEXT DEFAULT '',
    applicability TEXT[],
    origin_employee VARCHAR(255) DEFAULT '',
    verified_count INTEGER DEFAULT 0
)
"""

_PG_CREATE_EMPLOYEES = """\
CREATE TABLE IF NOT EXISTS employees (
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'tenant_admin',
    name VARCHAR(255) NOT NULL,
    character_name VARCHAR(255) NOT NULL DEFAULT '',
    display_name VARCHAR(255) DEFAULT '',
    description TEXT DEFAULT '',
    summary TEXT DEFAULT '',
    version VARCHAR(20) DEFAULT '1.0',
    tags TEXT[],
    author VARCHAR(255) DEFAULT '',
    triggers TEXT[],
    model VARCHAR(255) DEFAULT '',
    model_tier VARCHAR(50) DEFAULT '',
    agent_id VARCHAR(20),
    agent_status VARCHAR(20) DEFAULT 'active',
    avatar_prompt TEXT DEFAULT '',
    auto_memory BOOLEAN DEFAULT FALSE,
    kpi TEXT[],
    bio TEXT DEFAULT '',
    domains TEXT[],
    temperature DOUBLE PRECISION,
    max_tokens INTEGER,
    tools TEXT[],
    context TEXT[],
    permissions_json TEXT,
    api_key VARCHAR(512) DEFAULT '',
    base_url VARCHAR(512) DEFAULT '',
    fallback_model VARCHAR(255) DEFAULT '',
    fallback_api_key VARCHAR(512) DEFAULT '',
    fallback_base_url VARCHAR(512) DEFAULT '',
    research_instructions TEXT DEFAULT '',
    body TEXT DEFAULT '',
    soul_content TEXT DEFAULT '',
    soul_version INTEGER DEFAULT 1,
    soul_updated_at TIMESTAMP,
    soul_updated_by VARCHAR(255),
    source_layer VARCHAR(20) DEFAULT 'builtin',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,
    PRIMARY KEY (tenant_id, name)
)
"""

_PG_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_soul_history_employee ON employee_soul_history(employee_name)",
    "CREATE INDEX IF NOT EXISTS idx_discussions_updated ON discussions(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_pipelines_updated ON pipelines(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_memories_employee ON memories(employee)",
    "CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)",
    "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_memories_shared ON memories(shared)",
    "CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags)",
    "CREATE INDEX IF NOT EXISTS idx_employees_agent_id ON employees(agent_id)",
    "CREATE INDEX IF NOT EXISTS idx_employees_agent_status ON employees(agent_status)",
    "CREATE INDEX IF NOT EXISTS idx_employees_tags ON employees USING GIN(tags)",
    "CREATE INDEX IF NOT EXISTS idx_employees_triggers ON employees USING GIN(triggers)",
]


def init_config_tables() -> None:
    """初始化配置存储表（仅 PG 模式）."""
    if not is_pg():
        logger.debug("SQLite 模式，跳过配置表初始化")
        return

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(_PG_CREATE_EMPLOYEE_SOULS)
        cur.execute(_PG_CREATE_DISCUSSIONS)
        cur.execute(_PG_CREATE_PIPELINES)
        cur.execute(_PG_CREATE_SOUL_HISTORY)
        cur.execute(_PG_CREATE_MEMORIES)
        cur.execute(_PG_CREATE_EMPLOYEES)
        for sql in _PG_CREATE_INDEXES:
            cur.execute(sql)

    logger.info("配置存储表初始化完成")


# ── Employee Soul 操作 ──


def get_soul(employee_name: str, tenant_id: str = DEFAULT_ADMIN_TENANT_ID) -> dict[str, Any] | None:
    """读取员工灵魂配置.

    Returns:
        包含 content, version, updated_at, updated_by 的字典，不存在返回 None
    """
    if not is_pg():
        return None

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT content, version, updated_at, updated_by, metadata FROM employee_souls WHERE employee_name = %s AND tenant_id = %s",
            (employee_name, tenant_id),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "employee_name": employee_name,
            "content": row[0],
            "version": row[1],
            "updated_at": row[2].isoformat() if row[2] else None,
            "updated_by": row[3],
            "metadata": json.loads(row[4]) if row[4] else {},
        }


def update_soul(
    employee_name: str,
    content: str,
    updated_by: str = "",
    metadata: dict[str, Any] | None = None,
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
) -> dict[str, Any]:
    """更新员工灵魂配置（自动版本递增 + 历史记录）.

    Returns:
        更新后的配置字典
    """
    if not is_pg():
        raise RuntimeError("配置存储仅支持 PG 模式")

    now = datetime.now(timezone.utc)
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

    with get_connection() as conn:
        cur = conn.cursor()

        # 获取当前版本
        cur.execute(
            "SELECT version, content FROM employee_souls WHERE employee_name = %s AND tenant_id = %s",
            (employee_name, tenant_id),
        )
        row = cur.fetchone()

        if row:
            # 更新现有记录
            old_version = row[0]
            old_content = row[1]
            new_version = old_version + 1

            # 保存历史版本
            cur.execute(
                """
                INSERT INTO employee_soul_history (employee_name, version, content, updated_at, updated_by, tenant_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, employee_name, version) DO NOTHING
                """,
                (employee_name, old_version, old_content, now, updated_by, tenant_id),
            )

            # 更新当前版本
            cur.execute(
                """
                UPDATE employee_souls
                SET content = %s, version = %s, updated_at = %s, updated_by = %s, metadata = %s
                WHERE employee_name = %s AND tenant_id = %s
                """,
                (content, new_version, now, updated_by, metadata_json, employee_name, tenant_id),
            )
        else:
            # 插入新记录
            new_version = 1
            cur.execute(
                """
                INSERT INTO employee_souls (employee_name, content, version, updated_at, updated_by, metadata, tenant_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (employee_name, content, new_version, now, updated_by, metadata_json, tenant_id),
            )

    # 同步到 employees 表（如果存在对应记录）
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE employees
                SET soul_content = %s, soul_version = %s, soul_updated_at = %s,
                    soul_updated_by = %s, updated_at = %s
                WHERE tenant_id = %s AND (name = %s OR character_name = %s)
                """,
                (
                    content,
                    new_version,
                    now,
                    updated_by,
                    now,
                    tenant_id,
                    employee_name,
                    employee_name,
                ),
            )
    except Exception as e:
        logger.debug("同步 soul 到 employees 表跳过: %s", e)

    return {
        "employee_name": employee_name,
        "version": new_version,
        "updated_at": now.isoformat(),
        "updated_by": updated_by,
    }


def list_souls(tenant_id: str = DEFAULT_ADMIN_TENANT_ID) -> list[dict[str, Any]]:
    """列出所有员工灵魂配置（不含 content）."""
    if not is_pg():
        return []

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT employee_name, version, updated_at, updated_by FROM employee_souls WHERE tenant_id = %s ORDER BY employee_name",
            (tenant_id,),
        )
        rows = cur.fetchall()
        return [
            {
                "employee_name": row[0],
                "version": row[1],
                "updated_at": row[2].isoformat() if row[2] else None,
                "updated_by": row[3],
            }
            for row in rows
        ]


def _generate_unique_agent_id() -> str:
    """生成唯一的 agent_id（AI + 4位随机数字）."""
    import random
    from pathlib import Path

    # 确定头像目录
    static_dir = Path(__file__).parent.parent.parent / "static" / "avatars"
    static_dir.mkdir(parents=True, exist_ok=True)

    # 尝试生成唯一 ID
    for _ in range(100):
        num = random.randint(1000, 9999)
        agent_id = f"AI{num}"
        avatar_path = static_dir / f"{agent_id}.webp"
        if not avatar_path.exists():
            return agent_id

    raise RuntimeError("无法生成唯一的 agent_id（100 次尝试后仍冲突）")


def _create_employee_files(
    name: str,
    character_name: str,
    metadata: dict[str, Any],
    soul_content: str,
) -> None:
    """创建员工的文件系统结构（employee.yaml + soul.md）."""
    from pathlib import Path

    import yaml

    # 确定员工目录
    emp_dir = Path(__file__).parent.parent.parent / "private" / "employees" / name
    emp_dir.mkdir(parents=True, exist_ok=True)

    # 创建 employee.yaml
    yaml_content = {
        "name": name,
        "character_name": character_name,
        "display_name": metadata.get("display_name", ""),
        "description": metadata.get("description", ""),
        "model": metadata.get("model", "claude-sonnet-4-6"),
        "model_tier": metadata.get("model_tier", "claude"),
        "tags": metadata.get("tags", []),
        "agent_id": metadata.get("agent_id"),
        "agent_status": metadata.get("agent_status", "active"),
        "avatar_prompt": metadata.get("avatar_prompt", ""),
    }

    yaml_path = emp_dir / "employee.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # 创建 soul.md
    soul_path = emp_dir / "soul.md"
    with open(soul_path, "w", encoding="utf-8") as f:
        f.write(soul_content)

    logger.info("已创建员工文件: %s", emp_dir)


def create_employee(
    name: str,
    character_name: str,
    display_name: str = "",
    description: str = "",
    model: str = "claude-sonnet-4-6",
    model_tier: str = "claude",
    tags: list[str] | None = None,
    soul_content: str = "",
    agent_status: str = "active",
    avatar_prompt: str = "",
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
) -> dict[str, Any]:
    """创建新员工（数据库 + 文件系统）.

    Args:
        name: 员工标识（slug，仅 [a-z0-9-]）
        character_name: 角色名/中文名
        display_name: 显示名称（可选）
        description: 职责描述
        model: 使用的模型
        model_tier: 模型档位
        tags: 标签列表
        soul_content: 初始 soul 配置
        agent_status: 员工状态（active/frozen/inactive）
        avatar_prompt: 头像生成 prompt（可选）

    Returns:
        创建结果字典（包含 agent_id, employee_name, version, created_at）

    Raises:
        ValueError: 员工已存在
        RuntimeError: agent_id 生成失败或配置存储不可用
    """
    if not is_pg():
        raise RuntimeError("配置存储仅支持 PG 模式")

    # 1. 检查员工是否已存在（同一租户内）
    if get_soul(character_name, tenant_id=tenant_id):
        raise ValueError(f"employee already exists: {character_name}")

    # 2. 生成唯一的 agent_id
    agent_id = _generate_unique_agent_id()

    # 3. 准备元数据
    metadata = {
        "name": name,
        "agent_id": agent_id,
        "display_name": display_name,
        "description": description,
        "model": model,
        "model_tier": model_tier,
        "tags": tags or [],
        "agent_status": agent_status,
        "avatar_prompt": avatar_prompt,
    }

    # 4. 写入数据库
    result = update_soul(
        employee_name=character_name,
        content=soul_content,
        updated_by="create_employee",
        metadata=metadata,
        tenant_id=tenant_id,
    )

    # 5. 创建文件系统
    try:
        _create_employee_files(name, character_name, metadata, soul_content)
    except Exception as e:
        logger.error("创建员工文件失败: %s", e, exc_info=True)
        # 不抛出异常，因为数据库已写入

    # 6. 返回结果（添加 agent_id）
    result["agent_id"] = agent_id
    return result


# ── Discussion 操作 ──


def get_discussion(name: str, tenant_id: str = DEFAULT_ADMIN_TENANT_ID) -> dict[str, Any] | None:
    """读取讨论会配置."""
    if not is_pg():
        return None

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT yaml_content, description, created_at, updated_at, metadata FROM discussions WHERE name = %s AND tenant_id = %s",
            (name, tenant_id),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "name": name,
            "yaml_content": row[0],
            "description": row[1],
            "created_at": row[2].isoformat() if row[2] else None,
            "updated_at": row[3].isoformat() if row[3] else None,
            "metadata": json.loads(row[4]) if row[4] else {},
        }


def create_discussion(
    name: str,
    yaml_content: str,
    description: str = "",
    metadata: dict[str, Any] | None = None,
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
) -> dict[str, Any]:
    """创建讨论会配置."""
    if not is_pg():
        raise RuntimeError("配置存储仅支持 PG 模式")

    now = datetime.now(timezone.utc)
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO discussions (name, yaml_content, description, created_at, updated_at, metadata, tenant_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (name, yaml_content, description, now, now, metadata_json, tenant_id),
        )

    return {
        "name": name,
        "description": description,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }


def update_discussion(
    name: str,
    yaml_content: str,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
) -> dict[str, Any]:
    """更新讨论会配置."""
    if not is_pg():
        raise RuntimeError("配置存储仅支持 PG 模式")

    now = datetime.now(timezone.utc)

    with get_connection() as conn:
        cur = conn.cursor()

        # 构建更新语句
        updates = ["yaml_content = %s", "updated_at = %s"]
        params: list[Any] = [yaml_content, now]

        if description is not None:
            updates.append("description = %s")
            params.append(description)

        if metadata is not None:
            updates.append("metadata = %s")
            params.append(json.dumps(metadata, ensure_ascii=False))

        params.extend([name, tenant_id])

        cur.execute(
            f"UPDATE discussions SET {', '.join(updates)} WHERE name = %s AND tenant_id = %s",
            tuple(params),
        )

    return {"name": name, "updated_at": now.isoformat()}


def list_discussions(tenant_id: str = DEFAULT_ADMIN_TENANT_ID) -> list[dict[str, Any]]:
    """列出所有讨论会配置（不含 yaml_content）."""
    if not is_pg():
        return []

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, description, created_at, updated_at FROM discussions WHERE tenant_id = %s ORDER BY name",
            (tenant_id,),
        )
        rows = cur.fetchall()
        return [
            {
                "name": row[0],
                "description": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
                "updated_at": row[3].isoformat() if row[3] else None,
            }
            for row in rows
        ]


# ── Pipeline 操作 ──


def get_pipeline(name: str, tenant_id: str = DEFAULT_ADMIN_TENANT_ID) -> dict[str, Any] | None:
    """读取流水线配置."""
    if not is_pg():
        return None

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT yaml_content, description, created_at, updated_at, metadata FROM pipelines WHERE name = %s AND tenant_id = %s",
            (name, tenant_id),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "name": name,
            "yaml_content": row[0],
            "description": row[1],
            "created_at": row[2].isoformat() if row[2] else None,
            "updated_at": row[3].isoformat() if row[3] else None,
            "metadata": json.loads(row[4]) if row[4] else {},
        }


def create_pipeline(
    name: str,
    yaml_content: str,
    description: str = "",
    metadata: dict[str, Any] | None = None,
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
) -> dict[str, Any]:
    """创建流水线配置."""
    if not is_pg():
        raise RuntimeError("配置存储仅支持 PG 模式")

    now = datetime.now(timezone.utc)
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO pipelines (name, yaml_content, description, created_at, updated_at, metadata, tenant_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (name, yaml_content, description, now, now, metadata_json, tenant_id),
        )

    return {
        "name": name,
        "description": description,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }


def update_pipeline(
    name: str,
    yaml_content: str,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
) -> dict[str, Any]:
    """更新流水线配置."""
    if not is_pg():
        raise RuntimeError("配置存储仅支持 PG 模式")

    now = datetime.now(timezone.utc)

    with get_connection() as conn:
        cur = conn.cursor()

        # 构建更新语句
        updates = ["yaml_content = %s", "updated_at = %s"]
        params: list[Any] = [yaml_content, now]

        if description is not None:
            updates.append("description = %s")
            params.append(description)

        if metadata is not None:
            updates.append("metadata = %s")
            params.append(json.dumps(metadata, ensure_ascii=False))

        params.extend([name, tenant_id])

        cur.execute(
            f"UPDATE pipelines SET {', '.join(updates)} WHERE name = %s AND tenant_id = %s",
            tuple(params),
        )

    return {"name": name, "updated_at": now.isoformat()}


def list_pipelines(tenant_id: str = DEFAULT_ADMIN_TENANT_ID) -> list[dict[str, Any]]:
    """列出所有流水线配置（不含 yaml_content）."""
    if not is_pg():
        return []

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, description, created_at, updated_at FROM pipelines WHERE tenant_id = %s ORDER BY name",
            (tenant_id,),
        )
        rows = cur.fetchall()
        return [
            {
                "name": row[0],
                "description": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
                "updated_at": row[3].isoformat() if row[3] else None,
            }
            for row in rows
        ]


# ── Employees 统一表操作 ──

# employees 表列名（INSERT 顺序）
_EMPLOYEE_COLUMNS = (
    "tenant_id",
    "name",
    "character_name",
    "display_name",
    "description",
    "summary",
    "version",
    "tags",
    "author",
    "triggers",
    "model",
    "model_tier",
    "agent_id",
    "agent_status",
    "avatar_prompt",
    "auto_memory",
    "kpi",
    "bio",
    "domains",
    "temperature",
    "max_tokens",
    "tools",
    "context",
    "permissions_json",
    "api_key",
    "base_url",
    "fallback_model",
    "fallback_api_key",
    "fallback_base_url",
    "research_instructions",
    "body",
    "soul_content",
    "soul_version",
    "soul_updated_at",
    "soul_updated_by",
    "source_layer",
    "created_at",
    "updated_at",
    "metadata",
)


def _employee_row_to_dict(row: tuple, columns: tuple[str, ...] | None = None) -> dict[str, Any]:
    """将 employees 表的一行转换为字典."""
    cols = columns or _EMPLOYEE_COLUMNS
    d: dict[str, Any] = {}
    for i, col in enumerate(cols):
        val = row[i]
        if col in ("created_at", "updated_at", "soul_updated_at") and val is not None:
            val = val.isoformat() if hasattr(val, "isoformat") else val
        d[col] = val
    return d


def get_employee_from_db(
    name: str,
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
) -> dict[str, Any] | None:
    """从 employees 表读取单个员工."""
    if not is_pg():
        return None

    cols = ", ".join(_EMPLOYEE_COLUMNS)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {cols} FROM employees WHERE tenant_id = %s AND name = %s",  # noqa: S608
            (tenant_id, name),
        )
        row = cur.fetchone()
        if not row:
            return None
        return _employee_row_to_dict(row)


def list_employees_from_db(
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
    *,
    active_only: bool = False,
) -> list[dict[str, Any]]:
    """从 employees 表列出员工."""
    if not is_pg():
        return []

    cols = ", ".join(_EMPLOYEE_COLUMNS)
    sql = f"SELECT {cols} FROM employees WHERE tenant_id = %s"  # noqa: S608
    params: list[Any] = [tenant_id]
    if active_only:
        sql += " AND agent_status = %s"
        params.append("active")
    sql += " ORDER BY name"

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        return [_employee_row_to_dict(row) for row in rows]


def upsert_employee_to_db(
    data: dict[str, Any],
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
) -> dict[str, Any]:
    """插入或更新 employees 表记录（UPSERT）.

    Args:
        data: 员工数据字典（至少包含 name）
        tenant_id: 租户 ID

    Returns:
        更新后的员工数据
    """
    if not is_pg():
        raise RuntimeError("employees 表仅支持 PG 模式")

    now = datetime.now(timezone.utc)
    name = data["name"]

    # 构建完整数据
    row_data = {
        "tenant_id": tenant_id,
        "name": name,
        "character_name": data.get("character_name", ""),
        "display_name": data.get("display_name", ""),
        "description": data.get("description", ""),
        "summary": data.get("summary", ""),
        "version": data.get("version", "1.0"),
        "tags": data.get("tags") or [],
        "author": data.get("author", ""),
        "triggers": data.get("triggers") or [],
        "model": data.get("model", ""),
        "model_tier": data.get("model_tier", ""),
        "agent_id": data.get("agent_id"),
        "agent_status": data.get("agent_status", "active"),
        "avatar_prompt": data.get("avatar_prompt", ""),
        "auto_memory": bool(data.get("auto_memory", False)),
        "kpi": data.get("kpi") or [],
        "bio": data.get("bio", ""),
        "domains": data.get("domains") or [],
        "temperature": data.get("temperature"),
        "max_tokens": data.get("max_tokens"),
        "tools": data.get("tools") or [],
        "context": data.get("context") or [],
        "permissions_json": json.dumps(data["permissions"], ensure_ascii=False)
        if data.get("permissions")
        else None,
        "api_key": data.get("api_key", ""),
        "base_url": data.get("base_url", ""),
        "fallback_model": data.get("fallback_model", ""),
        "fallback_api_key": data.get("fallback_api_key", ""),
        "fallback_base_url": data.get("fallback_base_url", ""),
        "research_instructions": data.get("research_instructions", ""),
        "body": data.get("body", ""),
        "soul_content": data.get("soul_content", ""),
        "soul_version": data.get("soul_version", 1),
        "soul_updated_at": data.get("soul_updated_at") or now,
        "soul_updated_by": data.get("soul_updated_by", ""),
        "source_layer": data.get("source_layer", "builtin"),
        "created_at": now,
        "updated_at": now,
        "metadata": json.dumps(data.get("metadata") or {}, ensure_ascii=False),
    }

    cols = ", ".join(_EMPLOYEE_COLUMNS)
    placeholders = ", ".join(["%s"] * len(_EMPLOYEE_COLUMNS))
    values = tuple(row_data[c] for c in _EMPLOYEE_COLUMNS)

    # ON CONFLICT 更新所有非主键字段
    update_cols = [c for c in _EMPLOYEE_COLUMNS if c not in ("tenant_id", "name", "created_at")]
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO employees ({cols}) VALUES ({placeholders}) "  # noqa: S608
            f"ON CONFLICT (tenant_id, name) DO UPDATE SET {update_set}",
            values,
        )

    row_data["created_at"] = now.isoformat()
    row_data["updated_at"] = now.isoformat()
    return row_data


def update_employee_soul_in_db(
    name: str,
    content: str,
    updated_by: str = "",
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
) -> dict[str, Any]:
    """更新 employees 表中的 soul 内容（自动版本递增 + 写入 soul_history）.

    Returns:
        更新后的信息字典
    """
    if not is_pg():
        raise RuntimeError("employees 表仅支持 PG 模式")

    now = datetime.now(timezone.utc)

    with get_connection() as conn:
        cur = conn.cursor()

        # 读取当前版本
        cur.execute(
            "SELECT soul_version, soul_content FROM employees WHERE tenant_id = %s AND name = %s",
            (tenant_id, name),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Employee not found: {name}")

        old_version = row[0] or 1
        old_content = row[1] or ""
        new_version = old_version + 1

        # 保存历史版本到 employee_soul_history
        cur.execute(
            """
            INSERT INTO employee_soul_history (employee_name, version, content, updated_at, updated_by, tenant_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (tenant_id, employee_name, version) DO NOTHING
            """,
            (name, old_version, old_content, now, updated_by, tenant_id),
        )

        # 更新 employees 表
        cur.execute(
            """
            UPDATE employees
            SET soul_content = %s, soul_version = %s, soul_updated_at = %s,
                soul_updated_by = %s, updated_at = %s
            WHERE tenant_id = %s AND name = %s
            """,
            (content, new_version, now, updated_by, now, tenant_id, name),
        )

    return {
        "employee_name": name,
        "version": new_version,
        "updated_at": now.isoformat(),
        "updated_by": updated_by,
    }


def delete_employee_from_db(
    name: str,
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
) -> bool:
    """从 employees 表删除员工.

    Returns:
        是否成功删除
    """
    if not is_pg():
        return False

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM employees WHERE tenant_id = %s AND name = %s",
            (tenant_id, name),
        )
        return cur.rowcount > 0


def copy_employee_to_tenant(
    source_name: str,
    target_tenant_id: str,
    source_tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
    new_name: str | None = None,
    new_character_name: str | None = None,
    customizations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """从来源租户复制员工到目标租户.

    Args:
        source_name: 来源员工 slug
        target_tenant_id: 目标租户 ID
        source_tenant_id: 来源租户 ID（默认 admin）
        new_name: 新员工名（可选，默认同原名）
        new_character_name: 新角色名（可选）
        customizations: 覆盖字段（可选），支持 description, model, tags 等

    Returns:
        复制后的员工数据字典

    Raises:
        ValueError: 来源员工不存在 / 目标租户内名字重复
        RuntimeError: 非 PG 模式
    """
    if not is_pg():
        raise RuntimeError("employees 表仅支持 PG 模式")

    # 1. 读取来源员工
    source = get_employee_from_db(source_name, tenant_id=source_tenant_id)
    if not source:
        raise ValueError(f"source employee not found: {source_tenant_id}/{source_name}")

    # 2. 确定目标名字
    target_name = new_name or source_name

    # 3. 检查目标租户内名字是否已存在
    existing = get_employee_from_db(target_name, tenant_id=target_tenant_id)
    if existing:
        raise ValueError(
            f"employee already exists in target tenant: {target_tenant_id}/{target_name}"
        )

    # 4. 生成新的 agent_id
    agent_id = _generate_unique_agent_id()

    # 5. 构建复制数据
    now = datetime.now(timezone.utc)
    customs = customizations or {}

    copy_data = {
        "name": target_name,
        "character_name": new_character_name or source.get("character_name", ""),
        "display_name": customs.get("display_name", source.get("display_name", "")),
        "description": customs.get("description", source.get("description", "")),
        "summary": customs.get("summary", source.get("summary", "")),
        "version": "1.0",
        "tags": customs.get("tags", source.get("tags") or []),
        "author": customs.get("author", source.get("author", "")),
        "triggers": customs.get("triggers", source.get("triggers") or []),
        "model": customs.get("model", source.get("model", "")),
        "model_tier": customs.get("model_tier", source.get("model_tier", "")),
        "agent_id": agent_id,
        "agent_status": customs.get("agent_status", "active"),
        "avatar_prompt": customs.get("avatar_prompt", source.get("avatar_prompt", "")),
        "auto_memory": source.get("auto_memory", False),
        "kpi": source.get("kpi") or [],
        "bio": customs.get("bio", source.get("bio", "")),
        "domains": customs.get("domains", source.get("domains") or []),
        "temperature": customs.get("temperature", source.get("temperature")),
        "max_tokens": customs.get("max_tokens", source.get("max_tokens")),
        "tools": source.get("tools") or [],
        "context": source.get("context") or [],
        "permissions": json.loads(source["permissions_json"])
        if source.get("permissions_json")
        else None,
        "api_key": source.get("api_key", ""),
        "base_url": source.get("base_url", ""),
        "fallback_model": source.get("fallback_model", ""),
        "fallback_api_key": source.get("fallback_api_key", ""),
        "fallback_base_url": source.get("fallback_base_url", ""),
        "research_instructions": customs.get(
            "research_instructions", source.get("research_instructions", "")
        ),
        "body": customs.get("body", source.get("body", "")),
        "soul_content": customs.get("soul_content", source.get("soul_content", "")),
        "soul_version": 1,
        "soul_updated_at": now,
        "soul_updated_by": "copy_employee",
        "source_layer": "db",
        "metadata": {
            "source_copied_from": f"{source_tenant_id}/{source_name}",
        },
    }

    # 6. 写入目标租户
    result = upsert_employee_to_db(copy_data, tenant_id=target_tenant_id)

    return result


def migrate_employees_to_db(
    project_dir: Path | None = None,
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
) -> dict[str, Any]:
    """从文件系统 + employee_souls 表迁移数据到 employees 表.

    幂等操作：已存在的记录不覆盖（ON CONFLICT DO NOTHING）。

    Returns:
        迁移统计信息
    """

    if not is_pg():
        return {"skipped": True, "reason": "not PG mode"}

    # discovery 模块未迁入，无法执行文件系统迁移
    try:
        from ensoul.discovery import _discover_employees_uncached  # noqa: F401
    except ImportError:
        return {"skipped": True, "reason": "discovery module not available"}
    from ensoul.paths import resolve_project_dir

    root = resolve_project_dir(project_dir)
    result = _discover_employees_uncached(root)

    now = datetime.now(timezone.utc)
    migrated = 0
    skipped = 0
    errors: list[str] = []

    for emp in result.employees.values():
        try:
            # 读取 employee.yaml 中的额外字段（bio, temperature, max_tokens, domains）
            bio = ""
            temperature = None
            max_tokens = None
            domains: list[str] = []
            if emp.source_path and emp.source_path.is_dir():
                yaml_path = emp.source_path / "employee.yaml"
                if yaml_path.exists():
                    import yaml

                    with open(yaml_path) as f:
                        yaml_config = yaml.safe_load(f) or {}
                    bio = yaml_config.get("bio", "")
                    temperature = yaml_config.get("temperature")
                    max_tokens = yaml_config.get("max_tokens")
                    raw_domains = yaml_config.get("domains", [])
                    domains = raw_domains if isinstance(raw_domains, list) else []

            # 读取 soul.md 内容
            soul_content = ""
            if emp.source_path:
                if emp.source_path.is_dir():
                    soul_path = emp.source_path / "soul.md"
                else:
                    soul_path = emp.source_path
                if soul_path.exists():
                    soul_content = soul_path.read_text(encoding="utf-8")

            # 从 employee_souls 表获取 soul 版本信息
            soul_version = 1
            soul_updated_at = now
            soul_updated_by = ""
            soul_row = get_soul(emp.character_name or emp.name, tenant_id=tenant_id)
            if soul_row:
                soul_content = soul_row.get("content") or soul_content
                soul_version = soul_row.get("version", 1)
                soul_updated_at_str = soul_row.get("updated_at")
                if soul_updated_at_str:
                    try:
                        soul_updated_at = datetime.fromisoformat(soul_updated_at_str)
                    except (ValueError, TypeError):
                        pass
                soul_updated_by = soul_row.get("updated_by", "")

            # 构建 permissions JSON
            permissions = None
            if emp.permissions:
                permissions = emp.permissions.model_dump()

            # 组装数据
            data = {
                "name": emp.name,
                "character_name": emp.character_name,
                "display_name": emp.display_name,
                "description": emp.description,
                "summary": emp.summary,
                "version": emp.version,
                "tags": emp.tags,
                "author": emp.author,
                "triggers": emp.triggers,
                "model": emp.model,
                "model_tier": emp.model_tier,
                "agent_id": emp.agent_id,
                "agent_status": emp.agent_status,
                "avatar_prompt": emp.avatar_prompt,
                "auto_memory": emp.auto_memory,
                "kpi": emp.kpi,
                "bio": bio,
                "domains": domains,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": emp.tools,
                "context": emp.context,
                "permissions": permissions,
                "api_key": emp.api_key,
                "base_url": emp.base_url,
                "fallback_model": emp.fallback_model,
                "fallback_api_key": emp.fallback_api_key,
                "fallback_base_url": emp.fallback_base_url,
                "research_instructions": emp.research_instructions,
                "body": emp.body,
                "soul_content": soul_content,
                "soul_version": soul_version,
                "soul_updated_at": soul_updated_at,
                "soul_updated_by": soul_updated_by,
                "source_layer": emp.source_layer,
            }

            # ON CONFLICT DO NOTHING — 幂等
            cols = ", ".join(_EMPLOYEE_COLUMNS)
            placeholders = ", ".join(["%s"] * len(_EMPLOYEE_COLUMNS))
            row_data = {
                **data,
                "tenant_id": tenant_id,
                "created_at": now,
                "updated_at": now,
                "metadata": json.dumps({}, ensure_ascii=False),
            }
            values = tuple(row_data.get(c) for c in _EMPLOYEE_COLUMNS)

            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    f"INSERT INTO employees ({cols}) VALUES ({placeholders}) "  # noqa: S608
                    "ON CONFLICT (tenant_id, name) DO NOTHING",
                    values,
                )
                if cur.rowcount > 0:
                    migrated += 1
                else:
                    skipped += 1

        except Exception as e:
            errors.append(f"{emp.name}: {e}")
            logger.warning("迁移员工 %s 失败: %s", emp.name, e, exc_info=True)

    logger.info("员工迁移完成: migrated=%d, skipped=%d, errors=%d", migrated, skipped, len(errors))
    return {
        "migrated": migrated,
        "skipped": skipped,
        "errors": errors,
        "total": len(result.employees),
    }


# ── 通用 KV 配置存储 ──

_PG_CREATE_KV_CONFIGS = """\
CREATE TABLE IF NOT EXISTS kv_configs (
    namespace VARCHAR(255) NOT NULL,
    key VARCHAR(255) NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (namespace, key)
)
"""


def _ensure_kv_table() -> None:
    """确保 kv_configs 表存在（幂等）."""
    if not is_pg():
        return
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(_PG_CREATE_KV_CONFIGS)


def put_config(namespace: str, key: str, value: str) -> None:
    """写入通用 KV 配置.

    Args:
        namespace: 命名空间（如 'soul_evolution'）
        key: 配置键
        value: 配置值（通常为 JSON 字符串）
    """
    if not is_pg():
        logger.debug("SQLite 模式，跳过 put_config")
        return

    _ensure_kv_table()
    now = datetime.now(timezone.utc)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO kv_configs (namespace, key, value, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (namespace, key) DO UPDATE
            SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
            """,
            (namespace, key, value, now),
        )


def get_config(namespace: str, key: str) -> str | None:
    """读取通用 KV 配置.

    Args:
        namespace: 命名空间
        key: 配置键

    Returns:
        配置值字符串，不存在返回 None
    """
    if not is_pg():
        return None

    _ensure_kv_table()

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT value FROM kv_configs WHERE namespace = %s AND key = %s",
            (namespace, key),
        )
        row = cur.fetchone()
        return row[0] if row else None
