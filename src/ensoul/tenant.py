"""多租户核心模块 — 租户管理、认证、依赖注入.

Phase 1: 基于 API key 的租户识别。每个租户分配唯一 API key，
通过 Bearer token 认证并注入 tenant_id 到请求上下文。

用法::

    from ensoul.tenant import get_current_tenant, TenantContext

    # 在 handler 中获取当前租户
    tenant = get_current_tenant(request)
"""

from __future__ import annotations

import hmac
import logging
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ensoul.database import get_connection, is_pg

logger = logging.getLogger(__name__)


# ── 数据模型 ──


@dataclass
class TenantContext:
    """当前请求的租户上下文."""

    tenant_id: str
    tenant_name: str
    is_admin: bool = False


# 默认管理员租户（向后兼容：原有单 token 映射到此租户）
DEFAULT_ADMIN_TENANT_ID = "tenant_admin"
DEFAULT_ADMIN_TENANT_NAME = "admin"


# ── Schema ──

_PG_CREATE_TENANTS = """\
CREATE TABLE IF NOT EXISTS tenants (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    api_key VARCHAR(255) NOT NULL UNIQUE,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
)
"""

_PG_CREATE_TENANTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_tenants_api_key ON tenants(api_key)",
    "CREATE INDEX IF NOT EXISTS idx_tenants_name ON tenants(name)",
]


def init_tenant_tables() -> None:
    """初始化 tenants 表（仅 PG 模式）."""
    if not is_pg():
        logger.debug("SQLite 模式，跳过 tenants 表初始化")
        return

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(_PG_CREATE_TENANTS)
        for sql in _PG_CREATE_TENANTS_INDEXES:
            cur.execute(sql)

    logger.info("tenants 表初始化完成")


# ── 租户 CRUD ──


def create_tenant(
    name: str,
    is_admin: bool = False,
    api_key: str | None = None,
    tenant_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """创建新租户.

    Args:
        name: 租户名称
        is_admin: 是否管理员（可访问所有租户数据）
        api_key: 自定义 API key（留空则自动生成）
        tenant_id: 自定义 tenant_id（留空则自动生成）
        metadata: 附加元数据

    Returns:
        租户信息字典
    """
    import json

    if not is_pg():
        raise RuntimeError("多租户仅支持 PG 模式")

    tid = tenant_id or f"tenant_{uuid.uuid4().hex[:12]}"
    key = api_key or secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    meta_json = json.dumps(metadata or {}, ensure_ascii=False)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO tenants (id, name, api_key, is_admin, created_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (tid, name, key, is_admin, now, meta_json),
        )

    return {
        "id": tid,
        "name": name,
        "api_key": key,
        "is_admin": is_admin,
        "created_at": now.isoformat(),
    }


def get_tenant_by_api_key(api_key: str) -> TenantContext | None:
    """通过 API key 查找租户（时序安全比较）.

    Returns:
        TenantContext 或 None
    """
    if not is_pg():
        return None

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, api_key, is_admin FROM tenants",
        )
        rows = cur.fetchall()

    # 时序安全比较，避免 timing attack
    for row in rows:
        if hmac.compare_digest(row[2], api_key):
            return TenantContext(
                tenant_id=row[0],
                tenant_name=row[1],
                is_admin=row[3],
            )
    return None


def get_tenant_by_id(tenant_id: str) -> TenantContext | None:
    """通过 tenant_id 查找租户."""
    if not is_pg():
        return None

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, is_admin FROM tenants WHERE id = %s",
            (tenant_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return TenantContext(
            tenant_id=row[0],
            tenant_name=row[1],
            is_admin=row[2],
        )


def list_tenants() -> list[dict[str, Any]]:
    """列出所有租户（不含 api_key）."""
    if not is_pg():
        return []

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, is_admin, created_at FROM tenants ORDER BY created_at")
        rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "is_admin": row[2],
                "created_at": row[3].isoformat() if row[3] else None,
            }
            for row in rows
        ]


def delete_tenant(tenant_id: str) -> bool:
    """删除租户（仅删 tenants 表记录，不删关联数据）.

    Returns:
        是否成功删除（False 表示未找到）
    """
    if not is_pg():
        raise RuntimeError("多租户仅支持 PG 模式")

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM tenants WHERE id = %s", (tenant_id,))
        return cur.rowcount > 0


def update_tenant(
    tenant_id: str,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """更新租户信息（仅允许改 name 和 metadata）.

    Returns:
        更新后的租户信息，未找到返回 None
    """
    import json

    if not is_pg():
        raise RuntimeError("多租户仅支持 PG 模式")

    if name is None and metadata is None:
        raise ValueError("至少需要提供 name 或 metadata")

    # 构建 SET 子句
    sets: list[str] = []
    params: list[Any] = []
    if name is not None:
        sets.append("name = %s")
        params.append(name)
    if metadata is not None:
        sets.append("metadata = %s")
        params.append(json.dumps(metadata, ensure_ascii=False))

    params.append(tenant_id)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE tenants SET {', '.join(sets)} WHERE id = %s",  # noqa: S608
            tuple(params),
        )
        if cur.rowcount == 0:
            return None

    # 返回更新后的信息
    tenant = get_tenant_by_id(tenant_id)
    if not tenant:
        return None
    return {
        "id": tenant.tenant_id,
        "name": tenant.tenant_name,
        "is_admin": tenant.is_admin,
    }


def get_tenant_detail(tenant_id: str) -> dict[str, Any] | None:
    """获取租户详细信息（含 metadata，不含 api_key）."""
    if not is_pg():
        return None

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, is_admin, created_at, metadata FROM tenants WHERE id = %s",
            (tenant_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "is_admin": row[2],
            "created_at": row[3].isoformat() if row[3] else None,
            "metadata": row[4],
        }


# ── 认证中间件 ──


def ensure_admin_tenant(admin_token: str) -> None:
    """确保管理员租户存在（服务启动时调用）.

    将原有的单一 Bearer token 注册为管理员租户，实现向后兼容。
    """
    if not is_pg():
        return

    existing = get_tenant_by_api_key(admin_token)
    if existing:
        return

    create_tenant(
        name=DEFAULT_ADMIN_TENANT_NAME,
        is_admin=True,
        api_key=admin_token,
        tenant_id=DEFAULT_ADMIN_TENANT_ID,
    )
    logger.info("管理员租户已创建: %s", DEFAULT_ADMIN_TENANT_ID)


def get_current_tenant(request: Any) -> TenantContext:
    """从请求中获取当前租户上下文.

    依赖 MultiTenantAuthMiddleware 已将 tenant 注入 request.state。
    如果未找到，返回管理员租户（向后兼容）。
    """
    tenant = getattr(getattr(request, "state", None), "tenant", None)
    if tenant is not None:
        return tenant

    # 向后兼容：未经过多租户中间件的请求视为管理员
    return TenantContext(
        tenant_id=DEFAULT_ADMIN_TENANT_ID,
        tenant_name=DEFAULT_ADMIN_TENANT_NAME,
        is_admin=True,
    )


# ── 多租户认证中间件 ──


class MultiTenantAuthMiddleware:
    """替代原有 BearerTokenMiddleware，支持多租户认证.

    工作流程:
    1. 提取 Bearer token
    2. 查 tenants 表找到对应租户
    3. 注入 request.state.tenant
    4. 如果 token 无效返回 401

    向后兼容: 如果数据库不可用或非 PG 模式，回退到原有单 token 校验。
    """

    def __init__(
        self,
        app: Any,
        *,
        admin_token: str,
        skip_paths: list[str] | None = None,
        shared_cache: dict[str, TenantContext] | None = None,
    ):
        self.app = app
        self.admin_token = admin_token
        self.skip_paths = skip_paths or ["/health"]
        # 缓存：api_key -> TenantContext（避免每次查库）
        # 支持外部传入共享 dict，方便 handler 清缓存
        self._cache: dict[str, TenantContext] = shared_cache if shared_cache is not None else {}

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from starlette.requests import Request
        from starlette.responses import JSONResponse

        request = Request(scope, receive)
        path = request.url.path

        # 跳过的路径
        if any(path == sp or path.startswith(sp + "/") for sp in self.skip_paths):
            await self.app(scope, receive, send)
            return

        # 提取 token
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            response = JSONResponse({"error": "unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return

        token = auth[7:]

        # 查缓存
        tenant = self._cache.get(token)

        if tenant is None:
            # 先尝试数据库查找
            try:
                tenant = get_tenant_by_api_key(token)
            except Exception:
                tenant = None

            # 回退：如果 DB 查不到但和 admin_token 匹配，创建管理员上下文
            if tenant is None and self.admin_token and hmac.compare_digest(token, self.admin_token):
                tenant = TenantContext(
                    tenant_id=DEFAULT_ADMIN_TENANT_ID,
                    tenant_name=DEFAULT_ADMIN_TENANT_NAME,
                    is_admin=True,
                )

            if tenant is None:
                response = JSONResponse({"error": "unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return

            # 缓存
            self._cache[token] = tenant

        # 注入到 request.state
        scope.setdefault("state", {})
        scope["state"]["tenant"] = tenant

        await self.app(scope, receive, send)

    def invalidate_cache(self, api_key: str | None = None) -> None:
        """清除缓存（租户增删时调用）."""
        if api_key:
            self._cache.pop(api_key, None)
        else:
            self._cache.clear()
