"""持久化记忆 — 每个员工独立的经验存储."""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def resolve_to_character_name(employee: str, project_dir: Path | None = None) -> str:
    """将 slug 或花名统一转换为花名（character_name）.

    通过 DiscoveryResult.get() 查找员工，有 CJK 检测。找不到时原样返回。
    公共函数，供 MemoryStoreDB 使用，避免逻辑重复。
    """
    if not employee or not isinstance(employee, str):
        return employee
    # 已经是中文名（含 CJK 字符）则直接返回
    if any("\u4e00" <= c <= "\u9fff" for c in employee):
        return employee
    try:
        from ensoul.discovery import discover_employees

        discovery = discover_employees(project_dir=project_dir)
        emp = discovery.get(employee)
        if emp and emp.character_name and isinstance(emp.character_name, str):
            return emp.character_name
    except Exception:
        pass
    return employee


class MemoryConfig(BaseModel):
    """记忆系统配置."""

    default_ttl_days: int = Field(default=0, description="默认 TTL 天数 (0=永不过期)")
    max_entries_per_employee: int = Field(default=0, description="每员工最大记忆条数 (0=不限)")
    confidence_half_life_days: float = Field(default=90.0, description="置信度衰减半衰期（天）")
    auto_index: bool = Field(default=True, description="写入时自动更新语义索引")


class MemoryEntry(BaseModel):
    """单条记忆."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], description="唯一 ID")
    employee: str = Field(description="员工标识符")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="创建时间"
    )
    category: Literal["decision", "estimate", "finding", "correction", "pattern"] = Field(
        description="记忆类别"
    )
    content: str = Field(description="记忆内容")
    source_session: str = Field(default="", description="来源 session ID")
    confidence: float = Field(default=1.0, description="置信度（被纠正后降低）")
    superseded_by: str = Field(default="", description="被哪条记忆覆盖")
    # Enhancement 1: 衰减 + 容量
    ttl_days: int = Field(default=0, description="生存期天数，0=永不过期")
    # 重要性与访问追踪
    importance: int = Field(
        default=3, description="重要性 1-5（5=最高，如决策/待办；1=最低，如闲聊背景）"
    )
    last_accessed: str = Field(default="", description="最后被加载到上下文的时间（ISO 格式）")
    # Enhancement 3: 跨员工共享
    tags: list[str] = Field(default_factory=list, description="语义标签")
    shared: bool = Field(default=False, description="是否加入共享记忆池")
    # 可见性控制
    visibility: Literal["open", "private"] = Field(
        default="open", description="可见性: open=公开, private=仅私聊可见"
    )
    # 信息分级（ISO 27001 四级分类）
    classification: Literal["public", "internal", "restricted", "confidential"] = Field(
        default="internal",
        description="信息分级: public=公开, internal=内部(默认), restricted=受限(需域匹配), confidential=机密(仅CEO)",
    )
    domain: list[str] = Field(
        default_factory=list,
        description="职能域标签，仅 restricted 级别使用，如 ['hr'], ['finance']",
    )
    # Pattern 专有字段（仅 category="pattern" 时使用）
    trigger_condition: str = Field(default="", description="触发条件：什么场景下该用此模式")
    applicability: list[str] = Field(default_factory=list, description="适用范围：角色/领域标签")
    origin_employee: str = Field(default="", description="来源员工：谁发现的此模式")
    verified_count: int = Field(default=0, description="被验证次数：其他员工确认有效的次数")
    # Phase 3-1: 记忆管线 — 结构化关键词与关联
    keywords: list[str] = Field(
        default_factory=list, description="结构化关键词，用于 Connect 阶段匹配"
    )
    linked_memories: list[str] = Field(default_factory=list, description="关联记忆 ID 列表（双向）")
    # Phase 4: 时序感知 — 事实有效期
    valid_from: datetime | None = Field(default=None, description="事实生效时间（UTC）")
    valid_until: datetime | None = Field(
        default=None, description="事实失效时间（UTC），NULL=永久有效"
    )


def get_memory_store(project_dir=None, tenant_id: str | None = None):
    """工厂函数：返回 MemoryStoreDB 实例.

    Args:
        project_dir: 项目目录
        tenant_id: 租户 ID（None 则使用默认管理员租户，向后兼容）

    Raises:
        RuntimeError: 当 MemoryStoreDB 初始化失败时
    """
    from ensoul.memory_store_db import MemoryStoreDB

    return MemoryStoreDB(project_dir=project_dir, tenant_id=tenant_id)
