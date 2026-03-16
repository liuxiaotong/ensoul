"""工具语义索引 — 为工具集生成 embedding，支持按自然语言搜索最相关的工具.

复用 crew.embedding 的 all-MiniLM-L6-v2 模型（384 维），
工具集是静态的，embedding 只需计算一次后缓存在模块级变量中。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ensoul.embedding import get_embedding, is_available

logger = logging.getLogger(__name__)

# ── 模块级缓存 ──
_tool_embeddings: dict[str, Any] | None = None


def _build_search_text(name: str, schema: dict[str, Any], use_cases: str = "") -> str:
    """构建工具的语义搜索文本.

    格式: "{name} {description} {use_cases}"
    """
    description = schema.get("description", "")
    parts = [name, description]
    if use_cases:
        parts.append(use_cases)
    return " ".join(parts)


def _ensure_index(
    tool_schemas: dict[str, dict[str, Any]] | None = None,
    use_cases_map: dict[str, str] | None = None,
) -> bool:
    """确保工具索引已构建.

    Args:
        tool_schemas: 工具 schema 字典。为 None 时从 tool_schema 模块自动获取。
        use_cases_map: 工具名 -> use_cases 文本映射。

    Returns:
        True 表示索引可用，False 表示 embedding 不可用。
    """
    global _tool_embeddings

    if np is None:
        logger.warning("numpy 不可用，工具语义搜索将不可用")
        return False

    if _tool_embeddings is not None:
        return True

    if not is_available():
        logger.warning("embedding 模型不可用，工具语义搜索将不可用")
        return False

    if tool_schemas is None:
        from ensoul.tool_schema import _TOOL_SCHEMAS

        tool_schemas = _TOOL_SCHEMAS

    if use_cases_map is None:
        from ensoul.tool_schema import TOOL_USE_CASES

        use_cases_map = TOOL_USE_CASES

    embeddings: dict[str, Any] = {}

    for name, schema in tool_schemas.items():
        use_cases = use_cases_map.get(name, "")
        text = _build_search_text(name, schema, use_cases)
        vec = get_embedding(text)
        if vec is not None:
            embeddings[name] = np.array(vec, dtype=np.float32)

    if not embeddings:
        logger.warning("未能为任何工具生成 embedding")
        return False

    _tool_embeddings = embeddings
    logger.info("工具语义索引构建完成，共 %d 个工具", len(embeddings))
    return True


async def search_tools(
    query: str,
    candidate_tools: Sequence[str] | None = None,
    top_k: int = 10,
    threshold: float = 0.3,
) -> list[tuple[str, float]]:
    """按自然语言搜索最相关的工具.

    Args:
        query: 自然语言查询（如 "帮我订会议室"）
        candidate_tools: 限定搜索范围（如员工有权限的工具列表），None 表示全部
        top_k: 返回数量上限
        threshold: 最低相似度阈值

    Returns:
        [(tool_name, score), ...] 按相似度降序排列
    """
    if np is None:
        return []

    if _tool_embeddings is None:
        ok = await asyncio.to_thread(_ensure_index)
        if not ok:
            return []
    elif not _ensure_index():
        return []

    assert _tool_embeddings is not None

    query_vec = get_embedding(query)
    if query_vec is None:
        return []

    query_arr = np.array(query_vec, dtype=np.float32)
    _candidates = set(candidate_tools) if candidate_tools is not None else None

    # 计算余弦相似度（embedding 已 normalize，直接点积）
    results: list[tuple[str, float]] = []
    for name, tool_vec in _tool_embeddings.items():
        if _candidates is not None and name not in _candidates:
            continue
        score = float(np.dot(query_arr, tool_vec))
        if score >= threshold:
            results.append((name, score))

    # 按分数降序排列，取 top_k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def reset_index() -> None:
    """重置索引缓存（仅用于测试）."""
    global _tool_embeddings
    _tool_embeddings = None
