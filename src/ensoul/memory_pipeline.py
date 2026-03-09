"""记忆管线 — Reflect -> Connect -> Store 三步处理.

Phase 3-1: 核心理念是没有坏记忆，只有未加工的原始数据。
但不是所有数据都值得进入记忆库。

Reflect: 用 LLM 提取结构化笔记，决定是否存储
Connect: 用关键词匹配找到关联记忆，决定 merge/link/new
Store:   执行实际的数据库写入
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import httpx

from ensoul.memory import MemoryEntry, resolve_to_character_name
from ensoul.memory_constants import (
    CONNECT_LLM_TIMEOUT,
    KEYWORD_LINK_THRESHOLD,
    KEYWORD_MERGE_THRESHOLD,
    SEMANTIC_CANDIDATE_MIN_SIMILARITY,
    SEMANTIC_LINK_THRESHOLD,
    SEMANTIC_MERGE_THRESHOLD,
    SEMANTIC_NOOP_THRESHOLD,
)

if TYPE_CHECKING:
    from ensoul.memory_store_db import MemoryStoreDB

logger = logging.getLogger(__name__)


def _parse_llm_json(text: str) -> dict | None:
    """从 LLM 响应中解析 JSON，自动剥离 markdown 代码块包裹."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


# ── 常量 ──

MAX_LINKED_MEMORIES = 20

# ── Anthropic client 惰性单例 ──

_anthropic_client: Any = None


def _get_anthropic_client() -> Any:
    """获取或创建 Anthropic client 单例（不传 timeout，由调用方按需指定）."""
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY 环境变量未设置")
        try:
            import anthropic

            _anthropic_client = anthropic.Anthropic(api_key=api_key)
        except ImportError as e:
            raise RuntimeError("anthropic SDK 未安装") from e
    return _anthropic_client


def _reset_anthropic_client() -> None:
    """重置 Anthropic client 单例（供测试 teardown 使用）."""
    global _anthropic_client
    _anthropic_client = None


# ── 数据结构 ──


@dataclass
class ReflectResult:
    """Reflect 阶段的输出：结构化笔记."""

    store: bool
    content: str
    category: Literal["decision", "finding", "pattern", "correction", "estimate"]
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    context: str = ""
    importance: int = 3
    confidence: float = 0.9
    trigger_condition: str = ""
    applicability: list[str] = field(default_factory=list)


@dataclass
class ConnectResult:
    """Connect 阶段的输出：关联决策."""

    action: Literal["merge", "link", "new", "update", "noop", "conflict"]
    entry: MemoryEntry
    merged_entry_id: str | None = None  # merge/update/noop/conflict 时关联的已有记忆 ID
    rationale: str = ""  # LLM 决策理由


# ── Reflect 阶段 ──

_REFLECT_PROMPT = """\
你是一个记忆提炼器。分析以下原始文本，判断是否值得存储为长期记忆。

规则：
- 值得存储的：决策及理由、发现的事实/规律、工作模式/最佳实践、纠正（之前错了现在对了）、估算经验
- 不值得存储的：纯过程性描述（"我打开了文件"）、重复已知信息、无事实/决策/模式的闲聊
- 提炼时要保留关键信息，去掉冗余
- 原子化：一条记忆 = 一个独立知识点，不要把多个无关事实塞进一条

importance 评分标准（1-5）：
  1 = 琐碎操作记录（如"改了个变量名"）
  2 = 普通发现（如"这个接口返回 JSON"）
  3 = 有价值的发现（如"API 限流阈值 100 req/min"）
  4 = 重要决策/教训（如"选用 Redis 做缓存因为…"）
  5 = 关键架构决策/安全事件（如"数据库迁移方案确定"）

员工：{employee}

原始文本：
{raw_text}

请返回 JSON（不要加 markdown 标记）：
{{
  "store": true或false,
  "content": "提炼后的内容（简洁但完整）",
  "category": "decision/finding/pattern/correction/estimate",
  "keywords": ["关键词1", "关键词2"],
  "tags": ["标签1"],
  "importance": 1-5,
  "confidence": 0.0-1.0（对这条记忆准确性的信心），
  "trigger_condition": "什么场景下该回忆这条（如：排查超时问题时、讨论缓存方案时）",
  "applicability": ["适用的角色或领域，如 backend、infra、全员"]
}}
"""

# 500 太短，丢失上下文导致 Reflect 质量差；2000 兼顾质量和 token 成本
_MAX_INPUT_LENGTH = 2000


def _call_llm(prompt: str, *, timeout: float = 30.0, max_tokens: int = 1024) -> str:
    """调用 Anthropic API 获取 LLM 响应.

    使用模块级惰性单例 client，timeout 在 messages.create() 层面指定。

    Args:
        prompt: 发送给 LLM 的提示文本
        timeout: API 调用超时秒数
        max_tokens: 最大返回 token 数

    Returns:
        LLM 响应的文本内容

    Raises:
        RuntimeError: API key 缺失或 SDK 未安装
    """
    client = _get_anthropic_client()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        timeout=httpx.Timeout(timeout),
    )
    return response.content[0].text


def reflect(raw_text: str, employee: str) -> ReflectResult | None:
    """Reflect 阶段：调 LLM 提取结构化笔记.

    Args:
        raw_text: 原始文本（超过 2000 字自动截断）
        employee: 员工标识符

    Returns:
        ReflectResult 结构化笔记，若 LLM 判断 store=false 则返回 None
    """
    # 输入截断
    truncated = raw_text[:_MAX_INPUT_LENGTH] if len(raw_text) > _MAX_INPUT_LENGTH else raw_text

    prompt = _REFLECT_PROMPT.format(employee=employee, raw_text=truncated)

    try:
        response_text = _call_llm(prompt)

        # 尝试解析 JSON（容忍 markdown 代码块包裹）
        data = _parse_llm_json(response_text)

        # 验证必需字段
        if not isinstance(data, dict) or "store" not in data:
            logger.warning("Reflect LLM 返回格式异常: %s", response_text[:200])
            return None

        if not data["store"]:
            logger.debug("Reflect 决定跳过: employee=%s", employee)
            return None

        # 校验 category
        valid_categories = {"decision", "finding", "pattern", "correction", "estimate"}
        category = data.get("category", "finding")
        if category not in valid_categories:
            category = "finding"

        # 解析 importance（限制 1-5 范围）
        raw_importance = data.get("importance", 3)
        try:
            importance = max(1, min(5, int(raw_importance)))
        except (TypeError, ValueError):
            importance = 3

        # 解析 confidence（限制 0.0-1.0 范围）
        raw_confidence = data.get("confidence", 0.9)
        try:
            confidence = max(0.0, min(1.0, float(raw_confidence)))
        except (TypeError, ValueError):
            confidence = 0.9

        return ReflectResult(
            store=True,
            content=data.get("content", truncated),
            category=category,
            keywords=data.get("keywords", []),
            tags=data.get("tags", []),
            context=data.get("context", ""),
            importance=importance,
            confidence=confidence,
            trigger_condition=data.get("trigger_condition", data.get("context", "")),
            applicability=data.get("applicability", []),
        )

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("Reflect 阶段解析失败: %s", e)
        return None
    except Exception as e:
        logger.error("Reflect 阶段异常: %s", e)
        return None


# ── Connect 阶段 ──


def _keyword_overlap(keywords_a: list[str], keywords_b: list[str]) -> float:
    """计算两组关键词的重叠度.

    Args:
        keywords_a: 第一组关键词
        keywords_b: 第二组关键词

    Returns:
        重叠度 0.0-1.0（交集/并集，Jaccard 系数）
    """
    if not keywords_a or not keywords_b:
        return 0.0
    set_a = {k.lower() for k in keywords_a}
    set_b = {k.lower() for k in keywords_b}
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _find_candidates_by_keywords(
    keywords: list[str], employee: str, store: MemoryStoreDB, limit: int = 5
) -> list[MemoryEntry]:
    """用 ILIKE 匹配关键词，查已有记忆候选.

    Args:
        keywords: 要匹配的关键词列表
        employee: 员工标识符
        store: 数据库存储实例
        limit: 最多返回候选数

    Returns:
        匹配到的记忆条目列表（最多 limit 条）
    """
    if not keywords:
        return []

    from ensoul.database import get_connection

    # 用 ANY 匹配：记忆的 keywords 数组中有任一元素 ILIKE 任一搜索关键词
    # 构建条件：对每个搜索关键词，检查 keywords 数组是否包含匹配项
    conditions = []
    params: list[str | int] = []

    employee_resolved = resolve_to_character_name(employee)

    for kw in keywords[:10]:  # 最多 10 个关键词避免查询过大
        conditions.append("EXISTS (SELECT 1 FROM unnest(keywords) AS k WHERE k ILIKE %s)")
        params.append(f"%{kw}%")

    if not conditions:
        return []

    keyword_condition = " OR ".join(conditions)

    with get_connection() as conn:
        import psycopg2.extras

        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"""
            SELECT id, employee, created_at, category, content,
                   source_session, confidence, superseded_by, ttl_days,
                   importance, last_accessed, tags, shared, visibility,
                   trigger_condition, applicability, origin_employee, verified_count,
                   classification, domain,
                   keywords, linked_memories
            FROM memories
            WHERE employee = %s
              AND tenant_id = %s
              AND (superseded_by = '' OR superseded_by IS NULL)
              AND ({keyword_condition})
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (employee_resolved, store._tenant_id, *params, limit),
        )
        rows = cur.fetchall()

    return [store._row_to_entry(row) for row in rows]


def _find_candidates_by_semantic(
    content: str,
    keywords: list[str],
    employee: str,
    store: MemoryStoreDB,
    limit: int = 5,
    min_similarity: float = SEMANTIC_CANDIDATE_MIN_SIMILARITY,
) -> list[tuple[MemoryEntry, float]]:
    """NG-2: 用向量相似度找候选记忆.

    Args:
        content: 新记忆内容
        keywords: 新记忆关键词
        employee: 员工标识符
        store: 数据库存储实例
        limit: 最多返回候选数
        min_similarity: 最低 cosine similarity 阈值

    Returns:
        (MemoryEntry, similarity) 元组列表，按相似度降序
    """
    try:
        from ensoul.embedding import build_embedding_text, get_embedding

        emb_text = build_embedding_text(content, keywords)
        query_vec = get_embedding(emb_text)
        if query_vec is None:
            return []
    except Exception:
        return []

    from ensoul.database import get_connection

    employee_resolved = resolve_to_character_name(employee)
    max_distance = 1.0 - min_similarity  # cosine distance 阈值

    try:
        with get_connection() as conn:
            import psycopg2.extras

            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                SELECT id, employee, created_at, category, content,
                       source_session, confidence, superseded_by, ttl_days,
                       importance, last_accessed, tags, shared, visibility,
                       trigger_condition, applicability, origin_employee, verified_count,
                       classification, domain,
                       keywords, linked_memories,
                       (embedding <=> %s) AS cosine_dist
                FROM memories
                WHERE employee = %s
                  AND tenant_id = %s
                  AND (superseded_by = '' OR superseded_by IS NULL)
                  AND embedding IS NOT NULL
                  AND (embedding <=> %s) < %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (
                    str(query_vec),
                    employee_resolved,
                    store._tenant_id,
                    str(query_vec),
                    max_distance,
                    str(query_vec),
                    limit,
                ),
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            sim = 1.0 - float(row["cosine_dist"])
            entry = store._row_to_entry(row)
            results.append((entry, sim))
        return results

    except Exception as e:
        logger.debug("semantic candidate search failed: %s", e)
        return []


_CONNECT_DECIDE_PROMPT = """\
你是一个记忆管理器。给定一条新记忆和若干已有候选记忆，判断应执行什么操作。

新记忆：
- 内容: {new_content}
- 类别: {new_category}
- 重要性: {new_importance}
- 关键词: {new_keywords}

已有候选记忆：
{candidates_text}

操作选项：
- ADD: 全新记忆，与已有候选无关
- UPDATE: 新记忆是旧记忆的更新版本（旧内容已过时），用新内容替换旧记忆
- MERGE: 新旧记忆语义高度重叠，应合并为一条
- NOOP: 已有候选中存在等价记忆，新记忆无需写入
- CONFLICT: 新记忆与旧记忆存在矛盾（事实冲突），需标记

判断规则：
1. 如果新记忆包含的信息已被某候选完全覆盖，选 NOOP
2. 如果新记忆是某候选的更新版本（同一主题但事实有变化），选 UPDATE
3. 如果新旧记忆描述同一件事但各有补充信息，选 MERGE
4. 如果新旧记忆在事实上矛盾（如不同的数值、相反的结论），选 CONFLICT
5. 如果新记忆与所有候选都不相关，选 ADD

请返回 JSON（不要加 markdown 标记）：
{{
  "action": "ADD|UPDATE|MERGE|NOOP|CONFLICT",
  "target_id": "候选记忆 ID（ADD 时为 null）",
  "rationale": "一句话理由"
}}
"""


def _llm_decide_action(
    note: ReflectResult,
    candidates: list[MemoryEntry],
) -> dict | None:
    """调 LLM 决定 Connect 操作.

    Args:
        note: 新记忆的 ReflectResult
        candidates: top-3 候选记忆

    Returns:
        解析后的决策 dict（含 action/target_id/rationale），失败返回 None
    """
    # 构建候选文本
    candidate_lines = []
    for i, c in enumerate(candidates[:3], 1):
        candidate_lines.append(
            f"候选 {i}:\n"
            f"  ID: {c.id}\n"
            f"  内容: {c.content[:500]}\n"
            f"  类别: {c.category}\n"
            f"  重要性: {c.importance}\n"
            f"  关键词: {', '.join(c.keywords)}\n"
            f"  创建时间: {c.created_at}"
        )

    prompt = _CONNECT_DECIDE_PROMPT.format(
        new_content=note.content[:500],
        new_category=note.category,
        new_importance=note.importance,
        new_keywords=", ".join(note.keywords),
        candidates_text="\n\n".join(candidate_lines),
    )

    try:
        response_text = _call_llm(prompt, timeout=float(CONNECT_LLM_TIMEOUT), max_tokens=384)

        # 解析 JSON（容忍 markdown 代码块包裹）
        data = _parse_llm_json(response_text)
        if data is None:
            raise json.JSONDecodeError("_parse_llm_json returned None", response_text, 0)

        # 校验必须字段
        action = data.get("action", "").upper()
        valid_actions = {"ADD", "UPDATE", "MERGE", "NOOP", "CONFLICT"}
        if action not in valid_actions:
            logger.warning("Connect LLM: 无效 action=%s，降级", action)
            return None

        target_id = data.get("target_id")
        rationale = data.get("rationale", "")

        # UPDATE/MERGE/NOOP/CONFLICT 必须有 target_id
        if action in {"UPDATE", "MERGE", "NOOP", "CONFLICT"} and not target_id:
            logger.warning("Connect LLM: action=%s 但 target_id 为空，降级", action)
            return None

        # 验证 target_id 在候选列表中
        candidate_ids = {c.id for c in candidates}
        if target_id and target_id not in candidate_ids:
            logger.warning("Connect LLM: target_id=%s 不在候选列表中，降级", target_id)
            return None

        return {"action": action, "target_id": target_id, "rationale": rationale}

    except json.JSONDecodeError as e:
        logger.warning("Connect LLM: JSON 解析失败: %s，降级到阈值逻辑", e)
        return None
    except Exception as e:
        logger.warning("Connect LLM: 调用异常: %s，降级到阈值逻辑", e)
        return None


def _execute_merge(
    target: MemoryEntry,
    note: ReflectResult,
    employee: str,
    store: MemoryStoreDB,
    rationale: str = "",
) -> ConnectResult:
    """执行 MERGE 操作：合并新旧内容到已有记忆.

    PG 模式下多步更新在同一事务中完成，保证原子性。
    """
    merged_content = f"{target.content}\n---\n{note.content}"
    if len(merged_content) > 3000:
        merged_content = f"{target.content[:1500]}\n---\n{note.content}"
    merged_keywords = list(set(target.keywords + note.keywords))
    merged_importance = max(note.importance, target.importance)

    # 尝试在单个事务中完成多步更新（PG 模式）
    from ensoul.database import is_pg, get_connection

    if is_pg():
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                # 合并为单条 UPDATE：content + importance + keywords
                cur.execute(
                    "UPDATE memories SET content = %s, importance = %s, keywords = %s "
                    "WHERE id = %s AND tenant_id = %s",
                    (merged_content, merged_importance, merged_keywords, target.id, store._tenant_id),
                )
        except Exception:
            logger.warning("merge: 事务更新失败 entry_id=%s，降级到非事务模式", target.id)
            store.update(
                target.id,
                employee,
                content=merged_content,
                importance=merged_importance if merged_importance > target.importance else None,
            )
            try:
                store.update_keywords(target.id, employee, merged_keywords)
            except Exception:
                logger.warning("merge: update_keywords 失败 entry_id=%s", target.id)
    else:
        # SQLite 模式：保持现有行为
        store.update(
            target.id,
            employee,
            content=merged_content,
            importance=merged_importance if merged_importance > target.importance else None,
        )
        try:
            store.update_keywords(target.id, employee, merged_keywords)
        except Exception:
            logger.warning("merge: update_keywords 失败 entry_id=%s，content 已更新", target.id)

    # 更新 embedding
    _update_embedding(target.id, merged_content, merged_keywords, store)

    updated_entry = target.model_copy(
        update={
            "content": merged_content,
            "keywords": merged_keywords,
            "importance": merged_importance,
        }
    )
    logger.info("connect merge: id=%s", target.id)
    return ConnectResult(
        action="merge",
        entry=updated_entry,
        merged_entry_id=target.id,
        rationale=rationale,
    )


def _execute_update(
    target: MemoryEntry,
    note: ReflectResult,
    employee: str,
    store: MemoryStoreDB,
    rationale: str = "",
    **store_kwargs: Any,
) -> ConnectResult:
    """执行 UPDATE 操作：写入新记忆，标记旧记忆失效.

    Phase 4: 旧记忆设 valid_until=now + superseded_by=新记忆ID，
    新记忆通过 linked_memories 指向旧记忆，保留完整变更历史。
    PG 模式下，新记忆 linked_memories 和旧记忆标记失效在同一事务中完成。
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    # 1. 写入新记忆
    new_entry = _store_new(note, employee, store, **store_kwargs)

    # 2+3. 新记忆 linked_memories + 旧记忆标记失效（尝试事务）
    from ensoul.database import is_pg, get_connection

    if is_pg():
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                # 新记忆 linked_memories 指向旧记忆
                cur.execute(
                    "UPDATE memories SET linked_memories = %s WHERE id = %s AND tenant_id = %s",
                    ([target.id], new_entry.id, store._tenant_id),
                )
                # 旧记忆标记失效
                cur.execute(
                    "UPDATE memories SET valid_until = %s, superseded_by = %s "
                    "WHERE id = %s AND tenant_id = %s",
                    (now, new_entry.id, target.id, store._tenant_id),
                )
        except Exception:
            logger.warning("update: 事务更新失败，降级到非事务模式 new_id=%s", new_entry.id)
            try:
                store.update_linked_memories(new_entry.id, employee, [target.id])
            except Exception:
                logger.warning(
                    "update: 新记忆 linked_memories 设置失败 new_id=%s", new_entry.id
                )
            store.update(target.id, employee, valid_until=now, superseded_by=new_entry.id)
    else:
        # SQLite 模式：保持现有行为
        try:
            store.update_linked_memories(new_entry.id, employee, [target.id])
        except Exception:
            logger.warning("update: 新记忆 linked_memories 设置失败 new_id=%s", new_entry.id)
        store.update(target.id, employee, valid_until=now, superseded_by=new_entry.id)

    logger.info(
        "connect update: new=%s supersedes=%s rationale=%s",
        new_entry.id,
        target.id,
        rationale,
    )
    return ConnectResult(
        action="update",
        entry=new_entry,
        merged_entry_id=target.id,
        rationale=rationale,
    )


def _execute_noop(
    target: MemoryEntry,
    employee: str,
    store: MemoryStoreDB,
    rationale: str = "",
) -> ConnectResult:
    """执行 NOOP 操作：丢弃新记忆，更新旧记忆的 last_accessed."""
    try:
        store.record_usage(target.id, employee)
    except Exception:
        logger.warning("noop: record_usage 失败 entry_id=%s", target.id)

    logger.info("connect noop: duplicate of id=%s rationale=%s", target.id, rationale)
    return ConnectResult(
        action="noop",
        entry=target,
        merged_entry_id=target.id,
        rationale=rationale,
    )


def _execute_conflict(
    target: MemoryEntry,
    note: ReflectResult,
    employee: str,
    store: MemoryStoreDB,
    rationale: str = "",
    **store_kwargs: Any,
) -> ConnectResult:
    """执行 CONFLICT 操作：写入新记忆并标记矛盾关系."""
    # 新记忆加 conflict tag
    conflict_tags = list(set(note.tags + ["conflict"]))
    note_with_conflict = ReflectResult(
        store=note.store,
        content=note.content,
        category=note.category,
        keywords=note.keywords,
        tags=conflict_tags,
        context=note.context,
        importance=note.importance,
        confidence=note.confidence,
        trigger_condition=note.trigger_condition,
        applicability=note.applicability,
    )
    new_entry = _store_new(note_with_conflict, employee, store, **store_kwargs)

    # 预计算旧记忆的 conflict tags（事务和降级分支共用）
    new_tags = list(set(target.tags) | {"conflict"})

    # 后续三步在同一事务中完成，避免中间失败导致数据不一致
    try:
        from ensoul.database import get_connection

        with get_connection() as conn:
            cur = conn.cursor()
            # 新记忆 linked_memories 指向矛盾的旧记忆
            cur.execute(
                "UPDATE memories SET linked_memories = %s WHERE id = %s AND tenant_id = %s",
                ([target.id], new_entry.id, store._tenant_id),
            )
            # 旧记忆 tags 加 conflict
            cur.execute(
                "UPDATE memories SET tags = %s WHERE id = %s AND tenant_id = %s",
                (new_tags, target.id, store._tenant_id),
            )
            # 旧记忆 linked_memories 加新记忆（限制上限）
            old_linked = list(target.linked_memories or [])
            if new_entry.id not in old_linked:
                old_linked.append(new_entry.id)
            old_linked = old_linked[-MAX_LINKED_MEMORIES:]
            cur.execute(
                "UPDATE memories SET linked_memories = %s WHERE id = %s AND tenant_id = %s",
                (old_linked, target.id, store._tenant_id),
            )
    except Exception:
        logger.warning("conflict: 事务更新失败 new_id=%s target_id=%s，降级到非事务模式", new_entry.id, target.id)
        # 降级：逐步执行各个操作
        try:
            store.update_linked_memories(new_entry.id, employee, [target.id])
        except Exception:
            logger.warning("conflict: 新记忆 linked_memories 设置失败 new_id=%s", new_entry.id)
        try:
            store.update(target.id, employee, tags=new_tags)
        except Exception:
            logger.warning("conflict: 旧记忆 tags 更新失败 target_id=%s", target.id)
        try:
            old_linked = list(target.linked_memories or [])
            if new_entry.id not in old_linked:
                old_linked.append(new_entry.id)
            old_linked = old_linked[-MAX_LINKED_MEMORIES:]
            store.update_linked_memories(target.id, employee, old_linked)
        except Exception:
            logger.warning("conflict: 旧记忆 linked_memories 更新失败 target_id=%s", target.id)

    logger.info(
        "connect conflict: new=%s conflicts_with=%s rationale=%s",
        new_entry.id,
        target.id,
        rationale,
    )
    return ConnectResult(
        action="conflict",
        entry=new_entry,
        merged_entry_id=target.id,
        rationale=rationale,
    )


def _execute_link(
    target: MemoryEntry,
    note: ReflectResult,
    employee: str,
    store: MemoryStoreDB,
    rationale: str = "",
    **store_kwargs: Any,
) -> ConnectResult:
    """执行 link 操作：新建记忆 + 建立双向链接."""
    new_entry = _store_new(note, employee, store, **store_kwargs)
    try:
        old_linked = list(target.linked_memories or [])
        if new_entry.id not in old_linked:
            old_linked.append(new_entry.id)
            old_linked = old_linked[-MAX_LINKED_MEMORIES:]
            store.update_linked_memories(target.id, employee, old_linked)
        merged_kw = list(set(target.keywords + note.keywords))
        store.update_keywords(target.id, employee, merged_kw)
    except Exception:
        logger.warning(
            "link: 关联更新部分失败 new_id=%s, candidate_id=%s",
            new_entry.id,
            target.id,
        )
    logger.info("connect link: new=%s linked_to=%s", new_entry.id, target.id)
    return ConnectResult(action="link", entry=new_entry, rationale=rationale)


def _update_embedding(
    entry_id: str,
    content: str,
    keywords: list[str],
    store: MemoryStoreDB,
) -> None:
    """更新记忆的 embedding 向量."""
    try:
        from ensoul.embedding import build_embedding_text, get_embedding

        emb_text = build_embedding_text(content, keywords)
        new_vec = get_embedding(emb_text)
        if new_vec is not None:
            from ensoul.database import get_connection as _gc

            with _gc() as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE memories SET embedding = %s WHERE id = %s AND tenant_id = %s",
                    (new_vec, entry_id, store._tenant_id),
                )
    except Exception:
        logger.warning(
            "embedding 更新失败 entry_id=%s，设置 embedding=NULL 以触发重新生成",
            entry_id,
        )
        try:
            from ensoul.database import get_connection as _gc2

            with _gc2() as _conn2:
                _cur2 = _conn2.cursor()
                _cur2.execute(
                    "UPDATE memories SET embedding = NULL WHERE id = %s AND tenant_id = %s",
                    (entry_id, store._tenant_id),
                )
        except Exception:
            logger.warning("embedding 置 NULL 也失败 entry_id=%s", entry_id)


def _connect_by_threshold(
    note: ReflectResult,
    employee: str,
    store: MemoryStoreDB,
    semantic_candidates: list[tuple[MemoryEntry, float]],
    keyword_candidates: list[MemoryEntry],
    **store_kwargs: Any,
) -> ConnectResult:
    """阈值降级逻辑：LLM 不可用时的兜底判定.

    保留原有 cosine/Jaccard 阈值判定逻辑。
    """
    if semantic_candidates:
        best_candidate, best_sim = semantic_candidates[0]

        if best_sim >= SEMANTIC_NOOP_THRESHOLD:
            return _execute_noop(
                best_candidate, employee, store,
                rationale=f"阈值降级: cosine={best_sim:.3f} >= {SEMANTIC_NOOP_THRESHOLD}",
            )
        elif best_sim >= SEMANTIC_MERGE_THRESHOLD and best_candidate.category == note.category:
            return _execute_merge(best_candidate, note, employee, store)
        elif best_sim >= SEMANTIC_LINK_THRESHOLD:
            return _execute_link(best_candidate, note, employee, store, **store_kwargs)
        else:
            entry = _store_new(note, employee, store, **store_kwargs)
            return ConnectResult(action="new", entry=entry)

    if not keyword_candidates:
        entry = _store_new(note, employee, store, **store_kwargs)
        return ConnectResult(action="new", entry=entry)

    # 找最高重叠度的候选
    best_overlap = 0.0
    best_candidate_kw: MemoryEntry | None = None
    for candidate in keyword_candidates:
        overlap = _keyword_overlap(note.keywords, candidate.keywords)
        if overlap > best_overlap:
            best_overlap = overlap
            best_candidate_kw = candidate

    if best_candidate_kw is None:
        entry = _store_new(note, employee, store, **store_kwargs)
        return ConnectResult(action="new", entry=entry)

    if best_overlap >= SEMANTIC_NOOP_THRESHOLD:
        return _execute_noop(
            best_candidate_kw, employee, store,
            rationale=f"阈值降级: keyword_overlap={best_overlap:.3f} >= {SEMANTIC_NOOP_THRESHOLD}",
        )
    elif best_overlap >= KEYWORD_MERGE_THRESHOLD and best_candidate_kw.category == note.category:
        return _execute_merge(best_candidate_kw, note, employee, store)
    elif best_overlap >= KEYWORD_LINK_THRESHOLD:
        return _execute_link(best_candidate_kw, note, employee, store, **store_kwargs)
    else:
        entry = _store_new(note, employee, store, **store_kwargs)
        return ConnectResult(action="new", entry=entry)


def connect(
    note: ReflectResult,
    employee: str,
    store: MemoryStoreDB,
    **store_kwargs: Any,
) -> ConnectResult:
    """Connect 阶段：找到关联记忆，用 LLM 决定操作.

    Phase 2: 优先用 LLM 判定操作（ADD/UPDATE/MERGE/NOOP/CONFLICT），
    LLM 不可用时降级到阈值逻辑。

    候选查找逻辑不变（语义优先，关键词降级）。

    Args:
        note: Reflect 阶段输出的结构化笔记
        employee: 员工标识符
        store: 数据库存储实例
        **store_kwargs: 透传给 store.add() 的额外参数

    Returns:
        ConnectResult 包含 action、结果 entry 和 rationale
    """
    # ── 候选查找（不变）──
    semantic_candidates = _find_candidates_by_semantic(
        note.content,
        note.keywords,
        employee,
        store,
    )

    keyword_candidates: list[MemoryEntry] = []
    if not semantic_candidates:
        keyword_candidates = _find_candidates_by_keywords(note.keywords, employee, store)

    # 收集所有候选 MemoryEntry（去重）
    all_candidates: list[MemoryEntry] = []
    seen_ids: set[str] = set()
    for entry, _sim in semantic_candidates:
        if entry.id not in seen_ids:
            all_candidates.append(entry)
            seen_ids.add(entry.id)
    for entry in keyword_candidates:
        if entry.id not in seen_ids:
            all_candidates.append(entry)
            seen_ids.add(entry.id)

    # 无候选时直接 ADD，不调 LLM（省钱）
    if not all_candidates:
        entry = _store_new(note, employee, store, **store_kwargs)
        return ConnectResult(action="new", entry=entry, rationale="无候选记忆，直接新增")

    # ── LLM 决策 ──
    decision = _llm_decide_action(note, all_candidates[:3])

    if decision is None:
        # LLM 失败，降级到阈值逻辑
        logger.info("connect: LLM 决策失败，降级到阈值逻辑")
        return _connect_by_threshold(
            note, employee, store, semantic_candidates, keyword_candidates, **store_kwargs
        )

    action = decision["action"]
    target_id = decision["target_id"]
    rationale = decision["rationale"]

    # 找到 target entry
    target: MemoryEntry | None = None
    if target_id:
        for c in all_candidates:
            if c.id == target_id:
                target = c
                break

    # ── 执行操作 ──
    if action == "ADD":
        entry = _store_new(note, employee, store, **store_kwargs)
        return ConnectResult(action="new", entry=entry, rationale=rationale)

    elif action == "UPDATE":
        if target is None:
            entry = _store_new(note, employee, store, **store_kwargs)
            return ConnectResult(
                action="new", entry=entry, rationale="UPDATE target 未找到，降级 ADD"
            )
        return _execute_update(target, note, employee, store, rationale=rationale, **store_kwargs)

    elif action == "MERGE":
        if target is None:
            entry = _store_new(note, employee, store, **store_kwargs)
            return ConnectResult(
                action="new", entry=entry, rationale="MERGE target 未找到，降级 ADD"
            )
        return _execute_merge(target, note, employee, store, rationale=rationale)

    elif action == "NOOP":
        if target is None:
            entry = _store_new(note, employee, store, **store_kwargs)
            return ConnectResult(
                action="new", entry=entry, rationale="NOOP target 未找到，降级 ADD"
            )
        return _execute_noop(target, employee, store, rationale=rationale)

    elif action == "CONFLICT":
        if target is None:
            entry = _store_new(note, employee, store, **store_kwargs)
            return ConnectResult(
                action="new", entry=entry, rationale="CONFLICT target 未找到，降级 ADD"
            )
        return _execute_conflict(
            target, note, employee, store, rationale=rationale, **store_kwargs
        )

    else:
        # 不应到达，但保底
        logger.warning("connect: 未知 action=%s，降级到阈值逻辑", action)
        return _connect_by_threshold(
            note, employee, store, semantic_candidates, keyword_candidates, **store_kwargs
        )


def _store_new(
    note: ReflectResult,
    employee: str,
    store: MemoryStoreDB,
    **store_kwargs: Any,
) -> MemoryEntry:
    """Store 阶段 - 新建记忆条目.

    Args:
        note: Reflect 阶段输出
        employee: 员工标识符
        store: 数据库存储实例
        **store_kwargs: 透传给 store.add() 的额外参数
            (source_session, ttl_days, shared, confidence 等)

    Returns:
        新创建的 MemoryEntry
    """
    # 原子写入 keywords，避免 add() + update_keywords() 两步非原子操作
    entry = store.add(
        employee=employee,
        category=note.category,
        content=note.content,
        tags=note.tags,
        keywords=note.keywords,
        **store_kwargs,
    )
    return entry


# ── 顶层入口 ──


def process_memory(
    raw_text: str,
    employee: str,
    store: MemoryStoreDB | None = None,
    skip_reflect: bool = False,
    source_session: str = "",
    **kwargs: Any,
) -> MemoryEntry | None:
    """记忆管线顶层入口：Reflect -> Connect -> Store.

    Args:
        raw_text: 原始文本
        employee: 员工标识符
        store: 数据库存储实例（None 则自动获取）
        skip_reflect: 跳过 Reflect 阶段（给 add_memory 等已结构化的路径用）
        source_session: 来源会话 ID
        **kwargs: skip_reflect=True 时，用于构造 ReflectResult 的参数：
            content, category, keywords, tags, context
            以及透传给 store.add() 的参数：
            ttl_days, shared, confidence, trigger_condition,
            applicability, origin_employee, classification, domain

    Returns:
        处理后的 MemoryEntry，Reflect 决定跳过时返回 None
    """
    # P2-S20: content 长度校验 — 防止超大文本进入管线中间步骤
    if len(raw_text) > 5000:
        logger.warning("content 超长截断: employee=%s len=%d", employee, len(raw_text))
        raw_text = raw_text[:5000]

    if store is None:
        from ensoul.memory import get_memory_store

        store = get_memory_store()

    # 分离 ReflectResult 专属参数和 store.add() 透传参数
    # importance/confidence/trigger_condition/applicability 既用于构造 ReflectResult
    # 也需要透传到 store.add()，所以不放入 _reflect_only_keys
    _reflect_only_keys = {"content", "category", "keywords", "tags", "context"}
    _store_extra = {k: v for k, v in kwargs.items() if k not in _reflect_only_keys}
    if source_session:
        _store_extra["source_session"] = source_session

    if skip_reflect:
        # 直接构造 ReflectResult，对 importance/confidence 做夹值校验
        importance = max(1, min(5, int(kwargs.get("importance", 3))))
        confidence = max(0.0, min(1.0, float(kwargs.get("confidence", 0.9))))
        note = ReflectResult(
            store=True,
            content=kwargs.get("content", raw_text),
            category=kwargs.get("category", "finding"),
            keywords=kwargs.get("keywords", []),
            tags=kwargs.get("tags", []),
            context=kwargs.get("context", ""),
            importance=importance,
            confidence=confidence,
            trigger_condition=kwargs.get("trigger_condition", ""),
            applicability=kwargs.get("applicability", []),
        )
    else:
        note = reflect(raw_text, employee)
        if note is None:
            return None

    # 透传 ReflectResult 的新字段到 store.add() kwargs
    _store_extra.setdefault("importance", note.importance)
    _store_extra.setdefault("confidence", note.confidence)
    if note.trigger_condition:
        _store_extra.setdefault("trigger_condition", note.trigger_condition)
    if note.applicability:
        _store_extra.setdefault("applicability", note.applicability)

    result = connect(note, employee, store, **_store_extra)
    return result.entry
