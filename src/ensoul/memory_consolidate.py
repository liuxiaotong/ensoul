"""碎片聚合 — 把同主题的 findings 合成为高质量 pattern.

Phase 4: 记忆库中有大量零散的 finding，很多是关于同一主题的碎片。
本模块提供定期聚合机制，把同主题的 findings 合成为一条可复用的 pattern。
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from ensoul.memory import MemoryEntry, resolve_to_character_name
from ensoul.memory_pipeline import _call_llm, _keyword_overlap, _parse_llm_json, process_memory

if TYPE_CHECKING:
    from ensoul.memory_store_db import MemoryStoreDB

logger = logging.getLogger(__name__)

# ── LLM Prompt ──

_SYNTHESIZE_PROMPT = """\
你是一个知识聚合器。把多条碎片发现合成为一条可复用的模式（pattern）。

要求：
- 合成为一条可复用的策略/规律，而不是简单拼接
- 提取合并后的关键词（去重、精炼）
- 标签应反映主题领域

碎片发现列表：
{findings}

请返回 JSON（不要加 markdown 标记）：
{{
  "content": "合成后的 pattern 内容（简洁但完整的策略/规律描述）",
  "keywords": ["关键词1", "关键词2"],
  "tags": ["标签1"]
}}
"""

# 每个 cluster 最多处理的 findings 数
_MAX_CLUSTER_SIZE = 10


# ── 聚类 ──


def find_clusters(
    employee: str,
    store: MemoryStoreDB,
    min_cluster_size: int = 3,
) -> list[list[MemoryEntry]]:
    """找到同一员工的 finding 聚类.

    算法：
    1. 查所有未 superseded 的 findings
    2. 对每对 finding 计算 keyword_overlap
    3. 用贪心聚类：从最高重叠对开始，overlap >= 0.4 的归为同一 cluster
    4. 只返回 size >= min_cluster_size 的 cluster

    Args:
        employee: 员工标识符
        store: 数据库存储实例
        min_cluster_size: 最小聚类大小

    Returns:
        聚类列表，每个聚类是一组 MemoryEntry
    """
    # 查所有未 superseded 的 findings
    findings = store.query(
        employee,
        category="finding",
        limit=500,
        include_expired=False,
    )

    if len(findings) < min_cluster_size:
        return []

    # 倒排索引：keyword → entry indices（只比较有共同 keyword 的 pair）
    keyword_index: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(findings):
        for kw in (f.keywords or []):
            keyword_index[kw.lower()].append(i)

    # 统计每对共享 keyword 数，筛选候选 pair
    pair_shared: Counter[tuple[int, int]] = Counter()
    for indices in keyword_index.values():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                pair = (min(indices[a], indices[b]), max(indices[a], indices[b]))
                pair_shared[pair] += 1

    # 只对有共享 keyword 的 pair 计算精确 overlap
    pairs: list[tuple[float, int, int]] = []
    for (i, j) in pair_shared:
        overlap = _keyword_overlap(findings[i].keywords, findings[j].keywords)
        if overlap >= 0.4:
            pairs.append((overlap, i, j))

    # 按重叠度降序排序
    pairs.sort(key=lambda x: x[0], reverse=True)

    # 贪心聚类：用 Union-Find
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for _overlap, i, j in pairs:
        union(i, j)

    # 收集聚类
    clusters_map: dict[int, list[int]] = {}
    for idx in range(len(findings)):
        root = find(idx)
        clusters_map.setdefault(root, []).append(idx)

    # 只返回 size >= min_cluster_size 的 cluster
    result: list[list[MemoryEntry]] = []
    for indices in clusters_map.values():
        if len(indices) >= min_cluster_size:
            cluster = [findings[i] for i in indices]
            result.append(cluster)

    return result


# ── 合成 ──


def _mark_superseded(
    entry_ids: list[str],
    new_id: str,
    employee: str,
    store: MemoryStoreDB,
) -> int:
    """把一批 findings 标记为被新 pattern 取代.

    Args:
        entry_ids: 要标记的记忆 ID 列表
        new_id: 新 pattern 的 ID
        employee: 员工标识符
        store: 数据库存储实例

    Returns:
        成功标记的条数
    """
    from ensoul.database import get_connection

    employee_resolved = resolve_to_character_name(employee)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE memories
            SET superseded_by = %s
            WHERE id = ANY(%s)
              AND employee = %s
              AND tenant_id = %s
              AND (superseded_by = '' OR superseded_by IS NULL)
            """,
            (new_id, entry_ids, employee_resolved, store._tenant_id),
        )
        return cur.rowcount


def synthesize_cluster(
    entries: list[MemoryEntry],
    employee: str,
    store: MemoryStoreDB,
) -> MemoryEntry | None:
    """用 LLM 把一个 cluster 的 findings 合成为一条 pattern.

    1. 把所有 entries 的 content 拼成上下文（按重要性排序截取 Top 10）
    2. 调 _call_llm 合成
    3. 用 store.add() 写入新 pattern
    4. 把原 findings 标记 superseded_by = 新 pattern 的 id
    5. 新 pattern 的 linked_memories 指向所有原 findings

    Args:
        entries: 同一 cluster 的 finding 列表
        employee: 员工标识符
        store: 数据库存储实例

    Returns:
        新创建的 pattern entry，失败返回 None
    """
    # 按重要性排序，截取 Top 10
    sorted_entries = sorted(entries, key=lambda e: e.importance, reverse=True)
    top_entries = sorted_entries[:_MAX_CLUSTER_SIZE]

    # 拼接 findings 内容
    findings_text = "\n".join(f"- [{i + 1}] {e.content}" for i, e in enumerate(top_entries))

    prompt = _SYNTHESIZE_PROMPT.format(findings=findings_text)

    try:
        response_text = _call_llm(prompt)

        # 解析 JSON（容忍 markdown 代码块包裹）
        data = _parse_llm_json(response_text)

        if not isinstance(data, dict) or "content" not in data:
            logger.warning("Synthesize LLM 返回格式异常: %s", response_text[:200])
            return None

        # 走管线写入新 pattern（skip_reflect 因内容已是 LLM 合成，但走 Connect 去重）
        new_pattern = process_memory(
            raw_text=data["content"],
            employee=employee,
            store=store,
            skip_reflect=True,
            category="pattern",
            content=data["content"],
            tags=data.get("tags", []),
            keywords=data.get("keywords", []),
            importance=4,
            confidence=1.0,
        )

        if new_pattern is None:
            logger.info(
                "consolidate: Connect 判定 NOOP，跳过合成 employee=%s",
                employee,
            )
            return None

        # 标记原 findings 为 superseded
        original_ids = [e.id for e in top_entries]
        _mark_superseded(original_ids, new_pattern.id, employee, store)

        # 更新 linked_memories
        try:
            store.update_linked_memories(new_pattern.id, employee, original_ids)
        except Exception:
            logger.warning(
                "consolidate: linked_memories 更新失败 pattern_id=%s",
                new_pattern.id,
            )

        logger.info(
            "consolidate: 合成 pattern=%s from %d findings, employee=%s",
            new_pattern.id,
            len(top_entries),
            employee,
        )

        return new_pattern.model_copy(update={"linked_memories": original_ids})

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("Synthesize 阶段解析失败: %s", e)
        return None
    except Exception as e:
        logger.error("Synthesize 阶段异常: %s", e)
        return None


def _update_importance(entry_id: str, importance: int, store: MemoryStoreDB) -> None:
    """直接更新 importance 字段（store.update 不支持 importance 参数）."""
    from ensoul.database import get_connection

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE memories SET importance = %s WHERE id = %s AND tenant_id = %s",
            (importance, entry_id, store._tenant_id),
        )


# ── 顶层入口 ──


def run_consolidation(
    store: MemoryStoreDB | None = None,
    dry_run: bool = False,
    employee: str | None = None,
) -> dict:
    """顶层入口：遍历所有员工，聚合碎片.

    Args:
        store: DB store 实例（None 则自动获取）
        dry_run: True 时只返回统计，不写库
        employee: 可选，指定单个员工

    Returns:
        {"employees_processed": N, "clusters_found": N,
         "patterns_created": N, "findings_superseded": N}
    """
    if store is None:
        from ensoul.memory import get_memory_store

        store = get_memory_store()

    stats = {
        "employees_processed": 0,
        "clusters_found": 0,
        "patterns_created": 0,
        "findings_superseded": 0,
    }

    # 确定要处理的员工列表
    if employee:
        employees = [employee]
    else:
        employees = store.list_employees()

    for emp in employees:
        stats["employees_processed"] += 1

        clusters = find_clusters(emp, store)
        stats["clusters_found"] += len(clusters)

        if dry_run:
            # dry_run 模式：统计但不写库
            for cluster in clusters:
                top_n = min(len(cluster), _MAX_CLUSTER_SIZE)
                stats["findings_superseded"] += top_n
                stats["patterns_created"] += 1
            continue

        for cluster in clusters:
            pattern = synthesize_cluster(cluster, emp, store)
            if pattern is not None:
                top_n = min(len(cluster), _MAX_CLUSTER_SIZE)
                stats["patterns_created"] += 1
                stats["findings_superseded"] += top_n

    logger.info(
        "consolidation 完成: %s",
        stats,
    )

    return stats
