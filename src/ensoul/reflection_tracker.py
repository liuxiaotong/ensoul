"""Reflection 效果追踪 — 统计反思记忆的召回率、质量等指标.

Phase 4 改动 16: 追踪 post-task reflection 产生的记忆是否被后续任务使用。
反思记忆通过 tags 中的 'origin:reflection' 标签识别。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def get_reflection_stats(
    since: str | None = None,
    until: str | None = None,
    tenant_id: str | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """统计反思记忆的效果指标.

    Args:
        since: 起始时间 ISO 8601（可选，过滤 created_at）
        until: 截止时间 ISO 8601（可选，过滤 created_at）
        tenant_id: 租户 ID（可选，默认用 DEFAULT_ADMIN_TENANT_ID）
        top_n: top_hit 返回条数

    Returns:
        统计结果 dict，包含:
        - total_reflections: 总反思记忆数
        - accessed_count: 被召回过的数量
        - hit_rate: accessed_count / total_reflections
        - avg_importance: 平均重要度
        - by_employee: 按员工分组的统计
        - by_category: 按 category 分组
        - top_hit: 被召回次数最多的 N 条
    """
    from ensoul.database import get_connection, is_pg
    from ensoul.tenant import DEFAULT_ADMIN_TENANT_ID

    tid = tenant_id or DEFAULT_ADMIN_TENANT_ID

    if not is_pg():
        logger.warning("reflection_tracker: 仅支持 PG 模式")
        return _empty_stats()

    # ── 基础条件 ──
    conditions = [
        "tenant_id = %s",
        "'origin:reflection' = ANY(tags)",
        "(superseded_by = '' OR superseded_by IS NULL)",
    ]
    params: list[Any] = [tid]

    if since:
        conditions.append("created_at >= %s")
        params.append(_parse_iso(since))
    if until:
        conditions.append("created_at <= %s")
        params.append(_parse_iso(until))

    where = " AND ".join(conditions)

    try:
        with get_connection() as conn:
            import psycopg2.extras

            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # 1) 总体统计
            cur.execute(
                f"""
                SELECT
                    COUNT(*) AS total,
                    COUNT(last_accessed) AS accessed,
                    COALESCE(AVG(importance), 0) AS avg_importance
                FROM memories
                WHERE {where}
                """,
                params,
            )
            row = cur.fetchone()
            total = int(row["total"])
            accessed = int(row["accessed"])
            avg_importance = round(float(row["avg_importance"]), 2)

            if total == 0:
                return _empty_stats()

            hit_rate = round(accessed / total, 4)

            # 2) 按员工分组
            cur.execute(
                f"""
                SELECT
                    employee,
                    COUNT(*) AS total,
                    COUNT(last_accessed) AS accessed,
                    COALESCE(AVG(importance), 0) AS avg_importance
                FROM memories
                WHERE {where}
                GROUP BY employee
                ORDER BY total DESC
                """,
                params,
            )
            by_employee = {
                r["employee"]: {
                    "total": int(r["total"]),
                    "accessed": int(r["accessed"]),
                    "hit_rate": round(int(r["accessed"]) / int(r["total"]), 4)
                    if int(r["total"]) > 0
                    else 0,
                    "avg_importance": round(float(r["avg_importance"]), 2),
                }
                for r in cur.fetchall()
            }

            # 3) 按 category 分组
            cur.execute(
                f"""
                SELECT
                    category,
                    COUNT(*) AS total,
                    COUNT(last_accessed) AS accessed,
                    COALESCE(AVG(importance), 0) AS avg_importance
                FROM memories
                WHERE {where}
                GROUP BY category
                ORDER BY total DESC
                """,
                params,
            )
            by_category = {
                r["category"]: {
                    "total": int(r["total"]),
                    "accessed": int(r["accessed"]),
                    "hit_rate": round(int(r["accessed"]) / int(r["total"]), 4)
                    if int(r["total"]) > 0
                    else 0,
                    "avg_importance": round(float(r["avg_importance"]), 2),
                }
                for r in cur.fetchall()
            }

            # 4) 被召回最多的 top N（按 recall_count 降序，只取有召回记录的）
            cur.execute(
                f"""
                SELECT id, employee, category, content, importance,
                       recall_count, last_accessed, created_at
                FROM memories
                WHERE {where}
                  AND recall_count > 0
                ORDER BY recall_count DESC
                LIMIT %s
                """,
                (*params, top_n),
            )
            top_hit = [
                {
                    "id": r["id"],
                    "employee": r["employee"],
                    "category": r["category"],
                    "content": r["content"][:200],
                    "importance": int(r["importance"]),
                    "recall_count": int(r["recall_count"]),
                    "last_accessed": r["last_accessed"].isoformat()
                    if r["last_accessed"] and hasattr(r["last_accessed"], "isoformat")
                    else str(r["last_accessed"] or ""),
                    "created_at": r["created_at"].isoformat()
                    if hasattr(r["created_at"], "isoformat")
                    else str(r["created_at"]),
                }
                for r in cur.fetchall()
            ]

    except Exception:
        logger.warning("reflection_tracker: 查询失败", exc_info=True)
        return _empty_stats()

    return {
        "total_reflections": total,
        "accessed_count": accessed,
        "hit_rate": hit_rate,
        "avg_importance": avg_importance,
        "by_employee": by_employee,
        "by_category": by_category,
        "top_hit": top_hit,
    }


def _empty_stats() -> dict[str, Any]:
    """返回零值统计结果."""
    return {
        "total_reflections": 0,
        "accessed_count": 0,
        "hit_rate": 0,
        "avg_importance": 0,
        "by_employee": {},
        "by_category": {},
        "top_hit": [],
    }


from datetime import datetime


def _parse_iso(s: str) -> datetime:
    """Parse ISO 8601 datetime string (inline replacement for crew.utils.parse_iso)."""
    return datetime.fromisoformat(s)
