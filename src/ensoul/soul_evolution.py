"""灵魂自动进化 -- 从记忆中提炼行为准则候选.

高频使用的 pattern 自动晋升为 soul.md 行为准则候选，
被多条 correction 推翻的旧规则自动标记归档候选。
候选存入数据库，由管理员审批后才真正更新 soul。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ensoul.config_store import get_config, put_config
from ensoul.memory import resolve_to_character_name

try:
    import httpx

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

logger = logging.getLogger(__name__)


@dataclass
class EvolutionCandidate:
    """进化候选项."""

    id: str = ""
    employee: str = ""
    action: str = ""  # "promote" 或 "archive"
    source_type: str = ""  # "pattern" 或 "correction"
    source_ids: list[str] = field(default_factory=list)
    content: str = ""  # 候选的行为准则文本
    reason: str = ""  # 为什么推荐这个变更
    confidence: float = 0.0  # 置信度 0-1
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典."""
        return asdict(self)


def _call_llm(prompt: str) -> str:
    """调用 LLM 生成文本.

    复用 memory_pipeline 中的 LLM 调用模式。

    Args:
        prompt: 提示文本

    Returns:
        LLM 响应文本

    Raises:
        RuntimeError: API key 缺失或 SDK 未安装
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY 环境变量未设置")

    try:
        import httpx

        timeout = httpx.Timeout(30.0)
    except ImportError:
        timeout = None  # type: ignore[assignment]

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except ImportError as e:
        raise RuntimeError("anthropic SDK 未安装") from e


def _keyword_overlap(keywords_a: list[str], keywords_b: list[str]) -> float:
    """计算两组关键词的 Jaccard 重叠度.

    Args:
        keywords_a: 第一组关键词
        keywords_b: 第二组关键词

    Returns:
        0.0-1.0 的重叠度
    """
    if not keywords_a or not keywords_b:
        return 0.0
    set_a = {k.lower() for k in keywords_a}
    set_b = {k.lower() for k in keywords_b}
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _get_existing_candidate_ids(employee: str) -> set[str]:
    """获取该员工已有候选的 source_ids，避免重复推荐.

    Returns:
        已在候选列表中的 source memory ID 集合
    """
    raw = get_config("soul_evolution", f"{employee}_candidates")
    if not raw:
        return set()
    try:
        candidates = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return set()

    existing_ids: set[str] = set()
    for c in candidates:
        for sid in c.get("source_ids", []):
            existing_ids.add(sid)
    return existing_ids


def find_promotion_candidates(
    employee: str,
    store: Any,
    min_verified: int = 3,
) -> list[EvolutionCandidate]:
    """找到可晋升的 pattern.

    查询条件：
    - category = 'pattern'
    - verified_count >= min_verified
    - superseded_by = '' (未被取代)
    - 不在已有候选中（避免重复推荐）

    对每个符合条件的 pattern：
    - 用 _call_llm 把 pattern content 改写为 soul.md 行为准则格式
    - 生成 reason（引用 verified_count 和使用场景）
    - confidence = min(1.0, verified_count / 5)

    Args:
        employee: 员工名
        store: MemoryStoreDB 实例
        min_verified: 最低验证次数阈值

    Returns:
        晋升候选列表
    """
    from ensoul.database import get_connection

    employee_resolved = resolve_to_character_name(employee)
    existing_ids = _get_existing_candidate_ids(employee_resolved)

    # 直接 SQL 查询高频验证的 pattern
    try:
        import psycopg2.extras

        cursor_factory = psycopg2.extras.RealDictCursor
    except ImportError:
        cursor_factory = None  # type: ignore[assignment]

    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=cursor_factory) if cursor_factory else conn.cursor()
        cur.execute(
            """
            SELECT id, employee, created_at, category, content,
                   source_session, confidence, superseded_by, ttl_days,
                   importance, last_accessed, tags, shared, visibility,
                   trigger_condition, applicability, origin_employee, verified_count,
                   classification, domain, keywords, linked_memories
            FROM memories
            WHERE employee = %s AND tenant_id = %s
              AND category = 'pattern'
              AND verified_count >= %s
              AND (superseded_by = '' OR superseded_by IS NULL)
            ORDER BY verified_count DESC
            """,
            (employee_resolved, store._tenant_id, min_verified),
        )
        rows = cur.fetchall()

    entries = [store._row_to_entry(row) for row in rows]

    # 过滤已有候选
    entries = [e for e in entries if e.id not in existing_ids]

    candidates: list[EvolutionCandidate] = []
    now = datetime.now(timezone.utc).isoformat()

    for entry in entries:
        verified = entry.verified_count
        # 用 LLM 改写为 soul 行为准则格式
        prompt = (
            f"将以下经验总结改写为一条简短的行为准则（祈使语气，一句话，中文）：\n\n"
            f"{entry.content}\n\n"
            f"只输出改写后的行为准则，不要解释。"
        )
        try:
            soul_rule = _call_llm(prompt).strip()
        except Exception as e:
            logger.warning("LLM 改写失败，使用原文: %s", e)
            soul_rule = entry.content

        reason = f"该 pattern 已被验证 {verified} 次，置信度高，建议晋升为行为准则。"

        candidate = EvolutionCandidate(
            id=uuid.uuid4().hex[:12],
            employee=employee_resolved,
            action="promote",
            source_type="pattern",
            source_ids=[entry.id],
            content=soul_rule,
            reason=reason,
            confidence=min(1.0, verified / 5),
            created_at=now,
        )
        candidates.append(candidate)

    return candidates


def find_archive_candidates(
    employee: str,
    store: Any,
    min_corrections: int = 2,
) -> list[EvolutionCandidate]:
    """找到应归档的旧规则.

    逻辑：
    1. 查 category='correction' 的记忆
    2. 按 tags 聚类（复用 _keyword_overlap）
    3. 同一主题的 corrections 数 >= min_corrections -> 生成 archive 候选
    4. content = 被推翻的规则描述
    5. reason = 引用具体 corrections

    Args:
        employee: 员工名
        store: MemoryStoreDB 实例
        min_corrections: 最低纠正次数阈值

    Returns:
        归档候选列表
    """
    # 查询所有 correction 记忆
    corrections = store.query(employee, category="correction", limit=200)
    if not corrections:
        return []

    existing_ids = _get_existing_candidate_ids(resolve_to_character_name(employee))

    # 按 tags 聚类
    clusters: list[list[Any]] = []
    for corr in corrections:
        if corr.id in existing_ids:
            continue

        placed = False
        for cluster in clusters:
            # 与 cluster 中第一条的 tags 比较
            overlap = _keyword_overlap(corr.tags, cluster[0].tags)
            if overlap >= 0.3:
                cluster.append(corr)
                placed = True
                break
        if not placed:
            clusters.append([corr])

    candidates: list[EvolutionCandidate] = []
    now = datetime.now(timezone.utc).isoformat()
    employee_resolved = resolve_to_character_name(employee)

    for cluster in clusters:
        if len(cluster) < min_corrections:
            continue

        # 合并 cluster 中的 correction 内容
        source_ids = [c.id for c in cluster]
        contents = [c.content for c in cluster]
        combined = "; ".join(contents[:5])

        reason = (
            f"共有 {len(cluster)} 条纠正记忆指向同一主题，"
            f"建议归档相关旧规则。涉及记忆: {', '.join(source_ids[:5])}"
        )

        candidate = EvolutionCandidate(
            id=uuid.uuid4().hex[:12],
            employee=employee_resolved,
            action="archive",
            source_type="correction",
            source_ids=source_ids,
            content=combined,
            reason=reason,
            confidence=min(1.0, len(cluster) / 4),
            created_at=now,
        )
        candidates.append(candidate)

    return candidates


def run_evolution_review(
    store: Any | None = None,
    employee: str | None = None,
) -> dict[str, Any]:
    """生成进化候选并存入数据库.

    Args:
        store: MemoryStoreDB 实例（None 则自动创建）
        employee: 可选，指定单个员工（None 则审查所有有记忆的员工）

    Returns:
        {
            "employees_reviewed": N,
            "promote_candidates": N,
            "archive_candidates": N,
            "candidates": [candidate_dict, ...]
        }
    """
    if store is None:
        from ensoul.memory import get_memory_store

        store = get_memory_store()

    # 确定要审查的员工列表
    if employee:
        employees = [employee]
    else:
        employees = store.list_employees()

    total_promote = 0
    total_archive = 0
    all_candidates: list[dict[str, Any]] = []

    for emp in employees:
        emp_resolved = resolve_to_character_name(emp)

        # 读取已有候选
        raw = get_config("soul_evolution", f"{emp_resolved}_candidates")
        existing: list[dict[str, Any]] = []
        if raw:
            try:
                existing = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                existing = []

        # 查找新候选
        promote = find_promotion_candidates(emp, store)
        archive = find_archive_candidates(emp, store)

        new_candidates = [c.to_dict() for c in promote + archive]

        if new_candidates:
            # 合并到已有候选
            merged = existing + new_candidates
            put_config(
                "soul_evolution",
                f"{emp_resolved}_candidates",
                json.dumps(merged, ensure_ascii=False),
            )

        total_promote += len(promote)
        total_archive += len(archive)
        all_candidates.extend(new_candidates)

    return {
        "employees_reviewed": len(employees),
        "promote_candidates": total_promote,
        "archive_candidates": total_archive,
        "candidates": all_candidates,
    }


# ---------------------------------------------------------------------------
# 审批执行（Phase 5）
# ---------------------------------------------------------------------------

_ENSOUL_API_BASE = os.environ.get("ENSOUL_API_BASE", "")


def _employee_slug_from_name(employee_name: str) -> str | None:
    """通过 character_name 反查员工 slug.

    Returns:
        slug 字符串，找不到返回 None
    """
    try:
        from ensoul.discovery import discover_employees
    except ImportError:
        logger.warning("discovery 模块不可用，无法查找 slug: %s", employee_name)
        return None

    try:
        discovery = discover_employees()
        for slug, emp in discovery.items():
            if emp.character_name == employee_name:
                return slug
        return None
    except Exception:
        logger.warning("无法通过 discover_employees 查找 slug: %s", employee_name)
        return None


def _get_ensoul_api_token() -> str | None:
    """从环境变量获取 ENSOUL API token."""
    return os.environ.get("ENSOUL_API_TOKEN")


def _update_soul_promote(employee_slug: str, rule_text: str) -> bool:
    """在 soul 的行为准则 section 追加一条规则.

    1. 调 API 获取当前 soul
    2. 找到 "## 行为准则" section
    3. 在该 section 末尾（下一个 ## 之前）追加 "- {rule_text}"
    4. PUT 写回 soul

    Returns:
        True 更新成功
    """
    token = _get_ensoul_api_token()
    if not token or not _HAS_HTTPX or not _ENSOUL_API_BASE:
        logger.warning("ENSOUL_API_TOKEN / ENSOUL_API_BASE 或 httpx 不可用，跳过 soul 更新")
        return False

    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = httpx.get(
            f"{_ENSOUL_API_BASE}/souls/{employee_slug}",
            headers=headers,
            timeout=15.0,
        )
        if resp.status_code != 200:
            logger.error("获取 soul 失败: HTTP %d", resp.status_code)
            return False

        data = resp.json()
        soul_text = data.get("soul", "")
        if not soul_text:
            logger.error("soul 内容为空: %s", employee_slug)
            return False

        # 在 "## 行为准则" section 末尾追加
        import re

        pattern = re.compile(r"(## 行为准则.*?)((?=\n## )|$)", re.DOTALL)
        match = pattern.search(soul_text)
        if match:
            insert_pos = match.end(1)
            new_line = f"\n- {rule_text}"
            updated = soul_text[:insert_pos] + new_line + soul_text[insert_pos:]
        else:
            # 没有"行为准则"section，追加到末尾
            updated = soul_text.rstrip() + f"\n\n## 行为准则\n\n- {rule_text}\n"

        put_resp = httpx.put(
            f"{_ENSOUL_API_BASE}/souls/{employee_slug}",
            headers=headers,
            json={"soul": updated},
            timeout=15.0,
        )
        if put_resp.status_code != 200:
            logger.error("写回 soul 失败: HTTP %d", put_resp.status_code)
            return False

        return True

    except Exception as exc:
        logger.exception("更新 soul promote 异常: %s", exc)
        return False


def _update_soul_archive(employee_slug: str, archive_note: str) -> bool:
    """在 soul 末尾添加归档说明.

    简化实现：不尝试删除旧规则（太危险），而是在 soul 末尾
    "## 自检清单" 之前加一行注释。

    Returns:
        True 更新成功
    """
    token = _get_ensoul_api_token()
    if not token or not _HAS_HTTPX or not _ENSOUL_API_BASE:
        logger.warning("ENSOUL_API_TOKEN / ENSOUL_API_BASE 或 httpx 不可用，跳过 soul 更新")
        return False

    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = httpx.get(
            f"{_ENSOUL_API_BASE}/souls/{employee_slug}",
            headers=headers,
            timeout=15.0,
        )
        if resp.status_code != 200:
            logger.error("获取 soul 失败: HTTP %d", resp.status_code)
            return False

        data = resp.json()
        soul_text = data.get("soul", "")
        if not soul_text:
            logger.error("soul 内容为空: %s", employee_slug)
            return False

        # 在 "## 自检清单" 之前插入归档注释
        note_line = f"\n<!-- archived: {archive_note} -->\n"
        selfcheck_idx = soul_text.find("## 自检清单")
        if selfcheck_idx != -1:
            updated = soul_text[:selfcheck_idx] + note_line + soul_text[selfcheck_idx:]
        else:
            # 没有自检清单，追加到末尾
            updated = soul_text.rstrip() + "\n" + note_line

        put_resp = httpx.put(
            f"{_ENSOUL_API_BASE}/souls/{employee_slug}",
            headers=headers,
            json={"soul": updated},
            timeout=15.0,
        )
        if put_resp.status_code != 200:
            logger.error("写回 soul 失败: HTTP %d", put_resp.status_code)
            return False

        return True

    except Exception as exc:
        logger.exception("更新 soul archive 异常: %s", exc)
        return False


def approve_candidate(
    candidate_id: str,
    employee: str,
    store: Any | None = None,
) -> dict[str, Any]:
    """批准一个进化候选，执行 soul 更新.

    流程：
    1. 从 config_store 找到该候选（遍历 {employee}_candidates）
    2. 根据 action 类型执行 promote/archive soul 更新
    3. 标记候选为已执行（从 candidates 列表移除，存入 {employee}_approved 列表）
    4. 记一条 decision 类记忆

    Args:
        candidate_id: 候选 ID
        employee: 员工名
        store: DB store（None 则自动创建）

    Returns:
        {"ok": True, "action": "promote/archive", "content": "...", "soul_updated": True/False}
    """
    # 1. 从 config_store 读取候选列表
    raw = get_config("soul_evolution", f"{employee}_candidates")
    if not raw:
        return {"ok": False, "error": f"员工 {employee} 没有待审批候选"}

    try:
        candidates = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"ok": False, "error": "候选列表解析失败"}

    # 找到目标候选
    target = None
    remaining = []
    for c in candidates:
        if c.get("id") == candidate_id:
            target = c
        else:
            remaining.append(c)

    if target is None:
        return {"ok": False, "error": f"候选 {candidate_id} 不存在"}

    action = target.get("action", "")
    content = target.get("content", "")

    # 2. 执行 soul 更新
    soul_updated = False
    slug = _employee_slug_from_name(employee)

    if slug:
        if action == "promote":
            soul_updated = _update_soul_promote(slug, content)
        elif action == "archive":
            soul_updated = _update_soul_archive(slug, content)
    else:
        logger.warning("找不到员工 %s 的 slug，跳过 soul 更新", employee)

    # 3. 更新 config_store：移除候选，存入 approved 列表
    put_config(
        "soul_evolution",
        f"{employee}_candidates",
        json.dumps(remaining, ensure_ascii=False),
    )

    # 存入 approved 列表
    approved_raw = get_config("soul_evolution", f"{employee}_approved")
    try:
        approved = json.loads(approved_raw) if approved_raw else []
    except (json.JSONDecodeError, TypeError):
        approved = []

    target["approved_at"] = datetime.now(timezone.utc).isoformat()
    target["soul_updated"] = soul_updated
    approved.append(target)
    put_config(
        "soul_evolution",
        f"{employee}_approved",
        json.dumps(approved, ensure_ascii=False),
    )

    # 4. 记 decision 类记忆
    if store is None:
        try:
            from ensoul.memory import get_memory_store

            store = get_memory_store()
        except Exception:
            logger.warning("无法创建 memory store，跳过 decision 记忆")
            store = None

    if store is not None:
        try:
            decision_content = (
                f"灵魂进化审批: {action} — {content}。"
                f"来源: {target.get('source_type', '')}，"
                f"source_ids: {target.get('source_ids', [])}"
            )
            from ensoul.memory_pipeline import process_memory

            process_memory(
                raw_text=decision_content,
                employee=employee,
                store=store,
                skip_reflect=True,
                category="decision",
                tags=["soul_evolution", action],
            )
        except Exception as exc:
            logger.warning("记录 decision 记忆失败: %s", exc)

    # 5. 标记 source corrections superseded_by（archive 场景）
    if action == "archive" and store is not None:
        for sid in target.get("source_ids", []):
            try:
                store.update(sid, superseded_by="archived")
            except Exception:
                pass

    return {
        "ok": True,
        "action": action,
        "content": content,
        "soul_updated": soul_updated,
    }


def reject_candidate(
    candidate_id: str,
    employee: str,
) -> dict[str, Any]:
    """拒绝一个进化候选.

    从 candidates 列表移除，存入 {employee}_rejected 列表。

    Args:
        candidate_id: 候选 ID
        employee: 员工名

    Returns:
        {"ok": True, "candidate_id": "..."}
    """
    raw = get_config("soul_evolution", f"{employee}_candidates")
    if not raw:
        return {"ok": False, "error": f"员工 {employee} 没有待审批候选"}

    try:
        candidates = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"ok": False, "error": "候选列表解析失败"}

    target = None
    remaining = []
    for c in candidates:
        if c.get("id") == candidate_id:
            target = c
        else:
            remaining.append(c)

    if target is None:
        return {"ok": False, "error": f"候选 {candidate_id} 不存在"}

    # 更新 candidates 列表
    put_config(
        "soul_evolution",
        f"{employee}_candidates",
        json.dumps(remaining, ensure_ascii=False),
    )

    # 存入 rejected 列表
    rejected_raw = get_config("soul_evolution", f"{employee}_rejected")
    try:
        rejected = json.loads(rejected_raw) if rejected_raw else []
    except (json.JSONDecodeError, TypeError):
        rejected = []

    target["rejected_at"] = datetime.now(timezone.utc).isoformat()
    rejected.append(target)
    put_config(
        "soul_evolution",
        f"{employee}_rejected",
        json.dumps(rejected, ensure_ascii=False),
    )

    return {"ok": True, "candidate_id": candidate_id}
