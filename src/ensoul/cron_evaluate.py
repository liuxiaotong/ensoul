"""过期决策扫描 — 自动回收评估结果."""

import logging
from datetime import datetime
from pathlib import Path

from ensoul.evaluation import EvaluationEngine
from ensoul.memory import get_memory_store
from ensoul.paths import resolve_project_dir
from ensoul.task_registry import TaskRecord, TaskRegistry

logger = logging.getLogger(__name__)


def _memory_exists_by_source(store, employee: str, source_session: str) -> bool:
    """检查指定 source_session 的记忆是否已存在（去重用）.

    NOTE: 此函数与后续 store.add 之间存在 TOCTOU 竞争窗口，
    但当前 cron 按员工串行调度，同一 session 不会被并行处理，风险极低。
    若需更严格保证，可在 MemoryStore.add 中做 upsert 或加 unique constraint。
    """
    try:
        existing = store.query(employee=employee, limit=200)
        for entry in existing:
            if getattr(entry, "source_session", "") == source_session:
                return True
    except Exception:
        pass
    return False


async def scan_overdue_decisions(
    project_dir: Path | None = None,
    tenant_id: str | None = None,
) -> dict:
    """扫描过期未评估的决策。

    逻辑：
    1. 调用 EvaluationEngine.list_overdue() 获取过期决策
    2. 有 task_id 的 -> 查 tasks.jsonl 看任务状态
       - 任务已完成 -> 自动调用 evaluate()，actual_outcome 用任务结果
       - 任务未完成 -> 记录为超期
    3. 没有 task_id 的 -> 生成提醒消息（返回给调用方，由 cron 投递）
    4. 超过 deadline 7 天仍未评估的 -> 自动写一条 finding 到员工记忆：
       "决策 [内容摘要] 超期 X 天未评估"

    Returns:
        dict with:
          - auto_evaluated: list[dict]  自动评估的决策（model_dump）
          - reminders: list[dict]  需要提醒的（employee, decision_id, content 摘要）
          - expired: list[dict]  超期 7 天自动关闭的决策（model_dump）
    """
    resolved_dir = resolve_project_dir(project_dir)
    engine = EvaluationEngine(project_dir=project_dir)
    today = datetime.now().strftime("%Y-%m-%d")
    overdue = engine.list_overdue(as_of=today)

    if not overdue:
        return {"auto_evaluated": [], "reminders": [], "expired": []}

    # 加载 TaskRegistry（只读查询）
    tasks_path = resolved_dir / ".ensoul" / "tasks.jsonl"
    registry: TaskRegistry | None = None
    if tasks_path.exists():
        registry = TaskRegistry(persist_path=tasks_path)

    auto_evaluated: list[dict] = []
    reminders: list[dict] = []
    expired: list[dict] = []

    for decision in overdue:
        days_overdue = _days_between(decision.deadline, today)

        # 有 task_id -> 查任务状态
        if decision.task_id and registry:
            task = registry.get(decision.task_id)
            if task and task.status == "completed":
                # 任务已完成 -> 自动 evaluate
                outcome = _task_result_summary(task)
                result = engine.evaluate(
                    decision_id=decision.id,
                    actual_outcome=outcome,
                    evaluation=f"[系统自动评估] 关联任务 {decision.task_id} 已完成。{outcome}",
                )
                if result:
                    auto_evaluated.append(result.model_dump())
                    logger.info(
                        "自动评估决策 %s（关联任务 %s 已完成）",
                        decision.id,
                        decision.task_id,
                    )
                continue

        # 超过 7 天 -> 自动写 finding 到员工记忆并关闭
        if days_overdue >= 7:
            content_summary = decision.content[:60]
            finding_text = (
                f"决策 [{content_summary}] 超期 {days_overdue} 天未评估（ID: {decision.id}）"
            )
            try:
                source_session_tag = f"cron:overdue:{decision.id}"
                store = get_memory_store(project_dir=project_dir, tenant_id=tenant_id)
                if not _memory_exists_by_source(store, decision.employee, source_session_tag):
                    from ensoul.memory_pipeline import process_memory

                    process_memory(
                        raw_text=finding_text,
                        employee=decision.employee,
                        store=store,
                        skip_reflect=True,
                        source_session=source_session_tag,
                        category="finding",
                    )
                logger.info(
                    "决策 %s 超期 %d 天，已写入 finding 记忆",
                    decision.id,
                    days_overdue,
                )
            except Exception:
                logger.warning("写入 finding 记忆失败: %s", decision.id, exc_info=True)

            # 自动关闭（evaluate 为超期）
            result = engine.evaluate(
                decision_id=decision.id,
                actual_outcome=f"超期 {days_overdue} 天未评估，系统自动关闭",
                evaluation=f"[系统自动关闭] 超过 deadline({decision.deadline}) {days_overdue} 天",
            )
            if result:
                expired.append(result.model_dump())
            continue

        # 其他情况 -> 生成提醒
        content_summary = decision.content[:60]
        reminders.append(
            {
                "employee": decision.employee,
                "decision_id": decision.id,
                "content": content_summary,
                "deadline": decision.deadline,
                "days_overdue": days_overdue,
            }
        )

    return {
        "auto_evaluated": auto_evaluated,
        "reminders": reminders,
        "expired": expired,
    }


def _days_between(date_str_a: str, date_str_b: str) -> int:
    """计算两个 ISO 日期字符串之间的天数差（b - a）."""
    try:
        a = datetime.strptime(date_str_a[:10], "%Y-%m-%d")
        b = datetime.strptime(date_str_b[:10], "%Y-%m-%d")
        return (b - a).days
    except (ValueError, TypeError):
        return 0


def _task_result_summary(task: TaskRecord) -> str:
    """从 TaskRecord 提取结果摘要."""
    if task.result:
        # 取 result 中的关键信息
        text = task.result.get("summary", "") or task.result.get("output", "")
        if text:
            return str(text)[:200]
        return f"任务结果: {str(task.result)[:200]}"
    return "任务已完成（无详细结果）"


def format_scan_report(results: dict) -> str | None:
    """将扫描结果格式化为可读报告。无内容返回 None。"""
    auto_evaluated = results.get("auto_evaluated", [])
    reminders = results.get("reminders", [])
    expired = results.get("expired", [])

    if not auto_evaluated and not reminders and not expired:
        return None

    parts = ["决策评估日报"]

    if auto_evaluated:
        parts.append(f"\n自动评估 ({len(auto_evaluated)} 条):")
        for d in auto_evaluated:
            parts.append(f"  - {d.get('employee', '?')}: {d.get('content', '')[:50]}")

    if reminders:
        parts.append(f"\n待回复 ({len(reminders)} 条):")
        for r in reminders:
            parts.append(f"  - {r['employee']}: {r['content']} (超期 {r['days_overdue']} 天)")

    if expired:
        parts.append(f"\n超期关闭 ({len(expired)} 条):")
        for d in expired:
            parts.append(f"  - {d.get('employee', '?')}: {d.get('content', '')[:50]}")

    return "\n".join(parts)
