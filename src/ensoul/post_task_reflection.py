"""任务后反思 — agent 完成/失败后自动回顾执行过程，提取可复用教训写入记忆."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── 环境变量开关 ──
CREW_POST_TASK_REFLECTION = os.environ.get("CREW_POST_TASK_REFLECTION", "1") != "0"

# ── 频率控制 ──
_MIN_ROUNDS_FOR_REFLECTION = 3  # 简单任务（≤3 轮）跳过（除非失败）
_MAX_REFLECTIONS_PER_HOUR = 5  # 每个员工每小时最多 5 次反思


@dataclass
class ReflectionItem:
    """单条反思记忆."""

    content: str
    category: str  # pattern / correction / finding
    importance: int  # 1-5
    keywords: list[str] = field(default_factory=list)
    trigger_condition: str = ""
    applicability: list[str] = field(default_factory=list)


@dataclass
class TaskExecutionContext:
    """任务执行上下文，用于反思分析."""

    employee_name: str
    task_description: str
    tool_rounds: int
    result_summary: str
    elapsed_seconds: float
    success: bool
    tools_used: list[str] = field(default_factory=list)


# ── 频率控制器 ──


class _ReflectionRateLimiter:
    """内存中的简单计数器，控制每个员工每小时反思次数."""

    def __init__(self) -> None:
        # employee_name -> list of timestamps
        self._timestamps: dict[str, list[float]] = defaultdict(list)

    def allow(self, employee_name: str) -> bool:
        """检查是否允许反思."""
        now = time.monotonic()
        cutoff = now - 3600  # 1 小时窗口
        # 清理过期记录
        self._timestamps[employee_name] = [t for t in self._timestamps[employee_name] if t > cutoff]
        return len(self._timestamps[employee_name]) < _MAX_REFLECTIONS_PER_HOUR

    def record(self, employee_name: str) -> None:
        """记录一次反思."""
        self._timestamps[employee_name].append(time.monotonic())

    def reset(self) -> None:
        """清空所有计数（测试用）."""
        self._timestamps.clear()


_rate_limiter = _ReflectionRateLimiter()


def get_rate_limiter() -> _ReflectionRateLimiter:
    """返回全局频率控制器."""
    return _rate_limiter


# ── 反思 Prompt ──

_REFLECTION_PROMPT = """\
你是一个任务回顾分析器。分析以下 agent 任务的执行过程，提取可复用的教训和模式。

规则：
- 只提取可复用的教训，不记流水账
- 每条教训必须是独立的、可操作的
- 如果没有值得记录的教训，返回空数组 []
- 最多返回 3 条

category 分类：
- pattern: 发现的高效工作模式、最佳实践
- correction: 之前错了现在纠正的认知、踩坑记录
- finding: 发现的事实、规律、技术细节

importance 评分（1-5）：
  1 = 琐碎（不太可能复用）
  2 = 普通发现
  3 = 有价值的教训（可在类似场景复用）
  4 = 重要教训（影响后续工作方式）
  5 = 关键教训（避免重大问题）

员工：{employee_name}

任务描述：{task_description}

执行情况：
- 结果：{status}
- 执行轮次：{tool_rounds}
- 耗时：{elapsed_seconds:.1f} 秒
- 使用工具：{tools_used}

执行摘要：
{result_summary}

请返回 JSON 数组（不要加 markdown 标记），每个元素包含：
{{
  "content": "提炼后的教训（简洁但完整）",
  "category": "pattern/correction/finding",
  "importance": 1-5,
  "keywords": ["关键词1", "关键词2"],
  "trigger_condition": "什么场景下该回忆这条",
  "applicability": ["适用的角色或领域"]
}}

如果没有值得记录的教训，直接返回 []
"""


def _should_reflect(ctx: TaskExecutionContext) -> bool:
    """判断是否需要反思."""
    if not CREW_POST_TASK_REFLECTION:
        return False

    # 失败任务强制反思
    if not ctx.success:
        return True

    # 简单任务（≤3 轮）跳过
    if ctx.tool_rounds <= _MIN_ROUNDS_FOR_REFLECTION:
        return False

    return True


def _build_prompt(ctx: TaskExecutionContext) -> str:
    """构造反思 prompt."""
    status = "成功" if ctx.success else "失败"
    tools_str = ", ".join(ctx.tools_used) if ctx.tools_used else "无"
    # 截断摘要防止 prompt 过长
    summary = ctx.result_summary[:2000] if len(ctx.result_summary) > 2000 else ctx.result_summary

    return _REFLECTION_PROMPT.format(
        employee_name=ctx.employee_name,
        task_description=ctx.task_description[:500],
        status=status,
        tool_rounds=ctx.tool_rounds,
        elapsed_seconds=ctx.elapsed_seconds,
        tools_used=tools_str,
        result_summary=summary,
    )


def _call_llm_for_reflection(prompt: str) -> str:
    """调用 LLM 进行反思（封装方便测试 mock）."""
    from ensoul.memory_pipeline import _call_llm

    return _call_llm(prompt, timeout=30.0, max_tokens=1024)


def _parse_reflection_response(raw: str) -> list[ReflectionItem]:
    """解析 LLM 返回的 JSON 数组."""
    # 尝试去除可能的 markdown 包裹
    text = raw.strip()
    if text.startswith("```"):
        # 去除 ```json 和 ```
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("反思结果 JSON 解析失败: %s", text[:200])
        return []

    if not isinstance(data, list):
        logger.warning("反思结果非数组: %s", type(data))
        return []

    items: list[ReflectionItem] = []
    valid_categories = {"pattern", "correction", "finding"}
    for entry in data[:3]:  # 最多 3 条
        if not isinstance(entry, dict):
            continue
        content = entry.get("content", "").strip()
        category = entry.get("category", "finding")
        if not content:
            continue
        if category not in valid_categories:
            category = "finding"
        importance = max(1, min(5, int(entry.get("importance", 3))))
        items.append(
            ReflectionItem(
                content=content,
                category=category,
                importance=importance,
                keywords=entry.get("keywords", []),
                trigger_condition=entry.get("trigger_condition", ""),
                applicability=entry.get("applicability", []),
            )
        )
    return items


async def run_post_task_reflection(
    ctx: TaskExecutionContext,
    *,
    project_dir: str | None = None,
    tenant_id: str | None = None,
    source_session: str = "",
) -> list[ReflectionItem]:
    """执行任务后反思，提取教训并写入记忆.

    异步执行，不阻塞主流程。

    Args:
        ctx: 任务执行上下文
        project_dir: 项目目录
        tenant_id: 租户 ID
        source_session: 来源会话 ID

    Returns:
        提取到的反思记忆列表（可能为空）
    """
    if not _should_reflect(ctx):
        logger.debug(
            "跳过反思: employee=%s rounds=%d success=%s",
            ctx.employee_name,
            ctx.tool_rounds,
            ctx.success,
        )
        return []

    # 频率控制
    limiter = get_rate_limiter()
    if not limiter.allow(ctx.employee_name):
        logger.info("反思频率限制: employee=%s", ctx.employee_name)
        return []

    prompt = _build_prompt(ctx)

    # 调用 LLM（在线程池中执行同步调用，避免阻塞事件循环）
    import asyncio

    loop = asyncio.get_running_loop()
    try:
        raw_response = await loop.run_in_executor(
            None,
            lambda: _call_llm_for_reflection(prompt),
        )
    except Exception:
        logger.warning("反思 LLM 调用失败: employee=%s", ctx.employee_name, exc_info=True)
        return []

    items = _parse_reflection_response(raw_response)
    if not items:
        logger.debug("反思无可记录教训: employee=%s", ctx.employee_name)
        return []

    # 写入记忆
    try:
        from ensoul.memory import get_memory_store
        from ensoul.memory_pipeline import process_memory

        mem_store = get_memory_store(project_dir=project_dir, tenant_id=tenant_id)

        for item in items:
            process_memory(
                raw_text=f"[反思] {item.content}",
                employee=ctx.employee_name,
                store=mem_store,
                skip_reflect=True,
                source_session=source_session,
                category=item.category,
                importance=item.importance,
                confidence=0.7,
                keywords=item.keywords,
                tags=["origin:reflection"],
                trigger_condition=item.trigger_condition,
                applicability=item.applicability,
                origin_employee=ctx.employee_name,
                ttl_days=90,
            )
    except Exception:
        logger.warning("反思记忆写入失败: employee=%s", ctx.employee_name, exc_info=True)
        return []

    # 记录频率
    limiter.record(ctx.employee_name)
    logger.info(
        "任务后反思完成: employee=%s items=%d success=%s rounds=%d",
        ctx.employee_name,
        len(items),
        ctx.success,
        ctx.tool_rounds,
    )
    return items
