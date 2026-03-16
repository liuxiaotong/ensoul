"""任务复杂度分类 + 动态模型路由.

Phase 10 改动 33 — 纯规则引擎，零 LLM 调用，零延迟。

根据任务消息内容和工具配置判断复杂度（simple / medium / complex），
然后从 organization.yaml 的 model_defaults 中选择对应 tier 的模型。
"""

from __future__ import annotations

import enum
import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── 复杂度枚举 ──


class TaskComplexity(str, enum.Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


# ── 路由决策 ──


@dataclass(frozen=True)
class RoutingDecision:
    """模型路由决策记录."""

    original_model: str
    routed_model: str
    complexity: TaskComplexity | None
    reason: str


# ── 复杂度关键词 ──

_COMPLEX_KEYWORDS: re.Pattern[str] = re.compile(
    r"设计|架构|重构|分析|审查|review|方案|对比|规划|迁移|优化策略|技术选型"
)

_SIMPLE_PATTERNS: re.Pattern[str] = re.compile(
    r"^(你好|hi|hello|hey|状态|ping|帮我查|在吗|在不在|测试)[？?！!。.\s]*$",
    re.IGNORECASE,
)

# ── 复杂度 → tier 映射 ──

_COMPLEXITY_TIER_MAP: dict[TaskComplexity, str] = {
    TaskComplexity.SIMPLE: "fast",
    TaskComplexity.MEDIUM: "default",
    TaskComplexity.COMPLEX: "strong",
}


# ── 公开 API ──


def classify_task_complexity(
    message: str,
    *,
    has_tools: bool = False,
    tool_count: int = 0,
) -> TaskComplexity:
    """纯规则引擎：根据消息长度、关键词、工具数量判断复杂度.

    规则优先级（从高到低）：
    1. 消息匹配问候/状态查询模式 → simple
    2. 消息 > 500 字 或 工具数 > 15 或 包含复杂关键词 → complex
    3. 消息 ≤ 50 字 且（无工具 或 工具数 ≤ 2）→ simple
    4. 其余 → medium
    """
    text = message if isinstance(message, str) else str(message)
    text_stripped = text.strip()
    text_len = len(text_stripped)

    # 规则 1：问候/状态查询
    if _SIMPLE_PATTERNS.match(text_stripped):
        return TaskComplexity.SIMPLE

    # 规则 2：复杂度高指标
    if text_len > 500:
        return TaskComplexity.COMPLEX
    if tool_count > 15:
        return TaskComplexity.COMPLEX
    if _COMPLEX_KEYWORDS.search(text_stripped):
        return TaskComplexity.COMPLEX

    # 规则 3：短消息 + 少量/无工具
    if text_len <= 50 and (not has_tools or tool_count <= 2):
        return TaskComplexity.SIMPLE

    # 规则 4：默认
    return TaskComplexity.MEDIUM


def route_model(
    complexity: TaskComplexity,
    *,
    employee_model: str | None = None,
    employee_tier: str | None = None,
    org: object | None = None,
) -> str:
    """根据复杂度选择模型.

    路由规则：
    - 员工显式指定了 model → 不覆盖，直接返回
    - 员工指定了 model_tier → 按 tier 从 org.model_defaults 查找
    - 否则按复杂度映射 tier（simple→fast, medium→default, complex→strong）

    找不到对应 tier 时 fallback 到 "default" tier，再找不到返回系统默认模型。
    """
    from ensoul.organization import get_default_model as _gdm

    # 员工显式指定 model，不覆盖
    if employee_model:
        return employee_model

    # 确定目标 tier：员工指定 tier 优先，否则按复杂度映射
    if employee_tier:
        target_tier = employee_tier
    else:
        target_tier = _COMPLEXITY_TIER_MAP.get(complexity, "default")

    # 从 organization 获取 tier 对应模型
    model_defaults = getattr(org, "model_defaults", None) if org else None

    if model_defaults:
        tier_cfg = model_defaults.get(target_tier)
        if tier_cfg and getattr(tier_cfg, "model", None):
            return tier_cfg.model
        # fallback 到 default tier
        if target_tier != "default":
            tier_cfg = model_defaults.get("default")
            if tier_cfg and getattr(tier_cfg, "model", None):
                return tier_cfg.model

    # 最终 fallback：系统默认模型
    return _gdm()


def make_routing_decision(
    message: str,
    *,
    has_tools: bool = False,
    tool_count: int = 0,
    current_model: str,
    employee_model: str | None = None,
    employee_tier: str | None = None,
    org: object | None = None,
) -> RoutingDecision:
    """一步到位：分类 + 路由 + 返回决策记录."""
    complexity = classify_task_complexity(
        message,
        has_tools=has_tools,
        tool_count=tool_count,
    )
    routed = route_model(
        complexity,
        employee_model=employee_model,
        employee_tier=employee_tier,
        org=org,
    )
    reason = (
        f"complexity={complexity.value}, "
        f"msg_len={len(message)}, "
        f"tool_count={tool_count}, "
        f"employee_model={'set' if employee_model else 'none'}"
    )
    return RoutingDecision(
        original_model=current_model,
        routed_model=routed,
        complexity=complexity,
        reason=reason,
    )


def is_routing_enabled() -> bool:
    """检查模型路由是否启用（环境变量 CREW_MODEL_ROUTING）."""
    return os.environ.get("CREW_MODEL_ROUTING", "1") != "0"


# ── 自适应升级 ──


def is_adaptive_upgrade_enabled() -> bool:
    """检查自适应升级是否启用（环境变量 CREW_ADAPTIVE_UPGRADE）."""
    return os.environ.get("CREW_ADAPTIVE_UPGRADE", "1") != "0"


def should_upgrade_model(
    *,
    rounds_without_progress: int,
    current_model: str,
    upgrade_threshold: int = 3,
) -> bool:
    """连续 N 轮无进展时建议升级模型.

    条件（全部满足）：
    1. 自适应升级功能开启
    2. rounds_without_progress >= upgrade_threshold
    3. 当前模型不是最强模型（strong tier）
    """
    if not is_adaptive_upgrade_enabled():
        return False
    if rounds_without_progress < upgrade_threshold:
        return False
    # 如果已经是 strong tier 模型，不再升级
    # 通过 get_upgrade_model 判断：如果返回 None 说明已是最强
    if get_upgrade_model(current_model=current_model) is None:
        return False
    return True


def get_upgrade_model(*, current_model: str, org: object | None = None) -> str | None:
    """获取升级目标模型（strong tier）. 已经是 strong 则返回 None."""
    model_defaults = getattr(org, "model_defaults", None) if org else None

    if model_defaults:
        tier_cfg = model_defaults.get("strong")
        if tier_cfg and getattr(tier_cfg, "model", None):
            # 如果当前模型已经是 strong tier 模型，返回 None
            if tier_cfg.model == current_model:
                return None
            return tier_cfg.model

    # 环境变量兜底
    env_model = os.environ.get("CREW_UPGRADE_MODEL")
    if env_model and env_model != current_model:
        return env_model

    return None


# ── 路由事件记录 ──


def record_routing_event(
    *,
    decision: RoutingDecision,
    employee_name: str,
    trigger: str,  # "task_start" / "tool_downgrade" / "adaptive_upgrade"
) -> None:
    """记录路由事件到 EventCollector."""
    try:
        from ensoul.event_collector import get_event_collector

        ec = get_event_collector()
        ec.record(
            event_type="model_routing",
            event_name=trigger,
            metadata={
                "original_model": decision.original_model,
                "routed_model": decision.routed_model,
                "complexity": decision.complexity.value if decision.complexity else None,
                "reason": decision.reason,
                "employee": employee_name,
            },
        )
    except Exception:
        logger.debug("record_routing_event failed", exc_info=True)


# ── 工具轮降级 ──

_TOOL_ROUND_TEXT_THRESHOLD = 200  # 上一轮文本超过此长度视为"深度推理"


def is_tool_downgrade_enabled() -> bool:
    """检查工具轮降级是否启用（环境变量 CREW_TOOL_DOWNGRADE）."""
    return os.environ.get("CREW_TOOL_DOWNGRADE", "1") != "0"


def should_downgrade_for_tool_round(
    last_response_text: str,
    tool_calls: list[dict],
    *,
    consecutive_tool_rounds: int = 0,
) -> bool:
    """判断当前轮是否可以降级到便宜模型.

    条件（全部满足才降级）：
    1. 功能开关打开（CREW_TOOL_DOWNGRADE != 0）
    2. 上一轮 LLM 输出文本很短（< 200 字）— 说明不在深度推理
    3. 工具调用都是只读工具（在 READONLY_TOOLS 集合中）
    4. 连续工具轮次 >= 1（第一轮不降，让模型先理解任务）
    """
    if not is_tool_downgrade_enabled():
        return False

    if consecutive_tool_rounds < 1:
        return False

    # 上一轮文本太长 → 说明在推理/分析，不降级
    if len((last_response_text or "").strip()) >= _TOOL_ROUND_TEXT_THRESHOLD:
        return False

    # 没有工具调用 → 不算工具轮
    if not tool_calls:
        return False

    # 所有工具调用都必须是只读
    from ensoul.tool_schema import is_readonly_tool

    for tc in tool_calls:
        tool_name = tc.get("name") or tc.get("function", {}).get("name", "")
        if not is_readonly_tool(tool_name):
            return False

    return True


def get_downgrade_model(*, org: object | None = None) -> str | None:
    """获取降级模型（fast tier），没有配置返回 None（不降级）.

    优先从 org.model_defaults["fast"] 获取，否则返回 None。
    """
    model_defaults = getattr(org, "model_defaults", None) if org else None

    if model_defaults:
        tier_cfg = model_defaults.get("fast")
        if tier_cfg and getattr(tier_cfg, "model", None):
            return tier_cfg.model

    # 环境变量兜底
    env_model = os.environ.get("CREW_DOWNGRADE_MODEL")
    if env_model:
        return env_model

    return None
