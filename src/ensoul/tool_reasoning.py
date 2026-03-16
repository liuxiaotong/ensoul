"""工具推理上下文 -- 基于灵魂、记忆、渠道生成工具选择指南.

在 system prompt 中注入一段上下文感知的工具选择指导，
让 LLM 在选工具时能"看到"渠道约束、Plan 建议和经验教训。
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── 渠道 → 指南映射 ──

_CHANNEL_GUIDANCE: dict[str, str] = {
    "antgather": (
        "你在蚁聚社区中与用户对话，长耗时操作（委派、搜索）应使用异步工具"
        "（delegate_async），避免用户干等。"
    ),
    "feishu": ("你在飞书群/私聊中，可以用同步操作，但复杂任务仍建议异步委派。"),
    "mcp": ("你在 MCP 内部调用中，优先用同步工具以保持调用链一致性。"),
}

# ── 易混淆工具对 ──

_CONFUSED_PAIRS: list[str] = [
    "delegate（同步，阻塞等待子任务完成）vs delegate_async（异步，立即返回 task_id）"
    "—— 蚁聚/飞书渠道优先用 delegate_async，MCP 渠道可用 delegate。",
]


def _channel_to_guidance(channel: str) -> str:
    """将渠道标识转换为工具选择指南文本.

    Args:
        channel: 渠道标识（如 antgather_dm, feishu_group, mcp 等）

    Returns:
        对应的指南文本，无匹配时返回空字符串
    """
    if not channel:
        return ""
    for prefix, guidance in _CHANNEL_GUIDANCE.items():
        if channel.startswith(prefix):
            return guidance
    return ""


def _extract_tool_lessons(corrections: list[str]) -> list[str]:
    """从 correction 文本中提取包含工具名的教训.

    Args:
        corrections: correction 类记忆的文本列表

    Returns:
        包含工具名引用的教训列表
    """
    from ensoul.tool_schema import AGENT_TOOLS

    lessons: list[str] = []
    for text in corrections:
        # 检查是否提到了任何已知工具名
        mentioned = [
            tool for tool in AGENT_TOOLS if re.search(r"\b" + re.escape(tool) + r"\b", text)
        ]
        if mentioned:
            # 截断到 200 字符
            truncated = text[:200] + ("..." if len(text) > 200 else "")
            lessons.append(truncated)
    return lessons


def _get_confused_pairs(channel: str) -> list[str]:
    """返回渠道相关的易混淆工具对.

    Args:
        channel: 渠道标识

    Returns:
        易混淆工具对的描述列表
    """
    # 目前所有渠道共享同一组易混淆工具对
    # 未来可按渠道过滤
    return _CONFUSED_PAIRS


def build_tool_reasoning_context(
    channel: str = "",
    plan_step_hint: str = "",
    employee_corrections: list[str] | None = None,
    employee_name: str = "",
) -> str:
    """构建工具推理提示，注入到 system prompt.

    基于渠道约束、Plan 步骤建议、员工踩坑经验和易混淆工具对
    生成一段指导 LLM 工具选择的上下文。

    Args:
        channel: 请求渠道标识
        plan_step_hint: 当前 Plan 步骤建议的工具
        employee_corrections: 员工 correction 类记忆文本列表
        employee_name: 员工名称（用于日志）

    Returns:
        格式化的工具推理上下文文本，无内容时返回空字符串
    """
    sections: list[str] = []

    # 1. 渠道约束
    if channel:
        channel_guidance = _channel_to_guidance(channel)
        if channel_guidance:
            sections.append(f"## 当前工具选择约束\n\n{channel_guidance}")

    # 2. Plan 步骤提示
    if plan_step_hint:
        sections.append(f"当前步骤建议工具: {plan_step_hint}")

    # 3. 经验教训（从 correction 记忆提取工具相关的）
    if employee_corrections:
        tool_lessons = _extract_tool_lessons(employee_corrections)
        if tool_lessons:
            sections.append(
                "## 工具使用教训（来自你的经验）\n\n"
                + "\n".join(f"- {lesson}" for lesson in tool_lessons)
            )

    # 4. 易混淆工具对警告（仅在有其他内容时才附加）
    if sections:
        confused_pairs = _get_confused_pairs(channel)
        if confused_pairs:
            sections.append("## 易混淆工具\n\n" + "\n".join(f"- {pair}" for pair in confused_pairs))

    result = "\n\n".join(sections) if sections else ""
    if result:
        logger.info(
            "工具推理上下文生成: employee=%s, channel=%s, sections=%d",
            employee_name,
            channel,
            len(sections),
        )
    return result
