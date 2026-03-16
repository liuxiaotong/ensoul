"""工具选择后校验 -- 基于渠道和经验的规则引擎.

在工具选好后、执行前进行轻量校验。初期只 log 不阻断。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 已知的工具名集合（延迟加载以避免循环导入）
_KNOWN_TOOLS: set[str] | None = None


def _get_known_tools() -> set[str]:
    """延迟加载已知工具名集合."""
    global _KNOWN_TOOLS
    if _KNOWN_TOOLS is None:
        from ensoul.tool_schema import AGENT_TOOLS

        _KNOWN_TOOLS = AGENT_TOOLS.copy()
    return _KNOWN_TOOLS


@dataclass
class ValidationResult:
    """工具选择校验结果.

    Attributes:
        valid: 是否通过校验（False 表示有问题，但初期不阻断）
        warning: 警告信息
        suggested_alternative: 建议替换的工具名
    """

    valid: bool
    warning: str = ""
    suggested_alternative: str = ""


def validate_tool_selection(
    tool_name: str,
    channel: str = "",
    plan_hint: str = "",
    employee_corrections: list[str] | None = None,
) -> ValidationResult:
    """校验工具选择是否合理.

    基于渠道-工具匹配、Plan hint 偏差和经验教训进行校验。
    初期只 log WARNING，不阻断执行。

    Args:
        tool_name: 选择的工具名
        channel: 请求渠道
        plan_hint: Plan 当前步骤建议的工具名
        employee_corrections: 员工 correction 记忆摘要列表

    Returns:
        ValidationResult 包含校验结果和可选建议
    """
    known_tools = _get_known_tools()

    # 规则 1：渠道-工具不匹配 — 蚁聚渠道用同步 delegate
    if tool_name == "delegate" and channel.startswith("antgather"):
        return ValidationResult(
            valid=False,
            warning="在蚁聚渠道使用同步 delegate 可能导致用户干等",
            suggested_alternative="delegate_async",
        )

    # 规则 2：Plan hint 偏差 — 实际选择与 Plan 建议不一致
    if plan_hint and tool_name != plan_hint and plan_hint in known_tools:
        return ValidationResult(
            valid=True,  # 不阻断，只提醒
            warning=f"Plan 建议用 {plan_hint}，实际选了 {tool_name}",
        )

    # 规则 3：correction 中的反面教训
    if employee_corrections:
        for corr_text in employee_corrections:
            # 检查是否有"不要用/避免用 tool_name"的模式
            avoid_patterns = [
                rf"不要.*\b{re.escape(tool_name)}\b",
                rf"避免.*\b{re.escape(tool_name)}\b",
                rf"\b{re.escape(tool_name)}\b.*导致.*问题",
                rf"\b{re.escape(tool_name)}\b.*失败",
            ]
            for pattern in avoid_patterns:
                if re.search(pattern, corr_text):
                    return ValidationResult(
                        valid=True,  # 不阻断，记录
                        warning=f"经验教训提示 {tool_name} 可能有问题: {corr_text[:100]}",
                    )

    return ValidationResult(valid=True)
