"""LLM 输出清洗 — 移除内部推理标签和工具调用 XML，只保留面向用户的内容."""

from __future__ import annotations

import re

# ── 内部标签模式 ──
# 匹配 <thinking>...</thinking> 及其内容（支持嵌套、跨行）
_THINKING_RE = re.compile(
    r"<thinking>.*?</thinking>",
    re.DOTALL,
)

# 匹配 <reflection>...</reflection> 及其内容
_REFLECTION_RE = re.compile(
    r"<reflection>.*?</reflection>",
    re.DOTALL,
)

# 匹配 <inner_monologue>...</inner_monologue> 及其内容
_INNER_MONOLOGUE_RE = re.compile(
    r"<inner_monologue>.*?</inner_monologue>",
    re.DOTALL,
)

# ── 工具调用 XML 模式 ──
# 匹配常见的 XML 工具调用块，如 <read_diary>...</read_diary>、<tool_use>...</tool_use> 等
# 策略：匹配已知的工具调用标签名 + 通用的蛇形命名标签（tool_xxx, read_xxx, write_xxx 等）
_TOOL_CALL_XML_RE = re.compile(
    r"<(?:tool_use|tool_call|function_call|"  # 通用工具调用标签
    r"read_\w+|write_\w+|create_\w+|delete_\w+|update_\w+|list_\w+|"  # CRUD 操作
    r"search_\w+|query_\w+|lookup_\w+|send_\w+|"  # 查询/发送操作
    r"add_\w+|get_\w+|check_\w+|schedule_\w+|"  # 其他常见操作
    r"delegate\w*|organize_\w+|find_\w+|cancel_\w+|"  # 编排操作
    r"web_search|run_python|file_read|file_write|"  # 具体工具名
    r"agent_file_\w+|feishu_\w+|read_feishu_\w+)"  # agent/飞书工具
    r">[^<]*(?:<[^/].*?)?</\w+>",
    re.DOTALL,
)

# ── 独立的 XML 工具调用块（更宽松：匹配任何以参数子标签开头的 XML 块）──
# 例如：<read_diary>\n  <user>Kai</user>\n  <count>7</count>\n</read_diary>
_XML_BLOCK_RE = re.compile(
    r"<([a-z_]+)>\s*\n(?:\s*<[a-z_]+>.*?</[a-z_]+>\s*\n?)+\s*</\1>",
    re.DOTALL,
)


def strip_internal_tags(text: str) -> str:
    """移除 LLM 输出中的内部标签和工具调用 XML.

    清除的内容：
    - <thinking>...</thinking> 推理过程
    - <reflection>...</reflection> 反思过程
    - <inner_monologue>...</inner_monologue> 内心独白
    - XML 格式的工具调用块（如 <read_diary>...</read_diary>）

    保留用户应该看到的自然语言内容。

    Args:
        text: LLM 原始输出文本

    Returns:
        清洗后的文本
    """
    if not text:
        return text

    result = text

    # 1. 移除 thinking/reflection/inner_monologue 块
    result = _THINKING_RE.sub("", result)
    result = _REFLECTION_RE.sub("", result)
    result = _INNER_MONOLOGUE_RE.sub("", result)

    # 2. 移除 XML 工具调用块（带参数子标签的结构化调用）
    result = _XML_BLOCK_RE.sub("", result)

    # 3. 移除已知名称的工具调用标签
    result = _TOOL_CALL_XML_RE.sub("", result)

    # 4. 移除方括号名字前缀（如【Employee】）
    result = re.sub(r"^【[^】]+】\s*", "", result)

    # 5. 清理残留空行（多个连续空行压缩为最多两个）
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()
