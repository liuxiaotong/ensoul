"""Observation Compressor — 用 LLM 将旧工具输出压缩为一句话摘要.

替代原有的硬删除策略，压缩可逆、保留关键引用（路径/URL/数字）。
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── 引用提取 ──────────────────────────────────────────────────────────

_RE_FILE_PATH = re.compile(r"(?:/[\w.\-]+){2,}")
_RE_URL = re.compile(r"https?://[^\s\]\"'<>,]+")
_RE_NUMERIC_ID = re.compile(r"\b(?:line|行|id|ID|#)\s*[:：]?\s*(\d+)\b")

_FALLBACK_PLACEHOLDER = "[工具输出已归档，如需回顾请重新调用]"

_COMPRESS_SYSTEM_PROMPT_TPL = (
    "你是一个工具输出压缩助手。将以下工具输出压缩为一句话摘要（不超过 {max_chars} 字）。\n"
    "要求：\n"
    "1. 一句话说清工具执行的结果\n"
    "2. 必须保留所有文件路径（如 /path/to/file）\n"
    "3. 必须保留所有 URL（http/https 链接）\n"
    "4. 必须保留所有关键数字（行号、计数、ID）\n"
    "5. 只输出摘要本身，不要加任何前缀或解释"
)

# 向后兼容：默认 80 字
_COMPRESS_SYSTEM_PROMPT = _COMPRESS_SYSTEM_PROMPT_TPL.format(max_chars=80)


def extract_references(text: str) -> list[str]:
    """从文本中提取文件路径、URL、数字 ID 等可逆引用."""
    refs: list[str] = []
    for m in _RE_URL.finditer(text):
        refs.append(m.group(0))
    for m in _RE_FILE_PATH.finditer(text):
        # 排除已被 URL 覆盖的路径
        path = m.group(0)
        if not any(path in u for u in refs):
            refs.append(path)
    for m in _RE_NUMERIC_ID.finditer(text):
        refs.append(m.group(0))
    return refs


async def compress_observation(
    content: str,
    tool_name: str = "",
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_chars: int = 80,
) -> str:
    """用 LLM 将单条工具输出压缩为一句话摘要.

    Parameters
    ----------
    max_chars : int
        压缩目标字数上限，默认 80。

    Returns
    -------
    str
        格式: ``[工具摘要] {摘要} (原始输出 {N} 字)``
        LLM 失败时降级为硬替换占位符。
    """
    # Guard: 已压缩的内容不再重复压缩
    if content.startswith("[工具摘要]") or content.startswith("[工具输出已归档"):
        return content

    from ensoul.executor import aexecute_prompt

    original_len = len(content)
    tool_hint = f"（工具: {tool_name}）" if tool_name else ""

    user_message = f"请压缩以下工具输出{tool_hint}：\n\n{content}"

    _system_prompt = _COMPRESS_SYSTEM_PROMPT_TPL.format(max_chars=max_chars)
    exec_kwargs: dict[str, Any] = {
        "system_prompt": _system_prompt,
        "user_message": user_message,
        "stream": False,
        "max_tokens": 256,
    }
    if model:
        exec_kwargs["model"] = model
    if api_key:
        exec_kwargs["api_key"] = api_key
    if base_url:
        exec_kwargs["base_url"] = base_url

    try:
        result = await aexecute_prompt(**exec_kwargs)
        summary = result.content if hasattr(result, "content") else str(result)
        summary = summary.strip()
        if not summary:
            return _FALLBACK_PLACEHOLDER

        # W-1: 检查摘要是否保留了原文中的关键引用，遗漏的追加到末尾
        original_refs = extract_references(content)
        if original_refs:
            missing = [r for r in original_refs if r not in summary]
            if missing:
                summary += " | 引用: " + ", ".join(missing)

        return f"[工具摘要] {summary} (原始输出 {original_len} 字)"
    except Exception:
        logger.warning("Observation 压缩失败，降级为硬替换", exc_info=True)
        return _FALLBACK_PLACEHOLDER
