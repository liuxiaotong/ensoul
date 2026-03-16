"""自动上下文压缩 — 当 input tokens 超过阈值时，将旧对话压缩为摘要."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _find_keep_start(messages: list[dict[str, Any]], keep_recent: int) -> int:
    """从消息列表尾部向前扫描，找到 keep_recent 个完整轮次的起点索引。

    一个完整轮次由以下消息组成（按顺序）：
    1. 一个 role=assistant 的消息（可能含 tool_calls）
    2. 紧随其后的所有 role=tool 响应消息（OAI 格式，0 到 N 条）
    3. 可选的后续 role=user 消息

    扫描从尾部向前进行，每遇到一个 assistant 消息就计为一个轮次的起点。
    确保不会将 assistant 消息压掉而保留其 tool 响应。
    """
    if not messages or keep_recent <= 0:
        return len(messages)

    rounds_found = 0
    i = len(messages) - 1
    candidate_start = len(messages)

    while i >= 0 and rounds_found < keep_recent:
        msg = messages[i]
        role = msg.get("role", "")

        if role == "assistant":
            # 找到一个轮次起点
            candidate_start = i
            rounds_found += 1
        elif role in ("user", "tool"):
            # user/tool 消息属于当前轮次（或前一个轮次的尾部）
            # 如果还没找到任何 assistant，这些消息也需要保留
            if rounds_found == 0:
                candidate_start = i
        i -= 1

    # 如果没找到足够轮次，保留所有消息（从头开始）
    if rounds_found < keep_recent:
        return 0

    return candidate_start


async def compact_context(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    keep_recent: int = 3,
    plan_state: str | None = None,
    todo_state: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """压缩旧对话，返回 (压缩后的 messages, 元数据).

    Parameters
    ----------
    messages:
        完整的 user/assistant 消息列表（不含 system prompt）。
    model:
        用于压缩的模型（建议用 fallback/fast model）。
    api_key:
        API key（可选，传给 aexecute_prompt）。
    base_url:
        API base URL（可选）。
    keep_recent:
        保留最近 N 轮原始消息（每轮 = 1 assistant + 1 tool_result，共 2 条）。
    plan_state:
        Plan-and-Execute 的计划状态文本（会包含在摘要中，不被压缩丢弃）。
    todo_state:
        动态 Todo 状态文本（会包含在摘要中，不被压缩丢弃）。

    Returns
    -------
    tuple[list[dict], dict]:
        (压缩后的消息列表, {original_count, compressed_count, summary_tokens})
    """
    from ensoul.executor import aexecute_prompt

    original_count = len(messages)

    # 从消息列表尾部向前扫描，找到 keep_recent 个完整轮次的起点。
    # 一个完整轮次 = 1 个 assistant（可能含 tool_calls）+ 其所有 tool 响应
    #               + 可选的后续 user 消息。
    # 这样避免 OAI 格式下 assistant + N tool 响应被拆断。
    keep_start = _find_keep_start(messages, keep_recent)
    keep_count = original_count - keep_start

    # 消息太少，不需要压缩
    if original_count <= keep_count + 2:
        return messages, {
            "original_count": original_count,
            "compressed_count": original_count,
            "summary_tokens": 0,
            "skipped": True,
        }

    # 分割：旧消息 vs 保留消息
    old_messages = messages[:keep_start]
    recent_messages = messages[keep_start:]

    # 将旧消息序列化为文本供 LLM 压缩
    old_text = _serialize_messages_for_summary(old_messages)

    # 构造压缩 prompt
    plan_section = ""
    if plan_state:
        plan_section = f"\n\n当前任务计划状态（必须保留）：\n{plan_state}"
    if todo_state:
        plan_section += f"\n\n当前 TODO 状态（必须保留）：\n{todo_state}"

    compress_prompt = (
        "你是一个对话压缩助手。请将以下对话历史压缩为简洁摘要，保留：\n"
        "- 任务目标和当前进度\n"
        "- 关键发现和中间结果\n"
        "- 已执行的工具调用及结果要点\n"
        "- 尚未完成的步骤\n"
        "- 出现的错误和已解决/未解决的问题\n"
        "格式：用 Markdown 列表，每点不超过 2 句话。只输出摘要，不要输出其他内容。"
        f"{plan_section}"
    )

    user_message = f"以下是需要压缩的对话历史：\n\n{old_text}"

    # 用 fallback model 执行压缩
    exec_kwargs: dict[str, Any] = {
        "system_prompt": compress_prompt,
        "user_message": user_message,
        "stream": False,
        "max_tokens": 2048,
    }
    if model:
        exec_kwargs["model"] = model
    if api_key:
        exec_kwargs["api_key"] = api_key
    if base_url:
        exec_kwargs["base_url"] = base_url

    result = await aexecute_prompt(**exec_kwargs)
    summary_text = result.content if hasattr(result, "content") else str(result)
    summary_tokens = result.output_tokens if hasattr(result, "output_tokens") else 0

    # 构造压缩摘要消息
    summary_message: dict[str, Any] = {
        "role": "user",
        "content": f"[上下文压缩摘要]\n{summary_text}",
    }

    # C1: 如果 recent_messages 第一条也是 user，插入 assistant 垫片消息
    # 避免连续 user 消息违反 Anthropic 交替规则
    if recent_messages and recent_messages[0].get("role") == "user":
        shim_message: dict[str, Any] = {
            "role": "assistant",
            "content": "好的，我已了解之前的对话上下文，继续执行任务。",
        }
        compressed_messages = [summary_message, shim_message] + recent_messages
    else:
        compressed_messages = [summary_message] + recent_messages

    meta = {
        "original_count": original_count,
        "compressed_count": len(compressed_messages),
        "summary_tokens": summary_tokens,
        "skipped": False,
    }

    return compressed_messages, meta


def _serialize_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    """将消息列表序列化为可读文本，供 LLM 压缩."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, list):
            # Anthropic 格式：content blocks
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        name = block.get("name", "?")
                        inp = block.get("input", {})
                        # 截断过长的工具输入
                        inp_str = str(inp)
                        if len(inp_str) > 200:
                            inp_str = inp_str[:200] + "..."
                        text_parts.append(f"[调用工具: {name}({inp_str})]")
                    elif btype == "image":
                        text_parts.append("[图片]")
                    elif btype == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, str) and len(tool_content) > 300:
                            tool_content = tool_content[:300] + "..."
                        text_parts.append(f"[工具结果: {tool_content}]")
            content = "\n".join(text_parts)
        elif isinstance(content, str) and len(content) > 500:
            content = content[:500] + "..."

        if content:
            parts.append(f"[{role}] {content}")

    return "\n\n".join(parts)
