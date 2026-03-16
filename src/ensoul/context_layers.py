"""MemGPT 式分层上下文管理.

将 agent context 分为三层:
- Core Memory: 始终在 system prompt，精简 soul + 当前任务 + 近期记忆摘要
- Working Memory: 动态管理，在 messages 中（工具调用、中间结果、todo）
- Archival Memory: 按需检索（全部历史记忆、Wiki 等通过工具获取）

设计原则:
- 不调 LLM，纯文本处理
- 不改 soul.md 格式，只在运行时提取
- budget_tokens 用字符数估算
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Token 估算常量：1 token ≈ 3 中文字符 或 4 英文字符
# 取保守值：1 token ≈ 3 字符（中英混合场景偏中文）
_CHARS_PER_TOKEN = 3

# 应该排除的章节关键词（归入 archival）
# 策略："非 archival 即保留"— 只排除明确标记为 archival 的章节，其余全部保留到 core memory
_ARCHIVAL_SECTION_KEYWORDS = [
    "历史经验",
    "高分范例",
    "上次教训",
    "同事",
    "眼里的同事",
    "组织信息",
    "生活",
    "情绪",
    "来历",
    "内心习惯",
    "日记",
    "状态",
    "焦虑",
]


def extract_soul_core(soul_text: str) -> str:
    """从完整 soul.md 中提取核心段落.

    保留策略:
    - 第一段（标题后到第一个 ## 之前）作为身份描述，始终保留
    - 包含核心关键词的章节保留
    - 排除历史经验、同事关系等可按需查询的内容

    Args:
        soul_text: 完整的 soul.md 文本

    Returns:
        精简后的 soul 核心文本
    """
    if not soul_text or not soul_text.strip():
        return ""

    lines = soul_text.split("\n")

    # 解析章节：(标题, 内容行列表)
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []

    for line in lines:
        if re.match(r"^##\s+", line):
            # 保存上一个章节
            if current_title or current_lines:
                sections.append((current_title, current_lines))
            current_title = line
            current_lines = []
        else:
            current_lines.append(line)

    # 保存最后一个章节
    if current_title or current_lines:
        sections.append((current_title, current_lines))

    # 筛选核心章节
    kept: list[str] = []

    for i, (title, content) in enumerate(sections):
        if i == 0 and not title.startswith("##"):
            # 第一段（标题前内容 / H1 标题），始终保留
            combined = "\n".join(content).strip()
            if combined:
                kept.append(combined)
            continue

        title_text = title.lstrip("#").strip()

        # 排除明确标记为 archival 的章节（历史经验、高分范例、同事近况等）
        is_archival = any(kw in title_text for kw in _ARCHIVAL_SECTION_KEYWORDS)
        if is_archival:
            continue

        # 非 archival 即保留 — 避免未匹配核心关键词的章节被静默丢弃
        section_text = title + "\n" + "\n".join(content)
        kept.append(section_text.strip())

    return "\n\n".join(kept)


def summarize_memories(memories: list[dict], limit: int = 5) -> str:
    """取 importance 最高的记忆生成一行摘要.

    Args:
        memories: 记忆字典列表，需包含 content, category 字段，
                  可选 importance 字段
        limit: 最多取几条

    Returns:
        格式化的记忆摘要文本，每条一行
    """
    if not memories:
        return ""

    # 按 importance 降序排列（默认 3）
    sorted_mems = sorted(
        memories,
        key=lambda m: m.get("importance", 3),
        reverse=True,
    )

    lines: list[str] = []
    for mem in sorted_mems[:limit]:
        category = mem.get("category", "finding")
        content = mem.get("content", "")
        # 先清理换行，再截断（避免换行符影响截断位置）
        content = content.replace("\n", " ").strip()
        # 截断到 80 字符
        if len(content) > 80:
            content = content[:80] + "..."
        if content:
            lines.append(f"- [{category}] {content}")

    return "\n".join(lines)


def format_progress_memories(progress_memories: list[dict]) -> str:
    """将进展记忆格式化为人类可读的文本.

    Args:
        progress_memories: 进展记忆列表，每条的 content 是 JSON 字符串

    Returns:
        格式化的进展文本
    """
    import json

    if not progress_memories:
        return ""

    lines: list[str] = []
    for mem in progress_memories:
        content = mem.get("content", "")
        try:
            data = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            continue

        goal = data.get("goal", "")
        status = data.get("status", "unknown")
        summary = data.get("summary", "")
        completed = data.get("completed", [])
        pending = data.get("pending", [])
        round_num = data.get("round", 0)

        status_label = "已完成" if status == "completed" else "已中断"
        lines.append(f"**目标**: {goal}")
        lines.append(f"**状态**: {status_label}（第 {round_num} 轮）")
        if summary:
            lines.append(f"**摘要**: {summary}")
        if completed:
            lines.append("**已完成步骤**:")
            for step in completed[:10]:
                lines.append(f"  - {step}")
        if pending:
            lines.append("**未完成步骤**:")
            for step in pending[:10]:
                lines.append(f"  - {step}")
        lines.append("")

    return "\n".join(lines).strip()


def build_core_memory(
    soul_text: str,
    task_text: str,
    memories: list[dict],
    *,
    budget_tokens: int = 2000,
    progress_memories: list[dict] | None = None,
) -> str:
    """构建 Core Memory 层内容.

    将精简 soul + 当前任务 + 近期记忆摘要组合，控制在 token 预算内。

    Args:
        soul_text: 完整 soul.md 文本
        task_text: 当前任务描述
        memories: 记忆列表（dict 格式）
        budget_tokens: token 预算（默认 2000）
        progress_memories: 前序进展记忆列表（可选）

    Returns:
        格式化的 core memory 文本
    """
    budget_chars = budget_tokens * _CHARS_PER_TOKEN

    # 1. 提取 soul 核心
    soul_core = extract_soul_core(soul_text)

    # 2. 生成记忆摘要
    memory_summary = summarize_memories(memories, limit=5)

    # 3. 组装
    parts: list[str] = []

    if soul_core:
        parts.append(f"## 身份\n\n{soul_core}")

    # 3.5 前序进展（在当前任务之前）
    if progress_memories:
        progress_text = format_progress_memories(progress_memories)
        if progress_text:
            parts.append(f"## 前序进展\n\n{progress_text}")

    if task_text:
        task_display = task_text if isinstance(task_text, str) else str(task_text)
        parts.append(f"## 当前任务\n\n{task_display}")

    if memory_summary:
        parts.append(f"## 近期相关记忆\n\n{memory_summary}")

    result = "\n\n".join(parts)

    # 4. 截断到预算
    if len(result) > budget_chars:
        # 优先保留身份和任务，截断记忆
        result = result[:budget_chars]
        # 确保不在 UTF-8 多字节字符中间截断（Python str 不会，但避免截断到半个 markdown 标记）
        # 找到最后一个完整行
        last_newline = result.rfind("\n")
        if last_newline > budget_chars * 0.8:
            result = result[:last_newline]
        result += "\n\n[...core memory truncated to fit budget...]"

    return result
