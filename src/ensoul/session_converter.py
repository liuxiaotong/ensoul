"""Session → Trajectory 转换器.

将 .ensoul/sessions/*.jsonl (事件流格式) 转换为 RewardEngine 可评分的 trajectory dict。
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# source 前缀判断: cli.* 开头的都是程序生成的
_SYNTHETIC_SOURCE_PREFIX = "cli."


def is_organic(start_event: dict) -> bool:
    """判断一个 session 是否为真实用户交互（非程序生成）.

    判定逻辑: metadata.source 以 "cli." 开头的是合成数据，其余为 organic。
    这样以后新增飞书、微信、web 等渠道都自动归为 organic。
    """
    source = start_event.get("metadata", {}).get("source", "")
    return not source.startswith(_SYNTHETIC_SOURCE_PREFIX)


def classify_session(start_event: dict) -> str:
    """返回 session 的分类标签，用于归档.

    返回值:
        "organic" — 真实用户交互（飞书、微信等）
        "training" — 批量训练数据（employee + cli.run）
        "pipeline" — 自动化流水线
        "discussion" — AI 间讨论
    """
    source = start_event.get("metadata", {}).get("source", "")
    session_type = start_event.get("session_type", "")

    if not source.startswith(_SYNTHETIC_SOURCE_PREFIX):
        return "organic"
    if session_type == "employee":
        return "training"
    if session_type == "pipeline":
        return "pipeline"
    if session_type == "discussion":
        return "discussion"
    return "synthetic"


def convert_session(session_path: Path) -> dict[str, Any] | None:
    """读取单个 session JSONL，返回 trajectory dict 或 None（跳过）.

    返回格式兼容 agentreward.RewardEngine.score()::

        {
            "task": "用户的问题或指令",
            "steps": [{"tool": "respond", "params": {}, "output": "回复内容"}],
            "outcome": {"success": true},
            "metadata": {"session_id": "...", "employee": "...", "model": "..."},
        }
    """
    lines = session_path.read_text("utf-8").strip().split("\n")
    if not lines:
        return None

    events = [json.loads(line) for line in lines if line.strip()]

    # 解析 start 事件
    start = next((e for e in events if e.get("event") == "start"), None)
    if start is None:
        return None

    subject = start.get("subject", "")
    session_type = start.get("session_type", "")
    start_meta = start.get("metadata", {})
    session_id = session_path.stem

    # 提取消息
    messages = [e for e in events if e.get("event") == "message"]
    if len(messages) < 2:
        return None

    # 找 user/prompt 消息和 assistant 消息
    user_msg = None
    assistant_msg = None

    for msg in messages:
        role = msg.get("role", "")
        if role in ("user", "prompt") and user_msg is None:
            user_msg = msg
        elif role == "assistant":
            assistant_msg = msg

    if assistant_msg is None:
        # 没有 assistant 回复，跳过（prompt-only session）
        return None

    # 提取 task
    task = _extract_task(user_msg, start_meta)
    if not task:
        return None

    # 提取 response
    response = assistant_msg.get("content", "").strip()
    if not response:
        return None

    # 提取 model
    model = assistant_msg.get("metadata", {}).get("model", "")
    if not model:
        model = start_meta.get("model", "")

    # 构建 trajectory
    return {
        "task": task,
        "steps": [
            {
                "tool": "respond",
                "params": {},
                "output": response,
            }
        ],
        "outcome": {"success": True},
        "metadata": {
            "session_id": session_id,
            "employee": subject,
            "model": model,
            "session_type": session_type,
            "source": start_meta.get("source", ""),
            "origin": "organic" if is_organic(start) else "synthetic",
            "timestamp": start.get("timestamp", ""),
        },
    }


def _extract_task(user_msg: dict | None, start_meta: dict) -> str:
    """从用户消息或 start 元数据中提取任务描述."""
    # 优先用 start args 里的 task
    args = start_meta.get("args", {})
    if isinstance(args, dict) and args.get("task"):
        return args["task"]

    if user_msg is None:
        return ""

    content = user_msg.get("content", "").strip()
    role = user_msg.get("role", "")

    if role == "user":
        # 真实用户输入，直接用
        return content

    if role == "prompt":
        # 系统 prompt，尝试从中提取用户指令
        # 如果整个 content 是系统 prompt（以 # 开头），提取 args.task 或标记为会话开始
        if content.startswith("#") or len(content) > 500:
            task = args.get("task", "") if isinstance(args, dict) else ""
            return task or "(会话开始)"
        return content

    return ""


def convert_sessions_batch(
    sessions_dir: Path,
    *,
    employee: str | None = None,
    since: str | None = None,
    limit: int | None = None,
    origin: str = "organic",
) -> list[dict[str, Any]]:
    """批量转换 sessions 目录下的所有 JSONL 文件.

    Args:
        sessions_dir: .ensoul/sessions/ 目录路径
        employee: 只转换特定员工（如 "ceo-assistant"）
        since: 只转换该日期之后的 session（如 "20260215"）
        limit: 最多转换 N 个
        origin: 来源过滤（默认 "organic" = 真实用户交互）。
            "organic" — 只取真实用户对话（排除 cli.* 生成的）
            "synthetic" — 只取程序生成的
            "all" — 不过滤

    Returns:
        转换成功的 trajectory 列表
    """
    trajectories = []
    skipped = 0

    for f in sorted(sessions_dir.glob("*.jsonl")):
        # 日期过滤
        if since and f.stem < since:
            continue

        # 读取 start 事件做过滤
        first_line = f.read_text("utf-8").split("\n")[0]
        start = json.loads(first_line)

        # 员工过滤
        if employee and start.get("subject") != employee:
            continue

        # origin 过滤（organic vs synthetic）
        if origin == "organic" and not is_organic(start):
            skipped += 1
            continue
        elif origin == "synthetic" and is_organic(start):
            skipped += 1
            continue

        traj = convert_session(f)
        if traj is None:
            skipped += 1
            continue

        trajectories.append(traj)

        if limit and len(trajectories) >= limit:
            break

    logger.info(
        "转换完成: %d 条 trajectory, 跳过 %d 条 (origin 过滤: %s)",
        len(trajectories),
        skipped,
        origin,
    )
    return trajectories


def archive_sessions(
    sessions_dir: Path,
    archive_root: Path,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """将合成数据从 sessions/ 归档到分类子目录.

    归档结构::

        archive_root/
            training/     ← employee + cli.run（批量训练数据）
            pipelines/    ← pipeline 执行记录
            discussions/  ← AI 间讨论

    organic 数据不动，留在 sessions/ 里。

    Args:
        sessions_dir: .ensoul/sessions/ 目录
        archive_root: 归档根目录（如 .ensoul/archive/）
        dry_run: True 时只统计不移动

    Returns:
        各分类移动的文件数
    """
    counts: dict[str, int] = {"training": 0, "pipeline": 0, "discussion": 0, "synthetic": 0}

    for f in sorted(sessions_dir.glob("*.jsonl")):
        first_line = f.read_text("utf-8").split("\n")[0]
        start = json.loads(first_line)

        category = classify_session(start)
        if category == "organic":
            continue

        counts[category] = counts.get(category, 0) + 1

        if not dry_run:
            # 按分类归档
            dest_dir = archive_root / category
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest_dir / f.name))

    return counts
