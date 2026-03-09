"""根据 session JSONL 摘要并写入记忆."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ensoul.memory import get_memory_store
from ensoul.memory_index import MemorySearchIndex
from ensoul.session_recorder import SessionRecorder

logger = logging.getLogger(__name__)


def _truncate(text: str, length: int = 120) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= length:
        return text
    return text[: length - 1] + "…"


class SessionMemoryWriter:
    """读取 session 记录并写入 MemoryStore."""

    def __init__(
        self,
        session_dir: Path | None = None,
        memory_dir: Path | None = None,
        *,
        project_dir: Path | None = None,
    ) -> None:
        self.recorder = SessionRecorder(session_dir=session_dir, project_dir=project_dir)
        self.store = get_memory_store(project_dir=project_dir)
        self.index = MemorySearchIndex(
            memory_dir=memory_dir, session_dir=session_dir, project_dir=project_dir
        )

    def capture(self, *, employee: str, session_id: str | None) -> str | None:
        if not session_id:
            return None
        entries = self.recorder.read_session(session_id)
        if not entries:
            return None
        summary = self._summarize(entries)
        if not summary:
            return None
        entry = self.store.add_from_session(
            employee=employee,
            session_id=session_id,
            summary=summary,
        )
        if entry:
            try:
                self.index.rebuild()
            except Exception as e:
                logger.warning("重建记忆索引失败: %s", e)
            return summary
        return None

    def _summarize(self, entries: list[dict[str, Any]]) -> str:
        start_meta: dict[str, Any] = {}
        messages: list[dict[str, Any]] = []
        for entry in entries:
            event = entry.get("event")
            if event == "start":
                start_meta = entry.get("metadata", {}) or {}
            elif event == "message":
                messages.append(entry)
        if not messages:
            return ""

        lines: list[str] = []
        args = start_meta.get("args")
        if isinstance(args, dict) and args:
            parts = [f"{k}={v}" for k, v in list(args.items())[:3]]
            lines.append("参数: " + ", ".join(parts))
        mode = start_meta.get("mode")
        if mode:
            lines.append(f"模式: {mode}")

        for msg in messages[:3]:
            role = msg.get("role", "message")
            snippet = _truncate(msg.get("content", ""))
            if not snippet:
                continue
            lines.append(f"- {role}: {snippet}")

        if len(messages) > 3:
            final = messages[-1]
            snippet = _truncate(final.get("content", ""))
            if snippet:
                lines.append(f"- final: {snippet}")

        lines.append(f"总消息: {len(messages)}")
        return "\n".join(lines)


__all__ = ["SessionMemoryWriter"]
