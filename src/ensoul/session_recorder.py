"""会话记录工具 — 将 CLI/MCP 生成的 prompt 以 JSONL 形式持久化."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ensoul.paths import resolve_project_dir

logger = logging.getLogger(__name__)


class SessionRecorder:
    """记录单次 CLI 调用或流水线执行的完整轨迹."""

    def __init__(self, session_dir: Path | None = None, *, project_dir: Path | None = None):
        self.session_dir = (
            session_dir
            if session_dir is not None
            else resolve_project_dir(project_dir) / ".ensoul" / "sessions"
        )

    def _ensure_dir(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        safe_id = session_id.replace("/", "_").replace("..", "_")
        return self.session_dir / f"{safe_id}.jsonl"

    def start(
        self,
        session_type: str,
        subject: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """创建 session 并写入第一条记录."""
        self._ensure_dir()
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "start",
            "session_type": session_type,
            "subject": subject,
            "metadata": metadata or {},
        }
        self._session_path(session_id).write_text(
            json.dumps(entry, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return session_id

    def _append(self, session_id: str, payload: dict[str, Any]) -> None:
        path = self._session_path(session_id)
        if not path.exists():
            raise ValueError(f"Session 不存在: {session_id}")
        payload.setdefault("timestamp", datetime.now().isoformat())
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def record_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """记录一条文本内容（prompt、计划、结果等）。"""
        self._append(
            session_id,
            {
                "event": "message",
                "role": role,
                "content": content,
                "metadata": metadata or {},
            },
        )

    def record_event(
        self,
        session_id: str,
        event: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """记录非文本事件，例如参数、输出路径等。"""
        self._append(
            session_id,
            {
                "event": event,
                "metadata": metadata or {},
            },
        )

    def finish(
        self,
        session_id: str,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        """结束 session，写入最终状态."""
        self._append(
            session_id,
            {
                "event": "end",
                "status": status,
                "detail": detail or "",
            },
        )

    def list_sessions(
        self,
        *,
        limit: int = 20,
        session_type: str | None = None,
        subject: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出最近的 session."""
        if not self.session_dir.is_dir():
            return []

        sessions: list[dict[str, Any]] = []
        files = sorted(self.session_dir.glob("*.jsonl"), reverse=True)
        for path in files:
            try:
                first_line = path.read_text(encoding="utf-8").splitlines()[0]
                data = json.loads(first_line)
            except (json.JSONDecodeError, IndexError, OSError) as e:
                logger.debug("跳过损坏的 session 文件 %s: %s", path.name, e)
                continue

            if session_type and data.get("session_type") != session_type:
                continue
            if subject and data.get("subject") != subject:
                continue

            sessions.append(
                {
                    "session_id": path.stem,
                    "session_type": data.get("session_type", ""),
                    "subject": data.get("subject", ""),
                    "started_at": data.get("timestamp", ""),
                }
            )
            if len(sessions) >= limit:
                break
        return sessions

    def read_session(self, session_id: str) -> list[dict[str, Any]]:
        """读取 session 的所有条目."""
        path = self._session_path(session_id)
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
