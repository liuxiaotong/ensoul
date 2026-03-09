"""工作日志 — JSONL 格式追踪员工工作记录."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from ensoul.models import WorkLogEntry
from ensoul.paths import resolve_project_dir

logger = logging.getLogger(__name__)


class WorkLogger:
    """工作日志管理器.

    日志以 JSONL 格式存储在指定目录下，
    每个 session 一个文件。
    """

    def __init__(self, log_dir: Path | None = None, *, project_dir: Path | None = None):
        """初始化日志管理器.

        Args:
            log_dir: 日志目录，默认 .crew/logs
            project_dir: 项目根目录，用于计算默认 log_dir
        """
        self.log_dir = (
            log_dir if log_dir is not None else resolve_project_dir(project_dir) / ".crew" / "logs"
        )

    def _ensure_dir(self) -> None:
        """确保日志目录存在."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _session_file(self, session_id: str) -> Path:
        """返回 session 文件路径."""
        safe_id = session_id.replace("/", "_").replace("..", "_")
        return self.log_dir / f"{safe_id}.jsonl"

    def create_session(
        self,
        employee_name: str,
        args: dict[str, str] | None = None,
        agent_id: str | None = None,
        detail: str | None = None,
    ) -> str:
        """创建新的工作 session.

        Args:
            employee_name: 员工名称
            args: 调用参数
            agent_id: 关联的平台 Agent ID

        Returns:
            session_id
        """
        self._ensure_dir()
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]

        entry = WorkLogEntry(
            employee_name=employee_name,
            action="session_start",
            detail=detail or f"员工 {employee_name} 开始工作",
            args=args or {},
            agent_id=agent_id,
        )

        with open(self._session_file(session_id), "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

        return session_id

    def add_entry(
        self,
        session_id: str,
        action: str,
        detail: str = "",
        severity: str = "info",
        metrics: dict[str, float] | None = None,
        links: list[str] | None = None,
    ) -> None:
        """向 session 追加日志条目.

        Args:
            session_id: 会话 ID
            action: 动作描述
            detail: 详细信息
        """
        session_file = self._session_file(session_id)
        if not session_file.exists():
            raise ValueError(f"Session 不存在: {session_id}")

        # 从第一条记录获取 employee_name
        lines = session_file.read_text(encoding="utf-8").splitlines()
        if not lines or not lines[0].strip():
            raise ValueError(f"Session 文件为空: {session_id}")
        try:
            first_entry = json.loads(lines[0])
        except json.JSONDecodeError as e:
            logger.warning("Session %s 首行 JSON 解析失败: %s", session_id, e)
            raise ValueError(f"Session 文件损坏: {session_id}") from e

        entry = WorkLogEntry(
            employee_name=first_entry.get("employee_name", "unknown"),
            action=action,
            detail=detail,
            severity=severity,
            metrics=metrics or {},
            links=links or [],
        )

        with open(session_file, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    def list_sessions(
        self,
        employee_name: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """列出工作 session.

        Args:
            employee_name: 按员工过滤
            limit: 返回条数

        Returns:
            session 摘要列表
        """
        if not self.log_dir.is_dir():
            return []

        sessions = []
        for f in sorted(self.log_dir.glob("*.jsonl"), reverse=True):
            try:
                first_entry: dict | None = None
                line_count = 0
                with f.open("r", encoding="utf-8") as fh:
                    for raw_line in fh:
                        line = raw_line.strip()
                        if not line:
                            continue
                        line_count += 1
                        if first_entry is None:
                            first_entry = json.loads(line)

                if not first_entry:
                    continue

                if employee_name and first_entry.get("employee_name") != employee_name:
                    continue

                sessions.append(
                    {
                        "session_id": f.stem,
                        "employee_name": first_entry.get("employee_name", ""),
                        "started_at": first_entry.get("timestamp", ""),
                        "entries": line_count,
                    }
                )

                if len(sessions) >= limit:
                    break
            except Exception:
                continue

        return sessions

    def get_session(self, session_id: str) -> list[dict]:
        """获取 session 的所有日志条目.

        Args:
            session_id: 会话 ID

        Returns:
            日志条目列表
        """
        session_file = self._session_file(session_id)
        if not session_file.exists():
            return []

        entries = []
        for line in session_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("跳过无效日志行: %s", line[:100])
                continue

        return entries
