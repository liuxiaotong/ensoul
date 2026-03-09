"""任务注册表 — 追踪 webhook 触发的异步任务，支持 JSONL 持久化."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field


class TaskRecord(BaseModel):
    """单个任务记录."""

    task_id: str = Field(description="任务唯一标识")
    status: Literal["pending", "running", "completed", "failed", "awaiting_approval"] = "pending"
    trigger: str = Field(description="触发来源: github / openclaw / generic / direct / cron")
    target_type: str = Field(description="目标类型: pipeline / employee")
    target_name: str = Field(description="目标名称")
    args: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    checkpoint: dict[str, Any] | None = Field(default=None, description="Pipeline 断点数据")
    owner: str | None = Field(default=None, description="任务创建者 ID（用于归属校验）")


class TaskRegistry:
    """任务注册表，内存缓存 + JSONL 持久化.

    Args:
        persist_path: JSONL 持久化文件路径（None 则仅内存模式）.
        max_history: 持久化保留的最大记录数（默认 5000，超出时截断旧记录）.
    """

    def __init__(
        self,
        persist_path: Path | None = None,
        max_history: int = 5000,
    ) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._persist_path = persist_path
        self._max_history = max_history
        self._lock = threading.Lock()

        if persist_path:
            self._load_history()

    def _load_history(self) -> None:
        """从 JSONL 加载历史任务."""
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            for line in self._persist_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = TaskRecord.model_validate_json(line)
                    self._tasks[record.task_id] = record
                    # 为已完成的任务创建已设置的 Event
                    evt = asyncio.Event()
                    if record.status in ("completed", "failed"):
                        evt.set()
                    self._events[record.task_id] = evt
                except Exception as e:
                    logger.warning("跳过无效的任务记录: %s", e)
                    continue
        except Exception as e:
            logger.warning("加载任务历史失败: %s", e)

    def _persist(self, record: TaskRecord) -> None:
        """追加写入 JSONL."""
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        except Exception as e:
            logger.warning("持久化任务记录失败 [%s]: %s", record.task_id, e)

    def _compact_if_needed(self) -> None:
        """如果记录过多，截断旧记录（原子写入）."""
        if self._persist_path is None:
            return
        with self._lock:
            if len(self._tasks) <= self._max_history:
                return
            # 按创建时间排序，保留最新的
            sorted_records = sorted(
                self._tasks.values(),
                key=lambda r: r.created_at,
            )
            to_remove = sorted_records[: len(sorted_records) - self._max_history]
            for r in to_remove:
                del self._tasks[r.task_id]
                self._events.pop(r.task_id, None)
            keep = sorted_records[len(to_remove) :]
            # 原子写入（锁内执行，避免 TOCTOU 竞态）
            try:
                lines = [r.model_dump_json() for r in keep]
                content = "\n".join(lines) + "\n"
                fd, tmp_path = tempfile.mkstemp(
                    dir=self._persist_path.parent,
                    suffix=".tmp",
                )
                fd_closed = False
                try:
                    os.write(fd, content.encode("utf-8"))
                    os.close(fd)
                    fd_closed = True
                    os.replace(tmp_path, self._persist_path)
                except Exception:
                    if not fd_closed:
                        os.close(fd)
                    if Path(tmp_path).exists():
                        os.unlink(tmp_path)
                    raise
            except Exception as e:
                logger.warning("任务记录压缩失败: %s", e)

    def create(
        self,
        trigger: str,
        target_type: str,
        target_name: str,
        args: dict[str, str] | None = None,
        owner: str | None = None,
    ) -> TaskRecord:
        """创建新任务."""
        task_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        record = TaskRecord(
            task_id=task_id,
            trigger=trigger,
            target_type=target_type,
            target_name=target_name,
            args=args or {},
            owner=owner,
        )
        with self._lock:
            self._tasks[task_id] = record
            self._events[task_id] = asyncio.Event()
        self._persist(record)
        self._compact_if_needed()
        return record

    def update(
        self,
        task_id: str,
        status: Literal["running", "completed", "failed", "awaiting_approval"],
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> TaskRecord | None:
        """更新任务状态."""
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            record.status = status
            if result is not None:
                record.result = result
            if error is not None:
                record.error = error
            if status in ("completed", "failed"):
                record.completed_at = datetime.now()
                event = self._events.get(task_id)
                if event:
                    event.set()
        self._persist(record)
        return record

    def update_checkpoint(self, task_id: str, checkpoint: dict[str, Any]) -> None:
        """更新任务的断点数据."""
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            record.checkpoint = checkpoint
        self._persist(record)

    def get(self, task_id: str) -> TaskRecord | None:
        """查询任务."""
        return self._tasks.get(task_id)

    def list_recent(self, n: int = 20) -> list[TaskRecord]:
        """列出最近 n 条任务."""
        sorted_records = sorted(
            self._tasks.values(),
            key=lambda r: r.created_at,
            reverse=True,
        )
        return sorted_records[:n]

    def list_by_status(
        self,
        status: Literal["pending", "running", "completed", "failed"],
        limit: int = 20,
    ) -> list[TaskRecord]:
        """按状态筛选任务，最新的在前."""
        filtered = [r for r in self._tasks.values() if r.status == status]
        return sorted(filtered, key=lambda r: r.created_at, reverse=True)[:limit]

    def snapshot(self) -> list[TaskRecord]:
        """返回所有任务记录的快照（防止外部直接访问 _tasks）."""
        with self._lock:
            return list(self._tasks.values())

    def list_by_type(self, target_type: str, limit: int = 20) -> list[TaskRecord]:
        """按目标类型筛选任务，最新的在前."""
        filtered = [r for r in self._tasks.values() if r.target_type == target_type]
        return sorted(filtered, key=lambda r: r.created_at, reverse=True)[:limit]

    async def wait(self, task_id: str, timeout: float = 300) -> TaskRecord | None:
        """等待任务完成（同步模式）."""
        event = self._events.get(task_id)
        if event is None:
            return None
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        return self._tasks.get(task_id)
