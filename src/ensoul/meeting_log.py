"""会议记录持久化 — 保存和查询讨论会历史."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from ensoul.paths import resolve_project_dir


class MeetingRecord(BaseModel):
    """会议记录元数据."""

    meeting_id: str = Field(description="会议 ID（时间戳格式）")
    name: str = Field(description="讨论会名称")
    topic: str = Field(description="议题")
    participants: list[str] = Field(description="参与者列表")
    mode: str = Field(description="模式: discussion / meeting")
    rounds: int = Field(description="轮次数")
    output_format: str = Field(default="summary", description="输出格式")
    started_at: str = Field(description="开始时间 (ISO)")
    args: dict[str, str] = Field(default_factory=dict, description="传入参数")


class MeetingLogger:
    """会议记录管理器.

    存储结构:
      {meetings_dir}/
        index.jsonl         — 元数据索引（每行一条 JSON）
        {meeting_id}.md     — 完整会议 prompt
    """

    def __init__(self, meetings_dir: Path | None = None, *, project_dir: Path | None = None):
        self.meetings_dir = (
            meetings_dir
            if meetings_dir is not None
            else resolve_project_dir(project_dir) / ".ensoul" / "meetings"
        )

    def save(
        self,
        discussion: Discussion,  # noqa: F821
        prompt: str,
        args: dict[str, str] | None = None,
    ) -> str:
        """保存会议记录，返回 meeting_id."""
        try:
            from ensoul.discussion import _resolve_rounds
        except ImportError:
            raise ImportError("ensoul.discussion module is required for MeetingLogger.save()")

        now = datetime.now()
        meeting_id = now.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

        self.meetings_dir.mkdir(parents=True, exist_ok=True)

        rounds = _resolve_rounds(discussion)
        rounds_count = rounds if isinstance(rounds, int) else len(rounds)

        record = MeetingRecord(
            meeting_id=meeting_id,
            name=discussion.name,
            topic=discussion.topic,
            participants=[p.employee for p in discussion.participants],
            mode=discussion.effective_mode,
            rounds=rounds_count,
            output_format=discussion.output_format,
            started_at=now.isoformat(),
            args=args or {},
        )

        # 追加到 index.jsonl
        index_path = self.meetings_dir / "index.jsonl"
        with index_path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

        # 写入完整 prompt
        prompt_path = self.meetings_dir / f"{meeting_id}.md"
        prompt_path.write_text(prompt, encoding="utf-8")

        return meeting_id

    def list(self, limit: int = 20, keyword: str | None = None) -> list[MeetingRecord]:
        """列出历史会议，最新在前."""
        index_path = self.meetings_dir / "index.jsonl"
        if not index_path.exists():
            return []

        records: list[MeetingRecord] = []
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = MeetingRecord(**json.loads(line))
                if keyword:
                    text = f"{record.name} {record.topic} {' '.join(record.participants)}"
                    if keyword.lower() not in text.lower():
                        continue
                records.append(record)
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug("跳过无效会议记录: %s", e)
                continue

        records.reverse()
        return records[:limit]

    def get(self, meeting_id: str) -> tuple[MeetingRecord, str] | None:
        """获取会议详情：元数据 + prompt 内容."""
        index_path = self.meetings_dir / "index.jsonl"
        if not index_path.exists():
            return None

        record: MeetingRecord | None = None
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("meeting_id") == meeting_id:
                    record = MeetingRecord(**data)
                    break
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug("跳过无效会议记录: %s", e)
                continue

        if record is None:
            return None

        prompt_path = self.meetings_dir / f"{meeting_id}.md"
        content = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

        return record, content
