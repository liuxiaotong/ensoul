"""评估闭环 — 追踪决策质量，回写纠正到记忆."""

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from ensoul.paths import file_lock, resolve_project_dir


class Decision(BaseModel):
    """会议中产出的决策."""

    id: str = Field(default_factory=lambda: "D" + uuid.uuid4().hex[:8], description="决策 ID")
    employee: str = Field(description="提出决策的员工")
    meeting_id: str = Field(default="", description="来源会议 ID")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="创建时间"
    )
    category: Literal["estimate", "recommendation", "commitment"] = Field(description="决策类别")
    content: str = Field(description="决策内容")
    expected_outcome: str = Field(default="", description="预期结果")
    status: Literal["pending", "evaluated"] = Field(default="pending", description="状态")
    actual_outcome: str = Field(default="", description="实际结果")
    evaluation: str = Field(default="", description="评估结论")
    deadline: str = Field(default="", description="截止日期 ISO 格式，如 2026-03-07")
    task_id: str = Field(default="", description="关联的任务 ID")


class EvaluationEngine:
    """评估引擎 — 管理决策追踪和回溯评估.

    存储结构:
      {eval_dir}/
        decisions.jsonl  — 所有决策记录
    """

    def __init__(
        self,
        eval_dir: Path | None = None,
        *,
        project_dir: Path | None = None,
        tenant_id: str | None = None,
    ):
        self._project_dir = project_dir
        self._tenant_id = tenant_id
        self.eval_dir = (
            eval_dir
            if eval_dir is not None
            else resolve_project_dir(project_dir) / ".ensoul" / "evaluations"
        )

    def _ensure_dir(self) -> None:
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def _decisions_file(self) -> Path:
        return self.eval_dir / "decisions.jsonl"

    def track(
        self,
        employee: str,
        category: Literal["estimate", "recommendation", "commitment"],
        content: str,
        expected_outcome: str = "",
        meeting_id: str = "",
        deadline: str = "",
        task_id: str = "",
    ) -> Decision:
        """记录一个待评估的决策.

        Args:
            employee: 提出决策的员工
            category: 决策类别
            content: 决策内容
            expected_outcome: 预期结果
            meeting_id: 来源会议 ID
            deadline: 截止日期 ISO 格式（可选）
            task_id: 关联的任务 ID（可选）

        Returns:
            Decision 对象
        """
        self._ensure_dir()
        decision = Decision(
            employee=employee,
            category=category,
            content=content,
            expected_outcome=expected_outcome,
            meeting_id=meeting_id,
            deadline=deadline,
            task_id=task_id,
        )
        with self._decisions_file().open("a", encoding="utf-8") as f:
            f.write(decision.model_dump_json() + "\n")
        return decision

    def evaluate(
        self,
        decision_id: str,
        actual_outcome: str,
        evaluation: str = "",
    ) -> Decision | None:
        """评估一个决策，记录实际结果.

        同时将评估结论写入对应员工的持久化记忆。

        Args:
            decision_id: 决策 ID
            actual_outcome: 实际结果
            evaluation: 评估结论（可选，为空则自动生成简要结论）

        Returns:
            更新后的 Decision，未找到返回 None
        """
        path = self._decisions_file()
        if not path.exists():
            return None

        with file_lock(path):
            lines = path.read_text(encoding="utf-8").splitlines()
            found_decision: Decision | None = None
            new_lines: list[str] = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    decision = Decision(**json.loads(line))
                except Exception:
                    new_lines.append(line)
                    continue

                if decision.id == decision_id:
                    decision.actual_outcome = actual_outcome
                    decision.evaluation = (
                        evaluation or f"预期: {decision.expected_outcome}; 实际: {actual_outcome}"
                    )
                    decision.status = "evaluated"
                    found_decision = decision

                new_lines.append(decision.model_dump_json())

            if found_decision is None:
                return None

            # 原子重写决策文件
            content = "\n".join(new_lines) + "\n"
            fd, tmp_path = tempfile.mkstemp(
                dir=path.parent,
                suffix=".tmp",
            )
            fd_closed = False
            try:
                os.write(fd, content.encode("utf-8"))
                os.close(fd)
                fd_closed = True
                os.replace(tmp_path, path)
            except Exception:
                if not fd_closed:
                    os.close(fd)
                if Path(tmp_path).exists():
                    os.unlink(tmp_path)
                raise

        # 将评估结论写入员工记忆（锁外执行，避免死锁，去重防止重复 correction）
        # NOTE: 此处先查后写存在 TOCTOU 竞争，但当前场景风险极低：
        # cron 按员工串行处理，同一 decision_id 不会被并行评估。
        # 若需更严格保证，可在 MemoryStore.add 中做 upsert 或加 unique constraint。
        try:
            from ensoul.memory import get_memory_store

            eval_source = f"eval:{decision_id}"
            store = get_memory_store(project_dir=self._project_dir, tenant_id=self._tenant_id)
            existing = store.query(
                employee=found_decision.employee, category="correction", limit=200
            )
            already_written = any(getattr(e, "source_session", "") == eval_source for e in existing)
            if not already_written:
                from ensoul.memory_pipeline import process_memory

                process_memory(
                    raw_text=found_decision.evaluation,
                    employee=found_decision.employee,
                    store=store,
                    skip_reflect=True,
                    source_session=eval_source,
                    category="correction",
                )
        except Exception:
            pass  # 记忆写入失败不影响评估

        return found_decision

    def list_decisions(
        self,
        employee: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[Decision]:
        """列出决策记录.

        Args:
            employee: 按员工过滤
            status: 按状态过滤 (pending / evaluated)
            limit: 最大返回条数

        Returns:
            决策列表（最新在前）
        """
        path = self._decisions_file()
        if not path.exists():
            return []

        decisions: list[Decision] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                decision = Decision(**json.loads(line))
            except Exception:
                continue

            if employee and decision.employee != employee:
                continue
            if status and decision.status != status:
                continue

            decisions.append(decision)

        decisions.reverse()
        return decisions[:limit]

    def get(self, decision_id: str) -> Decision | None:
        """获取单个决策详情."""
        path = self._decisions_file()
        if not path.exists():
            return None

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                decision = Decision(**json.loads(line))
                if decision.id == decision_id:
                    return decision
            except Exception:
                continue
        return None

    def list_overdue(self, as_of: str = "") -> list[Decision]:
        """列出已过期且未评估的决策.

        Args:
            as_of: 参考日期 ISO 格式（YYYY-MM-DD），默认今天

        Returns:
            过期的 pending 决策列表
        """
        if as_of:
            ref_date = as_of[:10]  # 取 YYYY-MM-DD 部分
        else:
            ref_date = datetime.now().strftime("%Y-%m-%d")

        path = self._decisions_file()
        if not path.exists():
            return []

        overdue: list[Decision] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                decision = Decision(**json.loads(line))
            except Exception:
                continue

            if decision.status != "pending":
                continue
            if not decision.deadline:
                continue
            # 比较日期字符串（ISO 格式天然支持字典序比较）
            if decision.deadline[:10] < ref_date:
                overdue.append(decision)

        return overdue

    def generate_evaluation_prompt(self, decision_id: str) -> str | None:
        """生成回溯评估 prompt.

        Args:
            decision_id: 决策 ID

        Returns:
            评估 prompt 文本，未找到返回 None
        """
        decision = self.get(decision_id)
        if decision is None:
            return None

        parts = [
            "# 决策回溯评估",
            "",
            f"**决策 ID**: {decision.id}",
            f"**员工**: {decision.employee}",
            f"**类别**: {decision.category}",
            f"**创建时间**: {decision.created_at}",
            "",
            "## 原始决策",
            "",
            decision.content,
            "",
        ]

        if decision.expected_outcome:
            parts.extend(
                [
                    "## 预期结果",
                    "",
                    decision.expected_outcome,
                    "",
                ]
            )

        if decision.actual_outcome:
            parts.extend(
                [
                    "## 实际结果",
                    "",
                    decision.actual_outcome,
                    "",
                ]
            )

        parts.extend(
            [
                "## 评估任务",
                "",
                "请分析：",
                "1. 预期与实际的偏差是什么？",
                "2. 偏差的原因是什么？",
                "3. 未来遇到类似决策时，应该如何调整？",
                "",
                "## 输出格式",
                "",
                "```json",
                "{",
                '  "deviation": "偏差描述",',
                '  "root_cause": "原因分析",',
                '  "lesson": "经验教训（一句话，将写入该员工的持久化记忆）"',
                "}",
                "```",
            ]
        )

        return "\n".join(parts)
