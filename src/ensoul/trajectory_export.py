"""轨迹数据集导出系统 — DS640102 格式."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TrajectoryAnnotation(BaseModel):
    """轨迹标注."""

    trajectory_id: str = Field(description="轨迹 ID")
    quality_score: float = Field(description="质量评分 0-1")
    annotator: str = Field(description="标注人")
    annotated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="标注时间",
    )
    notes: str = Field(default="", description="备注")


class TrainingExample(BaseModel):
    """训练样本 — DS640102 格式."""

    input: str = Field(description="输入：任务描述")
    output: str = Field(description="输出：最终结果")
    reasoning: str = Field(description="推理过程：步骤摘要")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")


class TrajectoryExporter:
    """轨迹数据集导出器."""

    def __init__(
        self,
        archive_dir: Path | None = None,
        annotations_dir: Path | None = None,
    ):
        """初始化导出器.

        Args:
            archive_dir: 轨迹归档目录，默认 /data/trajectory_archive
            annotations_dir: 标注存储目录，默认 /data/trajectory_annotations
        """
        self.archive_dir = archive_dir or Path("/data/trajectory_archive")
        self.annotations_dir = annotations_dir or Path("/data/trajectory_annotations")
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

    def load_trajectories(
        self,
        employee: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        min_quality: float = 0.0,
    ) -> list[dict[str, Any]]:
        """加载轨迹数据.

        Args:
            employee: 按员工过滤
            start_date: 起始日期
            end_date: 结束日期
            min_quality: 最低质量分数（需要先标注）

        Returns:
            轨迹列表
        """
        trajectories: list[dict[str, Any]] = []

        if not self.archive_dir.exists():
            return trajectories

        # 加载标注数据
        annotations = self._load_annotations()

        # 遍历归档目录
        for date_dir in sorted(self.archive_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            # 日期过滤
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                if start_date and dir_date < start_date:
                    continue
                if end_date and dir_date > end_date:
                    continue
            except ValueError:
                continue

            # 读取轨迹文件
            for traj_file in date_dir.glob("*.jsonl"):
                try:
                    for line in traj_file.read_text(encoding="utf-8").splitlines():
                        stripped = line.strip()
                        if not stripped:
                            continue

                        traj = json.loads(stripped)

                        # 向后兼容：旧格式是每行一个 step，新格式是每行一个完整轨迹
                        # 新格式有 trajectory_id 和 trajectory 字段
                        if "trajectory_id" in traj and "trajectory" in traj:
                            # 新格式（行业标准）
                            pass
                        elif "step_id" in traj or "tool_name" in traj:
                            # 旧格式（单个 step），跳过
                            logger.debug("跳过旧格式 step: %s", traj_file)
                            continue
                        else:
                            # 未知格式
                            logger.debug("跳过未知格式: %s", traj_file)
                            continue

                        # 员工过滤
                        if employee and traj.get("employee") != employee:
                            continue

                        # 质量过滤
                        traj_id = traj.get("trajectory_id", "")
                        if traj_id in annotations:
                            quality = annotations[traj_id].quality_score
                            if quality < min_quality:
                                continue
                            traj["quality_score"] = quality

                        trajectories.append(traj)

                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug("跳过损坏的轨迹文件: %s, error=%s", traj_file, e)
                    continue

        return trajectories

    def convert_to_training_example(self, trajectory: dict[str, Any]) -> TrainingExample:
        """将轨迹转换为训练样本.

        Args:
            trajectory: 轨迹数据（新格式或旧格式）

        Returns:
            训练样本
        """
        # 提取任务描述
        task = trajectory.get("task", "")
        if isinstance(task, dict):
            task = task.get("description", "")

        # 提取步骤（支持新旧格式）
        steps = trajectory.get("trajectory", trajectory.get("steps", []))

        # 构建推理过程
        reasoning_parts = []
        for i, step in enumerate(steps[:20], 1):  # 最多 20 步
            # 新格式：action.tool，旧格式：tool_name
            if "action" in step and isinstance(step["action"], dict):
                tool = step["action"].get("tool", "unknown")
            else:
                tool = step.get("tool_name", "unknown")

            thought = step.get("thought", "")[:200]  # 截取前 200 字符
            reasoning_parts.append(f"Step {i}: [{tool}] {thought}")

        reasoning = "\n".join(reasoning_parts)

        # 提取最终输出（最后一步的输出）
        output = ""
        if steps:
            last_step = steps[-1]
            # 新格式：result，旧格式：tool_output
            output = last_step.get("result", last_step.get("tool_output", ""))[:500]

        # 元数据
        metadata = {
            "employee": trajectory.get("employee", ""),
            "model": trajectory.get("model", ""),
            "channel": trajectory.get("channel", ""),
            "success": trajectory.get("success", True),
            "total_steps": len(steps),
            "trajectory_id": trajectory.get("trajectory_id", trajectory.get("task_id", "")),
        }

        # 从 metadata 字段提取额外信息（新格式）
        if "metadata" in trajectory and isinstance(trajectory["metadata"], dict):
            traj_meta = trajectory["metadata"]
            metadata["total_tokens"] = traj_meta.get("total_tokens", 0)
            metadata["duration_ms"] = traj_meta.get("duration_ms", 0)
        else:
            metadata["total_tokens"] = trajectory.get("total_tokens", 0)

        if "quality_score" in trajectory:
            metadata["quality_score"] = trajectory["quality_score"]

        return TrainingExample(
            input=task,
            output=output,
            reasoning=reasoning,
            metadata=metadata,
        )

    def export_dataset(
        self,
        output_file: Path,
        employee: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        min_quality: float = 0.0,
        max_samples: int = 0,
    ) -> dict[str, int]:
        """导出数据集.

        Args:
            output_file: 输出文件路径
            employee: 按员工过滤
            start_date: 起始日期
            end_date: 结束日期
            min_quality: 最低质量分数
            max_samples: 最大样本数（0 表示不限制）

        Returns:
            统计信息: {"total": 总数, "exported": 导出数}
        """
        trajectories = self.load_trajectories(
            employee=employee,
            start_date=start_date,
            end_date=end_date,
            min_quality=min_quality,
        )

        if max_samples > 0:
            trajectories = trajectories[:max_samples]

        exported = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for traj in trajectories:
                try:
                    example = self.convert_to_training_example(traj)
                    f.write(example.model_dump_json() + "\n")
                    exported += 1
                except Exception as e:
                    logger.warning("转换轨迹失败: %s", e)
                    continue

        logger.info("导出数据集: file=%s exported=%d", output_file, exported)

        return {"total": len(trajectories), "exported": exported}

    def _load_annotations(self) -> dict[str, TrajectoryAnnotation]:
        """加载所有标注数据.

        Returns:
            标注字典: {trajectory_id: annotation}
        """
        annotations: dict[str, TrajectoryAnnotation] = {}

        if not self.annotations_dir.exists():
            return annotations

        for anno_file in self.annotations_dir.glob("*.json"):
            try:
                data = json.loads(anno_file.read_text(encoding="utf-8"))
                anno = TrajectoryAnnotation(**data)
                annotations[anno.trajectory_id] = anno
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug("跳过损坏的标注文件: %s, error=%s", anno_file, e)
                continue

        return annotations

    def add_annotation(
        self,
        trajectory_id: str,
        quality_score: float,
        annotator: str,
        notes: str = "",
    ) -> TrajectoryAnnotation:
        """添加轨迹标注.

        Args:
            trajectory_id: 轨迹 ID
            quality_score: 质量评分 0-1
            annotator: 标注人
            notes: 备注

        Returns:
            标注对象
        """
        annotation = TrajectoryAnnotation(
            trajectory_id=trajectory_id,
            quality_score=quality_score,
            annotator=annotator,
            notes=notes,
        )

        # 保存到文件
        anno_file = self.annotations_dir / f"{trajectory_id}.json"
        anno_file.write_text(annotation.model_dump_json(indent=2), encoding="utf-8")

        logger.info(
            "添加轨迹标注: id=%s score=%.2f annotator=%s",
            trajectory_id,
            quality_score,
            annotator,
        )

        return annotation

    def get_annotation(self, trajectory_id: str) -> TrajectoryAnnotation | None:
        """获取轨迹标注.

        Args:
            trajectory_id: 轨迹 ID

        Returns:
            标注对象，不存在返回 None
        """
        anno_file = self.annotations_dir / f"{trajectory_id}.json"
        if not anno_file.exists():
            return None

        try:
            data = json.loads(anno_file.read_text(encoding="utf-8"))
            return TrajectoryAnnotation(**data)
        except (json.JSONDecodeError, ValueError):
            return None

    def list_annotations(
        self,
        min_quality: float = 0.0,
        annotator: str | None = None,
    ) -> list[TrajectoryAnnotation]:
        """列出标注.

        Args:
            min_quality: 最低质量分数
            annotator: 按标注人过滤

        Returns:
            标注列表（按时间倒序）
        """
        annotations = list(self._load_annotations().values())

        # 过滤
        filtered = []
        for anno in annotations:
            if anno.quality_score < min_quality:
                continue
            if annotator and anno.annotator != annotator:
                continue
            filtered.append(anno)

        # 按时间倒序
        filtered.sort(key=lambda a: a.annotated_at, reverse=True)
        return filtered
