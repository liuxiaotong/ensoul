"""Skills 系统 — 服务端自动触发机制."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ensoul.paths import file_lock, resolve_project_dir

logger = logging.getLogger(__name__)


class SkillTrigger(BaseModel):
    """Skill 触发配置."""

    type: Literal["semantic", "keyword", "always"] = Field(
        description="触发类型: semantic=语义匹配, keyword=关键词, always=总是触发"
    )
    keywords: list[str] = Field(
        default_factory=list, description="关键词列表（仅 keyword 类型使用）"
    )
    embedding_threshold: float = Field(
        default=0.75, description="语义相似度阈值（仅 semantic 类型使用）"
    )


class SkillAction(BaseModel):
    """Skill 执行动作."""

    type: Literal["query_memory", "load_checklist", "read_wiki", "custom"] = Field(
        description="动作类型"
    )
    params: dict[str, Any] = Field(default_factory=dict, description="动作参数")


class SkillMetadata(BaseModel):
    """Skill 元数据."""

    category: str = Field(default="general", description="分类: safety, quality, efficiency")
    priority: Literal["low", "medium", "high", "critical"] = Field(
        default="medium", description="优先级"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="创建时间"
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="更新时间"
    )


class Skill(BaseModel):
    """Skill 定义."""

    skill_id: str = Field(
        default_factory=lambda: "skill_" + uuid.uuid4().hex[:12], description="唯一 ID"
    )
    name: str = Field(description="Skill 名称")
    version: str = Field(default="0.1.0", description="版本号")
    employee: str = Field(description="所属员工")
    description: str = Field(description="触发条件描述（用于判断是否触发）")
    trigger: SkillTrigger = Field(description="触发配置")
    actions: list[SkillAction] = Field(description="执行动作列表")
    metadata: SkillMetadata = Field(default_factory=SkillMetadata, description="元数据")
    enabled: bool = Field(default=True, description="是否启用")


class SkillTriggerRecord(BaseModel):
    """Skill 触发记录."""

    trigger_id: str = Field(
        default_factory=lambda: "trig_" + uuid.uuid4().hex[:12], description="触发 ID"
    )
    skill_id: str = Field(description="Skill ID")
    employee: str = Field(description="员工名称")
    task: str = Field(description="任务描述")
    match_score: float = Field(description="匹配分数")
    triggered_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="触发时间"
    )
    execution_status: Literal["success", "failure", "skipped"] = Field(description="执行状态")
    execution_time_ms: int = Field(default=0, description="执行时间（毫秒）")
    actions_executed: int = Field(default=0, description="执行的动作数量")
    result: dict[str, Any] = Field(default_factory=dict, description="执行结果")


class SkillStore:
    """Skills 存储管理器.

    存储结构:
      {skills_dir}/
        {employee_name}/
          {skill_name}.json   — 每个 skill 一个文件
        triggers/
          {date}.jsonl        — 触发历史按日期分文件
    """

    def __init__(
        self,
        skills_dir: Path | None = None,
        *,
        project_dir: Path | None = None,
    ):
        self._project_dir = project_dir
        self.skills_dir = (
            skills_dir
            if skills_dir is not None
            else resolve_project_dir(project_dir) / ".ensoul" / "skills"
        )
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.triggers_dir = self.skills_dir / "triggers"
        self.triggers_dir.mkdir(parents=True, exist_ok=True)

    @property
    def project_dir(self) -> Path:
        """获取项目目录."""
        return resolve_project_dir(self._project_dir)

    def create_skill(self, skill: Skill) -> Skill:
        """创建 Skill."""
        employee_dir = self.skills_dir / skill.employee
        employee_dir.mkdir(parents=True, exist_ok=True)

        skill_file = employee_dir / f"{skill.name}.json"
        if skill_file.exists():
            raise ValueError(f"Skill {skill.name} already exists for {skill.employee}")

        with file_lock(skill_file):
            skill_file.write_text(skill.model_dump_json(indent=2), encoding="utf-8")

        logger.info(f"Created skill {skill.name} for {skill.employee}")
        return skill

    def get_skill(self, employee: str, skill_name: str) -> Skill | None:
        """获取 Skill."""
        skill_file = self.skills_dir / employee / f"{skill_name}.json"
        if not skill_file.exists():
            return None

        with file_lock(skill_file):
            data = json.loads(skill_file.read_text(encoding="utf-8"))
            return Skill(**data)

    def list_skills(self, employee: str | None = None) -> list[Skill]:
        """列出 Skills."""
        skills = []

        if employee:
            # 列出指定员工的 skills
            employee_dir = self.skills_dir / employee
            if employee_dir.exists():
                for skill_file in employee_dir.glob("*.json"):
                    try:
                        data = json.loads(skill_file.read_text(encoding="utf-8"))
                        skills.append(Skill(**data))
                    except Exception as e:
                        logger.warning(f"Failed to load skill {skill_file}: {e}")
        else:
            # 列出所有员工的 skills
            for employee_dir in self.skills_dir.iterdir():
                if employee_dir.is_dir() and employee_dir.name != "triggers":
                    for skill_file in employee_dir.glob("*.json"):
                        try:
                            data = json.loads(skill_file.read_text(encoding="utf-8"))
                            skills.append(Skill(**data))
                        except Exception as e:
                            logger.warning(f"Failed to load skill {skill_file}: {e}")

        return skills

    def update_skill(self, employee: str, skill_name: str, updates: dict[str, Any]) -> Skill:
        """更新 Skill."""
        skill = self.get_skill(employee, skill_name)
        if not skill:
            raise ValueError(f"Skill {skill_name} not found for {employee}")

        # 更新字段
        for key, value in updates.items():
            if hasattr(skill, key):
                setattr(skill, key, value)

        # 更新 updated_at
        skill.metadata.updated_at = datetime.now().isoformat()

        # 保存
        skill_file = self.skills_dir / employee / f"{skill_name}.json"
        with file_lock(skill_file):
            skill_file.write_text(skill.model_dump_json(indent=2), encoding="utf-8")

        logger.info(f"Updated skill {skill_name} for {employee}")
        return skill

    def delete_skill(self, employee: str, skill_name: str) -> bool:
        """删除 Skill."""
        skill_file = self.skills_dir / employee / f"{skill_name}.json"
        if not skill_file.exists():
            return False

        skill_file.unlink()
        logger.info(f"Deleted skill {skill_name} for {employee}")
        return True

    def record_trigger(self, record: SkillTriggerRecord) -> None:
        """记录触发历史."""
        date = datetime.now().strftime("%Y-%m-%d")
        trigger_file = self.triggers_dir / f"{date}.jsonl"

        with file_lock(trigger_file):
            with trigger_file.open("a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")

    def get_trigger_history(
        self,
        employee: str | None = None,
        skill_name: str | None = None,
        limit: int = 50,
        since: str | None = None,
    ) -> list[SkillTriggerRecord]:
        """获取触发历史."""
        records = []

        # 读取所有触发记录文件
        for trigger_file in sorted(self.triggers_dir.glob("*.jsonl"), reverse=True):
            if len(records) >= limit:
                break

            # 如果指定了 since，跳过更早的文件
            if since:
                file_date = trigger_file.stem
                if file_date < since[:10]:  # 比较日期部分
                    continue

            with file_lock(trigger_file):
                for line in trigger_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue

                    try:
                        record = SkillTriggerRecord(**json.loads(line))

                        # 过滤条件
                        if employee and record.employee != employee:
                            continue
                        if skill_name:
                            skill = self.get_skill(record.employee, skill_name)
                            if not skill or skill.skill_id != record.skill_id:
                                continue
                        if since and record.triggered_at < since:
                            continue

                        records.append(record)

                        if len(records) >= limit:
                            break
                    except Exception as e:
                        logger.warning(f"Failed to parse trigger record: {e}")

        return records[:limit]

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息."""
        all_skills = self.list_skills()

        stats = {
            "total_skills": len(all_skills),
            "by_employee": {},
            "by_category": {},
            "by_priority": {},
            "enabled_count": 0,
        }

        for skill in all_skills:
            # 按员工统计
            stats["by_employee"][skill.employee] = stats["by_employee"].get(skill.employee, 0) + 1

            # 按分类统计
            category = skill.metadata.category
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

            # 按优先级统计
            priority = skill.metadata.priority
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1

            # 启用数量
            if skill.enabled:
                stats["enabled_count"] += 1

        return stats
