"""Cron 配置 — 定时任务调度规则."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from ensoul.paths import resolve_project_dir


class DeliveryTarget(BaseModel):
    """投递目标（嵌入 cron 配置）."""

    type: Literal["webhook", "email", "feishu"] = Field(description="投递类型")
    url: str = Field(default="", description="Webhook URL / 飞书机器人 Webhook URL")
    headers: dict[str, str] = Field(default_factory=dict, description="自定义请求头")
    secret: str = Field(default="", description="飞书机器人签名密钥（可选）")
    to: str = Field(default="", description="收件人邮箱")
    subject: str = Field(default="", description="邮件主题（支持 {name} 占位符）")


class CronSchedule(BaseModel):
    """单条定时任务."""

    name: str = Field(description="任务名称（唯一标识）")
    cron: str = Field(description="cron 表达式，如 '0 9 * * *'")
    target_type: Literal["pipeline", "employee"] = Field(description="目标类型")
    target_name: str = Field(description="pipeline 或 employee 名称")
    args: dict[str, str] = Field(default_factory=dict, description="参数")
    delivery: list[DeliveryTarget] = Field(default_factory=list, description="投递目标")


class CronConfig(BaseModel):
    """Cron 调度器配置."""

    schedules: list[CronSchedule] = Field(default_factory=list, description="定时任务列表")


def load_cron_config(project_dir: Path | None = None) -> CronConfig:
    """从 .ensoul/cron.yaml 加载配置."""
    base = resolve_project_dir(project_dir)
    config_path = base / ".ensoul" / "cron.yaml"
    if not config_path.exists():
        return CronConfig()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not data:
        return CronConfig()
    return CronConfig(**data)


def validate_cron_config(config: CronConfig, project_dir: Path | None = None) -> list[str]:
    """校验 cron 配置.

    检查：
    - cron 表达式是否合法
    - 目标 pipeline / employee 是否存在
    - name 是否唯一
    """
    errors: list[str] = []

    # 检查 croniter 可用
    try:
        from croniter import croniter
    except ImportError:
        errors.append("croniter 未安装。请运行: pip install croniter")
        return errors

    # name 唯一性
    seen_names: set[str] = set()
    for schedule in config.schedules:
        if schedule.name in seen_names:
            errors.append(f"重复的任务名称: {schedule.name}")
        seen_names.add(schedule.name)

        # cron 表达式
        if not croniter.is_valid(schedule.cron):
            errors.append(f"无效的 cron 表达式 '{schedule.cron}'（任务: {schedule.name}）")

    # 检查目标是否存在
    try:
        from ensoul.discovery import discover_employees

        discovery = discover_employees(project_dir=project_dir)
        employee_names = set(discovery.employees.keys())
    except ImportError:
        employee_names = set()

    try:
        from ensoul.pipeline import discover_pipelines

        pipelines = discover_pipelines(project_dir=project_dir)
    except ImportError:
        pipelines = {}

    for schedule in config.schedules:
        if schedule.target_type == "pipeline" and schedule.target_name not in pipelines:
            errors.append(f"未找到 pipeline: {schedule.target_name}（任务: {schedule.name}）")
        elif schedule.target_type == "employee" and schedule.target_name not in employee_names:
            errors.append(f"未找到员工: {schedule.target_name}（任务: {schedule.name}）")

    return errors
