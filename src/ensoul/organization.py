"""组织架构加载 — 从 organization.yaml 读取团队、权限、路由 + 自动降级."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

from ensoul.models import Organization

logger = logging.getLogger(__name__)

_cache: Organization | None = None
_cache_time: float = 0.0
_cache_lock = threading.Lock()
_CACHE_TTL = float(os.environ.get("ENSOUL_ORG_CACHE_TTL", "30"))  # seconds


def load_organization(project_dir: Path | None = None) -> Organization:
    """加载组织架构配置.

    搜索顺序：
    1. {project_dir}/private/organization.yaml
    2. {project_dir}/.ensoul/organization.yaml
    3. 返回空 Organization（向后兼容）
    """
    global _cache, _cache_time

    with _cache_lock:
        if _cache is not None and (time.monotonic() - _cache_time) < _CACHE_TTL:
            return _cache

        candidates: list[Path] = []
        if project_dir:
            candidates.append(project_dir / "private" / "organization.yaml")
            candidates.append(project_dir / ".ensoul" / "organization.yaml")

        for path in candidates:
            if path.is_file():
                try:
                    data = yaml.safe_load(path.read_text(encoding="utf-8"))
                    org = Organization(**(data or {}))
                    logger.info("组织架构已加载: %s", path)
                    _cache = org
                    _cache_time = time.monotonic()
                    return org
                except Exception as e:
                    logger.warning("组织架构加载失败 (%s): %s", path, e)

        org = Organization()
        _cache = org
        _cache_time = time.monotonic()
        return org


def invalidate_cache() -> None:
    """清除缓存（测试用）."""
    global _cache, _cache_time
    with _cache_lock:
        _cache = None
        _cache_time = 0.0


def set_cache(org: Organization) -> None:
    """注入组织实例到缓存（测试用）."""
    global _cache, _cache_time
    with _cache_lock:
        _cache = org
        _cache_time = time.monotonic()


# ── 模型档位默认值 ──

_MODEL_TIER_FIELDS = (
    "model",
    "api_key",
    "base_url",
    "fallback_model",
    "fallback_api_key",
    "fallback_base_url",
)


def apply_model_defaults(
    employees: dict[str, Employee],  # noqa: F821
    org: Organization,
) -> None:
    """按 model_tier 从 organization.model_defaults 填充员工空字段.

    只填充空字段，不覆盖已有值（employee.yaml 优先）。
    """
    if not org.model_defaults:
        return

    for emp in employees.values():
        if not emp.model_tier:
            continue
        tier = org.model_defaults.get(emp.model_tier)
        if not tier:
            logger.debug("未知 model_tier: %s (employee=%s)", emp.model_tier, emp.name)
            continue
        for field in _MODEL_TIER_FIELDS:
            if not getattr(emp, field):
                object.__setattr__(emp, field, getattr(tier, field))


# ── 自动降级 ──
# 规则：连续 N 次任务失败 → 权限从 A 降到 C
# 只降不升，升级由管理员手动决定

_DOWNGRADE_THRESHOLD = 3  # 连续失败次数阈值
_DOWNGRADE_MAP = {"A": "C", "B": "C"}  # 降级映射

# 内存追踪：{employee_name: consecutive_failure_count}
_failure_tracker: dict[str, int] = {}
_failure_lock = threading.Lock()

# 权限覆盖：{employee_name: {"level": "C", "original": "A", "reason": "...", "since": "..."}}
_authority_overrides: dict[str, dict[str, str]] = {}


def _overrides_path(project_dir: Path | None) -> Path | None:
    """权限覆盖文件路径."""
    if project_dir is None:
        return None
    return project_dir / "private" / "authority_overrides.json"


def _load_overrides(project_dir: Path | None = None) -> dict[str, dict[str, str]]:
    """加载权限覆盖."""
    global _authority_overrides
    path = _overrides_path(project_dir)
    if path and path.is_file():
        try:
            _authority_overrides = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("加载权限覆盖失败: %s", e)
    return _authority_overrides


def _save_overrides(project_dir: Path | None) -> None:
    """持久化权限覆盖."""
    path = _overrides_path(project_dir)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_authority_overrides, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("保存权限覆盖失败: %s", e)


def get_effective_authority(
    org: Organization,
    employee_name: str,
    project_dir: Path | None = None,
) -> str | None:
    """获取生效的权限级别（覆盖优先于静态配置）."""
    overrides = _load_overrides(project_dir)
    override = overrides.get(employee_name)
    if override:
        return override["level"]
    return org.get_authority(employee_name)


def record_task_outcome(
    employee_name: str,
    success: bool,
    project_dir: Path | None = None,
) -> str | None:
    """记录任务结果，检查是否触发降级.

    Returns:
        降级信息字符串（如果触发），否则 None.
    """
    with _failure_lock:
        if success:
            _failure_tracker.pop(employee_name, None)
            return None

        count = _failure_tracker.get(employee_name, 0) + 1
        _failure_tracker[employee_name] = count

        if count < _DOWNGRADE_THRESHOLD:
            return None

        # 检查是否可以降级
        org = load_organization(project_dir=project_dir)
        current = org.get_authority(employee_name)
        if current is None or current not in _DOWNGRADE_MAP:
            return None

        # 已经被覆盖过的不再重复降级
        overrides = _load_overrides(project_dir)
        if employee_name in overrides:
            return None

        new_level = _DOWNGRADE_MAP[current]
        _authority_overrides[employee_name] = {
            "level": new_level,
            "original": current,
            "reason": f"连续 {count} 次任务失败",
            "since": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        _save_overrides(project_dir)
        _failure_tracker.pop(employee_name, None)

        msg = f"{employee_name} 权限从 {current} 降至 {new_level}（连续 {count} 次失败）"
        logger.warning(msg)
        return msg


def reset_overrides() -> None:
    """清除所有覆盖（测试用）."""
    global _authority_overrides
    _authority_overrides.clear()
    _failure_tracker.clear()
