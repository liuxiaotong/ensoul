"""权限守卫 — 运行时工具调用拦截 + 审计日志."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ensoul.models import Employee

logger = logging.getLogger(__name__)


class PermissionDenied(Exception):
    """工具调用被权限系统拒绝."""

    def __init__(self, employee_name: str, tool_name: str):
        self.employee_name = employee_name
        self.tool_name = tool_name
        super().__init__(f"员工 '{employee_name}' 无权调用工具 '{tool_name}'")


class ToolAuditLogger:
    """工具调用审计日志 — 记录 who/what/when/allowed.

    日志写入 ``{log_dir}/tool_usage.jsonl``，每行一条 JSON 记录::

        {"ts": 1718000000.0, "employee": "code-reviewer", "tool": "bash", "allowed": false}
    """

    def __init__(self, log_dir: str | Path = ".ensoul/audit") -> None:
        self._log_dir = Path(log_dir)
        self._log_file: Path | None = None

    def _ensure_dir(self) -> Path:
        if self._log_file is None:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_dir / "tool_usage.jsonl"
        return self._log_file

    def log(
        self,
        employee_name: str,
        tool_name: str,
        *,
        allowed: bool,
        **extra: object,
    ) -> None:
        """写入一条审计记录."""
        record = {
            "ts": time.time(),
            "employee": employee_name,
            "tool": tool_name,
            "allowed": allowed,
            **extra,
        }
        try:
            path = self._ensure_dir()
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except (OSError, TypeError):
            logger.debug("审计日志写入失败: %s", record)


# 全局审计日志实例（惰性初始化目录）
_audit_logger: ToolAuditLogger | None = None


def get_audit_logger(log_dir: str | Path = ".ensoul/audit") -> ToolAuditLogger:
    """获取或创建全局审计日志实例."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = ToolAuditLogger(log_dir)
    return _audit_logger


class PermissionGuard:
    """工具调用权限守卫.

    用法::

        guard = PermissionGuard(employee)
        msg = guard.check_soft("bash")  # None 表示允许，否则返回拒绝消息
    """

    # 始终允许的工具（任务完成 / 工具加载）
    _ALWAYS_ALLOWED = {"submit", "finish", "load_tools"}

    def __init__(self, employee: Employee, *, audit: bool = True) -> None:
        from ensoul.tool_schema import resolve_effective_tools

        self.employee_name = employee.name
        self.allowed = resolve_effective_tools(employee) | self._ALWAYS_ALLOWED
        self._audit = audit

    def _log(self, tool_name: str, allowed: bool) -> None:
        if self._audit:
            get_audit_logger().log(self.employee_name, tool_name, allowed=allowed)

    def check(self, tool_name: str) -> None:
        """检查工具调用是否允许，不允许则抛 PermissionDenied."""
        if tool_name not in self.allowed:
            self._log(tool_name, False)
            logger.warning(
                "权限拒绝: %s 尝试调用 %s",
                self.employee_name,
                tool_name,
            )
            raise PermissionDenied(self.employee_name, tool_name)
        self._log(tool_name, True)

    def check_soft(self, tool_name: str) -> str | None:
        """软检查 — 返回 None（允许）或拒绝消息."""
        if tool_name not in self.allowed:
            self._log(tool_name, False)
            return (
                f"工具 '{tool_name}' 不在允许列表中。"
                f"可用工具: {', '.join(sorted(self.allowed - self._ALWAYS_ALLOWED))}"
            )
        self._log(tool_name, True)
        return None
