"""轨迹录制 — 将 ensoul 员工执行数据写为 agentrecorder 标准 Trajectory 格式.

通过 contextvars 实现零侵入式录制：调用方用 TrajectoryCollector 上下文管理器
包裹执行流程，executor / agent_bridge 自动检测并记录每一步。

用法::

    with TrajectoryCollector("code-reviewer", "审查代码", model="claude-sonnet") as tc:
        result = execute_prompt(system_prompt=prompt, ...)
    traj = tc.finish()  # 写入 .ensoul/trajectories/trajectories.jsonl

依赖 agentrecorder 时输出标准 Trajectory JSONL；未安装时 fallback 为纯 JSON。
"""

from __future__ import annotations

import contextvars
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_collector_var: contextvars.ContextVar[TrajectoryCollector | None] = contextvars.ContextVar(
    "trajectory_collector", default=None
)

# slug → 中文名 缓存（进程级）
_NAME_CACHE: dict[str, str] = {}


def resolve_character_name(slug_or_name: str, *, project_dir: Path | None = None) -> str:
    """将 slug 统一映射为中文名 (character_name).

    如果已经是中文名则原样返回。discovery 不可用时返回原始值。
    """
    if not slug_or_name:
        return slug_or_name
    # 基本 CJK 区判断（\u4e00-\u9fff），覆盖所有常用中文字符，满足员工名检测需求
    if any("\u4e00" <= c <= "\u9fff" for c in slug_or_name):
        return slug_or_name
    # 缓存上限保护（正常 ~33 员工，超 200 说明有异常数据，清空重建）
    if len(_NAME_CACHE) > 200:
        _NAME_CACHE.clear()
    # 缓存命中
    if slug_or_name in _NAME_CACHE:
        return _NAME_CACHE[slug_or_name]
    # 查 discovery
    try:
        from ensoul.discovery import discover_employees

        discovery = discover_employees(project_dir=project_dir)
        emp = discovery.get(slug_or_name)
        if emp and emp.character_name:
            _NAME_CACHE[slug_or_name] = emp.character_name
            return emp.character_name
    except (ImportError, Exception):
        pass
    return slug_or_name


def is_hollow_trajectory(data: dict[str, Any]) -> bool:
    """判断轨迹是否为空壳（steps 全空或无实质内容）.

    公共函数，供 daily_eval.py / clean_trajectories.py 复用。
    """
    steps = data.get("steps", [])
    if not steps:
        return True
    # 全部 step 的 tool 都是 unknown/空 且 output 为空 → 空壳
    for s in steps:
        tool = s.get("tool") or (s.get("tool_call", {}) or {}).get("name") or s.get("tool_name", "")
        output = (
            s.get("output")
            or (s.get("tool_result", {}) or {}).get("output")
            or s.get("tool_output", "")
        )
        token_count = s.get("token_count", 0) or 0
        if tool not in ("unknown", "") or output or token_count > 0:
            return False
    return True


def extract_task_from_soul_prompt(text: str) -> str | None:
    """尝试从 soul prompt 中提取 ## 任务 之后的实际任务描述.

    公共函数，供 daily_eval.py / clean_trajectories.py / webhook_handlers.py 复用。
    返回提取到的任务文本，提取不到返回 None。
    """
    import re

    m = re.search(r"##\s*(?:本次)?任务\s*\n+(.+)", text, re.DOTALL)
    if m:
        task_text = m.group(1).strip()
        next_section = re.search(r"\n##\s", task_text)
        if next_section:
            task_text = task_text[: next_section.start()].strip()
        if task_text:
            return task_text
    return None


def _try_import_recorder():
    """尝试导入 agentrecorder，不可用时返回 None."""
    try:
        from agentrecorder.schema import (
            Outcome,
            Step,
            Trajectory,
        )
        from agentrecorder.schema import ToolCall as RecToolCall
        from agentrecorder.schema import TaskInfo
        from agentrecorder.schema import ToolResult as RecToolResult

        return Trajectory, Step, RecToolCall, RecToolResult, Outcome, TaskInfo
    except ImportError:
        return None


class TrajectoryCollector:
    """收集 LLM 调用数据，输出 agentrecorder 标准 Trajectory.

    使用 contextvars 实现跨函数传递，executor.py / agent_bridge.py 自动检测并记录。
    同时支持 sync 和 async 上下文。
    """

    def __init__(
        self,
        employee_name: str,
        task_description: str,
        model: str = "",
        *,
        channel: str = "cli",
        output_dir: Path | None = None,
    ):
        self.employee_name = employee_name
        self.task_description = task_description
        self.model = model
        self.channel = channel
        self.output_dir = output_dir or Path(".ensoul/trajectories")
        self._steps: list[dict[str, Any]] = []
        self._step_count = 0
        self._token: contextvars.Token | None = None
        self._pending: dict[str, Any] | None = None

    @classmethod
    def create_for_employee(
        cls,
        employee_name: str,
        task_description: str,
        channel: str,
        project_dir: Path,
    ) -> TrajectoryCollector:
        """工厂方法：resolve 中文名 + 设置 output_dir."""
        char_name = resolve_character_name(employee_name, project_dir=project_dir)
        return cls(
            char_name,
            task_description[:200],
            channel=channel,
            output_dir=project_dir / ".ensoul" / "trajectories",
        )

    @classmethod
    def try_create_for_employee(
        cls,
        employee_name: str,
        task_description: str,
        channel: str,
        project_dir: Path,
    ) -> TrajectoryCollector | None:
        """工厂方法：尝试创建 collector，失败时返回 None 而非抛异常."""
        try:
            return cls.create_for_employee(
                employee_name,
                task_description,
                channel=channel,
                project_dir=project_dir,
            )
        except Exception:
            return None

    def __enter__(self) -> TrajectoryCollector:
        self._token = _collector_var.set(self)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            _collector_var.reset(self._token)
            self._token = None

    @staticmethod
    def current() -> TrajectoryCollector | None:
        """获取当前上下文中的 collector（如果有）."""
        return _collector_var.get(None)

    # ── 单轮录制（execute_prompt） ──

    def add_prompt_step(
        self,
        content: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """记录 execute_prompt() 的单轮执行结果."""
        self._step_count += 1
        if not self.model:
            self.model = model
        self._steps.append(
            {
                "step_id": self._step_count,
                "thought": content,
                "tool_name": "respond",
                "tool_params": {},
                "tool_output": content,
                "tool_exit_code": 0,
                "timestamp": datetime.now().isoformat(),
                "token_count": input_tokens + output_tokens,
            }
        )

    # ── 多轮录制（execute_with_tools，分两步） ──

    def begin_tool_step(
        self,
        thought: str,
        tool_name: str,
        tool_params: dict[str, Any],
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
    ) -> None:
        """开始记录一个 tool-use 步骤（等待 complete_tool_step 提供工具执行结果）."""
        # 如果有上一个未完成的 pending step，先保存
        if self._pending is not None:
            self._pending.setdefault("tool_output", "")
            self._pending.setdefault("tool_exit_code", -1)
            self._steps.append(self._pending)
        self._step_count += 1
        if model and not self.model:
            self.model = model
        self._pending = {
            "step_id": self._step_count,
            "thought": thought,
            "tool_name": tool_name,
            "tool_params": tool_params,
            "timestamp": datetime.now().isoformat(),
            "token_count": input_tokens + output_tokens,
        }

    def complete_tool_step(self, tool_output: str, tool_exit_code: int = 0) -> None:
        """补全 pending 步骤的工具执行结果."""
        if self._pending is None:
            return
        self._pending["tool_output"] = tool_output
        self._pending["tool_exit_code"] = tool_exit_code
        self._steps.append(self._pending)
        self._pending = None

    # ── 临时文件存储（2026-03-02 记忆系统优化）──

    def _write_temp_file(self) -> None:
        """将工具调用序列写入临时文件 /data/trajectory_temp/{date}/session-{id}.jsonl."""
        try:
            from datetime import date as dt_date

            temp_base = Path("/data/trajectory_temp")
            date_str = dt_date.today().isoformat()
            date_dir = temp_base / date_str
            date_dir.mkdir(parents=True, exist_ok=True)

            session_id = f"{self.employee_name}-{uuid.uuid4().hex[:8]}"
            temp_file = date_dir / f"session-{session_id}.jsonl"

            with open(temp_file, "w", encoding="utf-8") as f:
                for step in self._steps:
                    f.write(json.dumps(step, ensure_ascii=False) + "\n")

            logger.debug("临时轨迹已写入: %s (%d 步)", temp_file, len(self._steps))
        except Exception as e:
            logger.warning("写入临时轨迹文件失败: %s", e)

    # ── 完整步骤一次性录制 ──

    def add_tool_step(
        self,
        thought: str,
        tool_name: str,
        tool_params: dict[str, Any],
        tool_output: str,
        tool_exit_code: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """记录一个已完成的 tool-use 步骤（含工具执行结果）."""
        self._step_count += 1
        self._steps.append(
            {
                "step_id": self._step_count,
                "thought": thought,
                "tool_name": tool_name,
                "tool_params": tool_params,
                "tool_output": tool_output,
                "tool_exit_code": tool_exit_code,
                "timestamp": datetime.now().isoformat(),
                "token_count": input_tokens + output_tokens,
            }
        )

    # ── 输出 ──

    def finish(self, *, success: bool = True, score: float = 0.0) -> Any:
        """完成录制，写入 Trajectory JSONL.

        Returns:
            agentrecorder Trajectory 对象（已安装时），否则返回 dict。
        """
        # 保存未完成的 pending step
        if self._pending is not None:
            self._pending.setdefault("tool_output", "")
            self._pending.setdefault("tool_exit_code", -1)
            self._steps.append(self._pending)
            self._pending = None

        if not self._steps:
            logger.debug("无步骤数据，跳过轨迹录制")
            return None

        # 写入临时文件（2026-03-02 记忆系统优化）
        self._write_temp_file()

        total_tokens = sum(s.get("token_count", 0) or 0 for s in self._steps)

        imports = _try_import_recorder()
        if imports is not None:
            Trajectory, Step, RecToolCall, RecToolResult, Outcome, TaskInfo = imports
            steps = [
                Step(
                    step_id=s["step_id"],
                    thought=s["thought"],
                    tool_call=RecToolCall(
                        name=s["tool_name"],
                        parameters=s["tool_params"],
                    ),
                    tool_result=RecToolResult(
                        output=s["tool_output"],
                        exit_code=s["tool_exit_code"],
                    ),
                    timestamp=s["timestamp"],
                    token_count=s.get("token_count"),
                )
                for s in self._steps
            ]
            # task 统一为 string（不包装成 TaskInfo dict），
            # 避免下游 daily_eval 等消费端遇到 dict 格式的 task
            task_str = self.task_description
            if isinstance(task_str, dict):
                task_str = task_str.get("description", "") or str(task_str)
            traj = Trajectory(
                task=TaskInfo(
                    task_id=f"ensoul-{uuid.uuid4().hex[:8]}",
                    description=task_str,
                    domain="ensoul",
                ),
                agent=f"ensoul/{self.employee_name}",
                model=self.model,
                steps=steps,
                outcome=Outcome(
                    success=success,
                    score=score,
                    total_steps=len(steps),
                    total_tokens=total_tokens,
                ),
                metadata={"employee": self.employee_name, "channel": self.channel},
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / "trajectories.jsonl"
            traj.to_jsonl(output_path)
            logger.info("轨迹已录制: %s (%d 步)", output_path, len(steps))
            return traj

        # fallback: agentrecorder 未安装，写纯 JSON
        # task 统一为 string
        task_str = self.task_description
        if isinstance(task_str, dict):
            task_str = task_str.get("description", "") or str(task_str)
        data = {
            "task_id": f"ensoul-{uuid.uuid4().hex[:8]}",
            "employee": self.employee_name,
            "channel": self.channel,
            "task": task_str,
            "model": self.model,
            "steps": self._steps,
            "success": success,
            "total_steps": len(self._steps),
            "total_tokens": total_tokens,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "trajectories.jsonl"
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        logger.info("轨迹已录制 (fallback JSON): %s (%d 步)", output_path, len(self._steps))
        return data
