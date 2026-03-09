"""轨迹提炼器 — 使用 Claude 分析轨迹并提取记忆草稿."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TrajectoryExtractor:
    """轨迹提炼器 — 分析轨迹价值并提取记忆."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        value_threshold: float = 0.7,
    ):
        """初始化提炼器.

        Args:
            api_key: Anthropic API key
            model: Claude 模型名称
            value_threshold: 价值评分阈值（0-1），低于此值的轨迹不提炼
        """
        self.api_key = api_key
        self.model = model
        self.value_threshold = value_threshold
        self._client = None

    def _get_client(self):
        """懒加载 Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError as e:
                raise RuntimeError("anthropic SDK 未安装") from e
        return self._client

    def analyze_trajectory(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        """分析轨迹并评估价值.

        Args:
            trajectory: 轨迹数据（包含 task, employee, steps 等字段）

        Returns:
            分析结果: {
                "value_score": float,  # 0-1 价值评分
                "reasoning": str,      # 评分理由
                "should_extract": bool # 是否值得提炼
            }
        """
        employee = trajectory.get("employee", "未知员工")
        task = trajectory.get("task", "")
        if isinstance(task, dict):
            task = task.get("description", "")
        steps = trajectory.get("steps", [])
        success = trajectory.get("success", True)

        # 构建分析 prompt
        prompt = self._build_analysis_prompt(employee, task, steps, success)

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            result = json.loads(content)

            # 验证返回格式
            if not isinstance(result, dict) or "value_score" not in result:
                logger.warning("Claude 返回格式异常: %s", content[:200])
                return {
                    "value_score": 0.0,
                    "reasoning": "分析失败：返回格式异常",
                    "should_extract": False,
                }

            value_score = float(result.get("value_score", 0.0))
            reasoning = result.get("reasoning", "")
            should_extract = value_score >= self.value_threshold

            return {
                "value_score": value_score,
                "reasoning": reasoning,
                "should_extract": should_extract,
            }

        except json.JSONDecodeError as e:
            logger.warning("Claude 返回 JSON 解析失败: %s", e)
            return {
                "value_score": 0.0,
                "reasoning": f"JSON 解析失败: {e}",
                "should_extract": False,
            }
        except Exception as e:
            logger.error("轨迹分析失败: %s", e)
            return {
                "value_score": 0.0,
                "reasoning": f"分析异常: {e}",
                "should_extract": False,
            }

    def extract_memories(self, trajectory: dict[str, Any]) -> list[dict[str, Any]]:
        """从轨迹中提取记忆草稿.

        Args:
            trajectory: 轨迹数据

        Returns:
            记忆草稿列表，每个草稿包含:
            {
                "employee": str,
                "category": str,  # decision/estimate/finding/correction/pattern
                "content": str,
                "tags": list[str],
                "confidence": float,
                "source_trajectory_id": str
            }
        """
        employee = trajectory.get("employee", "未知员工")
        task = trajectory.get("task", "")
        if isinstance(task, dict):
            task = task.get("description", "")
        steps = trajectory.get("steps", [])
        success = trajectory.get("success", True)
        trajectory_id = trajectory.get("task_id", "")

        # 构建提取 prompt
        prompt = self._build_extraction_prompt(employee, task, steps, success)

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            result = json.loads(content)

            # 验证返回格式
            if not isinstance(result, dict) or "memories" not in result:
                logger.warning("Claude 返回格式异常: %s", content[:200])
                return []

            memories = result.get("memories", [])
            if not isinstance(memories, list):
                logger.warning("memories 字段不是列表: %s", type(memories))
                return []

            # 补充字段
            for mem in memories:
                mem["employee"] = employee
                mem["source_trajectory_id"] = trajectory_id
                # 确保必需字段存在
                mem.setdefault("category", "finding")
                mem.setdefault("content", "")
                mem.setdefault("tags", [])
                mem.setdefault("confidence", 1.0)

            return memories

        except json.JSONDecodeError as e:
            logger.warning("Claude 返回 JSON 解析失败: %s", e)
            return []
        except Exception as e:
            logger.error("记忆提取失败: %s", e)
            return []

    def _build_analysis_prompt(
        self,
        employee: str,
        task: str,
        steps: list[dict[str, Any]],
        success: bool,
    ) -> str:
        """构建轨迹价值分析 prompt."""
        steps_summary = self._summarize_steps(steps)
        success_str = "成功" if success else "失败"

        return f"""你是一个轨迹价值评估专家。请分析以下工作轨迹的价值，判断是否值得提炼为记忆。

**员工**: {employee}
**任务**: {task}
**执行结果**: {success_str}
**步骤数**: {len(steps)}

**步骤摘要**:
{steps_summary}

**评估标准**:
1. **高价值 (0.8-1.0)**: 包含重要决策、踩坑教训、成功模式、架构洞察
2. **中等价值 (0.5-0.7)**: 包含有用的技术细节、工作流程、配置经验
3. **低价值 (0.0-0.4)**: 常规操作、简单查询、无特殊价值的重复工作

请返回 JSON 格式:
{{
  "value_score": 0.85,
  "reasoning": "包含重要的 API 设计决策和错误处理模式"
}}
"""

    def _build_extraction_prompt(
        self,
        employee: str,
        task: str,
        steps: list[dict[str, Any]],
        success: bool,
    ) -> str:
        """构建记忆提取 prompt."""
        steps_summary = self._summarize_steps(steps)
        success_str = "成功" if success else "失败"

        return f"""你是一个记忆提炼专家。请从以下工作轨迹中提取值得记录的经验和教训。

**员工**: {employee}
**任务**: {task}
**执行结果**: {success_str}
**步骤数**: {len(steps)}

**步骤详情**:
{steps_summary}

**记忆类别**:
- **decision**: 技术决策、方案选择
- **estimate**: 工作量估算、时间预测
- **finding**: 发现的问题、观察到的现象
- **correction**: 错误纠正、踩坑教训
- **pattern**: 可复用的工作模式、最佳实践

**提取要求**:
1. 每条记忆应该简洁明确（1-2 句话）
2. 包含足够的上下文，让其他人能理解
3. 标注合适的标签（如技术栈、模块名、问题类型）
4. 评估置信度（0-1，基于证据充分程度）

请返回 JSON 格式:
{{
  "memories": [
    {{
      "category": "correction",
      "content": "alembic migration 必须先检查字段是否存在再 add_column，否则重复执行会报错",
      "tags": ["alembic", "migration", "database"],
      "confidence": 1.0
    }}
  ]
}}

如果没有值得记录的内容，返回空列表: {{"memories": []}}
"""

    def _summarize_steps(self, steps: list[dict[str, Any]]) -> str:
        """将步骤列表总结为可读文本."""
        if not steps:
            return "（无步骤记录）"

        lines = []
        for i, step in enumerate(steps[:20], 1):  # 最多显示前 20 步
            tool = step.get("tool_name", "unknown")
            thought = step.get("thought", "")[:100]  # 截取前 100 字符
            output = step.get("tool_output", "")[:100]
            exit_code = step.get("tool_exit_code", 0)
            status = "✓" if exit_code == 0 else "✗"

            lines.append(f"{i}. [{status}] {tool}: {thought}")
            if output:
                lines.append(f"   输出: {output}")

        if len(steps) > 20:
            lines.append(f"... (还有 {len(steps) - 20} 步)")

        return "\n".join(lines)
