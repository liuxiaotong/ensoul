"""执行引擎 — 兼容层.

原 engine.py 已拆分为:
- engine_prompt.py: 纯 prompt 生成层（PromptEngine）
- engine_chat.py: LLM 调用 + 上下文编排层（CrewEngine）

本文件保留为兼容层，确保所有 `from ensoul.engine import CrewEngine` 不受影响。
"""

from ensoul.engine_chat import CrewEngine  # noqa: F401
from ensoul.engine_prompt import PromptEngine, _get_git_branch  # noqa: F401

__all__ = ["CrewEngine", "PromptEngine", "_get_git_branch"]
