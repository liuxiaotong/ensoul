"""Custom exception hierarchy for knowlyr-crew."""


class CrewError(Exception):
    """Base exception for all crew errors."""


class EmployeeNotFoundError(CrewError):
    """Employee not found by name or trigger."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"未找到员工: {name}")


class EmployeeValidationError(CrewError):
    """Employee definition failed validation."""

    def __init__(self, name: str, errors: list[str]):
        self.name = name
        self.errors = errors
        super().__init__(f"员工 '{name}' 校验失败: {'; '.join(errors)}")


class PipelineNotFoundError(CrewError):
    """Pipeline not found."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"未找到 pipeline: {name}")


class PipelineError(CrewError):
    """Pipeline execution error."""


class ExecutorError(CrewError):
    """LLM executor error."""


class ProviderError(CrewError, ValueError):
    """LLM provider detection/key resolution error.

    Inherits ValueError for backward compatibility with existing
    except (ValueError, ImportError) blocks.
    """


class TaskRegistryError(CrewError):
    """Task registry persistence error."""
