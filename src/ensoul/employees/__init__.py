"""内置员工目录."""

from pathlib import Path


def builtin_dir() -> Path:
    """返回内置员工目录的绝对路径."""
    return Path(__file__).parent
