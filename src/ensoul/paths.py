"""Path helpers for crew resources."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:  # pragma: no cover - only used on POSIX
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore

_GLOBAL_ENV_VAR = "KNOWLYR_CREW_GLOBAL_DIR"


def resolve_project_dir(project_dir: Path | None = None) -> Path:
    """Resolve project_dir, falling back to cwd if None."""
    return Path(project_dir) if project_dir else Path.cwd()


def _default_global_dir(project_dir: Path | None = None) -> Path:
    return resolve_project_dir(project_dir) / "private" / "employees"


def get_global_dir(project_dir: Path | None = None) -> Path:
    """Return the directory that stores global-level crew resources."""
    configured = os.environ.get(_GLOBAL_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return _default_global_dir(project_dir)


def get_global_templates_dir(project_dir: Path | None = None) -> Path:
    return get_global_dir(project_dir) / "templates"


def get_global_discussions_dir(project_dir: Path | None = None) -> Path:
    return get_global_dir(project_dir) / "discussions"


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """对文件加排他锁（跨进程安全，POSIX only）.

    在目标文件旁创建 .lock 文件，用 fcntl.flock 加排他锁。
    Windows 下退化为无锁（与 lanes.py 行为一致）。
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        if fcntl:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


__all__ = [
    "resolve_project_dir",
    "get_global_dir",
    "get_global_templates_dir",
    "get_global_discussions_dir",
    "file_lock",
]
