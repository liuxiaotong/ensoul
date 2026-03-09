"""Lane-based 串行调度工具."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from ensoul.paths import resolve_project_dir

try:  # pragma: no cover - only used on POSIX
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore

try:  # pragma: no cover - only used on Windows
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None  # type: ignore


class LaneLock:
    """基于文件锁的 Lane."""

    def __init__(
        self, lane_name: str, root: Path | None = None, *, project_dir: Path | None = None
    ):
        sanitized = lane_name.replace("/", "_").replace(":", "_")
        base = root if root is not None else resolve_project_dir(project_dir) / ".ensoul" / "lanes"
        self.path = base / f"{sanitized}.lock"
        self._fh: Any | None = None
        self._msvcrt_locked = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            elif msvcrt:
                self._lock_with_msvcrt(fh)
            fh.write(f"pid={os.getpid()}\n")
            fh.flush()
            self._fh = fh
        except Exception:
            fh.close()
            raise

    def release(self) -> None:
        if self._fh is None:
            return
        if fcntl:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        elif msvcrt and self._msvcrt_locked:
            try:
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self._msvcrt_locked = False
        self._fh.close()
        self._fh = None

    def __enter__(self) -> LaneLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self) -> None:
        if self._fh is not None:
            try:
                self.release()
            except Exception:
                pass

    def _lock_with_msvcrt(self, fh) -> None:
        """Windows msvcrt 锁实现."""
        fh.seek(0, os.SEEK_END)
        if fh.tell() == 0:
            fh.write(" ")
            fh.flush()
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        self._msvcrt_locked = True


@contextmanager
def lane_lock(
    lane_name: str, enabled: bool = True, *, project_dir: Path | None = None
) -> Iterator[None]:
    """按 lane_name 串行化执行."""
    if not enabled:
        with nullcontext():
            yield
        return
    lock = LaneLock(lane_name, project_dir=project_dir)
    with lock:
        yield


__all__ = ["lane_lock", "LaneLock"]
