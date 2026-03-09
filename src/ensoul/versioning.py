"""自动版本管理 — 内容哈希 + patch 自动递增 + 版本回滚."""

import hashlib
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def compute_content_hash(dir_path: Path) -> str:
    """计算目录内内容文件的 SHA256.

    只读 soul.md（员工唯一定义文件）。

    Args:
        dir_path: 员工目录路径

    Returns:
        SHA256 前 8 位十六进制字符串
    """
    hasher = hashlib.sha256()

    soul_path = dir_path / "soul.md"
    if soul_path.exists():
        hasher.update(soul_path.read_bytes())

    return hasher.hexdigest()[:8]


def _bump_patch(version: str) -> str:
    """将版本号的 patch 部分 +1.

    支持 "3.0" → "3.0.1" 和 "3.0.1" → "3.0.2" 格式。

    Args:
        version: 当前版本号

    Returns:
        递增后的版本号
    """
    parts = version.split(".")
    if len(parts) < 3:
        parts.append("1")
    else:
        parts[2] = str(int(parts[2]) + 1)
    return ".".join(parts)


def check_and_bump(dir_path: Path) -> tuple[str, bool]:
    """检查内容是否变更，若变更则 bump patch 并回写 employee.yaml.

    Args:
        dir_path: 员工目录路径

    Returns:
        (version, bumped) — 当前版本号和是否发生了 bump
    """
    config_path = dir_path / "employee.yaml"
    if not config_path.exists():
        return ("1.0", False)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        return ("1.0", False)
    version = str(config.get("version", "1.0"))
    stored_hash = config.get("_content_hash", "")

    current_hash = compute_content_hash(dir_path)

    if current_hash == stored_hash:
        return (version, False)

    # 内容变更 → bump patch
    new_version = _bump_patch(version)
    config["version"] = new_version
    config["_content_hash"] = current_hash

    try:
        import os
        import tempfile

        content = yaml.dump(config, allow_unicode=True, sort_keys=False, default_flow_style=False)
        fd, tmp = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        fd_closed = False
        try:
            os.write(fd, content.encode("utf-8"))
            os.close(fd)
            fd_closed = True
            os.replace(tmp, config_path)
        except Exception:
            if not fd_closed:
                os.close(fd)
            Path(tmp).unlink(missing_ok=True)
            raise
        logger.info("版本 bump: %s → %s (hash: %s)", version, new_version, current_hash)
    except OSError as e:
        logger.warning("无法回写 %s: %s", config_path, e)
        return (version, False)

    return (new_version, True)


# ── 版本历史与回滚 ──


@dataclass
class VersionEntry:
    """一条版本历史记录."""

    version: str
    commit_hash: str
    date: str
    message: str


def _git_repo_root(path: Path) -> Path | None:
    """获取 git 仓库根目录."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return Path(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def list_employee_versions(dir_path: Path) -> list[VersionEntry]:
    """列出员工目录的 git 历史版本.

    Returns:
        按时间倒序排列的版本列表（最新在前）
    """
    repo_root = _git_repo_root(dir_path)
    if not repo_root:
        return []

    try:
        rel_path = dir_path.relative_to(repo_root)
    except ValueError:
        return []

    # 获取该目录的 git 提交历史
    try:
        out = subprocess.check_output(
            ["git", "log", "--format=%H\t%ai\t%s", "--", str(rel_path)],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    versions: list[VersionEntry] = []
    for line in out.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        commit_hash, date_str, message = parts
        date = date_str[:10]  # YYYY-MM-DD

        # 从该 commit 读取 employee.yaml 的 version 字段
        version = _read_version_at_commit(repo_root, rel_path, commit_hash)
        versions.append(
            VersionEntry(
                version=version,
                commit_hash=commit_hash,
                date=date,
                message=message,
            )
        )

    return versions


def _read_version_at_commit(repo_root: Path, rel_path: Path, commit: str) -> str:
    """从指定 commit 读取 employee.yaml 的 version 字段."""
    yaml_path = f"{rel_path}/employee.yaml"
    try:
        content = subprocess.check_output(
            ["git", "show", f"{commit}:{yaml_path}"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        config = yaml.safe_load(content)
        if isinstance(config, dict):
            return str(config.get("version", "?"))
    except (subprocess.CalledProcessError, Exception):
        pass
    return "?"


def rollback_to(dir_path: Path, commit_hash: str) -> str:
    """从 git 历史恢复员工目录到指定 commit 的状态.

    只恢复 employee.yaml、prompt.md、soul.md 等内容文件，
    不影响其他文件。

    Returns:
        恢复后的版本号
    """
    repo_root = _git_repo_root(dir_path)
    if not repo_root:
        raise RuntimeError("不在 git 仓库中")

    try:
        rel_path = dir_path.relative_to(repo_root)
    except ValueError:
        raise RuntimeError(f"路径 {dir_path} 不在仓库 {repo_root} 中")

    # 获取该 commit 中该目录下的所有文件
    try:
        out = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", commit_hash, str(rel_path) + "/"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(f"无法读取 commit {commit_hash[:8]} 的文件列表")

    restored_files = []
    for file_path in out.strip().splitlines():
        if not file_path:
            continue
        try:
            content = subprocess.check_output(
                ["git", "show", f"{commit_hash}:{file_path}"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
            target = repo_root / file_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            restored_files.append(file_path)
        except subprocess.CalledProcessError:
            logger.warning("跳过无法恢复的文件: %s", file_path)

    logger.info("已恢复 %d 个文件从 commit %s", len(restored_files), commit_hash[:8])

    # 读取恢复后的版本号
    config_path = dir_path / "employee.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(config, dict):
            return str(config.get("version", "?"))
    return "?"
