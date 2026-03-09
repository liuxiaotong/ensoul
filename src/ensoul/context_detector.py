"""项目类型检测器 — 从文件结构和依赖推断项目技术栈."""

import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

# ── TTL 缓存（线程安全）──
_cache: dict[str, tuple[float, "ProjectInfo"]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 30.0  # seconds


class ProjectInfo(BaseModel):
    """项目类型检测结果."""

    project_type: str = Field(default="unknown", description="python/nodejs/go/rust/java/unknown")
    framework: str = Field(default="", description="fastapi/django/flask/express/react/vue/...")
    package_manager: str = Field(default="", description="uv/pip/npm/yarn/pnpm/cargo/...")
    test_framework: str = Field(default="", description="pytest/unittest/jest/vitest/go test/...")
    lint_tools: list[str] = Field(default_factory=list, description="ruff/eslint/golangci-lint/...")
    key_files: list[str] = Field(default_factory=list, description="检测到的关键配置文件")

    @property
    def display_label(self) -> str:
        """人类可读的项目类型标签，如 'Python (FastAPI)'."""
        if self.project_type == "unknown":
            return "unknown"
        label = self.project_type.capitalize()
        if self.framework:
            label += f" ({self.framework.capitalize()})"
        return label


def detect_project(
    project_dir: Path | None = None, *, cache_ttl: float | None = None
) -> ProjectInfo:
    """从文件存在性 + 依赖内容快速检测项目类型.

    带 TTL 缓存（默认 30s），cache_ttl=0 禁用。
    纯文件 I/O，不调用 subprocess。
    """
    from ensoul.paths import resolve_project_dir

    root = resolve_project_dir(project_dir)

    ttl = cache_ttl if cache_ttl is not None else _CACHE_TTL
    key = str(root)
    now = time.monotonic()

    with _cache_lock:
        if ttl > 0 and key in _cache:
            ts, result = _cache[key]
            if now - ts < ttl:
                return result

    info = _detect_project_uncached(root)

    if ttl > 0:
        with _cache_lock:
            _cache[key] = (now, info)

    return info


def _detect_project_uncached(root: Path) -> ProjectInfo:
    """实际执行文件系统检测的内部函数."""
    info = ProjectInfo()

    # ── Python ──
    pyproject = root / "pyproject.toml"
    setup_py = root / "setup.py"
    requirements = root / "requirements.txt"

    if pyproject.exists():
        info.project_type = "python"
        info.key_files.append("pyproject.toml")
        _detect_python_details(pyproject, info)
    elif setup_py.exists():
        info.project_type = "python"
        info.key_files.append("setup.py")
    elif requirements.exists():
        info.project_type = "python"
        info.key_files.append("requirements.txt")

    # ── Node.js ──
    if info.project_type == "unknown":
        package_json = root / "package.json"
        if package_json.exists():
            info.project_type = "nodejs"
            info.key_files.append("package.json")
            _detect_nodejs_details(package_json, info)

    # ── Go ──
    if info.project_type == "unknown":
        go_mod = root / "go.mod"
        if go_mod.exists():
            info.project_type = "go"
            info.key_files.append("go.mod")
            info.test_framework = "go test"

    # ── Rust ──
    if info.project_type == "unknown":
        cargo = root / "Cargo.toml"
        if cargo.exists():
            info.project_type = "rust"
            info.key_files.append("Cargo.toml")
            info.package_manager = "cargo"
            info.test_framework = "cargo test"

    # ── Java ──
    if info.project_type == "unknown":
        pom = root / "pom.xml"
        gradle = root / "build.gradle"
        if pom.exists():
            info.project_type = "java"
            info.key_files.append("pom.xml")
            info.package_manager = "maven"
        elif gradle.exists():
            info.project_type = "java"
            info.key_files.append("build.gradle")
            info.package_manager = "gradle"

    # ── 通用 lint 检测 ──
    _detect_lint_tools(root, info)

    return info


def _detect_python_details(pyproject: Path, info: ProjectInfo) -> None:
    """从 pyproject.toml 内容检测 Python 项目细节."""
    try:
        content = pyproject.read_text(encoding="utf-8")
    except OSError:
        return

    # 包管理器
    if (pyproject.parent / "uv.lock").exists():
        info.package_manager = "uv"
    elif (pyproject.parent / "poetry.lock").exists():
        info.package_manager = "poetry"
    elif (pyproject.parent / "Pipfile.lock").exists():
        info.package_manager = "pipenv"
    else:
        info.package_manager = "pip"

    # 测试框架
    if "[tool.pytest" in content or "pytest" in content.lower():
        info.test_framework = "pytest"
    else:
        info.test_framework = "unittest"

    # Lint 工具
    if "[tool.ruff" in content or "ruff" in content:
        info.lint_tools.append("ruff")
    if "[tool.black" in content:
        info.lint_tools.append("black")
    if "[tool.mypy" in content or "mypy" in content:
        info.lint_tools.append("mypy")
    if "[tool.flake8" in content or "flake8" in content:
        info.lint_tools.append("flake8")

    # 框架检测（从依赖名推断）
    content_lower = content.lower()
    if "fastapi" in content_lower:
        info.framework = "fastapi"
    elif "django" in content_lower:
        info.framework = "django"
    elif "flask" in content_lower:
        info.framework = "flask"
    elif "starlette" in content_lower:
        info.framework = "starlette"
    elif "sqlalchemy" in content_lower and not info.framework:
        info.framework = "sqlalchemy"


def _detect_nodejs_details(package_json: Path, info: ProjectInfo) -> None:
    """从 package.json 内容检测 Node.js 项目细节."""
    import json

    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    # 包管理器
    if (package_json.parent / "pnpm-lock.yaml").exists():
        info.package_manager = "pnpm"
    elif (package_json.parent / "yarn.lock").exists():
        info.package_manager = "yarn"
    else:
        info.package_manager = "npm"

    all_deps = set()
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        all_deps.update(data.get(key, {}).keys())

    # 测试框架
    if "jest" in all_deps:
        info.test_framework = "jest"
    elif "vitest" in all_deps:
        info.test_framework = "vitest"
    elif "mocha" in all_deps:
        info.test_framework = "mocha"

    # Lint
    if "eslint" in all_deps:
        info.lint_tools.append("eslint")
    if "prettier" in all_deps:
        info.lint_tools.append("prettier")
    if "typescript" in all_deps:
        info.lint_tools.append("typescript")

    # 框架
    if "next" in all_deps:
        info.framework = "nextjs"
    elif "react" in all_deps:
        info.framework = "react"
    elif "vue" in all_deps:
        info.framework = "vue"
    elif "express" in all_deps:
        info.framework = "express"
    elif "fastify" in all_deps:
        info.framework = "fastify"
    elif "@angular/core" in all_deps:
        info.framework = "angular"


def _detect_lint_tools(root: Path, info: ProjectInfo) -> None:
    """检测通用 lint 配置文件."""
    lint_files = {
        ".editorconfig": None,  # 不对应具体 lint 工具
        ".eslintrc": "eslint",
        ".eslintrc.js": "eslint",
        ".eslintrc.json": "eslint",
        ".prettierrc": "prettier",
        ".golangci.yml": "golangci-lint",
        ".golangci.yaml": "golangci-lint",
        "rustfmt.toml": "rustfmt",
        ".clippy.toml": "clippy",
    }
    for filename, tool in lint_files.items():
        if (root / filename).exists():
            info.key_files.append(filename)
            if tool and tool not in info.lint_tools:
                info.lint_tools.append(tool)
