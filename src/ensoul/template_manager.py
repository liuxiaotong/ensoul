"""模板库与渲染工具."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ensoul.paths import get_global_templates_dir

BUILTIN_TEMPLATES_DIR = Path(__file__).parent / "builtin_templates"
PROJECT_TEMPLATES_SUBDIR = ".ensoul/templates"


@dataclass
class TemplateRecord:
    """模板记录."""

    name: str
    path: Path
    layer: str


def _iter_template_files(dir_path: Path, layer: str) -> Iterable[TemplateRecord]:
    if not dir_path.is_dir():
        return []
    records: list[TemplateRecord] = []
    for file in sorted(dir_path.glob("*")):
        if file.is_file():
            records.append(TemplateRecord(name=file.stem, path=file, layer=layer))
    return records


def discover_templates(project_dir: Path | None = None) -> dict[str, TemplateRecord]:
    """发现可用模板，项目层覆盖全局，最终覆盖内置."""
    from ensoul.paths import resolve_project_dir

    root = resolve_project_dir(project_dir)
    records: dict[str, TemplateRecord] = {}

    for layer, directory in (
        ("builtin", BUILTIN_TEMPLATES_DIR),
        ("global", get_global_templates_dir(project_dir=project_dir)),
        ("project", root / PROJECT_TEMPLATES_SUBDIR),
    ):
        for record in _iter_template_files(directory, layer):
            records[record.name] = record

    return records


def load_template(name: str, project_dir: Path | None = None) -> TemplateRecord | None:
    """按名称获取模板记录."""
    templates = discover_templates(project_dir=project_dir)
    return templates.get(name)


_TEMPLATE_PATTERN = re.compile(r"{{\s*([^{}\s]+)\s*}}")


def render_template(content: str, variables: dict[str, str]) -> str:
    """使用 {{var}} 占位符渲染模板."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return variables.get(key, match.group(0))

    return _TEMPLATE_PATTERN.sub(replace, content)


def apply_template(
    name: str,
    variables: dict[str, str],
    output: Path,
    project_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """渲染模板并写出到指定路径."""
    record = load_template(name, project_dir=project_dir)
    if record is None:
        raise FileNotFoundError(f"模板不存在: {name}")

    content = record.path.read_text(encoding="utf-8")
    rendered = render_template(content, variables)

    if output.exists() and not overwrite:
        raise FileExistsError(f"目标已存在: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return output
