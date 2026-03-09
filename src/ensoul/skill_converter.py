"""EMPLOYEE.md <-> SKILL.md 双向转换器."""

import json
import re
from pathlib import Path

from ensoul.models import TOOL_TO_SKILL, Employee


def _tools_to_allowed_tools(tools: list[str]) -> str:
    """将工具名列表转为 allowed-tools 字符串.

    Args:
        tools: 工具名列表 (如 ['file_read', 'git', 'bash'])

    Returns:
        空格分隔的 SKILL.md 工具字符串 (如 'Read Bash(git:*) Bash')
    """
    if not tools:
        return ""
    seen: list[str] = []
    for tool in tools:
        skill_name = TOOL_TO_SKILL.get(tool, tool)
        if skill_name not in seen:
            seen.append(skill_name)
    return " ".join(seen)


def _args_to_argument_hint(employee: Employee) -> str:
    """将 Employee args 转为 argument-hint 字符串.

    Args:
        employee: Employee 对象

    Returns:
        如 '<target> [mode]'
    """
    if not employee.args:
        return ""
    parts = []
    for arg in employee.args:
        if arg.required:
            parts.append(f"<{arg.name}>")
        else:
            parts.append(f"[{arg.name}]")
    return " ".join(parts)


def _convert_named_to_positional(body: str, employee: Employee) -> str:
    """将正文中的命名变量 ($argname) 转为位置变量 ($0, $1).

    Args:
        body: Employee 正文
        employee: Employee 对象（提供 args 顺序）

    Returns:
        转换后的正文
    """
    result = body
    for i, arg in enumerate(employee.args):
        # 用 word boundary 匹配确保不会替换 $argname_extra 等
        result = re.sub(rf"\${re.escape(arg.name)}\b", f"${i}", result)
    return result


def _build_metadata_comment(employee: Employee) -> str:
    """构建 HTML 注释，保存专有元数据.

    仅包含 SKILL.md 无法原生表达的字段。
    """
    metadata: dict = {}

    if employee.display_name:
        metadata["display_name"] = employee.display_name
    if employee.character_name:
        metadata["character_name"] = employee.character_name
    if employee.summary:
        metadata["summary"] = employee.summary
    if employee.tags:
        metadata["tags"] = employee.tags
    if employee.triggers:
        metadata["triggers"] = employee.triggers
    if employee.author:
        metadata["author"] = employee.author
    if employee.version != "1.0":
        metadata["version"] = employee.version
    if employee.model:
        metadata["model"] = employee.model
    if employee.base_url:
        metadata["base_url"] = employee.base_url
    if employee.fallback_model:
        metadata["fallback_model"] = employee.fallback_model
    if employee.context:
        metadata["context"] = employee.context
    if employee.output and (employee.output.filename or employee.output.format != "markdown"):
        metadata["output"] = employee.output.model_dump(mode="json")

    if not metadata:
        return ""

    json_str = json.dumps(metadata, ensure_ascii=False, indent=2)
    return f"<!-- ensoul metadata {json_str} -->"


def employee_to_skill(employee: Employee) -> str:
    """将 Employee 转换为 SKILL.md 格式字符串.

    Args:
        employee: Employee 对象

    Returns:
        SKILL.md 格式的字符串
    """
    # 构建 frontmatter
    lines = ["---"]
    lines.append(f"name: {employee.name}")
    lines.append(f"description: {employee.description}")

    allowed_tools = _tools_to_allowed_tools(employee.tools)
    if allowed_tools:
        lines.append(f"allowed-tools: {allowed_tools}")

    argument_hint = _args_to_argument_hint(employee)
    if argument_hint:
        lines.append(f"argument-hint: {argument_hint}")

    if employee.model:
        lines.append(f"model: {employee.model}")

    lines.append("---")
    lines.append("")

    # 元数据注释
    metadata_comment = _build_metadata_comment(employee)
    if metadata_comment:
        lines.append(metadata_comment)
        lines.append("")

    # 正文（命名变量转位置变量）
    body = _convert_named_to_positional(employee.body, employee)
    lines.append(body)

    return "\n".join(lines)


def write_skill(employee: Employee, skills_dir: Path) -> Path:
    """将 Employee 写入为 SKILL.md 文件.

    按 Claude Code 目录结构: <skills_dir>/<name>/SKILL.md

    Args:
        employee: Employee 对象
        skills_dir: skills 根目录

    Returns:
        写入的文件路径
    """
    safe_name = employee.name.replace("/", "_").replace("..", "_")
    skill_dir = skills_dir / safe_name
    if not skill_dir.resolve().is_relative_to(skills_dir.resolve()):
        raise ValueError(f"不安全的员工名称: {employee.name}")
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_path = skill_dir / "SKILL.md"
    content = employee_to_skill(employee)
    skill_path.write_text(content, encoding="utf-8")

    return skill_path


def export_employee(employee: Employee, project_dir: Path) -> Path:
    """将单个 Employee 导出到 .claude/skills/ 目录.

    Args:
        employee: Employee 对象
        project_dir: 项目根目录

    Returns:
        写入的文件路径
    """
    skills_dir = project_dir / ".claude" / "skills"
    return write_skill(employee, skills_dir)


def export_all(employees: list[Employee], project_dir: Path) -> list[Path]:
    """批量导出所有 Employee 到 .claude/skills/ 目录.

    Args:
        employees: Employee 列表
        project_dir: 项目根目录

    Returns:
        写入的文件路径列表
    """
    return [export_employee(emp, project_dir) for emp in employees]


def sync_skills(
    employees: list[Employee],
    project_dir: Path,
    clean: bool = False,
) -> dict:
    """同步 Employee 到 .claude/skills/ 目录.

    Args:
        employees: Employee 列表
        project_dir: 项目根目录
        clean: 是否删除孤儿目录

    Returns:
        {"exported": [...], "removed": [...]} 操作报告
    """
    skills_dir = project_dir / ".claude" / "skills"

    exported = export_all(employees, project_dir)
    exported_names = {emp.name for emp in employees}

    removed: list[Path] = []
    if clean and skills_dir.is_dir():
        for child in skills_dir.iterdir():
            if child.is_dir() and child.name not in exported_names:
                # 仅删除包含 SKILL.md 的目录（安全检查）
                skill_file = child / "SKILL.md"
                if skill_file.exists():
                    skill_file.unlink()
                    # 如果目录为空则删除
                    if not any(child.iterdir()):
                        child.rmdir()
                    removed.append(child)

    return {
        "exported": [str(p) for p in exported],
        "removed": [str(p) for p in removed],
    }
