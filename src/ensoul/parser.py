"""EMPLOYEE.md / SKILL.md / 目录格式 解析器."""

import re
from pathlib import Path

import yaml

from ensoul.models import SKILL_TO_TOOL, Employee, EmployeeArg, EmployeeOutput, PermissionPolicy

# name 格式：仅小写字母、数字、连字符，不以连字符开头或结尾
NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


def _split_frontmatter(content: str) -> tuple[dict, str]:
    """分离 YAML frontmatter 和 Markdown 正文.

    Returns:
        (frontmatter_dict, body_text)
    """
    content = content.lstrip()
    if not content.startswith("---"):
        return {}, content

    lines = content.splitlines()
    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return {}, content

    yaml_str = "\n".join(lines[1:end_idx]).strip()
    body = "\n".join(lines[end_idx + 1 :]).strip()

    try:
        frontmatter = yaml.safe_load(yaml_str)
    except yaml.YAMLError:
        return {}, content

    if not isinstance(frontmatter, dict):
        return {}, content

    return frontmatter, body


def parse_employee(path: Path) -> Employee:
    """解析 EMPLOYEE.md 文件，返回 Employee 对象.

    Args:
        path: .md 文件路径

    Raises:
        ValueError: 解析失败或必填字段缺失
    """
    content = path.read_text(encoding="utf-8")
    return parse_employee_string(content, source_path=path)


def parse_employee_string(
    content: str,
    source_path: Path | None = None,
    source_layer: str = "builtin",
) -> Employee:
    """从字符串解析员工定义.

    Args:
        content: EMPLOYEE.md 文件内容
        source_path: 来源文件路径
        source_layer: 来源层（builtin / global / project）

    Raises:
        ValueError: 解析失败或必填字段缺失
    """
    frontmatter, body = _split_frontmatter(content)

    if not frontmatter:
        raise ValueError("缺少 YAML frontmatter（文件需以 --- 开头）")

    name = frontmatter.get("name")
    if not name:
        raise ValueError("缺少必填字段: name")

    description = frontmatter.get("description")
    if not description:
        raise ValueError("缺少必填字段: description")

    if not body:
        raise ValueError("Markdown 正文不能为空")

    # 解析 args
    raw_args = frontmatter.get("args", [])
    args = []
    if isinstance(raw_args, list):
        for item in raw_args:
            if isinstance(item, dict):
                args.append(EmployeeArg(**item))

    # 解析 output
    raw_output = frontmatter.get("output", {})
    output = EmployeeOutput(**raw_output) if isinstance(raw_output, dict) else EmployeeOutput()

    # 解析 tools
    raw_tools = frontmatter.get("tools", [])
    tools = raw_tools if isinstance(raw_tools, list) else []

    # 解析 context
    raw_context = frontmatter.get("context", [])
    context = raw_context if isinstance(raw_context, list) else []

    # 解析 permissions
    raw_perm = frontmatter.get("permissions")
    permissions = PermissionPolicy(**raw_perm) if isinstance(raw_perm, dict) else None

    return Employee(
        name=name,
        display_name=frontmatter.get("display_name", ""),
        character_name=frontmatter.get("character_name", ""),
        summary=frontmatter.get("summary", ""),
        version=str(frontmatter.get("version", "1.0")),
        description=description,
        tags=frontmatter.get("tags", []),
        author=frontmatter.get("author", ""),
        triggers=frontmatter.get("triggers", []),
        args=args,
        output=output,
        tools=tools,
        context=context,
        model_tier=str(frontmatter.get("model_tier", "")),
        model=str(frontmatter.get("model", "")),
        api_key=str(frontmatter.get("api_key", "")),
        base_url=str(frontmatter.get("base_url", "")),
        fallback_model=str(frontmatter.get("fallback_model", "")),
        fallback_api_key=str(frontmatter.get("fallback_api_key", "")),
        fallback_base_url=str(frontmatter.get("fallback_base_url", "")),
        agent_id=str(frontmatter["agent_id"]) if frontmatter.get("agent_id") is not None else None,
        avatar_prompt=frontmatter.get("avatar_prompt", ""),
        auto_memory=bool(frontmatter.get("auto_memory", False)),
        kpi=frontmatter.get("kpi", []),
        permissions=permissions,
        body=body,
        source_path=source_path,
        source_layer=source_layer,
    )


def parse_employee_dir(
    dir_path: Path,
    source_layer: str = "builtin",
) -> Employee:
    """从目录结构解析员工定义.

    目录结构:
        employee.yaml   — 纯配置（等同原 frontmatter）
        soul.md         — 员工唯一定义（角色灵魂）

    Args:
        dir_path: 员工目录路径
        source_layer: 来源层标识

    Raises:
        ValueError: 缺少必要文件或必填字段
    """
    config_path = dir_path / "employee.yaml"
    soul_path = dir_path / "soul.md"

    if not config_path.exists():
        raise ValueError(f"缺少 employee.yaml: {dir_path}")
    if not soul_path.exists():
        raise ValueError(f"缺少 soul.md: {dir_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"employee.yaml 格式无效: {dir_path}")

    name = config.get("name")
    if not name:
        raise ValueError(f"缺少必填字段 name: {config_path}")

    description = config.get("description")
    if not description:
        raise ValueError(f"缺少必填字段 description: {config_path}")

    # body 只读 soul.md（员工唯一定义）
    body = soul_path.read_text(encoding="utf-8")

    if not body.strip():
        raise ValueError(f"soul.md 正文不能为空: {dir_path}")

    # 解析 args
    raw_args = config.get("args", [])
    args = []
    if isinstance(raw_args, list):
        for item in raw_args:
            if isinstance(item, dict):
                args.append(EmployeeArg(**item))

    # 解析 output
    raw_output = config.get("output", {})
    output = EmployeeOutput(**raw_output) if isinstance(raw_output, dict) else EmployeeOutput()

    # 解析 tools
    raw_tools = config.get("tools", [])
    tools = raw_tools if isinstance(raw_tools, list) else []

    # 解析 context
    raw_context = config.get("context", [])
    context = raw_context if isinstance(raw_context, list) else []

    # 解析 permissions
    raw_perm = config.get("permissions")
    permissions = PermissionPolicy(**raw_perm) if isinstance(raw_perm, dict) else None

    version = str(config.get("version", "1.0"))

    return Employee(
        name=name,
        display_name=config.get("display_name", ""),
        character_name=config.get("character_name", ""),
        summary=config.get("summary", ""),
        version=version,
        description=description,
        tags=config.get("tags", []),
        author=config.get("author", ""),
        triggers=config.get("triggers", []),
        args=args,
        output=output,
        tools=tools,
        context=context,
        model_tier=str(config.get("model_tier", "")),
        model=str(config.get("model", "")),
        api_key=str(config.get("api_key", "")),
        base_url=str(config.get("base_url", "")),
        fallback_model=str(config.get("fallback_model", "")),
        fallback_api_key=str(config.get("fallback_api_key", "")),
        fallback_base_url=str(config.get("fallback_base_url", "")),
        agent_id=str(config["agent_id"]) if config.get("agent_id") is not None else None,
        agent_status=config.get("agent_status", "active"),
        avatar_prompt=config.get("avatar_prompt", ""),
        auto_memory=bool(config.get("auto_memory", False)),
        kpi=config.get("kpi", []),
        permissions=permissions,
        body=body,
        source_path=dir_path,
        source_layer=source_layer,
    )


def validate_employee(employee: Employee) -> list[str]:
    """校验员工定义，返回错误信息列表.

    Returns:
        错误信息列表，为空表示校验通过
    """
    errors = []

    # name 格式
    if not NAME_PATTERN.match(employee.name):
        errors.append(
            f"name '{employee.name}' 格式无效（仅允许小写字母、数字、连字符，"
            "不能以连字符开头或结尾）"
        )

    # name 长度
    if len(employee.name) > 64:
        errors.append(f"name 长度 {len(employee.name)} 超过 64 字符限制")

    # description 长度
    if len(employee.description) > 1024:
        errors.append(f"description 长度 {len(employee.description)} 超过 1024 字符限制")

    # body 非空
    if not employee.body.strip():
        errors.append("Markdown 正文不能为空")

    # args name 唯一性
    arg_names = [a.name for a in employee.args]
    if len(arg_names) != len(set(arg_names)):
        errors.append("args 中存在重复的参数名")

    # triggers 格式
    for trigger in employee.triggers:
        if not NAME_PATTERN.match(trigger):
            errors.append(f"trigger '{trigger}' 格式无效")

    return errors


# ── SKILL.md 解析 ──


def _parse_allowed_tools(tools_str: str) -> list[str]:
    """解析 allowed-tools 字符串为 Crew 工具名列表.

    处理带括号的语法，如 'Read Bash(git:*) Grep'。

    Args:
        tools_str: 空格分隔的工具字符串

    Returns:
        Crew 工具名列表（未映射的保留原名）
    """
    if not tools_str or not tools_str.strip():
        return []

    tokens: list[str] = []
    current = ""
    depth = 0

    for ch in tools_str:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == " " and depth == 0:
            if current:
                tokens.append(current)
                current = ""
        else:
            current += ch

    if current:
        tokens.append(current)

    # 映射到 Crew 工具名
    result = []
    for token in tokens:
        crew_name = SKILL_TO_TOOL.get(token)
        if crew_name:
            result.append(crew_name)
        else:
            result.append(token)

    return result


def _parse_argument_hint(hint_str: str) -> list[EmployeeArg]:
    """解析 argument-hint 字符串为 EmployeeArg 列表.

    格式: '<target> [mode]' — <> 表示必填，[] 表示可选。

    Args:
        hint_str: argument-hint 字符串

    Returns:
        EmployeeArg 列表
    """
    if not hint_str or not hint_str.strip():
        return []

    args: list[EmployeeArg] = []
    # 匹配 <name> 或 [name]
    for match in re.finditer(r"<([^>]+)>|\[([^\]]+)\]", hint_str):
        required_name = match.group(1)
        optional_name = match.group(2)
        if required_name:
            args.append(EmployeeArg(name=required_name, required=True))
        elif optional_name:
            args.append(EmployeeArg(name=optional_name, required=False))

    return args


def _extract_skill_metadata(body: str) -> tuple[dict, str]:
    """从 SKILL.md 正文中提取 HTML 注释里的元数据.

    查找 <!-- ensoul metadata {...} --> 格式的注释。

    Returns:
        (metadata_dict, clean_body)
    """
    pattern = r"<!--\s*ensoul\s+metadata\s+(.*?)\s*-->"
    match = re.search(pattern, body, re.DOTALL)
    if not match:
        return {}, body

    import json

    try:
        metadata = json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return {}, body

    clean_body = body[: match.start()] + body[match.end() :]
    clean_body = clean_body.strip()
    return metadata, clean_body


def _convert_skill_variables(body: str, args: list[EmployeeArg]) -> str:
    """将 SKILL.md 的位置变量 ($0, $1) 转为 Crew 的命名变量 ($argname).

    Args:
        body: SKILL.md 正文
        args: 已解析的参数列表

    Returns:
        转换后的正文
    """
    result = body

    # $ARGUMENTS → 所有参数名的组合（保留原样，仅供参考）
    # $0, $1, ... → $argname
    for i in range(len(args) - 1, -1, -1):
        result = result.replace(f"${i}", f"${args[i].name}")

    return result


def parse_skill_string(
    content: str,
    source_path: Path | None = None,
    skill_name: str | None = None,
) -> Employee:
    """从 SKILL.md 字符串解析为 Employee 对象.

    Args:
        content: SKILL.md 文件内容
        source_path: 来源文件路径
        skill_name: 技能名称（通常从目录名推断）

    Raises:
        ValueError: 解析失败或必填字段缺失
    """
    frontmatter, body = _split_frontmatter(content)

    if not frontmatter:
        raise ValueError("缺少 YAML frontmatter（文件需以 --- 开头）")

    name = skill_name or frontmatter.get("name")
    if not name:
        raise ValueError("缺少 name（SKILL.md 需在 frontmatter 或目录名中提供）")

    description = frontmatter.get("description", "")
    if not description:
        raise ValueError("缺少必填字段: description")

    if not body:
        raise ValueError("Markdown 正文不能为空")

    # 解析 allowed-tools
    tools_str = frontmatter.get("allowed-tools", "")
    tools = _parse_allowed_tools(str(tools_str)) if tools_str else []

    # 解析 argument-hint
    hint_str = frontmatter.get("argument-hint", "")
    args = _parse_argument_hint(str(hint_str)) if hint_str else []

    # 提取 HTML 注释中的 Crew 元数据
    metadata, clean_body = _extract_skill_metadata(body)

    # 转换位置变量为命名变量
    clean_body = _convert_skill_variables(clean_body, args)

    return Employee(
        name=name,
        display_name=metadata.get("display_name", ""),
        character_name=metadata.get("character_name", ""),
        summary=metadata.get("summary", ""),
        version=metadata.get("version", "1.0"),
        description=description,
        tags=metadata.get("tags", []),
        author=metadata.get("author", ""),
        triggers=metadata.get("triggers", []),
        args=args,
        output=EmployeeOutput(**metadata["output"]) if "output" in metadata else EmployeeOutput(),
        tools=tools,
        context=metadata.get("context", []),
        model=frontmatter.get("model", "") or metadata.get("model", ""),
        base_url=metadata.get("base_url", ""),
        fallback_model=metadata.get("fallback_model", ""),
        agent_id=str(metadata["agent_id"]) if metadata.get("agent_id") is not None else None,
        avatar_prompt=metadata.get("avatar_prompt", ""),
        body=clean_body,
        source_path=source_path,
        source_layer="skill",
    )


def parse_skill(path: Path, skill_name: str | None = None) -> Employee:
    """解析 SKILL.md 文件，返回 Employee 对象.

    Args:
        path: SKILL.md 文件路径
        skill_name: 技能名称（默认从父目录名推断）

    Raises:
        ValueError: 解析失败或必填字段缺失
    """
    content = path.read_text(encoding="utf-8")
    if skill_name is None:
        skill_name = path.parent.name
    return parse_skill_string(content, source_path=path, skill_name=skill_name)
