"""发现机制 — 内置 + .claude/skills + private/employees.

支持两种数据源（由 CREW_EMPLOYEE_SOURCE 环境变量控制）：
- ``db``（默认）：从 PostgreSQL employees 表读取
- ``filesystem``：原有文件系统扫描逻辑
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from ensoul.employees import builtin_dir
from ensoul.models import DiscoveryResult, Employee, PermissionPolicy
from ensoul.parser import parse_employee, parse_employee_dir, parse_skill, validate_employee

logger = logging.getLogger(__name__)

# Feature flag: db (默认) 或 filesystem
EMPLOYEE_SOURCE = os.environ.get("CREW_EMPLOYEE_SOURCE", "db")

# ── TTL 缓存 ──
_cache: dict[str, tuple[float, DiscoveryResult]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 30.0  # seconds

# Claude Code Skills 目录
SKILLS_DIR_NAME = ".claude/skills"

# 自定义员工目录（相对项目根）
PRIVATE_DIR_NAME = "private/employees"


def _scan_directory(
    dir_path: Path,
    layer: str,
) -> list[Employee]:
    """扫描目录中的员工定义（支持目录格式和文件格式）.

    扫描顺序:
    1. 目录格式: <dir_path>/<name>/employee.yaml → parse_employee_dir()
    2. 文件格式: <dir_path>/<name>.md → parse_employee()（已被目录覆盖的跳过）

    Args:
        dir_path: 要扫描的目录
        layer: 来源层标识（builtin / global / project）

    Returns:
        成功解析的员工列表（跳过解析失败的文件）
    """
    from ensoul.tool_schema import validate_permissions

    employees = []
    if not dir_path.is_dir():
        return employees

    seen_names: set[str] = set()

    # 1. 扫描目录格式的员工
    for item in sorted(dir_path.iterdir()):
        if not item.is_dir() or not (item / "employee.yaml").exists():
            continue
        try:
            # 可写层自动版本管理
            if layer in ("private",):
                try:
                    from ensoul.versioning import check_and_bump

                    check_and_bump(item)
                except Exception as e:
                    logger.debug("版本检查失败 %s: %s", item.name, e)

            emp = parse_employee_dir(item, source_layer=layer)

            errors = validate_employee(emp)
            if errors:
                logger.warning("跳过 %s: %s", item, "; ".join(errors))
                continue

            # 权限配置校验（警告级别，不阻止加载）
            for w in validate_permissions(emp):
                logger.debug("权限配置 %s: %s", emp.name, w)

            employees.append(emp)
            seen_names.add(emp.name)
        except ValueError as e:
            logger.warning("跳过 %s: %s", item, e)
        except Exception as e:
            logger.warning("跳过 %s: 未知错误 %s", item, e, exc_info=True)

    # 2. 扫描文件格式的员工（向后兼容），跳过已被目录覆盖的
    for md_file in sorted(dir_path.glob("*.md")):
        if md_file.name.startswith("_") or md_file.name == "README.md":
            continue
        if md_file.stem in seen_names:
            continue  # 目录格式优先
        try:
            emp = parse_employee(md_file)
            emp.source_layer = layer
            emp.source_path = md_file

            errors = validate_employee(emp)
            if errors:
                logger.warning("跳过 %s: %s", md_file, "; ".join(errors))
                continue

            employees.append(emp)
        except ValueError as e:
            logger.warning("跳过 %s: %s", md_file, e)
        except Exception as e:
            logger.warning("跳过 %s: 未知错误 %s", md_file, e, exc_info=True)

    return employees


def _scan_skills_directory(
    skills_dir: Path,
) -> list[Employee]:
    """扫描 .claude/skills/ 目录中的 SKILL.md 文件.

    Claude Code Skills 目录结构: <skills_dir>/<name>/SKILL.md

    Args:
        skills_dir: .claude/skills/ 目录

    Returns:
        成功解析的员工列表
    """
    employees = []
    if not skills_dir.is_dir():
        return employees

    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir():
            continue
        skill_file = child / "SKILL.md"
        if not skill_file.exists():
            continue
        try:
            emp = parse_skill(skill_file, skill_name=child.name)
            emp.source_layer = "skill"
            emp.source_path = skill_file

            errors = validate_employee(emp)
            if errors:
                logger.warning("跳过 %s: %s", skill_file, "; ".join(errors))
                continue

            employees.append(emp)
        except ValueError as e:
            logger.warning("跳过 %s: %s", skill_file, e)
        except Exception as e:
            logger.warning("跳过 %s: 未知错误 %s", skill_file, e, exc_info=True)

    return employees


def _merge_employee(
    emp: Employee,
    layer_name: str,
    employees: dict[str, Employee],
    trigger_map: dict[str, str],
    conflicts: list[dict],
) -> None:
    """合并单个员工到结果集（高层覆盖低层）."""
    if emp.name in employees:
        existing = employees[emp.name]
        conflicts.append(
            {
                "name": emp.name,
                "winner": f"{layer_name}:{emp.source_path}",
                "loser": f"{existing.source_layer}:{existing.source_path}",
            }
        )
        logger.info(
            "员工 '%s' 被 %s 层覆盖（原: %s 层）",
            emp.name,
            layer_name,
            existing.source_layer,
        )

    employees[emp.name] = emp

    for trigger in emp.triggers:
        if trigger in trigger_map and trigger_map[trigger] != emp.name:
            existing_name = trigger_map[trigger]
            conflicts.append(
                {
                    "type": "trigger",
                    "trigger": trigger,
                    "winner": emp.name,
                    "loser": existing_name,
                }
            )
        trigger_map[trigger] = emp.name


def discover_employees(
    project_dir: Path | None = None,
    *,
    cache_ttl: float | None = None,
    tenant_id: str | None = None,
) -> DiscoveryResult:
    """执行员工发现.

    数据源由 CREW_EMPLOYEE_SOURCE 环境变量控制：
    - ``db``（默认）：PG 可用时从 employees 表查询
    - ``filesystem``：原有文件系统扫描

    带 TTL 缓存（默认 30s），按 project_dir + tenant_id 分 key。
    cache_ttl=0 禁用缓存（适合 CLI / 测试场景）。

    优先级（filesystem 模式，高覆盖低）:
    1. 内置层: 包内 employees/*.md
    2. 技能层: {project_dir}/.claude/skills/<name>/SKILL.md
    3. private 层: {project_dir}/private/employees/
    """
    from ensoul.paths import resolve_project_dir

    root = resolve_project_dir(project_dir)

    ttl = cache_ttl if cache_ttl is not None else _CACHE_TTL
    tid = tenant_id or ""
    key = f"{root}:{tid}"
    now = time.monotonic()

    with _cache_lock:
        if ttl > 0 and key in _cache:
            ts, result = _cache[key]
            if now - ts < ttl:
                return result

    # 选择数据源
    source = os.environ.get("CREW_EMPLOYEE_SOURCE", "db")
    if source == "db":
        result = _discover_employees_from_db(root, tenant_id=tenant_id)
    else:
        result = _discover_employees_uncached(root)

    if ttl > 0:
        with _cache_lock:
            _cache[key] = (now, result)

    return result


def _db_row_to_employee(row: dict[str, Any]) -> Employee:
    """将 employees 表的字典转换为 Employee 对象."""
    # 解析 permissions
    permissions = None
    perm_json = row.get("permissions_json")
    if perm_json:
        try:
            permissions = PermissionPolicy(**json.loads(perm_json))
        except Exception:
            pass

    return Employee(
        name=row["name"],
        display_name=row.get("display_name", ""),
        character_name=row.get("character_name", ""),
        summary=row.get("summary", ""),
        version=row.get("version", "1.0"),
        description=row.get("description", ""),
        tags=row.get("tags") or [],
        author=row.get("author", ""),
        triggers=row.get("triggers") or [],
        args=[],  # args 暂不存入 DB，从 body 中解析
        tools=row.get("tools") or [],
        context=row.get("context") or [],
        model_tier=row.get("model_tier", ""),
        model=row.get("model", ""),
        api_key=row.get("api_key", ""),
        base_url=row.get("base_url", ""),
        fallback_model=row.get("fallback_model", ""),
        fallback_api_key=row.get("fallback_api_key", ""),
        fallback_base_url=row.get("fallback_base_url", ""),
        agent_id=row.get("agent_id"),
        agent_status=row.get("agent_status", "active"),
        avatar_prompt=row.get("avatar_prompt", ""),
        research_instructions=row.get("research_instructions", ""),
        auto_memory=bool(row.get("auto_memory", False)),
        kpi=row.get("kpi") or [],
        permissions=permissions,
        body=row.get("body", ""),
        source_path=None,  # DB 模式下没有文件路径
        source_layer=row.get("source_layer", "db"),
    )


def _discover_employees_from_db(
    root: Path,
    tenant_id: str | None = None,
) -> DiscoveryResult:
    """从 PostgreSQL employees 表查询员工.

    如果 PG 不可用，自动回退到文件系统扫描。
    """
    try:
        from ensoul.database import is_pg

        if not is_pg():
            logger.debug("非 PG 模式，回退文件系统发现")
            return _discover_employees_uncached(root)

        from ensoul.config_store import list_employees_from_db
        from ensoul.tenant import DEFAULT_ADMIN_TENANT_ID

        tid = tenant_id or DEFAULT_ADMIN_TENANT_ID
        rows = list_employees_from_db(tenant_id=tid)

        if not rows:
            # 仅 admin 租户回退文件系统（保底），普通租户返回空集
            if tid == DEFAULT_ADMIN_TENANT_ID:
                logger.info("admin 租户 employees 表为空，回退文件系统发现")
                return _discover_employees_uncached(root)
            logger.debug("租户 %s 无员工，返回空集", tid)
            return DiscoveryResult(employees={}, conflicts=[])

        employees: dict[str, Employee] = {}
        for row in rows:
            try:
                emp = _db_row_to_employee(row)
                employees[emp.name] = emp
            except Exception as e:
                logger.warning("从 DB 构造 Employee 失败 %s: %s", row.get("name"), e)

        # 按 model_tier 填充模型默认值
        from ensoul.organization import apply_model_defaults, load_organization

        org = load_organization(project_dir=root)
        apply_model_defaults(employees, org)

        return DiscoveryResult(employees=employees, conflicts=[])

    except Exception as e:
        logger.warning("DB 员工发现失败，回退文件系统: %s", e, exc_info=True)
        return _discover_employees_uncached(root)


def _discover_employees_uncached(root: Path) -> DiscoveryResult:
    """实际执行文件系统扫描的内部函数."""
    employees: dict[str, Employee] = {}
    trigger_map: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []

    # 低优先级：内置层
    for emp in _scan_directory(builtin_dir(), "builtin"):
        _merge_employee(emp, "builtin", employees, trigger_map, conflicts)

    # 中优先级：.claude/skills/
    skills_dir = root / SKILLS_DIR_NAME
    for emp in _scan_skills_directory(skills_dir):
        _merge_employee(emp, "skill", employees, trigger_map, conflicts)

    # 高优先级：private/employees/
    private_dir = root / PRIVATE_DIR_NAME
    for emp in _scan_directory(private_dir, "private"):
        _merge_employee(emp, "private", employees, trigger_map, conflicts)

    # 按 model_tier 填充模型默认值
    from ensoul.organization import apply_model_defaults, load_organization

    org = load_organization(project_dir=root)
    apply_model_defaults(employees, org)

    return DiscoveryResult(employees=employees, conflicts=conflicts)


def get_employee(
    name_or_trigger: str,
    project_dir: Path | None = None,
    *,
    tenant_id: str | None = None,
) -> Employee | None:
    """按名称或触发别名查找员工.

    Args:
        name_or_trigger: 员工名称或触发别名
        project_dir: 项目根目录
        tenant_id: 租户 ID（传入 discover_employees 做数据隔离）

    Returns:
        Employee 对象，未找到返回 None
    """
    result = discover_employees(project_dir=project_dir, tenant_id=tenant_id)
    return result.get(name_or_trigger)
