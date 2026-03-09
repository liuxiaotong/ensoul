"""记忆标签规范和词典 — 统一标签命名，提供自动建议."""

import re
from typing import Literal

# ============================================================================
# 标签词典定义
# ============================================================================

# 技术领域标签（层级结构）
TECH_TAGS = {
    "backend": ["api", "database", "migration", "performance", "security", "cache"],
    "api": ["rest", "graphql", "webhook", "auth", "handler"],
    "database": ["sql", "orm", "index", "query-optimization", "migration"],
    "frontend": ["ui", "css", "javascript", "react", "responsive", "component"],
    "ui": ["layout", "styling", "interaction", "accessibility"],
    "infra": ["deploy", "ci-cd", "docker", "monitoring", "server"],
    "deploy": ["github-actions", "rsync", "rollback"],
    "testing": ["unit-test", "integration-test", "e2e", "flaky-test", "coverage"],
}

# 项目标签
PROJECT_TAGS = [
    "knowlyr-crew",
    "knowlyr-id",
    "antgather",
    "knowlyr-website",
    "knowlyr-wiki",
]

# 记忆类型标签
MEMORY_TYPE_TAGS = [
    "architecture",  # 架构决策
    "bug-fix",  # Bug 修复
    "gotcha",  # 陷阱/坑
    "api-gotcha",  # API 陷阱
    "best-practice",  # 最佳实践
    "lesson-learned",  # 经验教训
    "workflow",  # 工作流程
    "code-review",  # 代码审查
    "incident",  # 事故
    "feature",  # 功能实现
]

# 状态标签
STATUS_TAGS = [
    "milestone",  # 里程碑
    "handoff-state",  # 交接状态
    "project-board",  # 项目看板
    "pending",  # 待处理
    "resolved",  # 已解决
    "in-progress",  # 进行中
]

# 特殊标签（系统保留）
RESERVED_TAGS = [
    "trajectory",  # 轨迹标记（会被拦截）
    "claude-code",  # Claude Code 来源
    "exemplar",  # 高分范例
    "high-score",  # 高分
    "wiki-pointer",  # Wiki 文档指针
]

# 第三方服务标签
SERVICE_TAGS = [
    "wecom",  # 企业微信
    "feishu",  # 飞书
    "github",  # GitHub
    "lark",  # Lark
]


# ============================================================================
# 标签规范化
# ============================================================================


def normalize_tag(tag: str) -> str:
    """规范化标签：小写、连字符、去除空格.

    规则：
    - 全小写
    - 下划线转连字符
    - 空格转连字符
    - 去除首尾空白
    - 去除特殊字符（仅保留字母、数字、连字符）

    Examples:
        >>> normalize_tag("Memory_System")
        'memory-system'
        >>> normalize_tag("API Gotcha")
        'api-gotcha'
        >>> normalize_tag("  backend  ")
        'backend'
    """
    if not tag or not isinstance(tag, str):
        return ""

    # 转小写
    tag = tag.lower().strip()

    # 下划线和空格转连字符
    tag = tag.replace("_", "-").replace(" ", "-")

    # 去除特殊字符（仅保留字母、数字、连字符、中文）
    tag = re.sub(r"[^a-z0-9\-\u4e00-\u9fff]", "", tag)

    # 去除连续的连字符
    tag = re.sub(r"-+", "-", tag)

    # 去除首尾连字符
    tag = tag.strip("-")

    return tag


def normalize_tags(tags: list[str]) -> list[str]:
    """批量规范化标签，去重并排序.

    Args:
        tags: 原始标签列表

    Returns:
        规范化后的标签列表（去重、排序）
    """
    if not tags or not isinstance(tags, list):
        return []

    normalized = [normalize_tag(tag) for tag in tags]
    # 去除空字符串
    normalized = [tag for tag in normalized if tag]
    # 去重并排序
    return sorted(set(normalized))


# ============================================================================
# 标签建议
# ============================================================================


def suggest_tags(
    category: Literal["decision", "estimate", "finding", "correction", "pattern"],
    content: str,
    existing_tags: list[str] | None = None,
) -> list[str]:
    """根据内容自动建议标签.

    Args:
        category: 记忆类别
        content: 记忆内容
        existing_tags: 已有标签（不重复建议）

    Returns:
        建议的标签列表（最多 5 个）
    """
    if not content:
        return []

    existing_tags = existing_tags or []
    existing_set = set(normalize_tags(existing_tags))
    suggestions = []

    content_lower = content.lower()

    # 中文关键词映射到英文标签
    chinese_keyword_map = {
        "数据库": "database",
        "接口": "api",
        "前端": "frontend",
        "后端": "backend",
        "部署": "deploy",
        "测试": "testing",
        "企微": "wecom",
        "飞书": "feishu",
    }

    # 先处理中文关键词映射
    for cn_keyword, en_tag in chinese_keyword_map.items():
        if cn_keyword in content and en_tag not in existing_set:
            suggestions.append(en_tag)

    # 1. 匹配技术领域标签
    for tech, subtags in TECH_TAGS.items():
        if tech in content_lower and tech not in existing_set:
            suggestions.append(tech)
        for subtag in subtags:
            if subtag in content_lower and subtag not in existing_set:
                suggestions.append(subtag)

    # 2. 匹配项目标签
    for project in PROJECT_TAGS:
        # 支持带连字符和不带连字符的匹配
        project_variants = [project, project.replace("-", "")]
        if any(variant in content_lower for variant in project_variants):
            if project not in existing_set:
                suggestions.append(project)

    # 3. 匹配服务标签
    for service in SERVICE_TAGS:
        if service in content_lower and service not in existing_set:
            suggestions.append(service)

    # 4. 根据 category 建议类型标签
    category_suggestions = {
        "correction": ["lesson-learned", "gotcha"],
        "pattern": ["best-practice", "workflow"],
        "finding": ["feature", "bug-fix"],
        "decision": ["architecture"],
    }

    if category in category_suggestions:
        for tag in category_suggestions[category]:
            if tag not in existing_set:
                # 检查内容是否相关
                if category == "correction" and any(
                    kw in content_lower for kw in ["陷阱", "注意", "避免", "错误"]
                ):
                    if tag == "gotcha":
                        suggestions.append(tag)
                elif category == "pattern" and any(
                    kw in content_lower for kw in ["模式", "方法", "流程", "步骤"]
                ):
                    if tag in ["best-practice", "workflow"]:
                        suggestions.append(tag)
                elif category == "finding":
                    if "bug" in content_lower or "修复" in content_lower or "问题" in content_lower:
                        if tag == "bug-fix":
                            suggestions.append(tag)
                    elif (
                        "功能" in content_lower
                        or "实现" in content_lower
                        or "新增" in content_lower
                    ):
                        if tag == "feature":
                            suggestions.append(tag)
                elif category == "decision" and tag == "architecture":
                    suggestions.append(tag)

    # 5. 特殊关键词匹配
    keyword_map = {
        "code-review": ["代码审查", "审查", "review"],
        "incident": ["事故", "故障", "崩溃", "高危"],
        "api-gotcha": ["api", "字段", "接口"],
    }

    for tag, keywords in keyword_map.items():
        if tag not in existing_set:
            if any(kw in content_lower for kw in keywords):
                # api-gotcha 需要同时包含 api 和陷阱相关词
                if tag == "api-gotcha":
                    if ("api" in content_lower or "接口" in content) and any(
                        kw in content_lower for kw in ["陷阱", "注意", "坑", "gotcha"]
                    ):
                        suggestions.append(tag)
                else:
                    suggestions.append(tag)

    # 6. 去重并限制数量
    suggestions = list(dict.fromkeys(suggestions))  # 保持顺序去重

    return suggestions[:5]  # 最多建议 5 个


# ============================================================================
# 标签验证
# ============================================================================


def validate_tags(tags: list[str]) -> dict:
    """验证标签是否符合规范.

    Args:
        tags: 待验证的标签列表

    Returns:
        {
            "valid": bool,
            "issues": list[str],  # 问题列表
            "normalized": list[str],  # 规范化后的标签
        }
    """
    if not tags or not isinstance(tags, list):
        return {
            "valid": True,
            "issues": [],
            "normalized": [],
        }

    issues = []
    normalized = []

    for tag in tags:
        if not isinstance(tag, str):
            issues.append(f"标签必须是字符串: {tag}")
            continue

        original = tag
        norm = normalize_tag(tag)

        if not norm:
            issues.append(f"标签为空或无效: '{original}'")
            continue

        # 检查长度
        if len(norm) > 50:
            issues.append(f"标签过长（最多 50 字符）: '{norm}'")
            continue

        # 检查是否包含保留标签
        if norm in RESERVED_TAGS:
            issues.append(f"标签 '{norm}' 是系统保留标签")
            continue

        normalized.append(norm)

    # 去重
    normalized = list(dict.fromkeys(normalized))

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "normalized": normalized,
    }


# ============================================================================
# 标签查询
# ============================================================================


def get_all_predefined_tags() -> dict:
    """获取所有预定义标签.

    Returns:
        {
            "tech": dict,
            "projects": list,
            "types": list,
            "status": list,
            "services": list,
            "reserved": list,
        }
    """
    return {
        "tech": TECH_TAGS,
        "projects": PROJECT_TAGS,
        "types": MEMORY_TYPE_TAGS,
        "status": STATUS_TAGS,
        "services": SERVICE_TAGS,
        "reserved": RESERVED_TAGS,
    }


def search_tags(query: str, limit: int = 10) -> list[str]:
    """搜索标签（模糊匹配）.

    Args:
        query: 搜索关键词
        limit: 最多返回数量

    Returns:
        匹配的标签列表
    """
    if not query:
        return []

    query_lower = query.lower()
    matches = []

    # 收集所有标签
    all_tags = set()

    # 技术标签
    for parent, children in TECH_TAGS.items():
        all_tags.add(parent)
        all_tags.update(children)

    # 其他标签
    all_tags.update(PROJECT_TAGS)
    all_tags.update(MEMORY_TYPE_TAGS)
    all_tags.update(STATUS_TAGS)
    all_tags.update(SERVICE_TAGS)

    # 模糊匹配
    for tag in all_tags:
        if query_lower in tag.lower():
            matches.append(tag)

    # 按相关度排序（完全匹配优先）
    matches.sort(
        key=lambda t: (
            t.lower() != query_lower,
            t.lower().index(query_lower) if query_lower in t.lower() else 999,
            t,
        )
    )

    return matches[:limit]
