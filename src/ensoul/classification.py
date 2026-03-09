"""信息分级体系 — Channel/Source/Clearance 映射逻辑.

根据对话渠道和员工身份，计算有效的信息访问许可等级，
用于 MemoryStore.query() 的分级过滤参数。

统一分级语言（人类 + AI 共用）：
    public      — 任何人可见（Wiki: open）
    internal    — 已登录/内部员工（Wiki: public, Crew 默认）
    restricted  — staff 角色/指定域（Wiki: internal + allowed_roles）
    confidential — 创始人/C-level（Wiki: draft, Crew: ceo-assistant）
"""

from __future__ import annotations

# 分级等级序（数值越大权限越高）
CLASSIFICATION_LEVELS: dict[str, int] = {
    "public": 0,
    "internal": 1,
    "restricted": 2,
    "confidential": 3,
}

# Channel → source_type 映射
CHANNEL_SOURCE_TYPE: dict[str, str] = {
    "external_dm": "external",
    "external_channel": "external",
    "external_mention": "external",
    "feishu_group": "internal",
    "feishu_dm": "internal",
    "feishu_bot": "internal",
    "wecom": "internal",
    "pull": "internal",  # Claude Code 派遣
    "api": "internal",  # 内部 API 调用
    "web": "external",  # Web widget
    "mcp": "internal",  # MCP server
}

# Source type → 默认 classification_max
SOURCE_TYPE_CLEARANCE: dict[str, str] = {
    "external": "public",  # 外部用户只能看 public
    "internal": "internal",  # 内部员工默认看到 internal
}

# Sender type → clearance（发送者身份，可提升渠道默认等级）
SENDER_TYPE_CLEARANCE: dict[str, str] = {
    "internal": "internal",  # 内部员工 → internal
    "agent": "internal",  # AI 员工 → internal
    "external": "public",  # 外部用户 → public
}

# Employee profile → clearance + domains
EMPLOYEE_CLEARANCE: dict[str, dict] = {
    "ceo-assistant": {"clearance": "confidential", "domains": []},  # C3，全域
    "hr-manager": {"clearance": "restricted", "domains": ["hr"]},
    "finance-expert": {"clearance": "restricted", "domains": ["finance"]},
    "legal-counsel": {"clearance": "restricted", "domains": ["legal"]},
    "security-auditor": {"clearance": "restricted", "domains": ["security"]},
    # 其他员工默认 C1
}
DEFAULT_CLEARANCE: dict[str, object] = {"clearance": "internal", "domains": []}


# Phase 3：外部对话输出控制 prompt（统一常量，供 webhook_handlers 和 sg_api_bridge 复用）
EXTERNAL_OUTPUT_CONTROL_PROMPT = """\
【信息输出规则 — 当前为外部用户对话】
你拥有的内部知识可以用于理解问题和做出判断，但回答中：
- 可以说：产品功能、公开文档内容、通用技术知识、操作指引
- 不可说：内部架构细节、部署配置、事故记录、Bug 详情
- 不可说：团队人员信息、内部决策过程、会议内容
- 不可说：财务数据、客户信息、合同细节、融资信息
- 不可说：密码、Token、API Key 等凭据
- 原则：说结论不说过程，说能力不说实现，说公开信息不说内部细节
- 如果不确定某信息是否可以公开，选择不说"""


def get_effective_clearance(
    employee_name: str,
    channel: str,
    sender_type: str = "",
) -> dict:
    """计算有效许可等级.

    取 min(员工自身许可, 对话场景许可)。
    场景许可 = max(渠道默认等级, sender_type 等级)。

    Args:
        employee_name: 员工标识（slug 或花名均可）
        channel: 对话渠道标识（如 "external_dm", "feishu_group" 等）
        sender_type: 发送者身份（"internal"/"external"/"agent"/""），
                     空字符串时不提升渠道默认等级（向后兼容）

    Returns:
        {
            "classification_max": str,        # 最高可查询的分级
            "allowed_domains": list[str]|None, # 允许的职能域（None 表示不限）
            "include_confidential": bool,      # 是否包含 confidential 级别
        }
    """
    # 空 channel 时默认最低权限（安全默认）
    if not channel:
        return {
            "classification_max": "public",
            "allowed_domains": None,
            "include_confidential": False,
        }

    # 1. 员工自身许可
    emp_clearance = EMPLOYEE_CLEARANCE.get(employee_name, DEFAULT_CLEARANCE)
    emp_level = str(emp_clearance["clearance"])
    emp_domains: list[str] = list(emp_clearance.get("domains", []))

    # 2. 场景许可 = max(渠道默认等级, sender_type 等级)
    source_type = CHANNEL_SOURCE_TYPE.get(channel, "external")  # 未知渠道默认外部
    channel_level = SOURCE_TYPE_CLEARANCE.get(source_type, "public")
    channel_level_num = CLASSIFICATION_LEVELS.get(channel_level, 0)

    # sender_type 提升（仅在提供时生效）
    if sender_type:
        sender_level = SENDER_TYPE_CLEARANCE.get(sender_type, "public")
        sender_level_num = CLASSIFICATION_LEVELS.get(sender_level, 0)
        scene_level_num = max(channel_level_num, sender_level_num)
    else:
        scene_level_num = channel_level_num

    # 3. 取较低的等级 min(员工, 场景)
    emp_level_num = CLASSIFICATION_LEVELS.get(emp_level, 1)
    effective_level_num = min(emp_level_num, scene_level_num)

    # 反查等级名称
    effective_level = "public"
    for name, num in CLASSIFICATION_LEVELS.items():
        if num == effective_level_num:
            effective_level = name
            break

    # 4. 域：只有 effective_level >= restricted 时才有意义
    effective_domains = emp_domains if effective_level_num >= 2 else []

    return {
        "classification_max": effective_level,
        "allowed_domains": effective_domains if effective_domains else None,
        "include_confidential": effective_level == "confidential",
    }
