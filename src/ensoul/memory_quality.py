"""记忆质量控制 — 防止低质量记忆污染系统."""

from typing import Literal


def check_memory_quality(
    category: Literal["decision", "estimate", "finding", "correction", "pattern"],
    content: str,
) -> dict:
    """检查记忆质量.

    Args:
        category: 记忆类别
        content: 记忆内容

    Returns:
        {
            'score': float,  # 0-1，>= 0.6 为合格
            'issues': list,  # 问题列表
            'suggestions': list  # 改进建议
        }
    """
    issues = []
    suggestions = []
    score = 0.0

    # 长度限制
    max_length = 500 if category in ["correction", "pattern", "decision"] else 1000
    min_length = 50

    content_len = len(content)

    # 1. 长度得分（0.3）
    if min_length <= content_len <= max_length:
        score += 0.3
    elif content_len < min_length:
        score += 0.0  # 太短直接不给分
        issues.append(f"内容过短（{content_len} 字符，最少需要 {min_length} 字符）")
        suggestions.append("补充更多细节：为什么、如何、影响是什么")
    else:
        score += 0.2
        issues.append(f"内容过长（{content_len} 字符，建议不超过 {max_length} 字符）")
        suggestions.append("提炼核心要点，或拆分成多条记忆")

    # 2. 关键词得分（0.4）
    keywords_map = {
        "correction": [
            "教训",
            "错误",
            "陷阱",
            "注意",
            "避免",
            "正确做法",
            "修正",
            "根因",
            "问题",
            "解决",
            "修复",
        ],
        "pattern": [
            "模式",
            "方法",
            "策略",
            "原则",
            "规律",
            "经验",
            "流程",
            "步骤",
            "最佳实践",
            "建议",
        ],
    }

    if category in keywords_map:
        keywords = keywords_map[category]
        has_keyword = any(kw in content for kw in keywords)
        if has_keyword:
            score += 0.4
        else:
            score += 0.1
            issues.append(f"缺少关键词（{category} 类型应包含：{', '.join(keywords[:5])} 等）")
            suggestions.append(f"明确说明这是什么类型的{category}，用关键词表达")
    else:
        # finding/decision/estimate 不强制关键词
        score += 0.4

    # 3. 结构得分（0.3）
    structure_score = 0.0

    # 3.1 不能以 [轨迹] 开头（严重扣分）
    if content.startswith("[轨迹]"):
        # 直接扣除大量分数，确保无法通过
        score -= 0.5
        issues.append("内容以 [轨迹] 开头，这是流水账标记")
        suggestions.append("移除 [轨迹] 标记，提炼真正的经验教训")
    else:
        structure_score += 0.15

    # 3.2 包含深度内容（0.15）
    # 扩展深度关键词：包括解释性、指导性、反思性词汇
    depth_keywords = [
        "为什么",
        "如何",
        "原因",
        "方法",
        "步骤",
        "因为",
        "所以",
        "导致",
        "避免",
        "注意",
        "原则",
        "策略",
        "教训",
        "正确做法",
        "流程",
        "目的",
        "影响",
        "后果",
        "风险",
        "建议",
        "最佳实践",
    ]
    if any(word in content for word in depth_keywords):
        structure_score += 0.15
    else:
        issues.append("缺少深度内容（为什么、如何、原因等）")
        suggestions.append("不要只说做了什么，要说明为什么这样做、如何做的")

    score += structure_score

    # 4. 流水账检测（额外扣分）
    trivial_patterns = [
        "修复了",
        "实现了",
        "完成了",
        "添加了",
        "更新了",
        "创建了",
        "删除了",
        "调整了",
    ]
    if any(pattern in content[:20] for pattern in trivial_patterns):
        # 如果开头就是这些词，可能是流水账
        if not any(word in content for word in ["为什么", "如何", "原因", "教训", "注意"]):
            issues.append("疑似流水账（只描述做了什么，没有说明为什么或如何）")
            suggestions.append("转换为经验：这次工作让你学到了什么？有什么要注意的？")

    return {
        "score": round(score, 2),
        "issues": issues,
        "suggestions": suggestions,
    }
