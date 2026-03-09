"""路由决策模块 — 判断消息是否需要工具调用.

供 engine.py、webhook_handlers.py 等上层模块直接引用，
避免引擎层反向依赖 webhook 层。
"""

# 工作关键词 — 命中任一则走完整 agent loop（带工具）
_WORK_KEYWORDS = frozenset(
    [
        "数据",
        "报表",
        "分析",
        "查一下",
        "查下",
        "帮我查",
        "统计",
        "日程",
        "日历",
        "会议",
        "审批",
        "待办",
        "任务",
        "委派",
        "安排",
        "创建",
        "删除",
        "更新",
        "发送",
        "通知",
        "催",
        "让他",
        "让她",
        "转告",
        "转发",
        "项目",
        "进度",
        "上线",
        "部署",
        "发布",
        "文档",
        "表格",
        "知识库",
        "github",
        "pr",
        "issue",
        "仓库",
        "邮件",
        "快递",
        "航班",
        "密码",
        "二维码",
        "短链",
        "记忆",
        "记住",
        "笔记",
        "写入",
        "网站",
        "网页",
        "链接",
    ]
)


def _needs_tools(text: str) -> bool:
    """判断消息是否需要工具（即不是纯闲聊）."""
    if not text or len(text) > 200:
        # 长消息通常是正式任务描述
        return True
    t = text.lower()
    # URL 通常需要工具处理（读取网页、链接预览等）
    if "http://" in t or "https://" in t:
        return True
    return any(kw in t for kw in _WORK_KEYWORDS)
