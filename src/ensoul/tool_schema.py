"""工具定义桥接 — ensoul employee.tools → LLM JSON Schema + sandbox 映射."""

from __future__ import annotations

from typing import Any

# ensoul tool name → sandbox tool name
ENSOUL_TO_SANDBOX: dict[str, str] = {
    "file_read": "file_read",
    "file_write": "file_write",
    "bash": "shell",
    "git": "git",
    "grep": "search",
    "glob": "search",
}

# 终止工具（agent loop 看到这些表示任务完成）
FINISH_TOOLS = {"submit", "finish"}

# LLM tool schemas（Anthropic 格式，OpenAI 自动转换）
_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "file_read": {
        "name": "file_read",
        "description": "读取文件内容。返回文件文本。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "start_line": {
                    "type": "integer",
                    "description": "起始行号 (0 表示从头)",
                    "default": 0,
                },
                "end_line": {
                    "type": "integer",
                    "description": "结束行号 (0 表示读到末尾)",
                    "default": 0,
                },
            },
            "required": ["path"],
        },
    },
    "file_write": {
        "name": "file_write",
        "description": "写入文件内容。自动创建父目录。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "文件内容"},
            },
            "required": ["path", "content"],
        },
    },
    "bash": {
        "name": "bash",
        "description": "在沙箱中执行 Shell 命令（bash -c）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell 命令"},
                "timeout": {
                    "type": "integer",
                    "description": "超时秒数",
                    "default": 300,
                },
            },
            "required": ["command"],
        },
    },
    "git": {
        "name": "git",
        "description": "执行 Git 操作（diff, log, status 等）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "subcommand": {
                    "type": "string",
                    "description": "Git 子命令 (diff, log, status, show 等)",
                },
                "args": {
                    "type": "string",
                    "description": "额外参数",
                    "default": "",
                },
            },
            "required": ["subcommand"],
        },
    },
    "grep": {
        "name": "grep",
        "description": "搜索代码内容（正则表达式匹配）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "搜索模式（正则）"},
                "path": {
                    "type": "string",
                    "description": "搜索路径",
                    "default": ".",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "文件过滤 (如 '*.py')",
                    "default": "",
                },
            },
            "required": ["pattern"],
        },
    },
    "glob": {
        "name": "glob",
        "description": "按文件名模式查找文件。",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "文件名模式 (如 '*.py')"},
                "path": {
                    "type": "string",
                    "description": "搜索路径",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
    },
    "submit": {
        "name": "submit",
        "description": "提交最终结果，结束任务。用结论或最终输出作为 result 参数。",
        "input_schema": {
            "type": "object",
            "properties": {
                "result": {"type": "string", "description": "最终结果/结论"},
            },
            "required": ["result"],
        },
    },
    "delegate": {
        "name": "delegate",
        "description": "同步委派任务给同事，等待完成后返回结果。",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_name": {
                    "type": "string",
                    "description": "员工名称",
                },
                "task": {
                    "type": "string",
                    "description": "任务描述",
                },
            },
            "required": ["employee_name", "task"],
        },
    },
    "delegate_async": {
        "name": "delegate_async",
        "description": "异步委派，立即返回任务 ID。适合并行派多人。",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_name": {
                    "type": "string",
                    "description": "员工名称",
                },
                "task": {
                    "type": "string",
                    "description": "任务描述",
                },
            },
            "required": ["employee_name", "task"],
        },
    },
    "check_task": {
        "name": "check_task",
        "description": "查询异步任务状态和结果。",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "任务 ID",
                },
            },
            "required": ["task_id"],
        },
    },
    "list_tasks": {
        "name": "list_tasks",
        "description": "列出最近异步任务。",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "running", "completed", "failed"],
                    "description": "按状态筛选",
                },
                "type": {
                    "type": "string",
                    "enum": ["employee", "meeting", "pipeline"],
                    "description": "按类型筛选",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回数量上限（默认 10）",
                    "default": 10,
                },
            },
        },
    },
    "organize_meeting": {
        "name": "organize_meeting",
        "description": "组织讨论会议。",
        "input_schema": {
            "type": "object",
            "properties": {
                "employees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "参会员工名称列表",
                },
                "topic": {
                    "type": "string",
                    "description": "会议议题",
                },
                "goal": {
                    "type": "string",
                    "description": "会议目标",
                },
                "rounds": {
                    "type": "integer",
                    "description": "讨论轮次（默认 2）",
                    "default": 2,
                },
            },
            "required": ["employees", "topic"],
        },
    },
    "check_meeting": {
        "name": "check_meeting",
        "description": "查会议结果。",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "会议 ID",
                },
            },
            "required": ["task_id"],
        },
    },
    "run_pipeline": {
        "name": "run_pipeline",
        "description": "执行预定义流水线，异步返回任务 ID。",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "流水线名称",
                },
                "args": {
                    "type": "object",
                    "description": "流水线参数",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["name"],
        },
    },
    "delegate_chain": {
        "name": "delegate_chain",
        "description": "按顺序委派多人，前步结果自动传给下步。异步返回任务 ID。",
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "employee_name": {
                                "type": "string",
                                "description": "目标员工名称",
                            },
                            "task": {
                                "type": "string",
                                "description": "任务描述，可用 {prev} 引用上一步结果",
                            },
                        },
                        "required": ["employee_name", "task"],
                    },
                    "description": "按顺序执行的步骤列表",
                },
            },
            "required": ["steps"],
        },
    },
    "route": {
        "name": "route",
        "description": "按路由模板发起委派链。",
        "input_schema": {
            "type": "object",
            "properties": {
                "template": {
                    "type": "string",
                    "description": "模板名称",
                },
                "task": {
                    "type": "string",
                    "description": "任务描述",
                },
                "overrides": {
                    "type": "object",
                    "description": "覆盖默认员工 {role: name}",
                },
            },
            "required": ["template", "task"],
        },
    },
    "query_cost": {
        "name": "query_cost",
        "description": "查询员工的 token 消耗和成本汇总。可按员工、按时间段筛选。",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee": {
                    "type": "string",
                    "description": "员工名称（可选，不填则查全部）",
                },
                "days": {
                    "type": "integer",
                    "description": "查询天数（默认 7）",
                },
            },
            "required": [],
        },
    },
    "schedule_task": {
        "name": "schedule_task",
        "description": "创建定时任务。",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "任务名称（唯一标识）",
                },
                "cron": {
                    "type": "string",
                    "description": "cron 表达式",
                },
                "employee_name": {
                    "type": "string",
                    "description": "执行的员工名称",
                },
                "task": {
                    "type": "string",
                    "description": "任务描述",
                },
            },
            "required": ["name", "cron", "employee_name", "task"],
        },
    },
    "list_schedules": {
        "name": "list_schedules",
        "description": "列出定时任务。",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    "cancel_schedule": {
        "name": "cancel_schedule",
        "description": "取消一个定时任务。",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "要取消的定时任务名称",
                },
            },
            "required": ["name"],
        },
    },
    "agent_file_read": {
        "name": "agent_file_read",
        "description": "读取项目文件。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（相对于项目根目录）",
                },
                "start_line": {
                    "type": "integer",
                    "description": "起始行号（可选，默认从头开始）",
                },
                "end_line": {
                    "type": "integer",
                    "description": "结束行号（可选，默认读到尾）",
                },
            },
            "required": ["path"],
        },
    },
    "agent_file_grep": {
        "name": "agent_file_grep",
        "description": "搜索项目文件内容。",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "正则表达式",
                },
                "path": {
                    "type": "string",
                    "description": "目录路径",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "文件名过滤（如 '*.py'）",
                },
            },
            "required": ["pattern"],
        },
    },
    "agent_bash": {
        "name": "agent_bash",
        "description": "在项目目录内执行 Shell 命令。不使用 shell 模式，通过 shlex 解析。",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "执行单条命令（不支持管道、重定向、&& 等 shell 特性，通过 shlex 解析为参数列表）",
                },
                "timeout": {
                    "type": "integer",
                    "description": "超时秒数（默认 120，最大 600）",
                    "default": 120,
                },
                "working_dir": {
                    "type": "string",
                    "description": "工作目录（相对于项目根目录，可选）",
                },
            },
            "required": ["command"],
        },
    },
    "agent_file_write": {
        "name": "agent_file_write",
        "description": "写入文件（新建或覆盖）。已有文件会自动备份。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "文件路径（相对于项目根目录）",
                },
                "content": {
                    "type": "string",
                    "description": "文件内容",
                },
                "create_dirs": {
                    "type": "boolean",
                    "description": "自动创建父目录（默认 true）",
                    "default": True,
                },
            },
            "required": ["file_path", "content"],
        },
    },
    "agent_file_edit": {
        "name": "agent_file_edit",
        "description": "精确编辑文件（search/replace）。编辑前自动备份。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "文件路径（相对于项目根目录）",
                },
                "old_string": {
                    "type": "string",
                    "description": "要替换的文本",
                },
                "new_string": {
                    "type": "string",
                    "description": "替换为的文本",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "替换所有匹配（默认 false，仅替换第一个且要求唯一）",
                    "default": False,
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    "agent_git": {
        "name": "agent_git",
        "description": "执行 Git 操作。command 不含 git 前缀，如 status、diff --stat。",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Git 子命令（不含 git 前缀）",
                },
            },
            "required": ["command"],
        },
    },
    "find_free_time": {
        "name": "find_free_time",
        "description": "查共同空闲时间。",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "飞书用户 open_id 列表",
                },
                "days": {
                    "type": "integer",
                    "description": "天数",
                    "default": 7,
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "分钟",
                    "default": 60,
                },
            },
            "required": ["user_ids"],
        },
    },
    "send_message": {
        "name": "send_message",
        "description": "给用户或AI同事发一条消息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient_id": {
                    "type": "integer",
                    "description": "收件人用户ID。Kai=1",
                },
                "content": {
                    "type": "string",
                    "description": "消息内容",
                },
            },
            "required": ["recipient_id", "content"],
        },
    },
    "list_agents": {
        "name": "list_agents",
        "description": "查看所有AI同事的列表和当前状态。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "web_search": {
        "name": "web_search",
        "description": "搜索互联网获取实时信息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回结果数量上限",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    "create_note": {
        "name": "create_note",
        "description": "保存笔记。",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "标题",
                },
                "content": {
                    "type": "string",
                    "description": "内容（markdown）",
                },
                "tags": {
                    "type": "string",
                    "description": "标签，逗号分隔",
                    "default": "",
                },
                "visibility": {
                    "type": "string",
                    "enum": ["open", "private"],
                    "default": "open",
                    "description": "可见性: open=公开, private=仅私聊可见",
                },
            },
            "required": ["title", "content"],
        },
    },
    "lookup_user": {
        "name": "lookup_user",
        "description": "查用户详情。",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "用户昵称",
                },
            },
            "required": ["name"],
        },
    },
    "read_notes": {
        "name": "read_notes",
        "description": "查看笔记。",
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "关键词",
                    "default": "",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回数量上限",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
    "project_status": {
        "name": "project_status",
        "description": "查询项目开发状态。refresh=true 实时刷新。",
        "input_schema": {
            "type": "object",
            "properties": {
                "refresh": {
                    "type": "boolean",
                    "description": "true=运行脚本实时刷新，false=读缓存（默认 false）",
                    "default": False,
                },
            },
            "required": [],
        },
    },
    "read_feishu_calendar": {
        "name": "read_feishu_calendar",
        "description": "查看飞书日历日程。",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "起始日期 YYYY-MM-DD，默认今天",
                },
                "days": {
                    "type": "integer",
                    "description": "查几天（1=今天，7=本周），默认 1",
                    "default": 1,
                },
            },
            "required": [],
        },
    },
    "delete_feishu_event": {
        "name": "delete_feishu_event",
        "description": "删除飞书日程。",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "要删除的日程 event_id",
                },
            },
            "required": ["event_id"],
        },
    },
    "create_feishu_task": {
        "name": "create_feishu_task",
        "description": "创建飞书待办任务。",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "任务标题",
                },
                "due": {
                    "type": "string",
                    "description": "截止日期 YYYY-MM-DD（可选）",
                },
                "description": {
                    "type": "string",
                    "description": "任务描述（可选）",
                    "default": "",
                },
            },
            "required": ["summary"],
        },
    },
    "list_feishu_tasks": {
        "name": "list_feishu_tasks",
        "description": "查看飞书待办任务列表。",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "返回条数，默认 20",
                    "default": 20,
                },
            },
            "required": [],
        },
    },
    "complete_feishu_task": {
        "name": "complete_feishu_task",
        "description": "完成飞书待办任务。",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "要完成的任务 task_id",
                },
            },
            "required": ["task_id"],
        },
    },
    "delete_feishu_task": {
        "name": "delete_feishu_task",
        "description": "删除飞书待办任务。",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "要删除的任务 task_id",
                },
            },
            "required": ["task_id"],
        },
    },
    "update_feishu_task": {
        "name": "update_feishu_task",
        "description": "更新飞书待办任务。",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "要更新的任务 task_id",
                },
                "summary": {
                    "type": "string",
                    "description": "新标题（可选）",
                },
                "due": {
                    "type": "string",
                    "description": "新截止日期 YYYY-MM-DD（可选）",
                },
                "description": {
                    "type": "string",
                    "description": "新描述（可选）",
                },
            },
            "required": ["task_id"],
        },
    },
    "get_datetime": {
        "name": "get_datetime",
        "description": "获取当前日期时间。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "calculate": {
        "name": "calculate",
        "description": "计算数学表达式。",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式",
                },
            },
            "required": ["expression"],
        },
    },
    "feishu_chat_history": {
        "name": "feishu_chat_history",
        "description": "读取飞书群/会话的最近消息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "string",
                    "description": "群聊 ID",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回消息条数，默认 10",
                    "default": 10,
                },
            },
            "required": ["chat_id"],
        },
    },
    "weather": {
        "name": "weather",
        "description": "查城市天气。",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名",
                },
                "days": {
                    "type": "integer",
                    "description": "天数",
                    "default": 3,
                },
            },
            "required": ["city"],
        },
    },
    "create_feishu_event": {
        "name": "create_feishu_event",
        "description": "创建飞书日历日程。",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "日程标题",
                },
                "date": {
                    "type": "string",
                    "description": "YYYY-MM-DD",
                },
                "start_hour": {
                    "type": "integer",
                    "description": "0-23",
                },
                "start_minute": {
                    "type": "integer",
                    "description": "0-59",
                    "default": 0,
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "分钟",
                    "default": 60,
                },
                "description": {
                    "type": "string",
                    "description": "描述",
                    "default": "",
                },
            },
            "required": ["summary", "date", "start_hour"],
        },
    },
    # ── 飞书文档工具 ──
    "search_feishu_docs": {
        "name": "search_feishu_docs",
        "description": "搜索飞书云文档。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "count": {"type": "integer", "description": "返回数量上限", "default": 10},
            },
            "required": ["query"],
        },
    },
    "read_feishu_doc": {
        "name": "read_feishu_doc",
        "description": "读取飞书文档内容。",
        "input_schema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "文档 ID"},
            },
            "required": ["document_id"],
        },
    },
    "create_feishu_doc": {
        "name": "create_feishu_doc",
        "description": "创建飞书文档。",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "文档标题"},
                "content": {"type": "string", "description": "文档正文"},
                "folder_token": {
                    "type": "string",
                    "description": "文件夹 token",
                    "default": "",
                },
            },
            "required": ["title", "content"],
        },
    },
    "send_feishu_group": {
        "name": "send_feishu_group",
        "description": "发消息到飞书群。",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": {"type": "string", "description": "群聊 ID"},
                "text": {"type": "string", "description": "消息内容"},
            },
            "required": ["chat_id", "text"],
        },
    },
    "send_feishu_file": {
        "name": "send_feishu_file",
        "description": "发送文件到飞书群。",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": {"type": "string", "description": "群聊 ID"},
                "file_name": {"type": "string", "description": "文件名"},
                "content": {"type": "string", "description": "文件内容"},
            },
            "required": ["chat_id", "file_name", "content"],
        },
    },
    "list_feishu_groups": {
        "name": "list_feishu_groups",
        "description": "列出飞书群列表。",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    # ── GitHub 工具 ──
    "github_prs": {
        "name": "github_prs",
        "description": "查看 GitHub 仓库的 Pull Requests 列表。",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "仓库（格式: owner/repo）"},
                "state": {
                    "type": "string",
                    "description": "状态: open/closed/all",
                    "default": "open",
                },
                "limit": {"type": "integer", "description": "返回数量上限", "default": 10},
            },
            "required": ["repo"],
        },
    },
    "github_issues": {
        "name": "github_issues",
        "description": "查看 GitHub 仓库的 Issues 列表。",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "仓库（格式: owner/repo）"},
                "state": {
                    "type": "string",
                    "description": "状态: open/closed/all",
                    "default": "open",
                },
                "labels": {"type": "string", "description": "标签过滤（逗号分隔）", "default": ""},
                "limit": {"type": "integer", "description": "返回数量上限", "default": 10},
            },
            "required": ["repo"],
        },
    },
    "github_repo_activity": {
        "name": "github_repo_activity",
        "description": "查看 GitHub 仓库最近活动：提交记录、贡献者。",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "仓库（格式: owner/repo）"},
                "days": {"type": "integer", "description": "查最近几天", "default": 7},
            },
            "required": ["repo"],
        },
    },
    # ── Notion 工具 ──
    "notion_search": {
        "name": "notion_search",
        "description": "搜索 Notion 页面和数据库。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "description": "返回数量上限", "default": 10},
            },
            "required": ["query"],
        },
    },
    "notion_read": {
        "name": "notion_read",
        "description": "读取 Notion 页面内容。传入页面 ID，返回正文文本。",
        "input_schema": {
            "type": "object",
            "properties": {
                "page_id": {"type": "string", "description": "Notion 页面 ID"},
            },
            "required": ["page_id"],
        },
    },
    "notion_create": {
        "name": "notion_create",
        "description": "在 Notion 创建新页面。",
        "input_schema": {
            "type": "object",
            "properties": {
                "parent_id": {"type": "string", "description": "父页面 ID"},
                "title": {"type": "string", "description": "页面标题"},
                "content": {"type": "string", "description": "页面内容"},
            },
            "required": ["parent_id", "title", "content"],
        },
    },
    "send_feishu_dm": {
        "name": "send_feishu_dm",
        "description": "给飞书用户发私聊消息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "open_id": {
                    "type": "string",
                    "description": "接收人 open_id",
                },
                "text": {
                    "type": "string",
                    "description": "消息内容",
                },
            },
            "required": ["open_id", "text"],
        },
    },
    "feishu_group_members": {
        "name": "feishu_group_members",
        "description": "查看飞书群成员列表。",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "string",
                    "description": "群聊 ID",
                },
            },
            "required": ["chat_id"],
        },
    },
    # ── 金融工具 ──
    "exchange_rate": {
        "name": "exchange_rate",
        "description": "查汇率。",
        "input_schema": {
            "type": "object",
            "properties": {
                "from": {
                    "type": "string",
                    "description": "货币代码",
                    "default": "USD",
                },
                "to": {
                    "type": "string",
                    "description": "目标货币，逗号分隔",
                    "default": "CNY",
                },
            },
            "required": [],
        },
    },
    "stock_price": {
        "name": "stock_price",
        "description": "查股价（A股/美股）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "股票代码",
                },
            },
            "required": ["symbol"],
        },
    },
    # ── 记忆工具 ──
    "track_decision": {
        "name": "track_decision",
        "description": "记录一个待评估的决策（技术选型、方案建议、工期预估等）。系统会在 deadline 后自动回收评估。",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["estimate", "recommendation", "commitment"],
                    "description": "决策类别: estimate=预估, recommendation=建议, commitment=承诺",
                },
                "content": {
                    "type": "string",
                    "description": "决策内容（如「选用 Redis 做缓存，预计 3 天完成」）",
                },
                "expected_outcome": {
                    "type": "string",
                    "description": "预期结果（可选）",
                    "default": "",
                },
                "deadline": {
                    "type": "string",
                    "description": "评估截止日期 ISO 格式（可选，默认 7 天后）",
                    "default": "",
                },
            },
            "required": ["category", "content"],
        },
    },
    "add_memory": {
        "name": "add_memory",
        "description": "记录持久记忆。",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "finding/decision/correction",
                    "default": "finding",
                },
                "content": {
                    "type": "string",
                    "description": "记忆内容",
                },
                "visibility": {
                    "type": "string",
                    "enum": ["open", "private"],
                    "default": "open",
                    "description": "可见性: open=公开, private=仅私聊可见",
                },
                "classification": {
                    "type": "string",
                    "enum": ["public", "internal", "restricted", "confidential"],
                    "default": "internal",
                    "description": "信息分级: public=可对外, internal=仅内部(默认), restricted=受限(需指定domain), confidential=机密",
                },
                "domain": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "职能域标签(仅restricted时使用)，如 ['hr'], ['finance']",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "语义标签列表（必填，至少一个）",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "关键词列表，用于后续关联匹配（必填，至少一个）",
                },
                "importance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "重要度 1-5（必填）",
                },
                "confidence": {
                    "type": "number",
                    "description": "置信度(0.0-1.0, 默认1.0)",
                    "default": 1.0,
                },
                "trigger_condition": {
                    "type": "string",
                    "description": "什么场景下该回忆这条（必填）",
                },
                "applicability": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "适用的角色或领域标签（必填，至少一个）",
                },
            },
            "required": [
                "content",
                "tags",
                "keywords",
                "importance",
                "trigger_condition",
                "applicability",
            ],
        },
    },
    "query_memory": {
        "name": "query_memory",
        "description": "查询员工的持久化记忆。可按类别过滤。",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee": {
                    "type": "string",
                    "description": "员工名称（slug）",
                },
                "category": {
                    "type": "string",
                    "enum": ["decision", "estimate", "finding", "correction", "pattern"],
                    "description": "按类别过滤（可选）",
                },
                "limit": {
                    "type": "integer",
                    "description": "最大返回条数（默认 20）",
                    "default": 20,
                },
                "classification_max": {
                    "type": "string",
                    "enum": ["public", "internal", "restricted", "confidential"],
                    "description": "最高信息分级（可选，不传则不过滤）",
                },
            },
            "required": ["employee"],
        },
    },
    # ── 生活助手 ──
    "translate": {
        "name": "translate",
        "description": "中英互译。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要翻译的文本"},
                "from_lang": {
                    "type": "string",
                    "description": "auto/zh/en/ja",
                    "default": "auto",
                },
                "to_lang": {
                    "type": "string",
                    "description": "目标语言",
                    "default": "",
                },
            },
            "required": ["text"],
        },
    },
    "countdown": {
        "name": "countdown",
        "description": "计算距离某日期的天数。",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "目标日期，YYYY-MM-DD 格式"},
                "event": {
                    "type": "string",
                    "description": "事件名称（可选，如「融资截止」）",
                    "default": "",
                },
            },
            "required": ["date"],
        },
    },
    "trending": {
        "name": "trending",
        "description": "查看热搜榜。",
        "input_schema": {
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "description": "weibo/zhihu",
                    "default": "weibo",
                },
                "limit": {"type": "integer", "description": "返回条数", "default": 15},
            },
            "required": [],
        },
    },
    # ── 飞书表格 ──
    "read_feishu_sheet": {
        "name": "read_feishu_sheet",
        "description": "读取飞书表格数据。",
        "input_schema": {
            "type": "object",
            "properties": {
                "spreadsheet_token": {
                    "type": "string",
                    "description": "飞书表格 token（从搜索结果或 URL 中获取）",
                },
                "sheet_id": {
                    "type": "string",
                    "description": "工作表 ID（可选，默认第一个 sheet）",
                    "default": "",
                },
                "range": {
                    "type": "string",
                    "description": "读取范围，如 A1:Z100（可选，默认前 100 行）",
                    "default": "",
                },
            },
            "required": ["spreadsheet_token"],
        },
    },
    "update_feishu_sheet": {
        "name": "update_feishu_sheet",
        "description": "写入飞书表格数据。",
        "input_schema": {
            "type": "object",
            "properties": {
                "spreadsheet_token": {
                    "type": "string",
                    "description": "飞书表格 token",
                },
                "sheet_id": {
                    "type": "string",
                    "description": "工作表 ID（可选，默认第一个 sheet）",
                    "default": "",
                },
                "range": {
                    "type": "string",
                    "description": "写入范围，如 A1:C3",
                },
                "values": {
                    "type": "string",
                    "description": "JSON 二维数组",
                },
            },
            "required": ["spreadsheet_token", "range", "values"],
        },
    },
    # ── 飞书审批 ──
    "list_feishu_approvals": {
        "name": "list_feishu_approvals",
        "description": "查看飞书审批列表。",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "PENDING/APPROVED/REJECTED/ALL",
                    "default": "PENDING",
                },
                "limit": {"type": "integer", "description": "返回条数", "default": 10},
            },
            "required": [],
        },
    },
    # ── 编码 & 开发辅助 ──
    "base64_codec": {
        "name": "base64_codec",
        "description": "Base64 编解码。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要编码或解码的文本"},
                "decode": {
                    "type": "boolean",
                    "description": "true=解码",
                    "default": False,
                },
            },
            "required": ["text"],
        },
    },
    "color_convert": {
        "name": "color_convert",
        "description": "颜色格式转换（HEX/RGB/HSL）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "color": {
                    "type": "string",
                    "description": "颜色值",
                },
            },
            "required": ["color"],
        },
    },
    "cron_explain": {
        "name": "cron_explain",
        "description": "解释或生成 cron 表达式。",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "cron 表达式或自然语言",
                },
            },
            "required": ["expression"],
        },
    },
    "regex_test": {
        "name": "regex_test",
        "description": "测试正则表达式。",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "正则表达式"},
                "text": {"type": "string", "description": "要测试的文本"},
                "replace": {
                    "type": "string",
                    "description": "替换字符串（可选，不填则只匹配）",
                    "default": "",
                },
            },
            "required": ["pattern", "text"],
        },
    },
    "hash_gen": {
        "name": "hash_gen",
        "description": "计算哈希值。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要计算哈希的文本"},
                "algorithm": {
                    "type": "string",
                    "description": "md5/sha1/sha256",
                    "default": "sha256",
                },
            },
            "required": ["text"],
        },
    },
    "url_codec": {
        "name": "url_codec",
        "description": "URL 编解码。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要编码或解码的文本/URL"},
                "decode": {
                    "type": "boolean",
                    "description": "true=解码",
                    "default": False,
                },
            },
            "required": ["text"],
        },
    },
    # ── 文本 & 开发工具 ──
    "text_extract": {
        "name": "text_extract",
        "description": "提取文本中的结构化信息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要提取信息的文本"},
                "extract_type": {
                    "type": "string",
                    "description": "email/phone/url/money/all",
                    "default": "all",
                },
            },
            "required": ["text"],
        },
    },
    "json_format": {
        "name": "json_format",
        "description": "格式化/压缩 JSON。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "JSON 文本或包含 JSON 的文本"},
                "compact": {
                    "type": "boolean",
                    "description": "true=压缩",
                    "default": False,
                },
            },
            "required": ["text"],
        },
    },
    "password_gen": {
        "name": "password_gen",
        "description": "生成随机密码。",
        "input_schema": {
            "type": "object",
            "properties": {
                "length": {"type": "integer", "description": "密码长度（默认 16）", "default": 16},
                "count": {"type": "integer", "description": "生成几个（默认 3）", "default": 3},
                "no_symbols": {
                    "type": "boolean",
                    "description": "是否不含特殊字符（默认 false）",
                    "default": False,
                },
            },
            "required": [],
        },
    },
    "ip_lookup": {
        "name": "ip_lookup",
        "description": "IP 地理位置查询。",
        "input_schema": {
            "type": "object",
            "properties": {
                "ip": {
                    "type": "string",
                    "description": "IP 地址（留空查本机公网 IP）",
                    "default": "",
                },
            },
            "required": [],
        },
    },
    "short_url": {
        "name": "short_url",
        "description": "生成短链接。",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要缩短的 URL"},
            },
            "required": ["url"],
        },
    },
    "word_count": {
        "name": "word_count",
        "description": "统计文本字数。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要统计的文本"},
            },
            "required": ["text"],
        },
    },
    # ── 实用工具 ──
    "unit_convert": {
        "name": "unit_convert",
        "description": "单位换算。",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "数值"},
                "from_unit": {"type": "string", "description": "原单位，如 km, lb, °F, GB"},
                "to_unit": {"type": "string", "description": "目标单位"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    "random_pick": {
        "name": "random_pick",
        "description": "随机选择或掷骰子。",
        "input_schema": {
            "type": "object",
            "properties": {
                "options": {
                    "type": "string",
                    "description": "逗号分隔选项，留空掷骰子",
                    "default": "",
                },
                "count": {"type": "integer", "description": "选几个（默认 1）", "default": 1},
            },
            "required": [],
        },
    },
    "holidays": {
        "name": "holidays",
        "description": "查询节假日和调休安排。",
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "年份（默认今年）", "default": 0},
                "month": {
                    "type": "integer",
                    "description": "月份（可选，查某月安排）",
                    "default": 0,
                },
            },
            "required": [],
        },
    },
    "timestamp_convert": {
        "name": "timestamp_convert",
        "description": "时间戳互转。",
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "时间戳或日期时间",
                },
            },
            "required": ["input"],
        },
    },
    "create_feishu_spreadsheet": {
        "name": "create_feishu_spreadsheet",
        "description": "创建飞书表格。",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "表格标题"},
                "folder_token": {
                    "type": "string",
                    "description": "文件夹 token（可选，默认根目录）",
                    "default": "",
                },
            },
            "required": ["title"],
        },
    },
    "feishu_contacts": {
        "name": "feishu_contacts",
        "description": "搜索飞书通讯录。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词（姓名）"},
                "limit": {"type": "integer", "description": "返回条数", "default": 5},
            },
            "required": ["query"],
        },
    },
    # ── 信息采集工具 ──
    "read_url": {
        "name": "read_url",
        "description": "读取网页内容，自动提取正文。适合读文章、文档、公告。",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "网页 URL"},
            },
            "required": ["url"],
        },
    },
    "rss_read": {
        "name": "rss_read",
        "description": "读取 RSS/Atom 订阅源，返回最近条目的标题、链接和摘要。",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "RSS/Atom 订阅源 URL"},
                "limit": {"type": "integer", "description": "返回条目数量上限", "default": 10},
            },
            "required": ["url"],
        },
    },
    # ── 飞书增强 ──
    "feishu_bitable": {
        "name": "feishu_bitable",
        "description": "读取飞书多维表格数据。",
        "input_schema": {
            "type": "object",
            "properties": {
                "app_token": {"type": "string", "description": "多维表格 app_token"},
                "table_id": {"type": "string", "description": "数据表 table_id"},
                "filter": {
                    "type": "string",
                    "description": "过滤条件（可选，如字段名=值）",
                    "default": "",
                },
                "limit": {"type": "integer", "description": "返回条数", "default": 20},
            },
            "required": ["app_token", "table_id"],
        },
    },
    "feishu_wiki": {
        "name": "feishu_wiki",
        "description": "搜索飞书知识库。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "description": "返回条数", "default": 10},
            },
            "required": ["query"],
        },
    },
    "approve_feishu": {
        "name": "approve_feishu",
        "description": "通过或拒绝飞书审批。",
        "input_schema": {
            "type": "object",
            "properties": {
                "instance_id": {"type": "string", "description": "审批实例 ID"},
                "action": {
                    "type": "string",
                    "description": "approve/reject",
                },
                "comment": {
                    "type": "string",
                    "description": "审批意见（可选）",
                    "default": "",
                },
            },
            "required": ["instance_id", "action"],
        },
    },
    # ── AI 能力 ──
    "summarize": {
        "name": "summarize",
        "description": "生成文本摘要。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要摘要的文本"},
                "style": {
                    "type": "string",
                    "description": "bullet/paragraph/oneline",
                    "default": "bullet",
                },
            },
            "required": ["text"],
        },
    },
    "sentiment": {
        "name": "sentiment",
        "description": "情感分析。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要分析的文本"},
            },
            "required": ["text"],
        },
    },
    # ── 效率工具 ──
    "email_send": {
        "name": "email_send",
        "description": "发送邮件。",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "收件人邮箱地址"},
                "subject": {"type": "string", "description": "邮件主题"},
                "body": {"type": "string", "description": "邮件正文"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    "email_read": {
        "name": "email_read",
        "description": "读取 Gmail 收件箱邮件。支持按数量、未读状态、搜索条件筛选。",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "最大返回数量（默认 10，上限 50）",
                    "default": 10,
                },
                "unread_only": {
                    "type": "boolean",
                    "description": "是否仅返回未读邮件",
                    "default": False,
                },
                "query": {
                    "type": "string",
                    "description": "Gmail 搜索语法（如 'from:alice subject:meeting'、'after:2026/03/01'）",
                },
                "body_length": {
                    "type": "integer",
                    "description": "正文摘要最大字符数（默认 200，上限 2000）",
                    "default": 200,
                },
            },
            "required": [],
        },
    },
    "qrcode": {
        "name": "qrcode",
        "description": "生成二维码。",
        "input_schema": {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "文本或 URL"},
                "size": {"type": "integer", "description": "图片尺寸（像素）", "default": 300},
            },
            "required": ["data"],
        },
    },
    "diff_text": {
        "name": "diff_text",
        "description": "对比文本差异。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text1": {"type": "string", "description": "原始文本"},
                "text2": {"type": "string", "description": "修改后的文本"},
            },
            "required": ["text1", "text2"],
        },
    },
    # ── 数据查询 ──
    "whois": {
        "name": "whois",
        "description": "WHOIS 查询。",
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "域名"},
            },
            "required": ["domain"],
        },
    },
    "dns_lookup": {
        "name": "dns_lookup",
        "description": "DNS 查询。",
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "域名"},
            },
            "required": ["domain"],
        },
    },
    "http_check": {
        "name": "http_check",
        "description": "检查网站可用性。",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要检查的 URL"},
            },
            "required": ["url"],
        },
    },
    # ── Python 代码执行 ──
    "run_python": {
        "name": "run_python",
        "description": "执行 Python 代码片段，用于解决内置工具做不到的事（网页抓取、数据处理、调 API、自定义计算）。可用库: httpx, bs4, json, re, datetime, math, csv, collections 等。用 print() 输出结果。",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python 代码"},
                "timeout": {
                    "type": "integer",
                    "description": "超时秒数（默认 30，最大 60）",
                },
            },
            "required": ["code"],
        },
    },
    # ── 生活服务 ──
    "express_track": {
        "name": "express_track",
        "description": "查快递物流。",
        "input_schema": {
            "type": "object",
            "properties": {
                "number": {"type": "string", "description": "快递单号"},
                "company": {
                    "type": "string",
                    "description": "快递公司代码",
                    "default": "",
                },
            },
            "required": ["number"],
        },
    },
    "flight_info": {
        "name": "flight_info",
        "description": "查航班信息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "flight_no": {"type": "string", "description": "航班号"},
                "date": {
                    "type": "string",
                    "description": "YYYY-MM-DD",
                    "default": "",
                },
            },
            "required": ["flight_no"],
        },
    },
    "aqi": {
        "name": "aqi",
        "description": "查空气质量。",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"},
            },
            "required": ["city"],
        },
    },
}

# 需要 agent loop 的工具（区别于 sandbox 工具）
AGENT_TOOLS = {
    "send_message",
    "list_agents",
    "delegate",
    "delegate_async",
    "check_task",
    "list_tasks",
    "organize_meeting",
    "check_meeting",
    "run_pipeline",
    "delegate_chain",
    "route",
    "query_cost",
    "schedule_task",
    "list_schedules",
    "cancel_schedule",
    "agent_file_read",
    "agent_file_grep",
    "find_free_time",
    "web_search",
    "create_note",
    "lookup_user",
    "read_notes",
    "project_status",
    "read_feishu_calendar",
    "delete_feishu_event",
    "create_feishu_event",
    "create_feishu_task",
    "list_feishu_tasks",
    "complete_feishu_task",
    "delete_feishu_task",
    "update_feishu_task",
    "feishu_chat_history",
    "weather",
    "get_datetime",
    "calculate",
    "send_feishu_dm",
    "feishu_group_members",
    "exchange_rate",
    "stock_price",
    # 记忆
    "add_memory",
    "query_memory",
    "track_decision",
    # 飞书文档
    "search_feishu_docs",
    "read_feishu_doc",
    "create_feishu_doc",
    "send_feishu_group",
    "send_feishu_file",
    "list_feishu_groups",
    # GitHub
    "github_prs",
    "github_issues",
    "github_repo_activity",
    # Notion
    "notion_search",
    "notion_read",
    "notion_create",
    # 信息采集
    "read_url",
    "rss_read",
    # 生活助手
    "translate",
    "countdown",
    "trending",
    # 飞书表格 & 审批
    "read_feishu_sheet",
    "update_feishu_sheet",
    "list_feishu_approvals",
    # 实用工具
    "unit_convert",
    "random_pick",
    "holidays",
    "timestamp_convert",
    "create_feishu_spreadsheet",
    "feishu_contacts",
    # 文本 & 开发
    "text_extract",
    "json_format",
    "password_gen",
    "ip_lookup",
    "short_url",
    "word_count",
    # 编码 & 开发辅助
    "base64_codec",
    "color_convert",
    "cron_explain",
    "regex_test",
    "hash_gen",
    "url_codec",
    # 飞书增强
    "feishu_bitable",
    "feishu_wiki",
    "approve_feishu",
    # AI 能力
    "summarize",
    "sentiment",
    # 效率工具
    "email_send",
    "email_read",
    "qrcode",
    "diff_text",
    # 数据查询
    "whois",
    "dns_lookup",
    "http_check",
    # 生活服务
    "express_track",
    "flight_info",
    "aqi",
    # 代码执行
    "run_python",
    # 代码操作
    "agent_bash",
    "agent_file_write",
    "agent_file_edit",
    "agent_git",
}

# 延迟加载工具 — 首轮只传名字列表，模型需要时调 load_tools 加载完整 schema
# 所有 agent 工具都延迟加载，首轮只有 load_tools + submit
DEFERRED_TOOLS = AGENT_TOOLS

# ── Phase 2: Allowlist 核心工具集 ──
# 员工默认只拥有这些工具，其余需通过外部 MCP 插件获取

CORE_TOOLS = {
    # 员工管理
    "send_message",
    "list_agents",
    "delegate",
    "delegate_async",
    "delegate_chain",
    "route",
    "check_task",
    "list_tasks",
    "organize_meeting",
    "check_meeting",
    "run_pipeline",
    "query_cost",
    "schedule_task",
    "list_schedules",
    "cancel_schedule",
    # 记忆与笔记
    "add_memory",
    "create_note",
    "read_notes",
    "track_decision",
    # 基础执行
    "get_datetime",
    "calculate",
    "lookup_user",
    "web_search",
    "read_url",
    "agent_file_read",
    "agent_file_grep",
    "project_status",
}

# 飞书工具（过渡期保留，Phase 3 拆为独立 MCP server 后移除）
FEISHU_TOOLS = {
    "read_feishu_calendar",
    "delete_feishu_event",
    "create_feishu_event",
    "find_free_time",
    "create_feishu_task",
    "list_feishu_tasks",
    "complete_feishu_task",
    "delete_feishu_task",
    "update_feishu_task",
    "feishu_chat_history",
    "send_feishu_dm",
    "feishu_group_members",
    "search_feishu_docs",
    "read_feishu_doc",
    "create_feishu_doc",
    "send_feishu_group",
    "send_feishu_file",
    "list_feishu_groups",
    "read_feishu_sheet",
    "update_feishu_sheet",
    "create_feishu_spreadsheet",
    "list_feishu_approvals",
    "feishu_contacts",
    "feishu_bitable",
    "feishu_wiki",
    "approve_feishu",
}

# ── 工具角色预设 ──

TOOL_ROLE_PRESETS: dict[str, set[str]] = {
    "readonly": {"file_read", "grep", "glob"},
    "developer": {
        "file_read",
        "file_write",
        "bash",
        "git",
        "grep",
        "glob",
        "agent_file_read",
        "agent_file_grep",
        "agent_file_write",
        "agent_file_edit",
        "agent_bash",
        "agent_git",
    },
    "agent-core": {"submit", "delegate", "delegate_async", "check_task", "list_tasks"},
    "meeting": {"organize_meeting", "check_meeting"},
    "pipeline": {"run_pipeline", "delegate_chain", "route"},
    "scheduling": {"schedule_task", "list_schedules", "cancel_schedule"},
    "feishu-read": {
        "read_feishu_calendar",
        "list_feishu_tasks",
        "feishu_chat_history",
        "list_feishu_groups",
        "feishu_group_members",
        "feishu_contacts",
        "search_feishu_docs",
        "read_feishu_doc",
        "read_feishu_sheet",
        "list_feishu_approvals",
        "feishu_bitable",
        "feishu_wiki",
    },
    "feishu-write": {
        "create_feishu_event",
        "delete_feishu_event",
        "create_feishu_task",
        "complete_feishu_task",
        "delete_feishu_task",
        "update_feishu_task",
        "create_feishu_doc",
        "send_feishu_group",
        "send_feishu_file",
        "send_feishu_dm",
        "update_feishu_sheet",
        "create_feishu_spreadsheet",
        "approve_feishu",
    },
    "github": {"github_prs", "github_issues", "github_repo_activity"},
    "notion": {"notion_search", "notion_read", "notion_create"},
    "web": {"web_search", "read_url", "rss_read"},
    "memory": {"add_memory", "create_note", "read_notes"},
    "admin": {
        "send_message",
        "list_agents",
        "lookup_user",
        "project_status",
    },
    "utilities": {
        "get_datetime",
        "calculate",
        "weather",
        "translate",
        "countdown",
        "trending",
        "exchange_rate",
        "stock_price",
        "unit_convert",
        "random_pick",
        "holidays",
        "timestamp_convert",
    },
    "dev-tools": {
        "base64_codec",
        "color_convert",
        "cron_explain",
        "regex_test",
        "hash_gen",
        "url_codec",
        "text_extract",
        "json_format",
        "password_gen",
        "ip_lookup",
        "short_url",
        "word_count",
        "diff_text",
        "whois",
        "dns_lookup",
        "http_check",
        "summarize",
        "sentiment",
        "qrcode",
        "run_python",
    },
    "life": {"express_track", "flight_info", "aqi", "email_send", "email_read"},
}

# ── 技能包 — load_tools 的语义分组，LLM 按包名一次加载 ──

SKILL_PACKS: dict[str, dict[str, Any]] = {
    "feishu": {
        "label": "飞书",
        "description": "日历/任务/文档/消息/审批/表格",
        "tools": TOOL_ROLE_PRESETS["feishu-read"] | TOOL_ROLE_PRESETS["feishu-write"],
    },
    "github": {
        "label": "GitHub",
        "description": "PR、Issue、仓库动态",
        "tools": TOOL_ROLE_PRESETS["github"],
    },
    "notion": {
        "label": "Notion",
        "description": "搜索/读取/创建页面",
        "tools": TOOL_ROLE_PRESETS["notion"],
    },
    "admin": {
        "label": "后台管理",
        "description": "用户查询/员工管理/系统健康/消息/成本",
        "tools": TOOL_ROLE_PRESETS["admin"] | {"query_cost"},
    },
    "delegate": {
        "label": "委派",
        "description": "委派任务给其他员工/检查任务状态/链式委派/路由",
        "tools": TOOL_ROLE_PRESETS["agent-core"] | TOOL_ROLE_PRESETS["pipeline"],
    },
    "web": {
        "label": "网络搜索",
        "description": "搜索/读网页/RSS",
        "tools": TOOL_ROLE_PRESETS["web"],
    },
    "memory": {
        "label": "记忆",
        "description": "添加记忆/创建笔记/读取笔记",
        "tools": TOOL_ROLE_PRESETS["memory"],
    },
    "scheduling": {
        "label": "日程",
        "description": "定时任务/会议安排/空闲时间查询",
        "tools": (
            TOOL_ROLE_PRESETS["scheduling"] | TOOL_ROLE_PRESETS["meeting"] | {"find_free_time"}
        ),
    },
    "utilities": {
        "label": "工具箱",
        "description": "计算/天气/汇率/翻译/时间/单位换算",
        "tools": TOOL_ROLE_PRESETS["utilities"],
    },
    "dev-tools": {
        "label": "开发工具",
        "description": "编码/JSON/正则/IP查询/文本处理/摘要",
        "tools": TOOL_ROLE_PRESETS["dev-tools"] | {"agent_file_read", "agent_file_grep"},
    },
    "code": {
        "label": "代码操作",
        "description": "代码操作（文件读写、Shell、Git）",
        "tools": {
            "agent_file_read",
            "agent_file_grep",
            "agent_file_write",
            "agent_file_edit",
            "agent_bash",
            "agent_git",
        },
    },
    "life": {
        "label": "生活服务",
        "description": "快递/航班/空气质量/邮件收发",
        "tools": TOOL_ROLE_PRESETS["life"],
    },
}

# 组合角色
TOOL_ROLE_PRESETS["feishu-admin"] = (
    TOOL_ROLE_PRESETS["feishu-read"] | TOOL_ROLE_PRESETS["feishu-write"]
)
TOOL_ROLE_PRESETS["all-agent"] = AGENT_TOOLS.copy()
TOOL_ROLE_PRESETS["all"] = AGENT_TOOLS | {"file_read", "file_write", "bash", "git", "grep", "glob"}

# ── 历史 Deny 清单（Phase 1 遗留，保留作为文档）──

# 工程师: 去掉与工程无关的生活/商务工具
DENY_ENGINEER = {"weather", "exchange_rate", "stock_price", "flight_info", "aqi", "express_track"}

# 研究员: 去掉管理后台写操作和委派
DENY_RESEARCHER = {"delegate_async", "delegate_chain", "route"}

# 商务: 去掉 GitHub 和代码相关
DENY_BUSINESS = {"github_prs", "github_issues", "github_repo_activity"}

# 安全审计: 去掉可能影响审计独立性的写操作
DENY_SECURITY = {
    "delegate",
    "delegate_async",
    "delegate_chain",
    "route",
    "send_feishu_dm",
}

# 数据组: 去掉生活服务工具
DENY_DATA = {"flight_info", "aqi", "express_track"}

# 职能组: 去掉代码执行
DENY_FUNCTIONS = {"run_python"}

# ── 团队 profile 预设（Phase 2: allowlist 模式）──
# 基线 = CORE_TOOLS + FEISHU_TOOLS（过渡期），按职能微调

TOOL_ROLE_PRESETS["profile-base"] = CORE_TOOLS | FEISHU_TOOLS | {"run_python"}

# 工程师: 核心 + 飞书
TOOL_ROLE_PRESETS["profile-engineer"] = CORE_TOOLS | FEISHU_TOOLS

# 数据组: 核心 + 飞书 + 代码执行
TOOL_ROLE_PRESETS["profile-data"] = CORE_TOOLS | FEISHU_TOOLS | {"run_python"}

# 研究员: 核心 + 飞书，去掉委派（保持审计独立性）
TOOL_ROLE_PRESETS["profile-researcher"] = (CORE_TOOLS | FEISHU_TOOLS) - {
    "delegate_async",
    "delegate_chain",
    "route",
}

# 商务: 核心 + 飞书
TOOL_ROLE_PRESETS["profile-business"] = CORE_TOOLS | FEISHU_TOOLS

# 职能组: 核心 + 飞书
TOOL_ROLE_PRESETS["profile-functions"] = CORE_TOOLS | FEISHU_TOOLS

# 基础设施: 核心 + 飞书
TOOL_ROLE_PRESETS["profile-infra"] = CORE_TOOLS | FEISHU_TOOLS

# 安全审计: 核心 + 飞书，去掉影响独立性的写操作
TOOL_ROLE_PRESETS["profile-security"] = (CORE_TOOLS | FEISHU_TOOLS) - {
    "delegate",
    "delegate_async",
    "delegate_chain",
    "route",
    "send_feishu_dm",
}


def resolve_effective_tools(employee: Employee) -> set[str]:  # noqa: F821
    """计算员工的有效工具集.

    逻辑:
    1. permissions 为 None → 直接使用 employee.tools（向后兼容）
    2. permissions 存在 → 展开 roles + allow - deny，与 tools 取交集
    """
    if employee.permissions is None:
        effective = set(employee.tools)
        # Auto-include MCP Gateway tools
        for tool_name in _TOOL_SCHEMAS:
            if tool_name.startswith("mcp__"):
                effective.add(tool_name)
        return effective

    policy = employee.permissions

    # 展开角色
    from_roles: set[str] = set()
    for role_name in policy.roles:
        preset = TOOL_ROLE_PRESETS.get(role_name)
        if preset is not None:
            from_roles |= preset

    # 合并 allow，减去 deny
    effective = from_roles | set(policy.allow)
    effective -= set(policy.deny)

    # 与 tools 声明取交集（tools 是 LLM 可见的上限）
    if employee.tools:
        effective &= set(employee.tools)

    # Auto-include MCP Gateway tools (available to all employees)
    for tool_name in _TOOL_SCHEMAS:
        if tool_name.startswith("mcp__"):
            effective.add(tool_name)

    return effective


def validate_permissions(employee: Employee) -> list[str]:  # noqa: F821
    """校验权限配置，返回警告列表."""
    warnings: list[str] = []
    if employee.permissions is None:
        return warnings

    policy = employee.permissions

    for role in policy.roles:
        if role not in TOOL_ROLE_PRESETS:
            warnings.append(f"未知角色: '{role}'")

    all_known = TOOL_ROLE_PRESETS["all"] | {"submit", "finish"}
    for tool in policy.allow:
        if tool not in all_known:
            warnings.append(f"allow 中未知工具: '{tool}'")
    for tool in policy.deny:
        if tool not in all_known:
            warnings.append(f"deny 中未知工具: '{tool}'")

    # tools 中被 deny 排除的工具
    effective = resolve_effective_tools(employee)
    denied_from_tools = set(employee.tools) - effective
    if denied_from_tools and employee.tools:
        warnings.append(f"以下已声明工具被权限排除: {', '.join(sorted(denied_from_tools))}")

    return warnings


def _make_load_tools_schema(available: set[str]) -> dict[str, Any]:
    """构建 load_tools 元工具 schema — 按技能包分组展示."""
    # 按技能包分组
    pack_lines: list[str] = []
    covered: set[str] = set()
    for pack_name, pack_def in SKILL_PACKS.items():
        pack_tools = pack_def["tools"] & available
        if pack_tools:
            pack_lines.append(f"{pack_name}({pack_def['label']}): {pack_def['description']}")
            covered |= pack_tools

    remaining = sorted(available - covered)
    desc_parts: list[str] = ["加载工具后才能调用。"]
    if pack_lines:
        desc_parts.append("技能包(用包名一次加载整包):\n" + "\n".join(pack_lines))
    if remaining:
        desc_parts.append("其他: " + ", ".join(remaining))

    return {
        "name": "load_tools",
        "description": "\n".join(desc_parts),
        "input_schema": {
            "type": "object",
            "properties": {
                "names": {
                    "type": "string",
                    "description": "技能包名或工具名，逗号分隔。如 'feishu,admin' 或 'list_agents'",
                },
            },
            "required": ["names"],
        },
    }


def employee_tools_to_schemas(
    tools: list[str],
    *,
    defer: bool = True,
) -> tuple[list[dict[str, Any]], set[str]]:
    """将 employee 的 tools 列表转为 LLM tool schemas.

    Args:
        tools: employee.tools 列表，如 ["file_read", "bash", "git"]
        defer: 是否启用延迟加载（默认 True）

    Returns:
        (schemas, deferred_names) — schemas 是核心工具 + load_tools，
        deferred_names 是被延迟的工具名集合（空集表示无延迟）。
    """
    schemas: list[dict[str, Any]] = []
    seen: set[str] = set()
    deferred: set[str] = set()
    for tool_name in tools:
        if tool_name not in _TOOL_SCHEMAS or tool_name in seen:
            continue
        if defer and tool_name in DEFERRED_TOOLS:
            deferred.add(tool_name)
        else:
            schemas.append(_TOOL_SCHEMAS[tool_name])
            seen.add(tool_name)
    # Auto-include MCP Gateway tools (available to all employees)
    for tool_name in _TOOL_SCHEMAS:
        if tool_name.startswith("mcp__") and tool_name not in seen:
            if defer and tool_name in DEFERRED_TOOLS:
                deferred.add(tool_name)
            else:
                schemas.append(_TOOL_SCHEMAS[tool_name])
                seen.add(tool_name)

    # 有延迟工具时加 load_tools 元工具
    if deferred:
        schemas.append(_make_load_tools_schema(deferred))
    # 始终包含 submit
    if "submit" not in seen:
        schemas.append(_TOOL_SCHEMAS["submit"])
    return schemas, deferred


def get_tool_schema(name: str) -> dict[str, Any] | None:
    """按名称获取单个工具 schema."""
    return _TOOL_SCHEMAS.get(name)


def map_tool_call(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """将 LLM 返回的工具调用映射为 sandbox 格式.

    Args:
        tool_name: LLM 返回的工具名 (如 "bash")
        params: LLM 返回的参数

    Returns:
        {"tool": sandbox_tool_name, "params": mapped_params}
    """
    sandbox_name = ENSOUL_TO_SANDBOX.get(tool_name, tool_name)

    # 特殊映射：bash → shell (参数名不同)
    if tool_name == "bash" and sandbox_name == "shell":
        return {"tool": "shell", "params": params}

    # glob → search (使用 file_pattern)
    if tool_name == "glob" and sandbox_name == "search":
        return {
            "tool": "search",
            "params": {
                "pattern": "",
                "path": params.get("path", "."),
                "file_pattern": params.get("pattern", ""),
            },
        }

    return {"tool": sandbox_name, "params": params}


def is_finish_tool(tool_name: str) -> bool:
    """判断是否为终止工具."""
    return tool_name in FINISH_TOOLS
