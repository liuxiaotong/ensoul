"""权限请求管理模块

用于处理敏感工具的权限确认请求。
"""

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# 敏感工具列表及其风险级别
SENSITIVE_TOOLS = {
    "file_write": "high",
    "agent_file_write": "high",
    "file_delete": "high",
    "agent_file_delete": "high",
    "bash": "high",
    "agent_bash": "high",
    "git_push": "high",
    "deploy": "high",
}


@dataclass
class PermissionRequest:
    """权限请求对象"""

    request_id: str
    tool_name: str
    tool_params: dict[str, Any]
    risk_level: str
    message: str
    created_at: float
    event: asyncio.Event
    approved: bool | None = None
    target_user_id: str = ""


class PermissionManager:
    """权限请求管理器（单例）"""

    _instance: "PermissionManager | None" = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._requests: dict[str, PermissionRequest] = {}
        return cls._instance

    def create_request(
        self, tool_name: str, tool_params: dict[str, Any], target_user_id: str = ""
    ) -> PermissionRequest:
        """创建权限请求"""
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        risk_level = SENSITIVE_TOOLS.get(tool_name, "medium")
        message = self._generate_message(tool_name, tool_params)

        request = PermissionRequest(
            request_id=request_id,
            tool_name=tool_name,
            tool_params=tool_params,
            risk_level=risk_level,
            message=message,
            created_at=time.time(),
            event=asyncio.Event(),
            target_user_id=target_user_id,
        )

        self._requests[request_id] = request
        return request

    def _generate_message(self, tool_name: str, tool_params: dict[str, Any]) -> str:
        """生成第一人称权限请求消息"""
        if tool_name in ("file_write", "agent_file_write"):
            path = tool_params.get("path", "文件")
            return f"我需要写入文件 {path}，请确认"
        elif tool_name in ("bash", "agent_bash"):
            cmd = tool_params.get("command", "命令")
            return f"我需要执行命令：{cmd}，请确认"
        elif tool_name in ("file_delete", "agent_file_delete"):
            path = tool_params.get("path", "文件")
            return f"我需要删除文件 {path}，请确认"
        elif tool_name == "git_push":
            return "我需要推送代码到远程仓库，请确认"
        elif tool_name == "deploy":
            return "我需要部署到生产环境，请确认"
        else:
            return f"我需要执行 {tool_name} 操作，请确认"

    async def request_permission(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        timeout: float = 60.0,
        push_event_fn: Callable[[dict], None] | None = None,
        target_user_id: str = "",
    ) -> bool:
        """请求权限确认"""
        request = self.create_request(tool_name, tool_params, target_user_id=target_user_id)

        # 如果提供了推送函数，通过 SSE 推送事件
        if push_event_fn:
            push_event_fn(
                {
                    "type": "permission_request",
                    "request_id": request.request_id,
                    "tool_name": request.tool_name,
                    "tool_params": request.tool_params,
                    "risk_level": request.risk_level,
                    "message": request.message,
                    "target_user_id": request.target_user_id,
                }
            )

        # 等待用户响应
        try:
            await asyncio.wait_for(request.event.wait(), timeout=timeout)
            approved = request.approved or False
        except asyncio.TimeoutError:
            approved = False
        finally:
            # 清理请求
            self._requests.pop(request.request_id, None)

        return approved

    def get_request(self, request_id: str) -> PermissionRequest | None:
        """获取权限请求对象（用于鉴权校验）"""
        return self._requests.get(request_id)

    def respond(self, request_id: str, approved: bool) -> bool:
        """响应权限请求"""
        request = self._requests.get(request_id)
        if request is None:
            return False

        request.approved = approved
        request.event.set()
        return True

    def get_pending_requests(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """获取待处理的权限请求。

        Args:
            user_id: 如果指定，只返回 target_user_id 匹配的请求。
                     target_user_id 为空的请求视为广播，始终返回。
        """
        result = []
        for req in self._requests.values():
            if req.approved is not None:
                continue
            # 用户隔离：如果指定了 user_id，只返回目标用户的请求或广播请求
            if user_id is not None and req.target_user_id and req.target_user_id != user_id:
                continue
            result.append(
                {
                    "request_id": req.request_id,
                    "tool_name": req.tool_name,
                    "tool_params": req.tool_params,
                    "risk_level": req.risk_level,
                    "message": req.message,
                    "created_at": req.created_at,
                    "target_user_id": req.target_user_id,
                }
            )
        return result
