"""投递层 — 将任务结果推送到外部渠道（Webhook / 邮件 / 飞书）."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import smtplib
import time
from email.mime.text import MIMEText
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT = 30.0  # httpx POST 超时（秒）
SMTP_TIMEOUT = 10  # SMTP 连接超时（秒）
EMAIL_BODY_MAX_LENGTH = 2000  # 邮件正文截断长度
FEISHU_CARD_CONTENT_MAX = 4000  # 飞书卡片内容截断长度


class DeliveryTarget(BaseModel):
    """单个投递目标."""

    type: Literal["webhook", "email", "feishu"] = Field(description="投递类型")
    # webhook / feishu
    url: str = Field(default="", description="Webhook URL / 飞书机器人 Webhook URL")
    headers: dict[str, str] = Field(default_factory=dict, description="自定义请求头")
    # feishu
    secret: str = Field(default="", description="飞书机器人签名密钥（可选）")
    # email
    to: str = Field(default="", description="收件人邮箱")
    subject: str = Field(default="", description="邮件主题（支持 {name} 占位符）")


class DeliveryResult(BaseModel):
    """投递结果."""

    target_type: str
    success: bool
    detail: str = ""


async def deliver(
    targets: list[DeliveryTarget],
    *,
    task_name: str = "",
    task_result: dict[str, Any] | None = None,
    task_error: str | None = None,
) -> list[DeliveryResult]:
    """投递任务结果到所有目标.

    Args:
        targets: 投递目标列表.
        task_name: 任务名称（用于模板替换）.
        task_result: 任务结果数据.
        task_error: 任务错误信息（有值表示任务失败）.

    Returns:
        每个目标的投递结果.
    """
    if not targets:
        return []

    results = await asyncio.gather(
        *[
            _deliver_one(t, task_name=task_name, task_result=task_result, task_error=task_error)
            for t in targets
        ],
        return_exceptions=True,
    )

    out: list[DeliveryResult] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            out.append(
                DeliveryResult(
                    target_type=targets[i].type,
                    success=False,
                    detail=str(r),
                )
            )
        else:
            out.append(r)
    return out


async def _deliver_one(
    target: DeliveryTarget,
    *,
    task_name: str,
    task_result: dict[str, Any] | None,
    task_error: str | None,
) -> DeliveryResult:
    """投递到单个目标."""
    if target.type == "webhook":
        return await _deliver_webhook(
            target, task_name=task_name, task_result=task_result, task_error=task_error
        )
    elif target.type == "email":
        return await _deliver_email(
            target, task_name=task_name, task_result=task_result, task_error=task_error
        )
    elif target.type == "feishu":
        return await _deliver_feishu(
            target, task_name=task_name, task_result=task_result, task_error=task_error
        )
    else:
        return DeliveryResult(
            target_type=target.type, success=False, detail=f"未知投递类型: {target.type}"
        )


async def _deliver_webhook(
    target: DeliveryTarget,
    *,
    task_name: str,
    task_result: dict[str, Any] | None,
    task_error: str | None,
) -> DeliveryResult:
    """POST JSON 到 webhook URL."""
    if not target.url:
        return DeliveryResult(target_type="webhook", success=False, detail="URL 为空")

    try:
        import httpx
    except ImportError:
        return DeliveryResult(target_type="webhook", success=False, detail="httpx 未安装")

    payload: dict[str, Any] = {
        "task_name": task_name,
        "status": "failed" if task_error else "completed",
    }
    if task_error:
        payload["error"] = task_error
    if task_result:
        payload["result"] = task_result

    headers = {"Content-Type": "application/json"}
    headers.update(target.headers)

    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT) as client:
            resp = await client.post(target.url, json=payload, headers=headers)
        if resp.status_code < 400:
            return DeliveryResult(
                target_type="webhook", success=True, detail=f"HTTP {resp.status_code}"
            )
        else:
            return DeliveryResult(
                target_type="webhook", success=False, detail=f"HTTP {resp.status_code}"
            )
    except Exception as e:
        return DeliveryResult(target_type="webhook", success=False, detail=str(e))


def _validate_smtp_config(
    host: str,
    port_str: str,
    to: str,
) -> tuple[bool, str]:
    """校验 SMTP 配置.

    Returns:
        (ok, error_message) 元组.
    """
    if not host:
        return False, "SMTP_HOST 未配置"
    try:
        port = int(port_str)
    except (ValueError, TypeError):
        return False, f"SMTP_PORT 无效: {port_str}"
    if port < 1 or port > 65535:
        return False, f"SMTP_PORT 超出范围: {port}"
    if "@" not in to:
        return False, f"收件人地址无效 (缺少 @): {to}"
    return True, ""


async def _deliver_email(
    target: DeliveryTarget,
    *,
    task_name: str,
    task_result: dict[str, Any] | None,
    task_error: str | None,
) -> DeliveryResult:
    """通过 SMTP 发送邮件."""
    if not target.to:
        return DeliveryResult(target_type="email", success=False, detail="收件人为空")

    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port_str = os.environ.get("SMTP_PORT", "587")
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user)

    ok, err = _validate_smtp_config(smtp_host, smtp_port_str, target.to)
    if not ok:
        return DeliveryResult(target_type="email", success=False, detail=err)

    smtp_port = int(smtp_port_str)

    subject = target.subject.format(name=task_name) if target.subject else f"任务完成: {task_name}"

    # 构建邮件正文
    if task_error:
        body = f"任务 {task_name} 执行失败。\n\n错误: {task_error}"
    elif task_result:
        # 提取关键信息
        output = task_result.get("output", "")
        if isinstance(output, str) and len(output) > EMAIL_BODY_MAX_LENGTH:
            output = output[:EMAIL_BODY_MAX_LENGTH] + "\n...(已截断)"
        body = f"任务 {task_name} 执行完成。\n\n结果:\n{output}"
    else:
        body = f"任务 {task_name} 执行完成。"

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = target.to

    try:
        # 在线程池中执行阻塞的 SMTP 操作
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, _send_smtp, smtp_host, smtp_port, smtp_user, smtp_pass, msg
        )
        return DeliveryResult(target_type="email", success=True, detail=f"发送至 {target.to}")
    except Exception as e:
        return DeliveryResult(target_type="email", success=False, detail=str(e))


def _send_smtp(host: str, port: int, user: str, password: str, msg: MIMEText) -> None:
    """同步 SMTP 发送."""
    with smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT) as server:
        server.starttls()
        if user and password:
            server.login(user, password)
        server.send_message(msg)


# ── 飞书投递 ──


def _feishu_sign(secret: str, timestamp: int) -> str:
    """生成飞书机器人签名.

    算法: HMAC-SHA256(timestamp + "\\n" + secret) → base64
    """
    import base64

    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return base64.b64encode(hmac_code).decode("utf-8")


def _build_feishu_card(
    task_name: str,
    task_result: dict[str, Any] | None,
    task_error: str | None,
) -> dict[str, Any]:
    """构建飞书交互式卡片消息.

    成功: 绿色 header + markdown 内容
    失败: 红色 header + 错误信息
    """
    if task_error:
        header_color = "red"
        header_title = f"❌ {task_name or '任务'} — 执行失败"
        content = f"**错误信息:**\n\n{task_error}"
    else:
        header_color = "green"
        header_title = f"✅ {task_name or '任务'} — 执行完成"

        if task_result:
            output = task_result.get("output", "")
            if isinstance(output, str):
                if len(output) > FEISHU_CARD_CONTENT_MAX:
                    output = output[:FEISHU_CARD_CONTENT_MAX] + "\n\n...(已截断)"
                content = output
            else:
                import json

                content = json.dumps(output, ensure_ascii=False, indent=2)[:FEISHU_CARD_CONTENT_MAX]
        else:
            content = "任务已完成。"

    # 附加元信息
    elements: list[dict[str, Any]] = [
        {"tag": "markdown", "content": content},
    ]

    # metadata note（员工、模型、耗时）
    if task_result:
        meta_parts: list[str] = []
        if task_result.get("employee"):
            meta_parts.append(task_result["employee"])
        if task_result.get("model"):
            meta_parts.append(task_result["model"])
        if task_result.get("duration_ms") is not None:
            ms = task_result["duration_ms"]
            meta_parts.append(f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms}ms")
        if meta_parts:
            elements.append(
                {
                    "tag": "note",
                    "elements": [{"tag": "plain_text", "content": " · ".join(meta_parts)}],
                }
            )

    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": header_title},
                "template": header_color,
            },
            "elements": elements,
        },
    }


async def _deliver_feishu(
    target: DeliveryTarget,
    *,
    task_name: str,
    task_result: dict[str, Any] | None,
    task_error: str | None,
) -> DeliveryResult:
    """投递到飞书群机器人 Webhook."""
    if not target.url:
        return DeliveryResult(target_type="feishu", success=False, detail="URL 为空")

    try:
        import httpx
    except ImportError:
        return DeliveryResult(target_type="feishu", success=False, detail="httpx 未安装")

    payload = _build_feishu_card(task_name, task_result, task_error)

    # 签名校验
    if target.secret:
        timestamp = int(time.time())
        sign = _feishu_sign(target.secret, timestamp)
        payload["timestamp"] = str(timestamp)
        payload["sign"] = sign

    headers = {"Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT) as client:
            resp = await client.post(target.url, json=payload, headers=headers)
        body = resp.json() if resp.status_code < 500 else {}
        code = body.get("code", body.get("StatusCode", -1))

        if resp.status_code < 400 and code == 0:
            return DeliveryResult(target_type="feishu", success=True, detail="发送成功")
        else:
            msg = body.get("msg", body.get("StatusMessage", f"HTTP {resp.status_code}"))
            return DeliveryResult(target_type="feishu", success=False, detail=msg)
    except Exception as e:
        return DeliveryResult(target_type="feishu", success=False, detail=str(e))
