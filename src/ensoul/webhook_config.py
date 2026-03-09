"""Webhook 配置 — 路由规则、GitHub 签名验证、模板解析."""

from __future__ import annotations

import hashlib
import hmac
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from ensoul.paths import resolve_project_dir


class RouteTarget(BaseModel):
    """路由目标."""

    type: Literal["pipeline", "employee"] = Field(description="目标类型")
    name: str = Field(description="pipeline 或 employee 名称")
    args: dict[str, str] = Field(default_factory=dict, description="参数，支持 {{path}} 模板")


class WebhookRoute(BaseModel):
    """单条路由规则."""

    event: str = Field(description="事件类型，如 push / pull_request / *")
    target: RouteTarget


class WebhookConfig(BaseModel):
    """Webhook 服务器配置."""

    github_secret: str = Field(default="", description="GitHub webhook secret")
    routes: list[WebhookRoute] = Field(default_factory=list, description="路由规则列表")


def load_webhook_config(project_dir: Path | None = None) -> WebhookConfig:
    """从 .ensoul/webhook.yaml 加载配置."""
    base = resolve_project_dir(project_dir)
    config_path = base / ".ensoul" / "webhook.yaml"
    if not config_path.exists():
        return WebhookConfig()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return WebhookConfig()
    return WebhookConfig(**data)


def verify_github_signature(
    payload_body: bytes,
    signature: str | None,
    secret: str,
) -> bool:
    """验证 GitHub webhook HMAC-SHA256 签名.

    Args:
        payload_body: 原始请求 body.
        signature: X-Hub-Signature-256 header 值.
        secret: 配置的 webhook secret.

    Returns:
        签名是否有效.
    """
    if not signature or not secret:
        return False
    if not signature.startswith("sha256="):
        return False
    expected = (
        "sha256="
        + hmac.new(
            secret.encode("utf-8"),
            payload_body,
            hashlib.sha256,
        ).hexdigest()
    )
    return hmac.compare_digest(signature, expected)


def resolve_template(value: str, payload: dict[str, Any]) -> str:
    """解析 {{dotted.path}} 模板为 payload 中的实际值.

    Examples:
        >>> resolve_template("{{ref}}", {"ref": "refs/heads/main"})
        'refs/heads/main'
        >>> resolve_template("{{pull_request.head.ref}}", {"pull_request": {"head": {"ref": "feat/x"}}})
        'feat/x'
    """

    def _replace(match: re.Match) -> str:
        path = match.group(1).strip()
        obj: Any = payload
        for key in path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(key)
            else:
                return match.group(0)  # 无法解析，保留原样
            if obj is None:
                return match.group(0)
        return str(obj)

    return re.sub(r"\{\{(.+?)\}\}", _replace, value)


def resolve_target_args(target: RouteTarget, payload: dict[str, Any]) -> dict[str, str]:
    """解析路由目标参数中的模板."""
    return {k: resolve_template(v, payload) for k, v in target.args.items()}


def match_route(event_type: str, config: WebhookConfig) -> WebhookRoute | None:
    """按 event 类型匹配第一条路由规则."""
    for route in config.routes:
        if route.event == event_type or route.event == "*":
            return route
    return None
