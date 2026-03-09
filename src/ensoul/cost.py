"""Token 成本追踪 — 模型单价表、成本估算、历史查询."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ensoul.task_registry import TaskRegistry

logger = logging.getLogger(__name__)

# ── 模型单价表（每 1K tokens，USD）──
# 来源：各厂商官方定价，2025-Q2
COST_PER_1K: dict[str, dict[str, float]] = {
    # Anthropic（Opus 4.6/4.5 = $5/$25 per M, Sonnet 4/4.5 = $3/$15, Haiku 4.5 = $1/$5）
    "claude-opus-4-6": {"input": 0.005, "output": 0.025},
    "claude-opus-4-20250514": {"input": 0.005, "output": 0.025},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.001, "output": 0.005},
    # Moonshot / Kimi（kimi-k2.5 = ¥4.2/¥21 per M → $0.6/$3 per M）
    "kimi-k2.5": {"input": 0.0006, "output": 0.003},
    "kimi2.5": {"input": 0.0006, "output": 0.003},
    "moonshot-v1-auto": {"input": 0.0006, "output": 0.003},
    "moonshot-v1-32k": {"input": 0.000694, "output": 0.002778},
    "moonshot-v1-8k": {"input": 0.000278, "output": 0.001389},
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    # Google
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    # DeepSeek
    "deepseek-chat": {"input": 0.0002, "output": 0.0008},
}

# 代理商定价覆盖（base_url → 模型 → 价格）
# Add your proxy pricing overrides here, e.g.:
# PROXY_PRICE_OVERRIDES = {"https://proxy.example.com/v1": {"claude-opus-4-6": {"input": 0.001, "output": 0.005}}}
PROXY_PRICE_OVERRIDES: dict[str, dict[str, dict[str, float]]] = {}

_DEFAULT_PRICE = {"input": 0.001, "output": 0.003}


def _model_channel(model: str) -> str:
    """模型名称 → 渠道分类."""
    if model.startswith("claude"):
        return "claude"
    if model.startswith("kimi") or model.startswith("moonshot"):
        return "kimi"
    return "other"


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    base_url: str | None = None,
) -> float:
    """估算单次调用成本（USD）."""
    if base_url:
        proxy = PROXY_PRICE_OVERRIDES.get(base_url.rstrip("/"), {})
        prices = proxy.get(model) or COST_PER_1K.get(model, _DEFAULT_PRICE)
    else:
        prices = COST_PER_1K.get(model, _DEFAULT_PRICE)
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1000


def enrich_result_with_cost(result: dict[str, Any]) -> dict[str, Any]:
    """向任务结果中追加 cost_usd 字段."""
    model = result.get("model", "")
    input_tokens = result.get("input_tokens", 0)
    output_tokens = result.get("output_tokens", 0)
    base_url = result.get("base_url")
    if input_tokens or output_tokens:
        result["cost_usd"] = round(
            estimate_cost(model, input_tokens, output_tokens, base_url=base_url),
            6,
        )
    return result


# 触发源分类常量
WORK_TRIGGERS = {
    "github",
    "openclaw",
    "generic",
    "direct",
    "cron",
    "delegate_async",
    "delegate_chain",
}
CHAT_TRIGGERS = {"feishu"}


def query_cost_summary(
    registry: TaskRegistry,
    employee: str | None = None,
    days: int = 7,
    source: str | None = None,
) -> dict[str, Any]:
    """从任务注册表查询成本汇总.

    Args:
        source: 过滤触发源 — "work"(正式任务) / "chat"(闲聊) / None(全部)
    """
    cutoff = datetime.now() - timedelta(days=days)

    by_employee: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    by_trigger: dict[str, dict[str, Any]] = {}
    total_cost = 0.0
    total_tasks = 0

    for record in registry.list_recent(n=5000):
        if record.created_at < cutoff:
            continue
        if record.status != "completed" or not record.result:
            continue
        if employee and record.target_name != employee:
            continue
        # 按触发源过滤
        if source == "work" and record.trigger in CHAT_TRIGGERS:
            continue
        if source == "chat" and record.trigger not in CHAT_TRIGGERS:
            continue

        result = record.result
        emp_name = result.get("employee", record.target_name)
        model = result.get("model", "unknown")
        inp = result.get("input_tokens", 0)
        out = result.get("output_tokens", 0)
        cost = result.get("cost_usd", 0.0)

        if not cost and (inp or out):
            base_url = result.get("base_url")
            cost = estimate_cost(model, inp, out, base_url=base_url)

        # 按员工汇总
        if emp_name not in by_employee:
            by_employee[emp_name] = {
                "cost_usd": 0.0,
                "tasks": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_by_channel": {"claude": 0.0, "kimi": 0.0, "other": 0.0},
            }
        by_employee[emp_name]["cost_usd"] += cost
        by_employee[emp_name]["tasks"] += 1
        by_employee[emp_name]["input_tokens"] += inp
        by_employee[emp_name]["output_tokens"] += out
        by_employee[emp_name]["cost_by_channel"][_model_channel(model)] += cost

        # 按模型汇总
        if model not in by_model:
            by_model[model] = {"cost_usd": 0.0, "tasks": 0}
        by_model[model]["cost_usd"] += cost
        by_model[model]["tasks"] += 1

        # 按触发源汇总
        trigger = record.trigger or "unknown"
        if trigger not in by_trigger:
            by_trigger[trigger] = {"cost_usd": 0.0, "tasks": 0}
        by_trigger[trigger]["cost_usd"] += cost
        by_trigger[trigger]["tasks"] += 1

        total_cost += cost
        total_tasks += 1

    def _round_dict(d: dict) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}

    def _round_nested(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, float):
                out[k] = round(v, 4)
            elif isinstance(v, dict):
                out[k] = {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
            else:
                out[k] = v
        return out

    return {
        "period_days": days,
        "total_cost_usd": round(total_cost, 4),
        "total_tasks": total_tasks,
        "by_employee": {
            k: _round_nested(v)
            for k, v in sorted(by_employee.items(), key=lambda x: -x[1]["cost_usd"])
        },
        "by_model": {
            k: _round_dict(v) for k, v in sorted(by_model.items(), key=lambda x: -x[1]["cost_usd"])
        },
        "by_trigger": {
            k: _round_dict(v)
            for k, v in sorted(by_trigger.items(), key=lambda x: -x[1]["cost_usd"])
        },
    }


def calibrate_employee_costs(
    cost_summary: dict[str, Any],
    proxy_real_usd: float | None = None,
    moonshot_real_usd: float | None = None,
) -> dict[str, Any]:
    """用真实账单按 token 比例校准员工成本.

    原理：
        校准系数 = 实际账单 / 估算总额
        员工实际成本 = 员工估算成本 × 校准系数

    只修改 by_employee 中的 calibrated_cost_usd 字段，不覆盖原始估算。
    """
    by_employee = cost_summary.get("by_employee", {})
    if not by_employee:
        return cost_summary

    # 汇总各渠道估算总额
    claude_estimated = 0.0
    kimi_estimated = 0.0
    for emp_data in by_employee.values():
        cbc = emp_data.get("cost_by_channel", {})
        claude_estimated += cbc.get("claude", 0)
        kimi_estimated += cbc.get("kimi", 0)

    # 计算校准系数
    claude_factor = (
        (proxy_real_usd / claude_estimated) if (proxy_real_usd and claude_estimated > 0) else None
    )
    kimi_factor = (
        (moonshot_real_usd / kimi_estimated) if (moonshot_real_usd and kimi_estimated > 0) else None
    )

    calibrated_total = 0.0
    for emp_data in by_employee.values():
        cbc = emp_data.get("cost_by_channel", {})
        cal = 0.0
        # Claude 渠道校准
        if claude_factor is not None and cbc.get("claude", 0) > 0:
            cal += cbc["claude"] * claude_factor
        else:
            cal += cbc.get("claude", 0)
        # Kimi 渠道校准
        if kimi_factor is not None and cbc.get("kimi", 0) > 0:
            cal += cbc["kimi"] * kimi_factor
        else:
            cal += cbc.get("kimi", 0)
        # 其他渠道不校准
        cal += cbc.get("other", 0)
        emp_data["calibrated_cost_usd"] = round(cal, 4)
        calibrated_total += cal

    cost_summary["calibrated_total_usd"] = round(calibrated_total, 4)
    if claude_factor is not None:
        cost_summary["claude_calibration_factor"] = round(claude_factor, 4)
    if kimi_factor is not None:
        cost_summary["kimi_calibration_factor"] = round(kimi_factor, 4)

    # 按校准后成本重新排序
    cost_summary["by_employee"] = dict(
        sorted(
            by_employee.items(), key=lambda x: -(x[1].get("calibrated_cost_usd", x[1]["cost_usd"]))
        )
    )

    return cost_summary


# ── 代理商真实账单 ──


async def fetch_proxy_billing(
    api_key: str,
    base_url: str = "",
    days: int = 7,
) -> dict[str, Any] | None:
    """从代理商账单 API 拉取真实消耗数据.

    同时查询 7 日消耗和累计消耗。
    """
    import httpx

    end = datetime.now()
    start = end - timedelta(days=days)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    base = base_url.rstrip("/")
    url = f"{base}/dashboard/billing/usage"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # 7 日消耗
            resp = await client.get(
                url, params={"start_date": start_str, "end_date": end_str}, headers=headers
            )
            resp.raise_for_status()
            total_cents = resp.json().get("total_usage", 0)

            # 累计消耗（从 2024-01-01 至今）
            resp2 = await client.get(
                url, params={"start_date": "2024-01-01", "end_date": end_str}, headers=headers
            )
            resp2.raise_for_status()
            cumulative_cents = resp2.json().get("total_usage", 0)

            return {
                "total_usd": round(total_cents / 100, 4),
                "total_cents": total_cents,
                "cumulative_usd": round(cumulative_cents / 100, 4),
                "period_days": days,
                "start": start_str,
                "end": end_str,
            }
    except Exception as exc:
        logger.warning("proxy billing fetch failed: %s", exc)
        return None


async def fetch_proxy_balance(
    access_token: str,
    user_id: str = "",
    base_url: str = "",
) -> dict[str, Any] | None:
    """从代理商（new-api 兼容）管理 API 查询账户余额.

    需要 access_token（系统访问令牌）+ user_id（New-Api-User header）。
    """
    import httpx

    url = f"{base_url.rstrip('/')}/api/user/self"
    headers: dict[str, str] = {"Authorization": f"Bearer {access_token}"}
    if user_id:
        headers["New-Api-User"] = user_id

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("success"):
                logger.warning("proxy balance returned success=false: %s", data.get("message"))
                return None
            user = data.get("data", {})
            quota = user.get("quota", 0)
            used = user.get("used_quota", 0)
            # new-api quota 单位: 默认 500000 = $1
            quota_per_unit = 500000
            return {
                "balance_usd": round(quota / quota_per_unit, 2),
                "used_usd": round(used / quota_per_unit, 2),
            }
    except Exception as exc:
        logger.warning("proxy balance fetch failed: %s", exc)
        return None


# ── Moonshot/Kimi 余额查询 ──

# 人民币→美元汇率（近似，用于展示）
_CNY_TO_USD = 0.138


async def fetch_moonshot_balance(
    api_key: str,
    base_url: str = "https://api.moonshot.cn/v1",
) -> dict[str, Any] | None:
    """从 Moonshot API 查询 Kimi 账户余额.

    Returns:
        {"balance_cny": 56.85, "balance_usd": 7.85, "cash_cny": 50, "voucher_cny": 6.85}
        或 None（请求失败时）
    """
    import httpx

    url = f"{base_url.rstrip('/')}/users/me/balance"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("status"):
                logger.warning("moonshot balance returned status=false: %s", data)
                return None
            bal = data.get("data", {})
            total_cny = bal.get("available_balance", 0)
            return {
                "balance_cny": round(total_cny, 2),
                "balance_usd": round(total_cny * _CNY_TO_USD, 2),
                "cash_cny": round(bal.get("cash_balance", 0), 2),
                "voucher_cny": round(bal.get("voucher_balance", 0), 2),
            }
    except Exception as exc:
        logger.warning("moonshot balance fetch failed: %s", exc)
        return None


# ── 质量预评分解析 ──


def parse_quality_score(output: str) -> dict[str, Any] | None:
    """从员工输出末尾解析结构化质量评分.

    格式: {"score": 72, "critical": 3, "warning": 5, "suggestion": 4}
    """
    pattern = r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}'
    matches = re.findall(pattern, output or "")
    for m in reversed(matches):  # 从末尾开始找
        try:
            data = json.loads(m)
            if "score" in data and isinstance(data["score"], (int, float)):
                return data
        except json.JSONDecodeError:
            continue
    return None
