"""运行时指标收集器 — 记录 LLM 调用统计."""

from __future__ import annotations

import threading
import time
from collections import deque

_MAX_LATENCY_SAMPLES = 10000


class MetricsCollector:
    """线程安全的运行时指标收集器.

    记录 LLM 调用次数、token 用量、错误数等统计信息。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._total_calls = 0
        self._success_calls = 0
        self._failed_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._by_employee: dict[str, dict[str, int]] = {}
        self._by_provider: dict[str, dict] = {}
        self._latency_samples: deque[float] = deque(maxlen=_MAX_LATENCY_SAMPLES)
        self._latency_by_provider: dict[str, deque[float]] = {}

    def record_call(
        self,
        *,
        employee: str = "",
        provider: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        success: bool = True,
        error_type: str = "",
    ) -> None:
        """记录一次 LLM 调用."""
        with self._lock:
            self._total_calls += 1
            if success:
                self._success_calls += 1
            else:
                self._failed_calls += 1
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens

            if employee:
                if employee not in self._by_employee:
                    self._by_employee[employee] = {"calls": 0, "tokens": 0}
                self._by_employee[employee]["calls"] += 1
                self._by_employee[employee]["tokens"] += input_tokens + output_tokens

            if provider:
                if provider not in self._by_provider:
                    self._by_provider[provider] = {
                        "calls": 0,
                        "success": 0,
                        "failed": 0,
                        "errors": {},
                    }
                self._by_provider[provider]["calls"] += 1
                if success:
                    self._by_provider[provider]["success"] += 1
                else:
                    self._by_provider[provider]["failed"] += 1
                    if error_type:
                        errors = self._by_provider[provider].setdefault("errors", {})
                        errors[error_type] = errors.get(error_type, 0) + 1

    def record_latency(self, *, latency_ms: float, provider: str = "") -> None:
        """记录一次调用延迟（毫秒）."""
        with self._lock:
            self._latency_samples.append(latency_ms)
            if provider:
                if provider not in self._latency_by_provider:
                    self._latency_by_provider[provider] = deque(maxlen=_MAX_LATENCY_SAMPLES)
                self._latency_by_provider[provider].append(latency_ms)

    @staticmethod
    def _latency_stats(samples: deque[float] | list[float]) -> dict:
        """计算延迟统计信息."""
        if not samples:
            return {"count": 0, "mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "max_ms": 0}
        s = sorted(samples)
        return {
            "count": len(s),
            "mean_ms": round(sum(s) / len(s), 1),
            "p50_ms": round(s[len(s) // 2], 1),
            "p95_ms": round(s[int(len(s) * 0.95)], 1),
            "max_ms": round(s[-1], 1),
        }

    def snapshot(self) -> dict:
        """返回当前指标快照."""
        with self._lock:
            data = {
                "uptime_seconds": round(time.monotonic() - self._start_time),
                "calls": {
                    "total": self._total_calls,
                    "success": self._success_calls,
                    "failed": self._failed_calls,
                },
                "tokens": {
                    "input": self._input_tokens,
                    "output": self._output_tokens,
                },
                "by_employee": dict(self._by_employee),
                "by_provider": dict(self._by_provider),
                "latency": self._latency_stats(self._latency_samples),
                "latency_by_provider": {
                    p: self._latency_stats(samples)
                    for p, samples in self._latency_by_provider.items()
                },
            }
        # 追加 MCP Tool 使用率数据（在锁外获取，避免死锁）
        try:
            from ensoul.mcp_server import get_tool_metrics_collector

            data["tool_metrics"] = get_tool_metrics_collector().snapshot()
        except Exception:
            pass  # MCP 未安装或未初始化时跳过

        # 追加 events 表聚合数据
        try:
            from ensoul.event_collector import get_event_collector

            ec = get_event_collector()
            data["events_summary"] = ec.aggregate()
        except Exception:
            pass  # EventCollector 未初始化时跳过
        return data

    def reset(self) -> None:
        """重置所有指标."""
        with self._lock:
            self._start_time = time.monotonic()
            self._total_calls = 0
            self._success_calls = 0
            self._failed_calls = 0
            self._input_tokens = 0
            self._output_tokens = 0
            self._by_employee.clear()
            self._by_provider.clear()
            self._latency_samples.clear()
            self._latency_by_provider.clear()


# 全局单例
_collector = MetricsCollector()


def get_collector() -> MetricsCollector:
    """获取全局指标收集器."""
    return _collector
