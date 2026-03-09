"""认证与安全中间件."""

from __future__ import annotations

import hmac
import time as _time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


def _path_matches_skip(path: str, skip_paths: list[str]) -> bool:
    """检查请求路径是否匹配 skip 列表（精确匹配或前缀匹配）."""
    return any(path == sp or path.startswith(sp + "/") for sp in skip_paths)


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """校验 Authorization: Bearer <token>（时序安全比较）."""

    def __init__(self, app, *, token: str, skip_paths: list[str] | None = None):
        super().__init__(app)
        self.token = token
        self.skip_paths = skip_paths or ["/health"]

    async def dispatch(self, request: Request, call_next):
        if _path_matches_skip(request.url.path, self.skip_paths):
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        if (
            not self.token
            or not auth.startswith("Bearer ")
            or not hmac.compare_digest(auth[7:], self.token)
        ):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        return await call_next(request)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """拒绝超过大小限制的请求（默认 1MB）."""

    DEFAULT_MAX_REQUEST_BYTES = 1_048_576  # 1 MB

    def __init__(self, app, *, max_bytes: int = DEFAULT_MAX_REQUEST_BYTES):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    return JSONResponse(
                        {"error": "request too large"},
                        status_code=413,
                    )
            except ValueError:
                return JSONResponse(
                    {"error": "invalid content-length"},
                    status_code=400,
                )
        # 无 Content-Length 时（chunked transfer），流式读取并限制大小
        if request.method in ("POST", "PUT", "PATCH") and not content_length:
            body_parts: list[bytes] = []
            total_size = 0
            async for chunk in request.stream():
                total_size += len(chunk)
                if total_size > self.max_bytes:
                    return JSONResponse(
                        {"error": "request too large"},
                        status_code=413,
                    )
                body_parts.append(chunk)
            # 缓存 body 供下游 handler 读取
            request._body = b"".join(body_parts)  # noqa: SLF001
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """基于客户端 IP 的滑动窗口速率限制.

    注意: 速率数据存于进程内 dict，多 worker 部署时各进程独立计数。
    当前单 worker 部署无影响，扩容前需迁移到 Redis 或共享内存。

    Args:
        rate: 窗口期内最大请求数.
        window: 窗口大小（秒）.
        skip_paths: 跳过限制的路径列表.
    """

    _BUCKET_CLEANUP_THRESHOLD = 1000

    def __init__(
        self,
        app,
        *,
        rate: int = 60,
        window: float = 60.0,
        skip_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.rate = rate
        self.window = window
        self.skip_paths = skip_paths or ["/health", "/metrics"]
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        if _path_matches_skip(request.url.path, self.skip_paths):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = _time.monotonic()

        # 清理过期记录
        bucket = self._buckets[client_ip]
        cutoff = now - self.window
        self._buckets[client_ip] = [t for t in bucket if t > cutoff]

        if len(self._buckets[client_ip]) >= self.rate:
            return JSONResponse(
                {"error": "rate limit exceeded"},
                status_code=429,
            )

        self._buckets[client_ip].append(now)

        # 周期性清理空桶，防止内存泄漏
        if len(self._buckets) > self._BUCKET_CLEANUP_THRESHOLD:
            stale = [ip for ip, ts in self._buckets.items() if not ts]
            for ip in stale:
                del self._buckets[ip]

        return await call_next(request)
