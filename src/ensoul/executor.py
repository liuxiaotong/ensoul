"""LLM 执行器 — 支持 Anthropic / OpenAI / DeepSeek / Moonshot / Gemini / Zhipu / Qwen."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from ensoul.models import ToolCall, ToolExecutionResult
from ensoul.providers import (
    API_KEY_ENV_VARS,
    DEEPSEEK_BASE_URL,
    MOONSHOT_BASE_URL,
    QWEN_BASE_URL,
    ZHIPU_BASE_URL,
    Provider,
    detect_provider,
    resolve_api_key,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

# API 代理通用 key 环境变量名（当 base_url 指定时，优先用这个而非 provider 原生 key）
_PROXY_KEY_ENV = "LLM_PROXY_API_KEY"


def _resolve_key_for_context(
    provider: Provider,
    api_key: str | None,
    base_url: str | None,
) -> str:
    """根据调用上下文解析 API key.

    - 显式传入 api_key → 直接用
    - base_url 指定（走代理）→ 依次尝试: provider 原生 key → LLM_PROXY_API_KEY → "sk-proxy"
    - 无 base_url（直连）→ 要求 provider 原生 key（严格模式）
    """
    if api_key:
        return api_key
    if base_url:
        # 走代理时宽松解析：先看 provider key，没有就看通用代理 key
        env_var = API_KEY_ENV_VARS.get(provider, "")
        key = os.environ.get(env_var, "") if env_var else ""
        if key:
            return key
        proxy_key = os.environ.get(_PROXY_KEY_ENV, "")
        if proxy_key:
            return proxy_key
        # 代理通常不校验 key 或使用内部认证，给一个占位值避免 SDK 报空 key
        logger.info("base_url=%s 已指定但无 API key，使用占位值", base_url)
        return "sk-proxy-placeholder"
    # 直连模式：严格要求 provider key
    return resolve_api_key(provider, api_key)


# 注意: genai.configure() 修改全局状态，_gemini_lock 只保护 configure 调用。
# 当前单 API key 安全；若需多 key 并发，应改为 per-request client 实例。
_gemini_lock = threading.Lock()


def _is_retryable(exc: Exception) -> bool:
    """判断异常是否可重试."""
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in (408, 429) or status >= 500
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return False


def _retry_delay(attempt: int) -> float:
    """指数退避延迟（秒）: 2^attempt + random jitter."""
    return float(2**attempt) + random.uniform(0, 0.5)


# ── Lazy SDK imports ──


def _get_anthropic():
    """Lazy import anthropic SDK."""
    try:
        import anthropic

        return anthropic
    except ImportError:
        return None


def _get_openai():
    """Lazy import openai SDK."""
    try:
        import openai

        return openai
    except ImportError:
        return None


def _get_genai():
    """Lazy import google.generativeai SDK."""
    try:
        import google.generativeai as genai

        return genai
    except ImportError:
        return None


@dataclass
class ExecutionResult:
    """LLM 执行结果."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str


# ── Anthropic 实现 ──


def _anthropic_execute(
    system_prompt: str,
    user_message: str,
    api_key: str,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    stream: bool,
    on_chunk: Callable[[str], None] | None,
) -> ExecutionResult:
    anthropic = _get_anthropic()
    if anthropic is None:
        raise ImportError("anthropic SDK 未安装。请运行: pip install ensoul[execute]")

    client = anthropic.Anthropic(api_key=api_key)
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens or 4096,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    if stream and on_chunk:
        collected: list[str] = []
        with client.messages.stream(**kwargs) as resp:
            for text in resp.text_stream:
                collected.append(text)
                on_chunk(text)
            final = resp.get_final_message()
        return ExecutionResult(
            content="".join(collected),
            model=final.model,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            stop_reason=final.stop_reason,
        )

    resp = client.messages.create(**kwargs)
    content = resp.content[0].text if resp.content else ""
    return ExecutionResult(
        content=content,
        model=resp.model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        stop_reason=resp.stop_reason,
    )


async def _anthropic_aexecute(
    system_prompt: str,
    user_message: str,
    api_key: str,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    stream: bool,
    full_events: bool = False,
) -> ExecutionResult | AsyncIterator[str] | AsyncIterator[dict[str, Any]]:
    anthropic = _get_anthropic()
    if anthropic is None:
        raise ImportError("anthropic SDK 未安装。请运行: pip install ensoul[execute]")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens or 4096,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    if stream:
        if full_events:
            # 返回完整事件流
            async def _event_stream() -> AsyncIterator[dict[str, Any]]:
                collected_text: list[str] = []
                async with client.messages.stream(**kwargs) as stream_resp:
                    async for event in stream_resp:
                        event_type = event.type

                        if event_type == "content_block_start":
                            block = event.content_block
                            evt: dict[str, Any] = {
                                "type": "content_block_start",
                                "index": event.index,
                                "content_type": block.type,
                            }
                            if block.type == "tool_use":
                                evt["tool_name"] = block.name
                                evt["tool_use_id"] = block.id
                            yield evt

                        elif event_type == "content_block_delta":
                            delta = event.delta
                            evt = {
                                "type": "content_block_delta",
                                "index": event.index,
                            }
                            if delta.type == "text_delta":
                                collected_text.append(delta.text)
                                evt["content_type"] = "text"
                                evt["text"] = delta.text
                            elif delta.type == "thinking_delta":
                                evt["content_type"] = "thinking"
                                evt["thinking"] = delta.thinking
                            elif delta.type == "input_json_delta":
                                evt["content_type"] = "tool_use"
                                evt["tool_input"] = delta.partial_json
                            yield evt

                        elif event_type == "content_block_stop":
                            yield {
                                "type": "content_block_stop",
                                "index": event.index,
                            }

                        elif event_type == "message_delta":
                            evt = {"type": "message_delta"}
                            if hasattr(event.delta, "stop_reason") and event.delta.stop_reason:
                                evt["stop_reason"] = event.delta.stop_reason
                            yield evt

                    final = await stream_resp.get_final_message()
                _event_stream.result = ExecutionResult(  # type: ignore[attr-defined]
                    content="".join(collected_text),
                    model=final.model,
                    input_tokens=final.usage.input_tokens,
                    output_tokens=final.usage.output_tokens,
                    stop_reason=final.stop_reason,
                )

            return _event_stream()
        else:
            # 原有的 text_stream 逻辑（向后兼容）
            async def _stream() -> AsyncIterator[str]:
                collected: list[str] = []
                async with client.messages.stream(**kwargs) as resp:
                    async for text in resp.text_stream:
                        collected.append(text)
                        yield text
                    final = resp.get_final_message()
                _stream.result = ExecutionResult(  # type: ignore[attr-defined]
                    content="".join(collected),
                    model=final.model,
                    input_tokens=final.usage.input_tokens,
                    output_tokens=final.usage.output_tokens,
                    stop_reason=final.stop_reason,
                )

            return _stream()

    resp = await client.messages.create(**kwargs)
    content = resp.content[0].text if resp.content else ""
    return ExecutionResult(
        content=content,
        model=resp.model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        stop_reason=resp.stop_reason,
    )


# ── OpenAI 兼容实现（OpenAI + DeepSeek）──


def _openai_execute(
    system_prompt: str,
    user_message: str,
    api_key: str,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    stream: bool,
    on_chunk: Callable[[str], None] | None,
    base_url: str | None = None,
) -> ExecutionResult:
    openai = _get_openai()
    if openai is None:
        raise ImportError("openai SDK 未安装。请运行: pip install ensoul[openai]")

    client_kwargs: dict = {"api_key": api_key, "timeout": 30.0}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    kwargs: dict = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    # Kimi K2.5 默认开启 thinking，会导致 content 为空（tokens 全在 reasoning）
    if model.lower().startswith("kimi-"):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    if stream and on_chunk:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        collected: list[str] = []
        input_tokens = 0
        output_tokens = 0
        resp = client.chat.completions.create(**kwargs)
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                collected.append(text)
                on_chunk(text)
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
        return ExecutionResult(
            content="".join(collected),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason="stop",
        )

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0] if resp.choices else None
    content = choice.message.content if choice and choice.message else ""
    return ExecutionResult(
        content=content or "",
        model=resp.model or model,
        input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        stop_reason=choice.finish_reason if choice else "unknown",
    )


async def _openai_aexecute(
    system_prompt: str,
    user_message: str,
    api_key: str,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    stream: bool,
    base_url: str | None = None,
) -> ExecutionResult | AsyncIterator[str]:
    openai = _get_openai()
    if openai is None:
        raise ImportError("openai SDK 未安装。请运行: pip install ensoul[openai]")

    client_kwargs: dict = {"api_key": api_key, "timeout": 30.0}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.AsyncOpenAI(**client_kwargs)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    kwargs: dict = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    # Kimi K2.5 默认开启 thinking，会导致 content 为空（tokens 全在 reasoning）
    if model.lower().startswith("kimi-"):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    if stream:

        async def _stream() -> AsyncIterator[str]:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            collected: list[str] = []
            input_tokens = 0
            output_tokens = 0
            resp = await client.chat.completions.create(**kwargs)
            async for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    collected.append(text)
                    yield text
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0
            _stream.result = ExecutionResult(  # type: ignore[attr-defined]
                content="".join(collected),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason="stop",
            )

        return _stream()

    resp = await client.chat.completions.create(**kwargs)
    choice = resp.choices[0] if resp.choices else None
    content = choice.message.content if choice and choice.message else ""
    return ExecutionResult(
        content=content or "",
        model=resp.model or model,
        input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        stop_reason=choice.finish_reason if choice else "unknown",
    )


# ── Gemini 实现 ──


def _gemini_execute(
    system_prompt: str,
    user_message: str,
    api_key: str,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    stream: bool,
    on_chunk: Callable[[str], None] | None,
) -> ExecutionResult:
    genai = _get_genai()
    if genai is None:
        raise ImportError(
            "google-generativeai SDK 未安装。请运行: pip install ensoul[gemini]"
        )

    gen_config = {}
    if temperature is not None:
        gen_config["temperature"] = temperature
    if max_tokens is not None:
        gen_config["max_output_tokens"] = max_tokens

    with _gemini_lock:
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
            generation_config=gen_config or None,
        )

    if stream and on_chunk:
        response = client.generate_content(user_message, stream=True)
        collected = []
        for chunk in response:
            if chunk.text:
                collected.append(chunk.text)
                on_chunk(chunk.text)
        # Get usage from final response
        usage = getattr(response, "usage_metadata", None)
        return ExecutionResult(
            content="".join(collected),
            model=model,
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            stop_reason="stop",
        )

    response = client.generate_content(user_message)
    usage = getattr(response, "usage_metadata", None)
    return ExecutionResult(
        content=response.text or "",
        model=model,
        input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
        stop_reason="stop",
    )


async def _gemini_aexecute(
    system_prompt: str,
    user_message: str,
    api_key: str,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    stream: bool,
) -> ExecutionResult | AsyncIterator[str]:
    genai = _get_genai()
    if genai is None:
        raise ImportError(
            "google-generativeai SDK 未安装。请运行: pip install ensoul[gemini]"
        )

    gen_config = {}
    if temperature is not None:
        gen_config["temperature"] = temperature
    if max_tokens is not None:
        gen_config["max_output_tokens"] = max_tokens

    with _gemini_lock:
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
            generation_config=gen_config or None,
        )

    if stream:

        async def _stream() -> AsyncIterator[str]:
            response = await client.generate_content_async(user_message, stream=True)
            collected = []
            async for chunk in response:
                if chunk.text:
                    collected.append(chunk.text)
                    yield chunk.text
            usage = getattr(response, "usage_metadata", None)
            _stream.result = ExecutionResult(
                content="".join(collected),
                model=model,
                input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
                stop_reason="stop",
            )

        return _stream()

    response = await client.generate_content_async(user_message)
    usage = getattr(response, "usage_metadata", None)
    return ExecutionResult(
        content=response.text or "",
        model=model,
        input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
        stop_reason="stop",
    )


# ── 指标记录 ──


def _record_metrics(provider_name: str, result: ExecutionResult, elapsed: float) -> None:
    """记录成功调用的指标 (no-op: metrics 模块未迁入)."""


def _record_failure(provider_name: str, exc: Exception) -> None:
    """记录失败调用的指标 (no-op: metrics 模块未迁入)."""


def _record_trajectory(result: ExecutionResult) -> None:
    """记录轨迹 (no-op: trajectory 模块未迁入)."""


class _MetricsStreamWrapper:
    """包装异步流式迭代器，在流消费完毕后自动记录指标."""

    def __init__(self, inner: AsyncIterator[str], provider_name: str, t0: float):
        self._inner = inner
        self._provider_name = provider_name
        self._t0 = t0

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        try:
            return await self._inner.__anext__()
        except StopAsyncIteration:
            result = getattr(self._inner, "result", None)
            if result is not None:
                _record_metrics(self._provider_name, result, time.monotonic() - self._t0)
            raise

    @property
    def result(self):
        return getattr(self._inner, "result", None)


# ── 公开 API ──


def execute_prompt(
    *,
    system_prompt: str,
    user_message: str = "请开始执行上述任务。",
    api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool = True,
    on_chunk: Callable[[str], None] | None = None,
    fallback_model: str | None = None,
    fallback_api_key: str | None = None,
    fallback_base_url: str | None = None,
    base_url: str | None = None,
) -> ExecutionResult:
    """调用 LLM API 执行 prompt（支持 Anthropic / OpenAI / DeepSeek）.

    Args:
        system_prompt: 完整 system prompt（engine.prompt() 输出）.
        user_message: 用户消息.
        api_key: API key（None 时自动从环境变量解析）.
        model: 模型标识符（根据前缀自动检测提供商）.
        temperature: 采样温度（None 使用 API 默认值）.
        max_tokens: 最大输出 token 数.
        stream: 是否流式输出.
        on_chunk: 流式模式下的文本块回调.
        fallback_model: 所有重试耗尽后的备用模型.

    Returns:
        ExecutionResult.
    """
    provider = detect_provider(model)
    # base_url 指定时走 API 代理，不强制要求 provider 原生 key
    resolved_key = _resolve_key_for_context(provider, api_key, base_url)

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.monotonic()
            # base_url 显式指定时，强制走 OpenAI 兼容路径（API 代理）
            if base_url:
                result = _openai_execute(
                    system_prompt,
                    user_message,
                    resolved_key,
                    model,
                    temperature,
                    max_tokens,
                    stream,
                    on_chunk,
                    base_url=base_url,
                )
            elif provider == Provider.ANTHROPIC:
                result = _anthropic_execute(
                    system_prompt,
                    user_message,
                    resolved_key,
                    model,
                    temperature,
                    max_tokens,
                    stream,
                    on_chunk,
                )
            elif provider == Provider.GEMINI:
                result = _gemini_execute(
                    system_prompt,
                    user_message,
                    resolved_key,
                    model,
                    temperature,
                    max_tokens,
                    stream,
                    on_chunk,
                )
            else:
                effective_base_url = {
                    Provider.DEEPSEEK: DEEPSEEK_BASE_URL,
                    Provider.MOONSHOT: MOONSHOT_BASE_URL,
                    Provider.ZHIPU: ZHIPU_BASE_URL,
                    Provider.QWEN: QWEN_BASE_URL,
                }.get(provider)
                result = _openai_execute(
                    system_prompt,
                    user_message,
                    resolved_key,
                    model,
                    temperature,
                    max_tokens,
                    stream,
                    on_chunk,
                    base_url=effective_base_url,
                )
            _record_metrics(provider.value, result, time.monotonic() - t0)
            _record_trajectory(result)
            return result
        except Exception as e:
            if attempt < MAX_RETRIES and _is_retryable(e):
                delay = _retry_delay(attempt)
                logger.warning(
                    "LLM 调用失败 (attempt %d/%d), %0.1fs 后重试: %s",
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                    e,
                )
                time.sleep(delay)
                last_exc = e
            else:
                _record_failure(provider.value, e)
                last_exc = e
                break

    # 尝试 fallback 模型
    if fallback_model and last_exc is not None:
        logger.warning("切换到 fallback 模型: %s", fallback_model)
        _fallback_max = min(max_tokens, 8192) if max_tokens is not None else 8192
        try:
            return execute_prompt(
                system_prompt=system_prompt,
                user_message=user_message,
                api_key=fallback_api_key or api_key,
                model=fallback_model,
                temperature=temperature,
                max_tokens=_fallback_max,
                stream=stream,
                on_chunk=on_chunk,
                base_url=fallback_base_url,
            )
        except Exception as e:
            logger.debug("Fallback 模型也失败: %s", e)

    raise last_exc  # type: ignore[misc]


async def aexecute_prompt(
    *,
    system_prompt: str,
    user_message: str = "请开始执行上述任务。",
    api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool = True,
    full_events: bool = False,
    fallback_model: str | None = None,
    fallback_api_key: str | None = None,
    fallback_base_url: str | None = None,
    base_url: str | None = None,
) -> ExecutionResult | AsyncIterator[str] | AsyncIterator[dict[str, Any]]:
    """execute_prompt 的异步版本.

    当 stream=True 时返回 AsyncIterator[str]（调用者需用 async for 消费），
    完成后可通过 .result 属性获取 ExecutionResult。
    当 stream=False 时直接返回 ExecutionResult。
    当 full_events=True 时返回 AsyncIterator[dict[str, Any]]（完整事件流）。
    """
    provider = detect_provider(model)
    # base_url 指定时走 API 代理，不强制要求 provider 原生 key
    resolved_key = _resolve_key_for_context(provider, api_key, base_url)

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.monotonic()
            # base_url 显式指定时，强制走 OpenAI 兼容路径（API 代理）
            if base_url:
                result = await _openai_aexecute(
                    system_prompt,
                    user_message,
                    resolved_key,
                    model,
                    temperature,
                    max_tokens,
                    stream,
                    base_url=base_url,
                )
            elif provider == Provider.ANTHROPIC:
                result = await _anthropic_aexecute(
                    system_prompt,
                    user_message,
                    resolved_key,
                    model,
                    temperature,
                    max_tokens,
                    stream,
                    full_events,
                )
            elif provider == Provider.GEMINI:
                result = await _gemini_aexecute(
                    system_prompt,
                    user_message,
                    resolved_key,
                    model,
                    temperature,
                    max_tokens,
                    stream,
                )
            else:
                effective_base_url = {
                    Provider.DEEPSEEK: DEEPSEEK_BASE_URL,
                    Provider.MOONSHOT: MOONSHOT_BASE_URL,
                    Provider.ZHIPU: ZHIPU_BASE_URL,
                    Provider.QWEN: QWEN_BASE_URL,
                }.get(provider)
                result = await _openai_aexecute(
                    system_prompt,
                    user_message,
                    resolved_key,
                    model,
                    temperature,
                    max_tokens,
                    stream,
                    base_url=effective_base_url,
                )
            if not stream and isinstance(result, ExecutionResult):
                _record_metrics(provider.value, result, time.monotonic() - t0)
                _record_trajectory(result)
            elif stream:
                result = _MetricsStreamWrapper(result, provider.value, t0)
            return result
        except Exception as e:
            if attempt < MAX_RETRIES and _is_retryable(e):
                delay = _retry_delay(attempt)
                logger.warning(
                    "LLM 调用失败 (attempt %d/%d), %0.1fs 后重试: %s",
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
                last_exc = e
            else:
                _record_failure(provider.value, e)
                last_exc = e
                break

    # 尝试 fallback 模型
    if fallback_model and last_exc is not None:
        logger.warning("切换到 fallback 模型: %s", fallback_model)
        _fallback_max = min(max_tokens, 8192) if max_tokens is not None else 8192
        try:
            return await aexecute_prompt(
                system_prompt=system_prompt,
                user_message=user_message,
                api_key=fallback_api_key or api_key,
                model=fallback_model,
                temperature=temperature,
                max_tokens=_fallback_max,
                stream=stream,
                full_events=full_events,
                base_url=fallback_base_url,
            )
        except Exception as e:
            logger.debug("Async fallback 模型也失败: %s", e)

    raise last_exc  # type: ignore[misc]


# ── Tool-use API ──


def _anthropic_execute_with_tools(
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    api_key: str,
    model: str,
    max_tokens: int,
) -> ToolExecutionResult:
    anthropic = _get_anthropic()
    if anthropic is None:
        raise ImportError("anthropic SDK 未安装。请运行: pip install ensoul[execute]")

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
        tools=tools,
    )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in resp.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                )
            )

    return ToolExecutionResult(
        content="\n".join(text_parts),
        tool_calls=tool_calls,
        model=resp.model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        stop_reason=resp.stop_reason,
    )


async def _anthropic_aexecute_with_tools(
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    api_key: str,
    model: str,
    max_tokens: int,
) -> ToolExecutionResult:
    anthropic = _get_anthropic()
    if anthropic is None:
        raise ImportError("anthropic SDK 未安装。请运行: pip install ensoul[execute]")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
        tools=tools,
    )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in resp.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                )
            )

    return ToolExecutionResult(
        content="\n".join(text_parts),
        tool_calls=tool_calls,
        model=resp.model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        stop_reason=resp.stop_reason,
    )


def _openai_execute_with_tools(
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    api_key: str,
    model: str,
    max_tokens: int,
    base_url: str | None = None,
) -> ToolExecutionResult:
    openai = _get_openai()
    if openai is None:
        raise ImportError("openai SDK 未安装。请运行: pip install ensoul[openai]")

    client_kwargs: dict = {"api_key": api_key, "timeout": 30.0}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    # 转换 Anthropic 格式 tools 为 OpenAI 格式
    openai_tools = []
    for t in tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
        )

    full_messages = [{"role": "system", "content": system_prompt}] + messages
    kwargs: dict = {
        "model": model,
        "messages": full_messages,
        "tools": openai_tools,
        "max_tokens": max_tokens,
    }
    # Kimi K2.5 默认开启 thinking，但 tool call 场景与 thinking 不兼容，关闭之
    if model.lower().startswith("kimi-"):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0] if resp.choices else None
    content = (choice.message.content or "") if choice and choice.message else ""

    tool_calls: list[ToolCall] = []
    if choice and choice.message and choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                )
            )

    return ToolExecutionResult(
        content=content,
        tool_calls=tool_calls,
        model=resp.model or model,
        input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        stop_reason=choice.finish_reason if choice else "unknown",
    )


async def _openai_aexecute_with_tools(
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    api_key: str,
    model: str,
    max_tokens: int,
    base_url: str | None = None,
) -> ToolExecutionResult:
    openai = _get_openai()
    if openai is None:
        raise ImportError("openai SDK 未安装。请运行: pip install ensoul[openai]")

    client_kwargs: dict = {"api_key": api_key, "timeout": 30.0}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.AsyncOpenAI(**client_kwargs)

    openai_tools = []
    for t in tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
        )

    full_messages = [{"role": "system", "content": system_prompt}] + messages
    kwargs: dict = {
        "model": model,
        "messages": full_messages,
        "tools": openai_tools,
        "max_tokens": max_tokens,
    }
    # Kimi K2.5 默认开启 thinking，但 tool call 场景与 thinking 不兼容，关闭之
    if model.lower().startswith("kimi-"):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    resp = await client.chat.completions.create(**kwargs)
    choice = resp.choices[0] if resp.choices else None
    content = (choice.message.content or "") if choice and choice.message else ""

    tool_calls: list[ToolCall] = []
    if choice and choice.message and choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                )
            )

    return ToolExecutionResult(
        content=content,
        tool_calls=tool_calls,
        model=resp.model or model,
        input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        stop_reason=choice.finish_reason if choice else "unknown",
    )


def execute_with_tools(
    *,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    api_key: str | None = None,
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 4096,
    base_url: str | None = None,
    fallback_model: str | None = None,
    fallback_api_key: str | None = None,
    fallback_base_url: str | None = None,
) -> ToolExecutionResult:
    """带工具调用的 LLM 执行.

    Args:
        system_prompt: 系统 prompt.
        messages: 对话历史 [{"role": "user"/"assistant"/"tool", "content": ...}].
        tools: LLM 工具定义（Anthropic 格式）.
        api_key: API key（None 时自动解析）.
        model: 模型标识符.
        max_tokens: 最大输出 token 数.

    Returns:
        ToolExecutionResult — 包含文本内容和工具调用.
    """
    provider = detect_provider(model)
    resolved_key = _resolve_key_for_context(provider, api_key, base_url)

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.monotonic()
            # base_url 显式指定时，强制走 OpenAI 兼容路径（API 代理）
            if base_url:
                result = _openai_execute_with_tools(
                    system_prompt,
                    messages,
                    tools,
                    resolved_key,
                    model,
                    max_tokens,
                    base_url=base_url,
                )
            elif provider == Provider.ANTHROPIC:
                result = _anthropic_execute_with_tools(
                    system_prompt,
                    messages,
                    tools,
                    resolved_key,
                    model,
                    max_tokens,
                )
            elif provider in (
                Provider.OPENAI,
                Provider.DEEPSEEK,
                Provider.MOONSHOT,
                Provider.ZHIPU,
                Provider.QWEN,
            ):
                effective_base_url = {
                    Provider.DEEPSEEK: DEEPSEEK_BASE_URL,
                    Provider.MOONSHOT: MOONSHOT_BASE_URL,
                    Provider.ZHIPU: ZHIPU_BASE_URL,
                    Provider.QWEN: QWEN_BASE_URL,
                }.get(provider)
                result = _openai_execute_with_tools(
                    system_prompt,
                    messages,
                    tools,
                    resolved_key,
                    model,
                    max_tokens,
                    base_url=effective_base_url,
                )
            else:
                raise ValueError(f"Provider {provider} 暂不支持 tool_use")
            _record_metrics(
                provider.value,
                ExecutionResult(
                    content=result.content,
                    model=result.model,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    stop_reason=result.stop_reason,
                ),
                time.monotonic() - t0,
            )
            return result
        except Exception as e:
            if attempt < MAX_RETRIES and _is_retryable(e):
                delay = _retry_delay(attempt)
                logger.warning(
                    "Tool-use LLM 调用失败 (attempt %d/%d), %0.1fs 后重试: %s",
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                    e,
                )
                time.sleep(delay)
                last_exc = e
            else:
                _record_failure(provider.value, e)
                last_exc = e
                break

    # 尝试 fallback 模型
    if fallback_model and last_exc is not None:
        logger.warning("Tool-use 切换到 fallback 模型: %s", fallback_model)
        _fallback_max = min(max_tokens, 8192) if max_tokens is not None else 8192
        try:
            return execute_with_tools(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                api_key=fallback_api_key or api_key,
                model=fallback_model,
                max_tokens=_fallback_max,
                base_url=fallback_base_url,
            )
        except Exception as e:
            logger.debug("Tool-use fallback 模型也失败: %s", e)

    raise last_exc  # type: ignore[misc]


async def aexecute_with_tools(
    *,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    api_key: str | None = None,
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 4096,
    base_url: str | None = None,
    fallback_model: str | None = None,
    fallback_api_key: str | None = None,
    fallback_base_url: str | None = None,
) -> ToolExecutionResult:
    """execute_with_tools 的异步版本."""
    provider = detect_provider(model)
    resolved_key = _resolve_key_for_context(provider, api_key, base_url)

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.monotonic()
            # base_url 显式指定时，强制走 OpenAI 兼容路径（API 代理）
            if base_url:
                result = await _openai_aexecute_with_tools(
                    system_prompt,
                    messages,
                    tools,
                    resolved_key,
                    model,
                    max_tokens,
                    base_url=base_url,
                )
            elif provider == Provider.ANTHROPIC:
                result = await _anthropic_aexecute_with_tools(
                    system_prompt,
                    messages,
                    tools,
                    resolved_key,
                    model,
                    max_tokens,
                )
            elif provider in (
                Provider.OPENAI,
                Provider.DEEPSEEK,
                Provider.MOONSHOT,
                Provider.ZHIPU,
                Provider.QWEN,
            ):
                effective_base_url = {
                    Provider.DEEPSEEK: DEEPSEEK_BASE_URL,
                    Provider.MOONSHOT: MOONSHOT_BASE_URL,
                    Provider.ZHIPU: ZHIPU_BASE_URL,
                    Provider.QWEN: QWEN_BASE_URL,
                }.get(provider)
                result = await _openai_aexecute_with_tools(
                    system_prompt,
                    messages,
                    tools,
                    resolved_key,
                    model,
                    max_tokens,
                    base_url=effective_base_url,
                )
            else:
                raise ValueError(f"Provider {provider} 暂不支持 tool_use")
            _record_metrics(
                provider.value,
                ExecutionResult(
                    content=result.content,
                    model=result.model,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    stop_reason=result.stop_reason,
                ),
                time.monotonic() - t0,
            )
            return result
        except Exception as e:
            if attempt < MAX_RETRIES and _is_retryable(e):
                delay = _retry_delay(attempt)
                logger.warning(
                    "Async tool-use LLM 调用失败 (attempt %d/%d), %0.1fs 后重试: %s",
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
                last_exc = e
            else:
                _record_failure(provider.value, e)
                last_exc = e
                break

    # 尝试 fallback 模型
    if fallback_model and last_exc is not None:
        logger.warning("Async tool-use 切换到 fallback 模型: %s", fallback_model)
        _fallback_max = min(max_tokens, 8192) if max_tokens is not None else 8192
        try:
            return await aexecute_with_tools(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                api_key=fallback_api_key or api_key,
                model=fallback_model,
                max_tokens=_fallback_max,
                base_url=fallback_base_url,
            )
        except Exception as e:
            logger.debug("Async tool-use fallback 模型也失败: %s", e)

    raise last_exc  # type: ignore[misc]


# StreamWithToolsResult 和 astream_final_response 已移除（v2 流式 agent loop 不再需要）
