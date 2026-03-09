"""LLM 调用 + 上下文编排层 — 调用 executor，处理 memory，输出清洗.

从 engine.py 拆分而来，包含所有有副作用的操作（LLM 调用、DB 读写、网络请求）。
继承 PromptEngine 获得 prompt 生成能力。
"""

import asyncio
import logging
import time as _time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ensoul.engine_prompt import PromptEngine
from ensoul.models import Employee

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class CrewEngine(PromptEngine):
    """数字员工执行引擎 — 完整版（含 LLM 调用和记忆管理）.

    继承 PromptEngine 的 prompt 生成能力，添加:
    - chat() — 统一对话接口
    - _load_memories_sync() — 同步记忆加载
    - _load_memories_parallel() — 异步并行记忆加载
    - _fire_memory_write() — 异步记忆写回
    """

    def __init__(self, project_dir: Path | None = None, tenant_id: str | None = None):
        super().__init__(project_dir=project_dir, tenant_id=tenant_id)

    def _load_memories_sync(
        self,
        employee: Employee,
        rendered: str,
        max_visibility: str,
        classification_max: str | None = None,
        allowed_domains: list[str] | None = None,
        include_confidential: bool = False,
    ) -> list[str]:
        """同步加载记忆（串行），返回 prompt parts."""
        from ensoul.memory import get_memory_store
        from ensoul.memory_cache import get_prompt_cached

        memory_store = get_memory_store(project_dir=self.project_dir, tenant_id=self.tenant_id)
        parts: list[str] = []

        # classification 参数包
        _cls_kwargs: dict = {}
        if classification_max is not None:
            _cls_kwargs["classification_max"] = classification_max
        if allowed_domains is not None:
            _cls_kwargs["allowed_domains"] = allowed_domains
        _cls_kwargs["include_confidential"] = include_confidential

        # 获取同团队成员
        _team_members: list[str] | None = None
        try:
            from ensoul.organization import load_organization as _load_org

            _org = _load_org(project_dir=self.project_dir)
            _tid = _org.get_team(employee.name)
            if _tid:
                _team_members = _org.teams[_tid].members
        except Exception:
            pass

        # 1. 历史经验
        memory_text = get_prompt_cached(
            employee.name,
            query=rendered,
            store=memory_store,
            employee_tags=employee.tags,
            max_visibility=max_visibility,
            team_members=_team_members,
            **_cls_kwargs,
        )
        if memory_text:
            parts.extend(["", "---", "", "## 历史经验", "", memory_text])

        # 2. 上次教训（corrections）
        try:
            corrections = memory_store.query(
                employee.name,
                category="correction",
                limit=3,
                max_visibility=max_visibility,
                **_cls_kwargs,
            )
            if corrections:
                lesson_lines = []
                for c in corrections:
                    if "待改进:" in c.content:
                        focus = c.content.split("待改进:")[-1].strip()
                        lesson_lines.append(f"- ⚠ {focus}")
                    else:
                        lesson_lines.append(f"- {c.content}")
                parts.extend(
                    [
                        "",
                        "---",
                        "",
                        "## 上次教训",
                        "",
                        "以下是你最近任务的自检结果，本次注意改进：",
                        "",
                    ]
                    + lesson_lines
                )
        except Exception:
            pass

        # 3. 高分范例（exemplars）
        try:
            exemplars = memory_store.query(
                employee.name,
                category="finding",
                limit=3,
                max_visibility=max_visibility,
                **_cls_kwargs,
            )
            exemplars = [e for e in exemplars if "exemplar" in (e.tags or [])]
            if exemplars:
                ex_lines = [f"- {e.content}" for e in exemplars]
                parts.extend(
                    [
                        "",
                        "---",
                        "",
                        "## 高分范例",
                        "",
                        "以下是你近期表现优秀的任务案例，可作为参考：",
                        "",
                    ]
                    + ex_lines
                )
        except Exception:
            pass

        # 4. 可复用工作模式（patterns）
        try:
            patterns = memory_store.query_patterns(
                employee=employee.name,
                applicability=employee.tags,
                limit=5,
            )
            if patterns:
                pattern_lines = []
                for p in patterns:
                    verified = f" ✓{p.verified_count}" if p.verified_count > 0 else ""
                    trigger = f" [触发: {p.trigger_condition}]" if p.trigger_condition else ""
                    origin = (
                        f" ({p.origin_employee})" if p.origin_employee != employee.name else ""
                    )
                    pattern_lines.append(f"- {p.content}{trigger}{origin}{verified}")
                parts.extend(
                    [
                        "",
                        "---",
                        "",
                        "## 可参考的工作模式",
                        "",
                        "以下是团队验证过的有效工作模式，适用时可直接采用：",
                        "",
                    ]
                    + pattern_lines
                )
        except Exception:
            pass

        return parts

    async def _load_memories_parallel(
        self,
        employee: Employee,
        rendered: str,
        max_visibility: str,
        classification_max: str | None = None,
        allowed_domains: list[str] | None = None,
        include_confidential: bool = False,
    ) -> list[str]:
        """异步并行加载记忆（4 个 DB 查询同时执行），返回 prompt parts.

        将原来串行 ~500ms-1s 的 4 次查询改为并行，预期降到 ~200ms。
        """
        from ensoul.memory import get_memory_store
        from ensoul.memory_cache import get_prompt_cached

        memory_store = get_memory_store(project_dir=self.project_dir, tenant_id=self.tenant_id)
        parts: list[str] = []

        # classification 参数包
        _cls_kwargs: dict = {}
        if classification_max is not None:
            _cls_kwargs["classification_max"] = classification_max
        if allowed_domains is not None:
            _cls_kwargs["allowed_domains"] = allowed_domains
        _cls_kwargs["include_confidential"] = include_confidential

        # 获取同团队成员
        _team_members: list[str] | None = None
        try:
            from ensoul.organization import load_organization as _load_org

            _org = _load_org(project_dir=self.project_dir)
            _tid = _org.get_team(employee.name)
            if _tid:
                _team_members = _org.teams[_tid].members
        except Exception:
            pass

        # 4 个同步 DB 查询 → asyncio.to_thread 并行
        def _q_cached():
            return get_prompt_cached(
                employee.name,
                query=rendered,
                store=memory_store,
                employee_tags=employee.tags,
                max_visibility=max_visibility,
                team_members=_team_members,
                **_cls_kwargs,
            )

        def _q_corrections():
            try:
                return memory_store.query(
                    employee.name,
                    category="correction",
                    limit=3,
                    max_visibility=max_visibility,
                    **_cls_kwargs,
                )
            except Exception:
                return []

        def _q_exemplars():
            try:
                results = memory_store.query(
                    employee.name,
                    category="finding",
                    limit=3,
                    max_visibility=max_visibility,
                    **_cls_kwargs,
                )
                return [e for e in results if "exemplar" in (e.tags or [])]
            except Exception:
                return []

        def _q_patterns():
            try:
                return memory_store.query_patterns(
                    employee=employee.name,
                    applicability=employee.tags,
                    limit=5,
                )
            except Exception:
                return []

        # 并行执行
        memory_text, corrections, exemplars, patterns = await asyncio.gather(
            asyncio.to_thread(_q_cached),
            asyncio.to_thread(_q_corrections),
            asyncio.to_thread(_q_exemplars),
            asyncio.to_thread(_q_patterns),
        )

        # 组装 parts
        if memory_text:
            parts.extend(["", "---", "", "## 历史经验", "", memory_text])

        if corrections:
            lesson_lines = []
            for c in corrections:
                if "待改进:" in c.content:
                    focus = c.content.split("待改进:")[-1].strip()
                    lesson_lines.append(f"- ⚠ {focus}")
                else:
                    lesson_lines.append(f"- {c.content}")
            parts.extend(
                ["", "---", "", "## 上次教训", "", "以下是你最近任务的自检结果，本次注意改进：", ""]
                + lesson_lines
            )

        if exemplars:
            ex_lines = [f"- {e.content}" for e in exemplars]
            parts.extend(
                [
                    "",
                    "---",
                    "",
                    "## 高分范例",
                    "",
                    "以下是你近期表现优秀的任务案例，可作为参考：",
                    "",
                ]
                + ex_lines
            )

        if patterns:
            pattern_lines = []
            for p in patterns:
                verified = f" ✓{p.verified_count}" if p.verified_count > 0 else ""
                trigger = f" [触发: {p.trigger_condition}]" if p.trigger_condition else ""
                origin = f" ({p.origin_employee})" if p.origin_employee != employee.name else ""
                pattern_lines.append(f"- {p.content}{trigger}{origin}{verified}")
            parts.extend(
                [
                    "",
                    "---",
                    "",
                    "## 可参考的工作模式",
                    "",
                    "以下是团队验证过的有效工作模式，适用时可直接采用：",
                    "",
                ]
                + pattern_lines
            )

        return parts

    async def chat(
        self,
        *,
        employee_id: str,
        message: str,
        channel: str,
        sender_id: str,
        max_visibility: str = "internal",
        stream: bool = False,
        context_only: bool = False,
        message_history: list[dict[str, Any]] | None = None,
        model: str | None = None,
        sender_type: str = "",
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """统一对话接口 — prompt 构建 -> LLM 调用 -> 记忆写回全流程.

        Args:
            employee_id: AI 员工 slug
            message: 用户输入文本
            channel: 渠道标识（如 dm / lark / internal / webhook 等，可配置）
            sender_id: 发送方 ID
            max_visibility: 记忆可见性上限，默认 internal
            stream: 是否 SSE 流式（真流式逐 token 推送）
            context_only: 仅返回 prompt+记忆不调 LLM
            message_history: 历史消息列表，格式 [{"role": "user"/"assistant", "content": "..."}]
            model: 覆盖员工配置中的模型
            sender_type: 发送方类型

        Returns:
            非流式: {"reply": str, "employee_id": str, "memory_updated": bool,
                     "tokens_used": int, "latency_ms": int}
            流式: AsyncIterator[dict] 产生 {"delta": str, "done": False} 和
                  {"done": True, "employee_id": str, "tokens_used": int, ...}
            context_only: {"prompt": str, "memories": list, "budget_remaining": int}
        """
        from ensoul.discovery import discover_employees
        from ensoul.routing import _needs_tools
        from ensoul.tool_schema import AGENT_TOOLS

        # ── 1. 参数校验 ──
        if not employee_id:
            raise ValueError("employee_id 不能为空")
        if not message:
            raise ValueError("message 不能为空")
        if not channel:
            raise ValueError("channel 不能为空")
        if not sender_id:
            raise ValueError("sender_id 不能为空")

        # ── 2. 查找员工 ──
        discovery = discover_employees(project_dir=self.project_dir)
        emp = discovery.get(employee_id)
        if emp is None:
            from ensoul.exceptions import EmployeeNotFoundError

            raise EmployeeNotFoundError(employee_id)

        # ── 3. prompt 构建 + 记忆并行加载 ──
        # 计算信息分级
        from ensoul.classification import get_effective_clearance

        _clearance = get_effective_clearance(emp.name, channel, sender_type=sender_type)
        _cls_max = _clearance["classification_max"]
        _cls_domains = _clearance["allowed_domains"]
        _cls_confidential = _clearance["include_confidential"]

        # 先生成不含记忆的 prompt 骨架（快速，无 DB 查询）
        base_prompt = self.prompt(
            emp,
            max_visibility=max_visibility,
            skip_memory=True,
            classification_max=_cls_max,
            allowed_domains=_cls_domains,
            include_confidential=_cls_confidential,
        )
        rendered = self.render(emp)

        # 并行加载记忆（4 个 DB 查询同时执行 ~200ms，vs 串行 ~500ms-1s）
        try:
            memory_parts = await self._load_memories_parallel(
                emp,
                rendered,
                max_visibility,
                classification_max=_cls_max,
                allowed_domains=_cls_domains,
                include_confidential=_cls_confidential,
            )
            if memory_parts:
                # 把记忆注入到 prompt 中（插入到安全准则之前）
                _safety_marker = "\n---\n\n## 安全准则"
                if _safety_marker in base_prompt:
                    idx = base_prompt.index(_safety_marker)
                    memory_block = "\n".join(memory_parts)
                    system_prompt = base_prompt[:idx] + memory_block + base_prompt[idx:]
                else:
                    system_prompt = base_prompt + "\n".join(memory_parts)
            else:
                system_prompt = base_prompt
        except Exception as _mem_err:
            logger.debug("并行记忆加载失败，使用无记忆 prompt: %s", _mem_err)
            system_prompt = base_prompt

        # 聊天格式指令：不加名字前缀
        system_prompt += "\n\n【聊天格式】\n直接回复内容，不要在开头加【名字】或任何方括号前缀。"

        # ── 4. context_only 模式：直接返回 prompt + 记忆，不调 LLM ──
        if context_only:
            try:
                from ensoul.memory import get_memory_store

                mem_store = get_memory_store(
                    project_dir=self.project_dir, tenant_id=self.tenant_id
                )
                raw_memories = mem_store.query(
                    employee_id,
                    limit=10,
                    max_visibility=max_visibility,
                )
                memories_list = [
                    {
                        "content": m.content if hasattr(m, "content") else m.get("content", ""),
                        "category": m.category
                        if hasattr(m, "category")
                        else m.get("category", ""),
                        "importance": m.importance
                        if hasattr(m, "importance")
                        else m.get("importance", 0),
                        "tags": (m.tags if hasattr(m, "tags") else m.get("tags")) or [],
                    }
                    for m in raw_memories
                ]
            except Exception:
                memories_list = []

            # 粗估剩余预算（默认 4096 token）
            prompt_tokens = len(system_prompt) // 2
            budget_remaining = max(0, 4096 - prompt_tokens)

            return {
                "prompt": system_prompt,
                "memories": memories_list,
                "budget_remaining": budget_remaining,
            }

        # ── 5. fast/full 路由决策 ──
        has_tools = any(t in AGENT_TOOLS for t in (emp.tools or []))
        needs_tools = _needs_tools(message)

        # fast path 条件
        _is_short = len(message) <= 30
        use_fast_path = emp.fallback_model and not needs_tools and (has_tools or _is_short)

        # ── 6. 构建 user_message（含历史上下文嵌入）──
        if message_history:
            history_text = "\n".join(
                f"[{'assistant' if m.get('role') == 'assistant' else 'user'}]"
                f" {m.get('content', '') if isinstance(m.get('content'), str) else '[图片消息]' if m.get('content') is not None else ''}"
                for m in message_history[-50:]  # 最多保留最近 50 条（200K token 窗口）
            )
            full_user_message: str | list[dict[str, Any]] = (
                f"[历史对话]\n{history_text}\n\n[当前消息]\n{message}"
            )
        else:
            full_user_message = message

        # ── 7. LLM 调用 ──
        t0 = _time.monotonic()
        # 确定模型和参数
        effective_user_msg = (
            full_user_message if isinstance(full_user_message, str) else "请开始执行上述任务。"
        )

        if use_fast_path:
            chat_model = emp.fallback_model
            chat_api_key = emp.fallback_api_key or None
            chat_base_url = emp.fallback_base_url or None
            chat_fallback_model = None
            chat_fallback_api_key = None
            chat_fallback_base_url = None
        else:
            chat_model = model or emp.model or DEFAULT_MODEL
            chat_api_key = emp.api_key or None
            chat_base_url = emp.base_url or None
            chat_fallback_model = emp.fallback_model or None
            chat_fallback_api_key = emp.fallback_api_key or None
            chat_fallback_base_url = emp.fallback_base_url or None

        # ── 7a. 流式模式：返回 AsyncIterator ──
        if stream:
            return self._chat_stream(
                employee_id=employee_id,
                system_prompt=system_prompt,
                user_message=effective_user_msg,
                channel=channel,
                sender_id=sender_id,
                message=message,
                chat_model=chat_model,
                chat_api_key=chat_api_key,
                chat_base_url=chat_base_url,
                chat_fallback_model=chat_fallback_model,
                chat_fallback_api_key=chat_fallback_api_key,
                chat_fallback_base_url=chat_fallback_base_url,
                use_fast_path=use_fast_path,
                t0=t0,
                max_visibility=max_visibility,
            )

        # ── 7b. 非流式模式（原有逻辑）──
        try:
            from ensoul.executor import aexecute_prompt

            result = await aexecute_prompt(
                system_prompt=system_prompt,
                user_message=effective_user_msg,
                api_key=chat_api_key,
                model=chat_model,
                stream=False,
                base_url=chat_base_url,
                fallback_model=chat_fallback_model,
                fallback_api_key=chat_fallback_api_key,
                fallback_base_url=chat_fallback_base_url,
            )

        except (ValueError, ImportError) as e:
            logger.warning("chat() LLM 调用失败 emp=%s: %s", employee_id, e)
            result = None

        elapsed_ms = int((_time.monotonic() - t0) * 1000)

        if result is None:
            return {
                "reply": "",
                "employee_id": employee_id,
                "memory_updated": False,
                "tokens_used": 0,
                "latency_ms": elapsed_ms,
            }

        # ── 8. 输出清洗 ──
        from ensoul.output_sanitizer import strip_internal_tags

        reply_text = strip_internal_tags(result.content)  # type: ignore[union-attr]
        tokens_used = (
            result.input_tokens + result.output_tokens  # type: ignore[union-attr]
        )

        # ── 9. 记忆异步写回（fire-and-forget）──
        self._fire_memory_write(
            employee_id,
            channel,
            sender_id,
            message,
            reply_text,
            max_visibility,
        )

        logger.info(
            "chat() 完成 [%s] emp=%s channel=%s latency=%dms tokens=%d",
            "fast" if use_fast_path else "full",
            employee_id,
            channel,
            elapsed_ms,
            tokens_used,
        )

        return {
            "reply": reply_text,
            "employee_id": employee_id,
            "memory_updated": False,
            "tokens_used": tokens_used,
            "latency_ms": elapsed_ms,
        }

    async def _chat_stream(
        self,
        *,
        employee_id: str,
        system_prompt: str,
        user_message: str,
        channel: str,
        sender_id: str,
        message: str,
        chat_model: str,
        chat_api_key: str | None,
        chat_base_url: str | None,
        chat_fallback_model: str | None,
        chat_fallback_api_key: str | None,
        chat_fallback_base_url: str | None,
        use_fast_path: bool,
        t0: float,
        max_visibility: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """真流式 chat — 逐 token 产生 SSE 事件."""
        from ensoul.executor import aexecute_prompt

        try:
            token_stream = await aexecute_prompt(
                system_prompt=system_prompt,
                user_message=user_message,
                api_key=chat_api_key,
                model=chat_model,
                stream=True,
                base_url=chat_base_url,
                fallback_model=chat_fallback_model,
                fallback_api_key=chat_fallback_api_key,
                fallback_base_url=chat_fallback_base_url,
            )
        except (ValueError, ImportError) as e:
            logger.warning("chat() stream LLM 调用失败 emp=%s: %s", employee_id, e)
            yield {
                "done": True,
                "employee_id": employee_id,
                "tokens_used": 0,
                "latency_ms": int((_time.monotonic() - t0) * 1000),
                "error": str(e),
            }
            return

        # 逐 token 推送
        collected: list[str] = []
        async for token in token_stream:
            collected.append(token)
            yield {"delta": token, "done": False}

        # 完成：获取最终统计
        elapsed_ms = int((_time.monotonic() - t0) * 1000)
        tokens_used = 0
        result_obj = getattr(token_stream, "result", None)
        if result_obj is None:
            # _MetricsStreamWrapper 包装：result 可能在 _inner 上
            _inner = getattr(token_stream, "_inner", None)
            if _inner:
                result_obj = getattr(_inner, "result", None)

        if result_obj:
            tokens_used = result_obj.input_tokens + result_obj.output_tokens

        # 记忆异步写回
        full_reply = "".join(collected)
        from ensoul.output_sanitizer import strip_internal_tags

        clean_reply = strip_internal_tags(full_reply)
        self._fire_memory_write(
            employee_id,
            channel,
            sender_id,
            message,
            clean_reply,
            max_visibility,
        )

        logger.info(
            "chat() stream 完成 [%s] emp=%s channel=%s latency=%dms tokens=%d",
            "fast" if use_fast_path else "full",
            employee_id,
            channel,
            elapsed_ms,
            tokens_used,
        )

        yield {
            "done": True,
            "employee_id": employee_id,
            "tokens_used": tokens_used,
            "latency_ms": elapsed_ms,
        }

    def _fire_memory_write(
        self,
        employee_id: str,
        channel: str,
        sender_id: str,
        message: str,
        reply_text: str,
        max_visibility: str,
    ) -> None:
        """异步写回记忆（fire-and-forget）."""
        try:
            from ensoul.memory import get_memory_store
            from ensoul.memory_pipeline import process_memory

            store = get_memory_store(project_dir=self.project_dir, tenant_id=self.tenant_id)

            async def _write_memory() -> None:
                try:
                    combined = (
                        f"[{channel}] {sender_id}: {message[:200]}\n\n回复: {reply_text[:500]}"
                    )
                    await asyncio.to_thread(
                        process_memory,
                        raw_text=combined,
                        employee=employee_id,
                        store=store,
                        source_session="",
                        visibility="open",
                        tags=[channel, "chat"],
                    )
                    logger.debug("chat() 记忆写回成功: emp=%s", employee_id)
                except Exception as _me:
                    logger.debug("chat() 记忆写回失败: %s", _me)

            _task = asyncio.create_task(_write_memory())
            _ = _task  # 防止 GC
        except Exception:
            pass
