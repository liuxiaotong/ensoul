"""回复后处理 — 决定是否将对话产出推送到记忆.

推送门槛：预计 95% 对话不触发，只有含决策/纠正/长产出的对话才写入记忆。
"""

import atexit
import logging
import re
import threading
import weakref

from ensoul.memory import get_memory_store
from ensoul.memory_cache import invalidate
from ensoul.memory_pipeline import process_memory

# daemon=False 线程退出前等待完成，避免记忆写入被截断
_active_threads: list[weakref.ref] = []


def _join_active_threads():
    for ref in _active_threads:
        t = ref()
        if t is not None and t.is_alive():
            t.join(timeout=15)


atexit.register(_join_active_threads)

logger = logging.getLogger(__name__)

# 决策关键词 — 需要更明确的决策语境
_DECISION_KEYWORDS = re.compile(
    r"(?:我们|最终|经过讨论)?决定(?:了|要|用|使用|采用|不|\w)"
    r"|(?:我们|最终|经过讨论)?确定(?:了|要|用|使用|采用|不)"
    r"|方案定了|最终选|统一(?:用|使用|改为)"
    r"|(?:正式)?采用|(?:正式)?弃用|(?:明确)?禁止"
)

# 纠正关键词 — 排除口头禅"其实"
_CORRECTION_KEYWORDS = re.compile(
    r"说错了|(?:需要)?纠正|更正一下|之前(?:说|写|理解)错"
    r"|搞错了|误解了|实际上应该"
    r"|(?:之前|上次).{0,6}(?:有问题|有误|不对|不准确)"
)

# 产出长度阈值（字符数）
_LONG_OUTPUT_THRESHOLD = 200

# 最小对话轮数（低于此轮数不检查长产出）
_MIN_TURNS_FOR_FINDING = 3


def should_push(
    reply: str,
    turn_count: int = 1,
) -> tuple[bool, str]:
    """判断是否应该将回复推送到记忆.

    Args:
        reply: 回复文本
        turn_count: 对话轮数

    Returns:
        (should_push, category) — category 为 "decision" / "correction" / "finding" / ""
    """
    if not reply or not reply.strip():
        return False, ""

    # 决策词 → 推 decision
    if _DECISION_KEYWORDS.search(reply):
        return True, "decision"

    # 纠正词 → 推 correction
    if _CORRECTION_KEYWORDS.search(reply):
        return True, "correction"

    # 长产出 + 多轮对话 → 推 finding
    if turn_count >= _MIN_TURNS_FOR_FINDING and len(reply) > _LONG_OUTPUT_THRESHOLD:
        return True, "finding"

    return False, ""


def _do_push(
    employee: str,
    reply: str,
    session_id: str,
    store,
    recalled_ids: list[str] | None = None,
) -> None:
    """后台线程执行记忆管线."""
    try:
        entry = process_memory(
            raw_text=reply,
            employee=employee,
            store=store,
            skip_reflect=False,
            source_session=session_id,
        )
        if entry:
            invalidate(employee)
            logger.info(
                "记忆管线推送成功: employee=%s id=%s",
                employee,
                entry.id,
            )
        else:
            logger.info("记忆管线决定跳过: employee=%s", employee)
    except Exception as e:
        logger.error("记忆管线推送失败: employee=%s error=%s", employee, e)


def push_if_needed(
    *,
    employee: str,
    reply: str,
    turn_count: int = 1,
    session_id: str = "",
    store=None,
    max_retries: int = 2,
    timeout: float = 10.0,
    recalled_ids: list[str] | None = None,
) -> bool:
    """检查并推送回复到记忆（如果满足门槛）.

    LLM 管线调用在后台线程异步执行，不阻塞调用方。

    Args:
        employee: 员工名称
        reply: 回复文本
        turn_count: 对话轮数
        session_id: 会话 ID（用于去重）
        store: MemoryStore 实例
        max_retries: 保留参数（已废弃，管线内部有错误处理）
        timeout: 保留参数（已废弃）
        recalled_ids: 本轮召回的记忆 ID 列表，push 成功时标记为 useful

    Returns:
        True 如果已触发管线（不等待结果），False 如果未触发
    """
    do_push, category = should_push(reply, turn_count)
    if not do_push or not category:
        return False

    if store is None:
        store = get_memory_store()

    # 后台线程异步执行管线（LLM 调用 2-5 秒，不阻塞）
    t = threading.Thread(
        target=_do_push,
        args=(employee, reply, session_id, store, recalled_ids),
        daemon=False,  # 非 daemon：进程退出前等待记忆写入完成
    )
    # 清理已完成的线程引用
    _active_threads[:] = [ref for ref in _active_threads if ref() is not None and ref().is_alive()]
    t.start()
    _active_threads.append(weakref.ref(t))
    return True
