"""Embedding 生成模块 — 为记忆系统提供向量语义检索能力.

使用 sentence-transformers 的 all-MiniLM-L6-v2 模型（384 维）。
当 sentence-transformers 不可用时，优雅降级：不生成 embedding，不阻塞写入。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# 模型维度常量
EMBEDDING_DIM: int = 384

# Module-level singleton：首次加载后缓存模型实例
_model: object | None = None
_model_load_attempted: bool = False
_model_load_failed_at: float | None = None  # 加载失败时间戳，用于 cooldown 重试
_MODEL_RETRY_COOLDOWN: float = 300.0  # 5 分钟 cooldown


def _load_model() -> object | None:
    """懒加载 sentence-transformers 模型（单例）.

    加载失败后允许在 cooldown（5 分钟）后重试，避免首次失败后永不恢复。

    Returns:
        SentenceTransformer 模型实例，或 None（不可用时）
    """
    global _model, _model_load_attempted, _model_load_failed_at

    if _model is not None:
        return _model

    if _model_load_attempted:
        # 曾成功加载但模型为 None（ImportError），不重试
        if _model_load_failed_at is None:
            return None
        # 曾失败（运行时错误），检查 cooldown
        if time.time() - _model_load_failed_at < _MODEL_RETRY_COOLDOWN:
            return None
        # cooldown 已过，允许重试
        logger.info("embedding 模型加载 cooldown 已过，尝试重新加载")

    _model_load_attempted = True

    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _model_load_failed_at = None  # 加载成功，清除失败标记
        logger.info("sentence-transformers 模型 all-MiniLM-L6-v2 加载成功")
        return _model
    except ImportError:
        logger.warning("sentence-transformers 未安装，embedding 功能不可用")
        # ImportError 不设 failed_at，因为不重试（库未安装不会自己修复）
        return None
    except Exception as e:
        _model_load_failed_at = time.time()
        logger.error("sentence-transformers 模型加载失败: %s", e)
        return None


def get_embedding(text: str) -> list[float] | None:
    """为文本生成 embedding 向量.

    Args:
        text: 输入文本（建议 content + " " + " ".join(keywords)）

    Returns:
        384 维浮点向量，或 None（模型不可用时）
    """
    model = _load_model()
    if model is None:
        return None

    try:
        # encode 返回 numpy array，转为 Python list
        vector = model.encode(text, normalize_embeddings=True)  # type: ignore[union-attr]
        return vector.tolist()
    except Exception as e:
        logger.error("embedding 生成失败: %s", e)
        return None


def build_embedding_text(content: str, keywords: list[str] | None = None) -> str:
    """构造用于 embedding 的输入文本.

    将 content 和 keywords 拼接，与 A-MEM 论文一致。

    Args:
        content: 记忆内容
        keywords: 关键词列表

    Returns:
        拼接后的文本
    """
    parts = [content]
    if keywords:
        parts.append(" ".join(keywords))
    return " ".join(parts)


def is_available() -> bool:
    """检查 embedding 功能是否可用.

    Returns:
        True 表示模型已加载或可以加载
    """
    return _load_model() is not None


def reset() -> None:
    """重置模型状态（仅用于测试）."""
    global _model, _model_load_attempted, _model_load_failed_at
    _model = None
    _model_load_attempted = False
    _model_load_failed_at = None
