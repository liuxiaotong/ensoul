"""记忆系统阈值常量 — 去重、关联、合并的统一配置.

所有记忆相关的阈值都集中在这里，方便调参和审计。
memory_pipeline.py / memory_store_db.py 共用。
"""

# ── 语义（embedding cosine similarity）阈值 ──

SEMANTIC_NOOP_THRESHOLD = 0.95
"""Connect 阶段：语义相似度 >= 此值 → NOOP（跳过写入，避免高度重复记忆拼接）."""

SEMANTIC_MERGE_THRESHOLD = 0.85
"""Connect 阶段：语义相似度 >= 此值 且同 category → merge."""

SEMANTIC_LINK_THRESHOLD = 0.35
"""Connect 阶段：语义相似度 >= 此值 → link（建立关联）."""

SEMANTIC_CANDIDATE_MIN_SIMILARITY = 0.3
"""_find_candidates_by_semantic 最低相似度过滤."""

# ── 关键词（Jaccard 系数）阈值 ──

KEYWORD_MERGE_THRESHOLD = 0.7
"""Connect 降级模式：关键词重叠度 >= 此值 且同 category → merge."""

KEYWORD_LINK_THRESHOLD = 0.3
"""Connect 降级模式：关键词重叠度 >= 此值 → link."""

# ── NG-2 写入时去重（memory_store_db._try_dedup_merge）──

DEDUP_MERGE_THRESHOLD = 0.90
"""add() 写入前去重：cosine similarity >= 此值 且同 category → 合并到旧记忆."""

# ── NG-3 自动关联（memory_store_db._auto_link_similar）──

AUTO_LINK_THRESHOLD = 0.35
"""add() 写入后自动关联：cosine similarity >= 此值 → 建立双向链接."""

AUTO_LINK_LIMIT = 3
"""自动关联最多链接的相似记忆数."""

# ── Connect LLM 决策 ──

CONNECT_LLM_TIMEOUT = 30
"""Connect 阶段 LLM 决策调用超时秒数."""

# ── 信息分级 ──

CLASSIFICATION_LEVELS = {"public": 0, "internal": 1, "restricted": 2, "confidential": 3}
"""信息分级等级序，用于 classification_max 过滤."""

# ── 类别中文标签 ──

CATEGORY_LABELS: dict[str, str] = {
    "decision": "决策",
    "estimate": "估算",
    "finding": "发现",
    "correction": "纠正",
    "pattern": "模式",
}
"""记忆类别 -> 中文标签映射，供 _format_entries / _category_label 等共用."""
