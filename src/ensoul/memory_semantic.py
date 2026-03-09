"""语义搜索和智能推荐系统."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ensoul.memory import MemoryEntry, get_memory_store

logger = logging.getLogger(__name__)


class SemanticSearchResult(BaseModel):
    """语义搜索结果."""

    memory_id: str = Field(description="记忆 ID")
    employee: str = Field(description="员工名")
    category: str = Field(description="类别")
    content: str = Field(description="内容")
    tags: list[str] = Field(default_factory=list, description="标签")
    confidence: float = Field(description="置信度")
    relevance_score: float = Field(description="相关性分数 0-1")
    match_reason: str = Field(description="匹配原因")


class MemoryRecommendation(BaseModel):
    """记忆推荐."""

    memory_id: str = Field(description="记忆 ID")
    employee: str = Field(description="员工名")
    category: str = Field(description="类别")
    content: str = Field(description="内容")
    tags: list[str] = Field(default_factory=list, description="标签")
    confidence: float = Field(description="置信度")
    recommendation_score: float = Field(description="推荐分数 0-1")
    recommendation_reason: str = Field(description="推荐原因")


class SemanticSearchEngine:
    """语义搜索引擎."""

    def __init__(self, memory_store=None):
        """初始化搜索引擎.

        Args:
            memory_store: 记忆存储，默认使用全局实例
        """
        self.memory_store = memory_store or get_memory_store()

    def search(
        self,
        query: str,
        employee: str | None = None,
        category: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> list[SemanticSearchResult]:
        """语义搜索记忆.

        Args:
            query: 搜索查询
            employee: 按员工过滤
            category: 按类别过滤
            min_confidence: 最低置信度
            limit: 最大返回数量

        Returns:
            搜索结果列表（按相关性降序）
        """
        # 加载所有记忆
        all_memories = self._load_all_memories(employee, category, min_confidence)

        if not all_memories:
            return []

        # 计算相关性分数
        results = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for memory in all_memories:
            score, reason = self._calculate_relevance(memory, query_lower, query_terms)
            if score > 0:
                results.append(
                    SemanticSearchResult(
                        memory_id=memory.id,
                        employee=memory.employee,
                        category=memory.category,
                        content=memory.content,
                        tags=memory.tags,
                        confidence=memory.confidence,
                        relevance_score=score,
                        match_reason=reason,
                    )
                )

        # 按相关性降序排序
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

    def recommend_for_task(
        self,
        task_description: str,
        employee: str,
        limit: int = 5,
    ) -> list[MemoryRecommendation]:
        """为任务推荐相关记忆.

        Args:
            task_description: 任务描述
            employee: 员工名
            limit: 最大推荐数量

        Returns:
            推荐列表（按推荐分数降序）
        """
        # 加载员工记忆 + 共享记忆
        employee_memories = self._load_all_memories(employee=employee, min_confidence=0.5)
        shared_memories = self._load_shared_memories(exclude_employee=employee, min_confidence=0.7)

        all_memories = employee_memories + shared_memories

        if not all_memories:
            return []

        # 计算推荐分数
        recommendations = []
        task_lower = task_description.lower()
        task_terms = set(task_lower.split())

        for memory in all_memories:
            score, reason = self._calculate_recommendation_score(
                memory, task_lower, task_terms, employee
            )
            if score > 0:
                recommendations.append(
                    MemoryRecommendation(
                        memory_id=memory.id,
                        employee=memory.employee,
                        category=memory.category,
                        content=memory.content,
                        tags=memory.tags,
                        confidence=memory.confidence,
                        recommendation_score=score,
                        recommendation_reason=reason,
                    )
                )

        # 按推荐分数降序排序
        recommendations.sort(key=lambda r: r.recommendation_score, reverse=True)
        return recommendations[:limit]

    def find_similar_memories(
        self,
        memory_id: str,
        limit: int = 5,
    ) -> list[SemanticSearchResult]:
        """查找相似记忆.

        Args:
            memory_id: 记忆 ID
            limit: 最大返回数量

        Returns:
            相似记忆列表（按相似度降序）
        """
        # 加载目标记忆
        target = self._find_memory_by_id(memory_id)
        if not target:
            return []

        # 加载所有记忆（排除自己）
        all_memories = self._load_all_memories()
        all_memories = [m for m in all_memories if m.id != memory_id]

        if not all_memories:
            return []

        # 计算相似度
        results = []
        target_terms = set(target.content.lower().split())
        target_tags = set(target.tags)

        for memory in all_memories:
            score, reason = self._calculate_similarity(memory, target, target_terms, target_tags)
            if score > 0:
                results.append(
                    SemanticSearchResult(
                        memory_id=memory.id,
                        employee=memory.employee,
                        category=memory.category,
                        content=memory.content,
                        tags=memory.tags,
                        confidence=memory.confidence,
                        relevance_score=score,
                        match_reason=reason,
                    )
                )

        # 按相似度降序排序
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

    def _load_all_memories(
        self,
        employee: str | None = None,
        category: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[MemoryEntry]:
        """加载所有记忆."""
        memories = []

        if employee:
            # 加载指定员工的记忆
            entries = self.memory_store.query(employee=employee, limit=1000)
            memories.extend(entries)
        else:
            # 加载所有员工的记忆（通过 list_employees 接口，兼容 DB 和文件版）
            for emp_name in self.memory_store.list_employees():
                entries = self.memory_store.query(employee=emp_name, limit=1000)
                memories.extend(entries)

        # 过滤
        filtered = []
        for memory in memories:
            if category and memory.category != category:
                continue
            if memory.confidence < min_confidence:
                continue
            filtered.append(memory)

        return filtered

    def _load_shared_memories(
        self,
        exclude_employee: str | None = None,
        min_confidence: float = 0.7,
    ) -> list[MemoryEntry]:
        """加载共享记忆（排除指定员工）."""
        memories = self._load_all_memories(min_confidence=min_confidence)
        if exclude_employee:
            memories = [m for m in memories if m.employee != exclude_employee and m.shared]
        else:
            memories = [m for m in memories if m.shared]
        return memories

    def _find_memory_by_id(self, memory_id: str) -> MemoryEntry | None:
        """根据 ID 查找记忆."""
        all_memories = self._load_all_memories()
        for memory in all_memories:
            if memory.id == memory_id:
                return memory
        return None

    def _calculate_relevance(
        self,
        memory: MemoryEntry,
        query_lower: str,
        query_terms: set[str],
    ) -> tuple[float, str]:
        """计算相关性分数.

        Returns:
            (score, reason) - 分数 0-1，匹配原因
        """
        score = 0.0
        reasons = []

        content_lower = memory.content.lower()

        # 1. 内容完全匹配（权重 0.5）
        if query_lower in content_lower:
            score += 0.5
            reasons.append("内容完全匹配")

        # 2. 标签匹配（权重 0.3）
        memory_tags_lower = {tag.lower() for tag in memory.tags}
        matching_tags = query_terms & memory_tags_lower
        if matching_tags:
            tag_score = len(matching_tags) / max(len(query_terms), 1) * 0.3
            score += tag_score
            reasons.append(f"标签匹配: {', '.join(matching_tags)}")

        # 3. 词项匹配（权重 0.3，提高权重）
        content_terms = set(content_lower.split())
        matching_terms = query_terms & content_terms
        if matching_terms:
            term_score = len(matching_terms) / max(len(query_terms), 1) * 0.3
            score += term_score
            reasons.append(f"关键词匹配: {len(matching_terms)}/{len(query_terms)}")

        # 4. 部分词匹配（权重 0.2）- 对中文友好
        if score == 0:
            for term in query_terms:
                if len(term) >= 2 and term in content_lower:
                    score += 0.2
                    reasons.append(f"部分匹配: {term}")
                    break

        reason = "; ".join(reasons) if reasons else "无匹配"
        return min(score, 1.0), reason

    def _calculate_recommendation_score(
        self,
        memory: MemoryEntry,
        task_lower: str,
        task_terms: set[str],
        employee: str,
    ) -> tuple[float, str]:
        """计算推荐分数.

        Returns:
            (score, reason) - 分数 0-1，推荐原因
        """
        score = 0.0
        reasons = []

        # 1. 类别相关性（权重 0.3）
        if memory.category == "correction":
            score += 0.3
            reasons.append("错误纠正记忆")
        elif memory.category == "pattern":
            score += 0.25
            reasons.append("成功模式")
        elif memory.category == "decision":
            score += 0.2
            reasons.append("决策记录")

        # 2. 内容相关性（权重 0.4）
        if task_lower in memory.content.lower():
            score += 0.4
            reasons.append("任务描述匹配")
        else:
            content_terms = set(memory.content.lower().split())
            matching_terms = task_terms & content_terms
            if matching_terms:
                term_score = len(matching_terms) / max(len(task_terms), 1) * 0.4
                score += term_score
                reasons.append(f"关键词匹配: {len(matching_terms)}/{len(task_terms)}")

        # 3. 置信度（权重 0.2）
        score += memory.confidence * 0.2
        reasons.append(f"置信度: {memory.confidence:.2f}")

        # 4. 共享记忆加分（权重 0.1）
        if memory.shared and memory.employee != employee:
            score += 0.1
            reasons.append(f"来自 {memory.employee} 的共享经验")

        reason = "; ".join(reasons) if reasons else "无推荐理由"
        return min(score, 1.0), reason

    def _calculate_similarity(
        self,
        memory: MemoryEntry,
        target: MemoryEntry,
        target_terms: set[str],
        target_tags: set[str],
    ) -> tuple[float, str]:
        """计算相似度.

        Returns:
            (score, reason) - 分数 0-1，相似原因
        """
        score = 0.0
        reasons = []

        # 1. 类别相同（权重 0.3）
        if memory.category == target.category:
            score += 0.3
            reasons.append(f"相同类别: {memory.category}")

        # 2. 标签重叠（权重 0.4）
        memory_tags = set(memory.tags)
        common_tags = memory_tags & target_tags
        if common_tags:
            tag_score = len(common_tags) / max(len(target_tags), 1) * 0.4
            score += tag_score
            reasons.append(f"共同标签: {', '.join(common_tags)}")

        # 3. 内容相似（权重 0.3）
        memory_terms = set(memory.content.lower().split())
        common_terms = memory_terms & target_terms
        if common_terms:
            term_score = len(common_terms) / max(len(target_terms), 1) * 0.3
            score += term_score
            reasons.append(f"内容相似度: {len(common_terms)}/{len(target_terms)}")

        reason = "; ".join(reasons) if reasons else "无相似性"
        return min(score, 1.0), reason
