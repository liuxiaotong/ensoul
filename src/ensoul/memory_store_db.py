"""Memory 数据库存储 — PostgreSQL 版本的 MemoryStore.

将记忆从 JSONL 文件迁移到数据库，提供与文件版本相同的接口。
"""

from __future__ import annotations

import logging
import math
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from ensoul.database import get_connection, is_pg
from ensoul.memory import MemoryEntry, resolve_to_character_name
from ensoul.memory_constants import AUTO_LINK_LIMIT, AUTO_LINK_THRESHOLD, CATEGORY_LABELS, CLASSIFICATION_LEVELS, DEDUP_MERGE_THRESHOLD
DEFAULT_ADMIN_TENANT_ID = "admin"

logger = logging.getLogger(__name__)

# ── Recency 衰减参数 ──
# recency_decay = exp(-λ × days_since_access)，λ=0.01 时半衰期约 69 天
RECENCY_LAMBDA = 0.01


# ── Schema 定义 ──

_PG_CREATE_MEMORIES = """\
CREATE TABLE IF NOT EXISTS memories (
    id VARCHAR(12) PRIMARY KEY,
    employee VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    category VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    source_session VARCHAR(255) DEFAULT '',
    confidence FLOAT DEFAULT 1.0,
    superseded_by VARCHAR(12) DEFAULT '',
    ttl_days INTEGER DEFAULT 0,
    importance INTEGER DEFAULT 3,
    last_accessed TIMESTAMP,
    tags TEXT[],
    shared BOOLEAN DEFAULT FALSE,
    visibility VARCHAR(20) DEFAULT 'open',
    trigger_condition TEXT DEFAULT '',
    applicability TEXT[],
    origin_employee VARCHAR(255) DEFAULT '',
    verified_count INTEGER DEFAULT 0
)
"""

_PG_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memories_employee ON memories(employee)",
    "CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)",
    "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_memories_shared ON memories(shared)",
    "CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags)",
]


def init_memory_tables() -> None:
    """初始化 memories 表（仅 PG 模式）."""
    if not is_pg():
        logger.debug("SQLite 模式，跳过 memories 表初始化")
        return

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(_PG_CREATE_MEMORIES)
        for sql in _PG_CREATE_INDEXES:
            cur.execute(sql)

        # 幂等添加 classification 列（信息分级）
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN classification VARCHAR(20) DEFAULT 'internal';
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)
        # 幂等添加 domain 列（职能域标签）
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN domain TEXT[] DEFAULT '{}';
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)
        # 索引
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_classification ON memories(classification)"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories USING GIN(domain)")

        # Phase 3-1: 幂等添加 keywords 列
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN keywords text[] DEFAULT '{}';
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)
        # Phase 3-1: 幂等添加 linked_memories 列
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN linked_memories text[] DEFAULT '{}';
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)
        # 注意：keywords 列的查询使用 unnest + ILIKE 子串匹配，GIN 索引无法加速。
        # 当前数据量不需要索引；后续量大时可考虑 pg_trgm 或改为精确匹配再加 GIN。
        cur.execute("DROP INDEX IF EXISTS idx_memories_keywords")

        # Phase 4: 幂等添加 recall_count 列（召回效果闭环）
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN recall_count INTEGER DEFAULT 0;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)

        # NG-4: 幂等添加 q_value 列（效用评分）
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN q_value REAL DEFAULT 0.5;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)

        # NG-1: pgvector 向量语义检索
        # 安装 pgvector 扩展（需要 superuser 或已预装）
        cur.execute("SAVEPOINT sp_pgvector_ext")
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception:
            # pgvector 扩展未安装时不阻塞初始化
            cur.execute("ROLLBACK TO SAVEPOINT sp_pgvector_ext")
            logger.warning("pgvector 扩展不可用，向量检索功能将被禁用")
        else:
            cur.execute("RELEASE SAVEPOINT sp_pgvector_ext")

        # 幂等添加 embedding 列（vector(384) = all-MiniLM-L6-v2 输出维度）
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN embedding vector(384);
            EXCEPTION
                WHEN duplicate_column THEN NULL;
                WHEN undefined_object THEN NULL;
            END $$;
        """)

        # ivfflat 索引：需要表中至少有足够行才能建 lists=100 的索引
        # 用 IF NOT EXISTS 保持幂等
        cur.execute("SAVEPOINT sp_embedding_idx")
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding
                ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """)
        except Exception:
            # 索引创建失败（如 vector 类型不存在或数据不足），不阻塞
            cur.execute("ROLLBACK TO SAVEPOINT sp_embedding_idx")
            logger.warning("embedding 索引创建失败，将在数据回填后重试")
        else:
            cur.execute("RELEASE SAVEPOINT sp_embedding_idx")

        # P2-S17: (employee, tenant_id) 复合索引 — 几乎所有查询都同时筛这两列
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_employee_tenant"
            " ON memories(employee, tenant_id)"
        )

        # P2-S14: 幂等添加 last_decayed_at 列（防止同日重复衰减）
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN last_decayed_at TIMESTAMPTZ;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)

        # Phase 4: 时序感知 — 事实有效期
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN valid_from TIMESTAMP;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN valid_until TIMESTAMP;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)
        # 回填：valid_from = created_at（已有记忆）
        cur.execute("UPDATE memories SET valid_from = created_at WHERE valid_from IS NULL")

        # P2-S24: q_value 范围约束 [0, 1]
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD CONSTRAINT chk_q_value CHECK (q_value >= 0 AND q_value <= 1);
            EXCEPTION WHEN duplicate_object THEN NULL;
            END $$;
        """)

    # ── memory_feedback 表（显式反馈：helpful/not_helpful/outdated/incorrect）──
    _init_memory_feedback_table(conn)

    logger.info("memories 表初始化完成")


def _init_memory_feedback_table(conn: Any) -> None:
    """幂等创建 memory_feedback 表."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memory_feedback (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            employee TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            submitted_by TEXT NOT NULL,
            context TEXT DEFAULT '',
            comment TEXT DEFAULT '',
            tenant_id TEXT DEFAULT 'default',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_memory_id ON memory_feedback(memory_id)")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_employee ON memory_feedback(employee, tenant_id)"
    )


def _thompson_rescore(rows: list[dict]) -> list[dict]:
    """对低召回记忆用 Thompson Sampling 重排."""
    rescored = []
    for row in rows:
        score = row.get("hybrid_score", 0) or 0
        rc = row.get("recall_count", 0) or 0
        vc = row.get("verified_count", 0) or 0
        if rc < 5:
            # Beta 采样：用采样值替代固定 q_value 的贡献
            sampled_q = random.betavariate(vc + 1, max(rc - vc, 0) + 1)
            # 替换 q_value 部分（原 0.15 * q_value -> 0.15 * sampled_q）
            base_score = score - 0.15 * (row.get("q_value", 0.5) or 0.5)
            score = base_score + 0.15 * sampled_q
        rescored.append((score, row))
    rescored.sort(key=lambda x: (-x[0], -x[1].get("importance", 0)))
    return [r[1] for r in rescored]


def _apply_recency(rows: list[dict], lam: float = RECENCY_LAMBDA) -> list[dict]:
    """给已排序行加入 recency_decay 分数并重排.

    recency_decay = exp(-λ × days_since_access)。
    SQL 层计算的 hybrid_score 权重总和为 0.85，这里补上 0.15 × recency_decay。
    """
    now = datetime.now(timezone.utc)
    scored = []
    for row in rows:
        base = row.get("hybrid_score", 0) or 0
        # 优先用 last_accessed，为空则降级 created_at
        ts = row.get("last_accessed") or row.get("created_at")
        if ts is None:
            recency = 0.0
        else:
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if ts.tzinfo is None:
                # W-1: naive datetime 假设为 UTC（DB 层统一存 UTC）
                ts = ts.replace(tzinfo=timezone.utc)
            days = max((now - ts).total_seconds() / 86400.0, 0.0)
            recency = math.exp(-lam * days)
        final = base + 0.15 * recency
        row["hybrid_score"] = final
        scored.append((final, row))
    scored.sort(key=lambda x: (-x[0], -x[1].get("importance", 0)))
    return [r[1] for r in scored]


# ── MemoryStoreDB 实现 ──


class MemoryStoreDB:
    """数据库版 MemoryStore.

    提供与文件版本相同的接口，但数据存储在 PostgreSQL 中。
    """

    def __init__(self, project_dir: Any = None, tenant_id: str | None = None):
        """初始化数据库存储.

        Args:
            project_dir: 项目目录（兼容参数，数据库版不使用）
            tenant_id: 租户 ID（None 则使用默认管理员租户，向后兼容）
        """
        self._project_dir = project_dir
        self._tenant_id = tenant_id or DEFAULT_ADMIN_TENANT_ID
        if not is_pg():
            raise RuntimeError("MemoryStoreDB 仅支持 PostgreSQL 模式")
        import psycopg2.extras

        self._dict_cursor_factory = psycopg2.extras.RealDictCursor

    def _resolve_to_character_name(self, employee: str) -> str:
        """将 slug 或花名统一转换为花名（character_name）.

        委托给模块级公共函数 resolve_to_character_name()。
        """
        return resolve_to_character_name(employee, project_dir=self._project_dir)

    def _get_query_embedding(self, keywords: list[str]) -> list[float] | None:
        """为查询关键词生成 embedding 向量.

        Args:
            keywords: 搜索关键词列表

        Returns:
            384 维浮点向量，或 None（不可用时静默降级）
        """
        try:
            from ensoul.embedding import build_embedding_text, get_embedding

            query_text = build_embedding_text(" ".join(keywords))
            return get_embedding(query_text)
        except Exception:
            return None

    def _row_to_entry(self, row: dict) -> MemoryEntry:
        """将数据库行（RealDictCursor dict）转换为 MemoryEntry."""
        return MemoryEntry(
            id=row["id"],
            employee=row["employee"],
            created_at=row["created_at"].isoformat()
            if hasattr(row["created_at"], "isoformat")
            else str(row["created_at"]),
            category=row["category"],
            content=row["content"],
            source_session=row.get("source_session") or "",
            confidence=float(row.get("confidence", 1.0)),
            superseded_by=row.get("superseded_by") or "",
            ttl_days=int(row.get("ttl_days", 0)),
            importance=int(row.get("importance", 3)),
            last_accessed=row["last_accessed"].isoformat()
            if row.get("last_accessed") and hasattr(row["last_accessed"], "isoformat")
            else (str(row["last_accessed"]) if row.get("last_accessed") else ""),
            tags=list(row.get("tags") or []),
            shared=bool(row.get("shared", False)),
            visibility=row.get("visibility") or "open",
            trigger_condition=row.get("trigger_condition") or "",
            applicability=list(row.get("applicability") or []),
            origin_employee=row.get("origin_employee") or "",
            classification=row.get("classification") or "internal",
            domain=list(row.get("domain") or []),
            keywords=list(row.get("keywords") or []),
            linked_memories=list(row.get("linked_memories") or []),
            valid_from=row.get("valid_from"),
            valid_until=row.get("valid_until"),
        )

    def add(
        self,
        employee: str,
        category: Literal["decision", "estimate", "finding", "correction", "pattern"],
        content: str,
        source_session: str = "",
        confidence: float = 1.0,
        ttl_days: int = 0,
        tags: list[str] | None = None,
        shared: bool = False,
        visibility: Literal["open", "private"] = "open",
        trigger_condition: str = "",
        applicability: list[str] | None = None,
        origin_employee: str = "",
        classification: Literal["public", "internal", "restricted", "confidential"] = "internal",
        domain: list[str] | None = None,
        keywords: list[str] | None = None,
        importance: int = 3,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
    ) -> MemoryEntry:
        """添加一条记忆.

        Returns:
            MemoryEntry 对象
        """
        employee = self._resolve_to_character_name(employee)

        # 防御性夹值：importance 限制 1-5 范围
        importance = max(1, min(5, importance))

        # 防御性截断（S4：以防 API 层校验被绕过）
        content = content[:5000] if len(content) > 5000 else content

        # 12 位 hex = 48 bit 熵，碰撞概率 ~1/2.8e14，当前数据量安全
        entry_id = uuid.uuid4().hex[:12]
        created_at = datetime.now(timezone.utc)

        # pattern 默认共享
        if category == "pattern":
            shared = True

        # 准备数据
        tags_list = tags or []
        applicability_list = applicability or []
        origin_emp = origin_employee or employee
        domain_list = domain or []
        # 支持 keywords 原子写入，避免先 INSERT 空再 UPDATE 的两步操作
        keywords_list = keywords or []

        # NG-1: 生成 embedding 向量（失败不阻塞写入）
        embedding_vector = None
        try:
            from ensoul.embedding import build_embedding_text, get_embedding

            emb_text = build_embedding_text(content, keywords_list)
            embedding_vector = get_embedding(emb_text)
        except Exception as e:
            logger.debug("embedding 生成跳过: %s", e)

        # NG-2: 轻量去重 — correction 类型不做去重（每次纠正都应独立记录）
        if embedding_vector is not None and category != "correction":
            merged = self._try_dedup_merge(
                employee,
                embedding_vector,
                content,
                keywords_list,
                category,
            )
            if merged is not None:
                # NG-3: dedup merge 后也做自动关联（best-effort）
                try:
                    self._auto_link_similar(merged.id, employee, embedding_vector)
                except Exception:
                    logger.debug("auto-link after dedup failed", exc_info=True)
                return merged

        # Phase 4: valid_from 默认为 created_at
        effective_valid_from = valid_from if valid_from is not None else created_at

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO memories (
                    id, employee, created_at, category, content,
                    source_session, confidence, superseded_by, ttl_days,
                    importance, last_accessed, tags, shared, visibility,
                    trigger_condition, applicability, origin_employee, verified_count,
                    classification, domain, tenant_id,
                    keywords, linked_memories, embedding,
                    valid_from, valid_until
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s
                )
                """,
                (
                    entry_id,
                    employee,
                    created_at,
                    category,
                    content,
                    source_session,
                    confidence,
                    "",
                    ttl_days,
                    importance,
                    None,
                    tags_list,
                    shared,
                    visibility,
                    trigger_condition,
                    applicability_list,
                    origin_emp,
                    0,
                    classification,
                    domain_list,
                    self._tenant_id,
                    keywords_list,  # keywords: 原子写入传入值
                    [],  # linked_memories
                    embedding_vector,  # NG-1: embedding 向量（可能为 None）
                    effective_valid_from,
                    valid_until,
                ),
            )

        # NG-3: INSERT 后自动关联相似记忆（best-effort）
        if embedding_vector is not None:
            try:
                self._auto_link_similar(entry_id, employee, embedding_vector)
            except Exception:
                logger.debug("auto-link after insert failed", exc_info=True)

        return MemoryEntry(
            id=entry_id,
            employee=employee,
            created_at=created_at.isoformat(),
            category=category,
            content=content,
            source_session=source_session,
            confidence=confidence,
            superseded_by="",
            ttl_days=ttl_days,
            importance=importance,
            last_accessed="",
            tags=tags_list,
            shared=shared,
            visibility=visibility,
            trigger_condition=trigger_condition,
            applicability=applicability_list,
            origin_employee=origin_emp,
            classification=classification,
            domain=domain_list,
            keywords=keywords_list,
            linked_memories=[],
            valid_from=effective_valid_from,
            valid_until=valid_until,
        )

    def _try_dedup_merge(
        self,
        employee: str,
        embedding_vector: list[float],
        content: str,
        keywords: list[str],
        category: str,
    ) -> MemoryEntry | None:
        """NG-2: 去重合并 — 查同员工同租户 top-1 最相似记忆，>=DEDUP_MERGE_THRESHOLD 则合并.

        Args:
            employee: 员工名称（已 resolve）
            embedding_vector: 新记忆的 embedding 向量
            content: 新记忆内容
            keywords: 新记忆关键词
            category: 新记忆类别

        Returns:
            合并后的 MemoryEntry（如果触发合并），否则 None
        """
        try:
            with get_connection() as conn:
                import psycopg2.extras

                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute(
                    """
                    WITH q AS (SELECT %s::vector AS qvec)
                    SELECT id, content, keywords, category, embedding,
                           employee, created_at, source_session, confidence,
                           superseded_by, ttl_days, importance, last_accessed,
                           tags, shared, visibility, trigger_condition,
                           applicability, origin_employee, verified_count,
                           classification, domain, linked_memories,
                           (embedding <=> q.qvec) AS cosine_dist
                    FROM memories, q
                    WHERE employee = %s AND tenant_id = %s
                      AND (superseded_by = '' OR superseded_by IS NULL)
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> q.qvec
                    LIMIT 1
                    FOR UPDATE
                    """,
                    (str(embedding_vector), employee, self._tenant_id),
                )
                row = cur.fetchone()

                if row is None:
                    return None

                cosine_distance = float(row["cosine_dist"])
                similarity = 1.0 - cosine_distance

                if similarity < DEDUP_MERGE_THRESHOLD or row["category"] != category:
                    return None

                # 触发合并：UPDATE 旧记忆（同一事务，行锁保护）
                merged_content = f"{row['content']}\n---\n{content}"
                if len(merged_content) > 3000:
                    merged_content = f"{row['content'][:1500]}\n---\n{content}"

                old_keywords = list(row.get("keywords") or [])
                merged_keywords = list(set(old_keywords + keywords))

                # 重新生成 embedding（用合并后内容）
                new_embedding = None
                try:
                    from ensoul.embedding import build_embedding_text, get_embedding

                    emb_text = build_embedding_text(merged_content, merged_keywords)
                    new_embedding = get_embedding(emb_text)
                except Exception:
                    new_embedding = embedding_vector  # fallback: 用新记忆的向量

                cur.execute(
                    """
                    UPDATE memories
                    SET content = %s, keywords = %s, embedding = %s
                    WHERE id = %s AND employee = %s AND tenant_id = %s
                    """,
                    (
                        merged_content,
                        merged_keywords,
                        new_embedding,
                        row["id"],
                        employee,
                        self._tenant_id,
                    ),
                )

            logger.info(
                "dedup: merged into %s (similarity=%.2f)",
                row["id"],
                similarity,
            )

            # 返回更新后的 entry
            return self._row_to_entry(
                {
                    **row,
                    "content": merged_content,
                    "keywords": merged_keywords,
                }
            )

        except Exception as e:
            logger.debug("dedup 查询失败，回退到正常写入: %s", e)
            return None

    def _auto_link_similar(
        self, entry_id: str, employee: str, embedding_vector: list[float]
    ) -> None:
        """NG-3: 自动关联 — 查同员工 top-N 相似记忆，>= AUTO_LINK_THRESHOLD 建立双向链接.

        阈值说明: 短文本(50-200字)的 sentence-transformers 余弦相似度
        通常在 0.2-0.5 区间，阈值见 memory_constants.AUTO_LINK_THRESHOLD。

        Best-effort：失败只 log 不报错。
        """
        import psycopg2
        import psycopg2.extras

        try:
            with get_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute(
                    """
                    WITH q AS (SELECT %s::vector AS qvec)
                    SELECT id, linked_memories, 1.0 - (embedding <=> q.qvec) AS similarity
                    FROM memories, q
                    WHERE employee = %s AND tenant_id = %s
                      AND id != %s
                      AND (superseded_by = '' OR superseded_by IS NULL)
                      AND embedding IS NOT NULL
                      AND (1.0 - (embedding <=> q.qvec)) >= %s
                    ORDER BY embedding <=> q.qvec
                    LIMIT %s
                    """,
                    (
                        embedding_vector,
                        employee,
                        self._tenant_id,
                        entry_id,
                        AUTO_LINK_THRESHOLD,
                        AUTO_LINK_LIMIT,
                    ),
                )
                candidates = cur.fetchall()

            if not candidates:
                return

            linked_ids: list[str] = [row["id"] for row in candidates]

            # 在同一个事务中完成所有 UPDATE（新记忆 + 旧记忆双向关联）
            with get_connection() as conn:
                cur = conn.cursor()
                # 更新新记忆的 linked_memories
                cur.execute(
                    "UPDATE memories SET linked_memories = %s WHERE id = %s AND tenant_id = %s",
                    (linked_ids, entry_id, self._tenant_id),
                )
                # 更新旧记忆的 linked_memories（原子操作，避免并发覆盖）
                for row in candidates:
                    cur.execute(
                        """
                        UPDATE memories
                        SET linked_memories = linked_memories || ARRAY[%s::text]
                        WHERE id = %s AND tenant_id = %s
                          AND NOT (%s = ANY(COALESCE(linked_memories, '{}')))
                        """,
                        (entry_id, row["id"], self._tenant_id, entry_id),
                    )

            logger.debug(
                "auto-link: %s linked to %s",
                entry_id,
                linked_ids,
            )

        except (psycopg2.Error, ValueError, KeyError) as e:
            logger.debug("auto-link failed (best-effort): %s", e)
        except Exception:
            logger.exception("auto-link unexpected error")

    # 信息分级等级序（用于 classification_max 过滤）
    _CLASSIFICATION_LEVELS = CLASSIFICATION_LEVELS

    def find_similar_by_content(
        self,
        employee: str,
        content: str,
        category: str | None = None,
        threshold: float = 0.85,
        limit: int = 3,
    ) -> list[tuple[dict[str, Any], float]]:
        """查找与给定内容语义相似的记忆（用于去重预检）.

        Args:
            employee: 员工名称
            content: 待比较内容
            category: 按类别过滤（可选）
            threshold: 余弦相似度阈值（0-1），默认 0.85
            limit: 最多返回条数

        Returns:
            [(memory_dict, similarity_score), ...] 按相似度降序
        """
        employee = self._resolve_to_character_name(employee)

        # 生成 embedding
        try:
            from ensoul.embedding import build_embedding_text, get_embedding

            emb_text = build_embedding_text(content)
            embedding_vector = get_embedding(emb_text)
        except Exception:
            return []

        if embedding_vector is None:
            return []

        # cosine_distance = 1 - cosine_similarity
        max_distance = 1.0 - threshold

        conditions = [
            "employee = %s",
            "tenant_id = %s",
            "(superseded_by = '' OR superseded_by IS NULL)",
            "embedding IS NOT NULL",
            "(embedding <=> %s) <= %s",
        ]
        params: list[Any] = [
            employee,
            self._tenant_id,
            str(embedding_vector),
            max_distance,
        ]

        if category:
            conditions.append("category = %s")
            params.append(category)

        where_clause = " AND ".join(conditions)

        with get_connection() as conn:
            import psycopg2.extras

            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                f"""
                SELECT *, (embedding <=> %s) AS cosine_dist
                FROM memories
                WHERE {where_clause}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (str(embedding_vector), *params, str(embedding_vector), limit),
            )
            rows = cur.fetchall()

        result: list[tuple[dict[str, Any], float]] = []
        for row in rows:
            similarity = 1.0 - float(row["cosine_dist"])
            entry = self._row_to_entry(row)
            result.append((entry.model_dump(), similarity))

        return result

    def query(
        self,
        employee: str,
        category: str | None = None,
        limit: int = 20,
        min_confidence: float = 0.0,
        include_expired: bool = False,
        max_visibility: str = "private",
        sort_by: str = "created_at",
        min_importance: int = 0,
        update_access: bool = False,
        classification_max: str | None = None,
        allowed_domains: list[str] | None = None,
        include_confidential: bool = False,
        search_text: str | None = None,
    ) -> list[MemoryEntry]:
        """查询员工记忆.

        Args:
            employee: 员工名称
            category: 按类别过滤（可选）
            limit: 最大返回条数
            min_confidence: 最低置信度
            include_expired: 是否包含已过期条目
            max_visibility: 可见性上限
            sort_by: 排序方式
            min_importance: 最低重要性
            update_access: 是否更新 last_accessed
            classification_max: 最高信息分级（可选，按等级过滤）
            allowed_domains: 允许的职能域（可选，restricted 级别需域匹配）
            include_confidential: 是否包含 confidential 级别记忆（默认 False）
            search_text: 文本搜索（可选，ILIKE 子串匹配，支持中文）

        Returns:
            记忆列表（MemoryEntry）
        """
        employee = self._resolve_to_character_name(employee)

        # 构建查询（租户隔离）
        conditions = [
            "employee = %s",
            "tenant_id = %s",
            "(superseded_by = '' OR superseded_by IS NULL)",
        ]
        params: list[Any] = [employee, self._tenant_id]

        if category:
            conditions.append("category = %s")
            params.append(category)

        if min_confidence > 0:
            conditions.append("confidence >= %s")
            params.append(min_confidence)

        if min_importance > 0:
            conditions.append("importance >= %s")
            params.append(min_importance)

        if max_visibility != "private":
            conditions.append("visibility = 'open'")

        if not include_expired:
            # 过滤过期记忆：ttl_days > 0 且 created_at + ttl_days < now
            conditions.append(
                "(ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())"
            )
            # Phase 4: 时序过滤 — 排除已失效记忆
            conditions.append("(valid_until IS NULL OR valid_until > NOW())")

        # 文本搜索（ILIKE 子串匹配，对中文友好）
        # 按空格拆分为多个词，每个词用 AND 连接，支持 "skip_paths 误删" 这类多词查询
        if search_text:
            tokens = search_text.strip().split()
            for token in tokens:
                conditions.append("content ILIKE %s")
                params.append(f"%{token}%")

        # 信息分级过滤
        if not include_confidential:
            conditions.append("COALESCE(classification, 'internal') != 'confidential'")

        if classification_max is not None:
            max_level = self._CLASSIFICATION_LEVELS.get(classification_max, 1)
            allowed_classifications = [
                k for k, v in self._CLASSIFICATION_LEVELS.items() if v <= max_level
            ]
            placeholders = ", ".join(["%s"] * len(allowed_classifications))
            conditions.append(f"COALESCE(classification, 'internal') IN ({placeholders})")
            params.extend(allowed_classifications)

        # [W7] restricted 域匹配移到 SQL 层
        if allowed_domains is not None:
            domain_placeholders = ", ".join(["%s"] * len(allowed_domains))
            conditions.append(
                f"(COALESCE(classification, 'internal') != 'restricted' OR domain && ARRAY[{domain_placeholders}]::text[])"
            )
            params.extend(allowed_domains)

        # 排序
        # 当有 search_text 时，按 category 权重排序（知识类优先）
        if search_text:
            order_by = (
                "CASE category "
                "WHEN 'pattern' THEN 1 "
                "WHEN 'decision' THEN 2 "
                "WHEN 'correction' THEN 3 "
                "WHEN 'estimate' THEN 4 "
                "WHEN 'finding' THEN 5 "
                "ELSE 6 END, "
                "importance DESC, created_at DESC"
            )
        elif sort_by == "importance":
            order_by = "importance DESC, created_at DESC"
        elif sort_by == "confidence":
            order_by = "confidence DESC, created_at DESC"
        else:
            order_by = "created_at DESC"

        # 执行查询
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
            sql = f"""
                SELECT id, employee, created_at, category, content,
                       source_session, confidence, superseded_by, ttl_days,
                       importance, last_accessed, tags, shared, visibility,
                       trigger_condition, applicability, origin_employee, verified_count,
                       classification, domain,
                       keywords, linked_memories
                FROM memories
                WHERE {" AND ".join(conditions)}
                ORDER BY {order_by}
                LIMIT %s
            """
            params.append(limit)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        # 转换为 MemoryEntry
        results: list[MemoryEntry] = []
        entry_ids: list[str] = []
        for row in rows:
            entry = self._row_to_entry(row)
            results.append(entry)
            entry_ids.append(entry.id)

        # Phase 4：审计日志
        logger.info(
            "memory_access: employee=%s classification_max=%s allowed_domains=%s "
            "include_confidential=%s returned=%d channel=unknown",
            employee,
            classification_max or "none",
            allowed_domains or "none",
            include_confidential,
            len(results),
        )

        # 更新访问时间
        if update_access and entry_ids:
            self._update_last_accessed(employee, entry_ids)

        return results

    def _update_last_accessed(self, employee: str, entry_ids: list[str]) -> None:
        """批量更新记忆的 last_accessed 时间戳."""
        if not entry_ids:
            return

        now = datetime.now(timezone.utc)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE memories
                SET last_accessed = %s
                WHERE id = ANY(%s) AND employee = %s AND tenant_id = %s
                """,
                (now, entry_ids, employee, self._tenant_id),
            )

    def query_shared(
        self,
        tags: list[str] | None = None,
        exclude_employee: str = "",
        limit: int = 10,
        min_confidence: float = 0.3,
    ) -> list[MemoryEntry]:
        """查询跨员工的共享记忆.

        Args:
            tags: 按标签过滤（任一匹配即可）
            exclude_employee: 排除指定员工
            limit: 最大返回条数
            min_confidence: 最低置信度

        Returns:
            共享记忆列表（MemoryEntry）
        """
        exclude_employee = self._resolve_to_character_name(exclude_employee)

        conditions = [
            "shared = TRUE",
            "tenant_id = %s",
            "(superseded_by = '' OR superseded_by IS NULL)",
        ]
        params: list[Any] = [self._tenant_id]

        if exclude_employee:
            conditions.append("employee != %s")
            params.append(exclude_employee)

        if min_confidence > 0:
            conditions.append("confidence >= %s")
            params.append(min_confidence)

        # 过滤过期记忆
        conditions.append("(ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())")
        # Phase 4: 时序过滤
        conditions.append("(valid_until IS NULL OR valid_until > NOW())")

        # 标签过滤（任一匹配）
        if tags:
            conditions.append("tags && %s")
            params.append(tags)

        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
            sql = f"""
                SELECT id, employee, created_at, category, content,
                       source_session, confidence, superseded_by, ttl_days,
                       importance, last_accessed, tags, shared, visibility,
                       trigger_condition, applicability, origin_employee, verified_count,
                       classification, domain,
                       keywords, linked_memories
                FROM memories
                WHERE {" AND ".join(conditions)}
                ORDER BY created_at DESC
                LIMIT %s
            """
            params.append(limit)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        results: list[MemoryEntry] = []
        for row in rows:
            results.append(self._row_to_entry(row))

        return results

    def query_patterns(
        self,
        employee: str = "",
        applicability: list[str] | None = None,
        limit: int = 10,
        min_confidence: float = 0.3,
    ) -> list[MemoryEntry]:
        """查询可复用的工作模式（跨员工）.

        Args:
            employee: 当前员工（排除自己的）
            applicability: 按适用范围标签过滤
            limit: 最大返回条数
            min_confidence: 最低置信度

        Returns:
            pattern 列表（MemoryEntry）
        """
        conditions = [
            "category = 'pattern'",
            "tenant_id = %s",
            "(superseded_by = '' OR superseded_by IS NULL)",
        ]
        params: list[Any] = [self._tenant_id]

        if employee:
            employee = self._resolve_to_character_name(employee)
            conditions.append("employee != %s")
            params.append(employee)

        if min_confidence > 0:
            conditions.append("confidence >= %s")
            params.append(min_confidence)

        # 过滤过期记忆
        conditions.append("(ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())")
        # Phase 4: 时序过滤
        conditions.append("(valid_until IS NULL OR valid_until > NOW())")

        # 适用范围过滤
        if applicability:
            conditions.append("applicability && %s")
            params.append(applicability)

        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
            sql = f"""
                SELECT id, employee, created_at, category, content,
                       source_session, confidence, superseded_by, ttl_days,
                       importance, last_accessed, tags, shared, visibility,
                       trigger_condition, applicability, origin_employee, verified_count,
                       classification, domain,
                       keywords, linked_memories
                FROM memories
                WHERE {" AND ".join(conditions)}
                ORDER BY verified_count DESC, created_at DESC
                LIMIT %s
            """
            params.append(limit)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        results: list[MemoryEntry] = []
        for row in rows:
            results.append(self._row_to_entry(row))

        return results

    def delete(self, entry_id: str, employee: str | None = None) -> bool:
        """删除指定的记忆条目.

        Args:
            entry_id: 记忆条目 ID
            employee: 员工名（可选）

        Returns:
            True 如果删除成功，False 如果未找到
        """
        with get_connection() as conn:
            cur = conn.cursor()
            if employee:
                employee = self._resolve_to_character_name(employee)
                cur.execute(
                    "DELETE FROM memories WHERE id = %s AND employee = %s AND tenant_id = %s",
                    (entry_id, employee, self._tenant_id),
                )
            else:
                cur.execute(
                    "DELETE FROM memories WHERE id = %s AND tenant_id = %s",
                    (entry_id, self._tenant_id),
                )

            return cur.rowcount > 0

    def update_confidence(self, entry_id: str, confidence: float) -> bool:
        """更新记忆的置信度.

        Args:
            entry_id: 记忆条目 ID
            confidence: 新的置信度

        Returns:
            True 如果更新成功，False 如果未找到
        """
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE memories SET confidence = %s WHERE id = %s AND tenant_id = %s",
                (confidence, entry_id, self._tenant_id),
            )
            return cur.rowcount > 0

    def count(self, employee: str | None = None) -> int:
        """返回有效记忆条数.

        Args:
            employee: 员工名称。None 时返回全租户总数。
        """
        with get_connection() as conn:
            cur = conn.cursor()
            if employee:
                employee = self._resolve_to_character_name(employee)
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM memories
                    WHERE employee = %s AND tenant_id = %s
                      AND (superseded_by = '' OR superseded_by IS NULL)
                      AND (ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())
                      AND (valid_until IS NULL OR valid_until > NOW())
                    """,
                    (employee, self._tenant_id),
                )
            else:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM memories
                    WHERE tenant_id = %s
                      AND (superseded_by = '' OR superseded_by IS NULL)
                      AND (ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())
                      AND (valid_until IS NULL OR valid_until > NOW())
                    """,
                    (self._tenant_id,),
                )
            row = cur.fetchone()
            return row[0] if row else 0

    def load_employee_entries(self, employee: str) -> list[MemoryEntry]:
        """加载指定员工的全部记忆条目（不做过滤）。

        公开接口，与文件版 MemoryStore.load_employee_entries() 对齐。
        """
        employee = self._resolve_to_character_name(employee)

        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
            cur.execute(
                """
                SELECT id, employee, created_at, category, content,
                       source_session, confidence, superseded_by, ttl_days,
                       importance, last_accessed, tags, shared, visibility,
                       trigger_condition, applicability, origin_employee, verified_count,
                       classification, domain,
                       keywords, linked_memories
                FROM memories
                WHERE employee = %s AND tenant_id = %s
                ORDER BY created_at ASC
                """,
                (employee, self._tenant_id),
            )
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    @staticmethod
    def is_expired(entry: MemoryEntry) -> bool:
        """检查记忆是否已过期（公开接口，与文件版对齐）."""
        ttl = entry.ttl_days
        if ttl <= 0:
            return False
        try:
            created = datetime.fromisoformat(entry.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
            return age_days > ttl
        except (ValueError, TypeError):
            return False

    def list_employees(self) -> list[str]:
        """列出有记忆的员工（当前租户）."""
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT DISTINCT employee FROM memories WHERE tenant_id = %s ORDER BY employee",
                (self._tenant_id,),
            )
            rows = cur.fetchall()
            return [row[0] for row in rows]

    def correct(
        self,
        employee: str,
        old_id: str,
        new_content: str,
        source_session: str = "",
    ) -> MemoryEntry | None:
        """纠正一条记忆：标记旧记忆为 superseded，创建新记忆.

        Args:
            employee: 员工名称
            old_id: 要纠正的记忆 ID
            new_content: 纠正后的内容
            source_session: 来源 session

        Returns:
            新创建的纠正记忆，如果旧记忆不存在返回 None
        """
        employee = self._resolve_to_character_name(employee)

        # 12 位 hex = 48 bit 熵，碰撞概率 ~1/2.8e14，当前数据量安全
        new_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)

        with get_connection() as conn:
            cur = conn.cursor()
            # 标记旧记忆为 superseded，同时清除 embedding 释放 ivfflat 索引空间
            cur.execute(
                """
                UPDATE memories
                SET superseded_by = %s, confidence = 0.0, embedding = NULL
                WHERE id = %s AND employee = %s AND tenant_id = %s
                """,
                (new_id, old_id, employee, self._tenant_id),
            )
            if cur.rowcount == 0:
                return None

            # 创建纠正记忆
            cur.execute(
                """
                INSERT INTO memories (
                    id, employee, created_at, category, content,
                    source_session, confidence, superseded_by, ttl_days,
                    importance, last_accessed, tags, shared, visibility,
                    trigger_condition, applicability, origin_employee, verified_count,
                    classification, domain, tenant_id
                ) VALUES (
                    %s, %s, %s, 'correction', %s,
                    %s, 1.0, '', 0,
                    3, NULL, '{}', FALSE, 'open',
                    '', '{}', %s, 0,
                    'internal', '{}', %s
                )
                """,
                (new_id, employee, now, new_content, source_session, employee, self._tenant_id),
            )

        return MemoryEntry(
            id=new_id,
            employee=employee,
            created_at=now.isoformat(),
            category="correction",
            content=new_content,
            source_session=source_session,
            confidence=1.0,
        )

    def format_for_prompt(
        self,
        employee: str,
        limit: int = 10,
        query: str = "",
        employee_tags: list[str] | None = None,
        max_visibility: str = "open",
        team_members: list[str] | None = None,
        classification_max: str | None = None,
        allowed_domains: list[str] | None = None,
        include_confidential: bool = False,
    ) -> str:
        """格式化记忆为可注入 prompt 的文本.

        Args:
            employee: 员工名称
            limit: 最大条数
            query: 查询上下文（有值时走 search_text 语义搜索，返回相关记忆）
            employee_tags: 员工标签（用于匹配共享记忆）
            max_visibility: 可见性上限
            team_members: 同团队成员名列表
            classification_max: 最高信息分级（可选）
            allowed_domains: 允许的职能域（可选）
            include_confidential: 是否包含 confidential 级别

        Returns:
            Markdown 格式的记忆文本，无记忆时返回空字符串
        """
        employee = self._resolve_to_character_name(employee)
        parts: list[str] = []

        # 个人记忆（透传 classification 参数）
        # L2 优化：有 query 时走 search_text 语义搜索，返回最相关 Top-N
        entries = self.query(
            employee,
            limit=limit,
            max_visibility=max_visibility,
            classification_max=classification_max,
            allowed_domains=allowed_domains,
            include_confidential=include_confidential,
            search_text=query if query else None,
        )
        if entries:
            parts.append(self._format_entries(entries))

        # 跨员工共享记忆
        shared_entries = self.query_shared(
            tags=employee_tags,
            exclude_employee=employee,
            limit=max(3, limit // 3),
        )
        if shared_entries:
            lines = []
            for entry in shared_entries:
                tag_str = f" [{', '.join(entry.tags)}]" if entry.tags else ""
                cat = self._category_label(entry.category)
                conf = f" (置信度: {entry.confidence:.0%})" if entry.confidence < 1.0 else ""
                trigger = (
                    f" [触发: {entry.trigger_condition}]"
                    if entry.category == "pattern" and entry.trigger_condition
                    else ""
                )
                lines.append(
                    f"- [{cat}]{conf}{tag_str}{trigger} ({entry.employee}) {entry.content}"
                )
            parts.append("\n### 团队共享经验\n\n" + "\n".join(lines))

        # 同团队成员的公开记忆
        if team_members:
            team_entries = self.query_team(
                team_members,
                exclude_employee=employee,
                limit=max(3, limit // 3),
            )
            if team_entries:
                lines = []
                for entry in team_entries:
                    cat = self._category_label(entry.category)
                    conf = f" (置信度: {entry.confidence:.0%})" if entry.confidence < 1.0 else ""
                    lines.append(f"- [{cat}]{conf} ({entry.employee}) {entry.content}")
                parts.append("\n### 队友近况\n\n" + "\n".join(lines))

        return "\n".join(parts)

    @staticmethod
    def _category_label(category: str) -> str:
        """将类别转为中文标签."""
        return CATEGORY_LABELS.get(category, category)

    @staticmethod
    def _format_entries(entries: list[MemoryEntry]) -> str:
        """格式化记忆条目列表为 Markdown."""
        lines = []
        for entry in entries:
            category_label = CATEGORY_LABELS.get(entry.category, entry.category)
            conf = f" (置信度: {entry.confidence:.0%})" if entry.confidence < 1.0 else ""
            proxied = " ⚠️模拟讨论记录，非实际工作" if "proxied" in entry.tags else ""
            trigger = (
                f" [触发: {entry.trigger_condition}]"
                if entry.category == "pattern" and entry.trigger_condition
                else ""
            )
            lines.append(f"- [{category_label}]{conf}{proxied}{trigger} {entry.content}")
        return "\n".join(lines)

    def query_team(
        self,
        members: list[str],
        exclude_employee: str = "",
        limit: int = 5,
        min_confidence: float = 0.3,
    ) -> list[MemoryEntry]:
        """查询指定团队成员的公开记忆（不要求 shared=True）.

        Args:
            members: 团队成员名列表
            exclude_employee: 排除指定员工（通常是当前员工自身）
            limit: 最大返回条数
            min_confidence: 最低有效置信度
        """
        members = [self._resolve_to_character_name(m) for m in members]
        exclude_employee = self._resolve_to_character_name(exclude_employee)

        member_set = set(members) - {exclude_employee}
        if not member_set:
            return []

        conditions = [
            "employee = ANY(%s)",
            "tenant_id = %s",
            "(superseded_by = '' OR superseded_by IS NULL)",
            "visibility = 'open'",
            "(ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())",
            "(valid_until IS NULL OR valid_until > NOW())",
        ]
        params: list[Any] = [list(member_set), self._tenant_id]

        if min_confidence > 0:
            conditions.append("confidence >= %s")
            params.append(min_confidence)

        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
            sql = f"""
                SELECT id, employee, created_at, category, content,
                       source_session, confidence, superseded_by, ttl_days,
                       importance, last_accessed, tags, shared, visibility,
                       trigger_condition, applicability, origin_employee, verified_count,
                       classification, domain,
                       keywords, linked_memories
                FROM memories
                WHERE {" AND ".join(conditions)}
                ORDER BY created_at DESC
                LIMIT %s
            """
            params.append(limit)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def verify_pattern(self, pattern_id: str) -> bool:
        """验证一条 pattern（verified_count +1）.

        Returns:
            True 如果找到并更新，False 如果未找到
        """
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE memories
                SET verified_count = verified_count + 1
                WHERE id = %s AND category = 'pattern' AND tenant_id = %s
                """,
                (pattern_id, self._tenant_id),
            )
            return cur.rowcount > 0

    def update(
        self,
        entry_id: str,
        employee: str,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        confidence: float | None = None,
        importance: int | None = None,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        valid_until: datetime | None = ...,  # type: ignore[assignment]
        superseded_by: str | None = None,
    ) -> bool:
        """更新一条记忆.

        Args:
            entry_id: 记忆 ID
            employee: 员工名称
            content: 新内容（None 不更新）
            tags: 完全替换标签列表（None 不更新）
            confidence: 新置信度（None 不更新）
            importance: 新重要度（None 不更新）
            add_tags: 追加标签
            remove_tags: 移除标签
            valid_until: 事实失效时间（sentinel ...=不更新, None=清除, datetime=设置）
            superseded_by: 取代此记忆的新记忆 ID（None 不更新）

        Returns:
            True 更新成功，False 未找到
        """
        employee = self._resolve_to_character_name(employee)

        # 先查出当前状态
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
            cur.execute(
                "SELECT id, tags FROM memories WHERE id = %s AND employee = %s AND tenant_id = %s",
                (entry_id, employee, self._tenant_id),
            )
            row = cur.fetchone()
            if not row:
                return False

            # 构建 SET 子句
            sets: list[str] = []
            params: list[Any] = []

            if content is not None:
                sets.append("content = %s")
                params.append(content)

            if confidence is not None:
                sets.append("confidence = %s")
                params.append(confidence)

            if importance is not None:
                sets.append("importance = %s")
                params.append(importance)

            # Phase 4: valid_until 时序感知
            if valid_until is not ...:
                sets.append("valid_until = %s")
                params.append(valid_until)

            if superseded_by is not None:
                sets.append("superseded_by = %s")
                params.append(superseded_by)

            # 标签处理
            current_tags = list(row.get("tags") or [])

            if tags is not None:
                # 完全替换
                current_tags = tags

            if add_tags:
                current_tags = list(set(current_tags + add_tags))

            if remove_tags:
                current_tags = [t for t in current_tags if t not in remove_tags]

            if tags is not None or add_tags or remove_tags:
                sets.append("tags = %s")
                params.append(current_tags)

            if not sets:
                return True  # 没有需要更新的字段

            params.extend([entry_id, employee, self._tenant_id])
            cur.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = %s AND employee = %s AND tenant_id = %s",
                tuple(params),
            )
            return cur.rowcount > 0

    def update_keywords(self, entry_id: str, employee: str, keywords: list[str]) -> bool:
        """更新记忆的结构化关键词.

        Args:
            entry_id: 记忆 ID
            employee: 员工名称
            keywords: 新的关键词列表（完全替换）

        Returns:
            True 更新成功，False 未找到
        """
        employee = self._resolve_to_character_name(employee)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE memories SET keywords = %s WHERE id = %s AND employee = %s AND tenant_id = %s",
                (keywords, entry_id, employee, self._tenant_id),
            )
            return cur.rowcount > 0

    def update_linked_memories(self, entry_id: str, employee: str, linked_ids: list[str]) -> bool:
        """更新记忆的关联记忆 ID 列表.

        Args:
            entry_id: 记忆 ID
            employee: 员工名称
            linked_ids: 关联记忆 ID 列表（完全替换）

        Returns:
            True 更新成功，False 未找到
        """
        employee = self._resolve_to_character_name(employee)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE memories SET linked_memories = %s WHERE id = %s AND employee = %s AND tenant_id = %s",
                (linked_ids, entry_id, employee, self._tenant_id),
            )
            return cur.rowcount > 0

    def expand_linked_memories(
        self, entries: list[MemoryEntry], max_linked: int = 2
    ) -> dict[str, list[MemoryEntry]]:
        """NG-3: 展开关联记忆 — 对每条 entry 取前 max_linked 个 linked ID 批量查询.

        Args:
            entries: 一组 MemoryEntry
            max_linked: 每条记忆最多展开的关联记忆数

        Returns:
            dict: {entry_id: [linked_entry_1, linked_entry_2, ...]}
            不递归展开（只展一层）。
        """
        # 收集所有需要查询的 linked IDs
        all_linked_ids: list[str] = []
        entry_to_linked: dict[str, list[str]] = {}
        for entry in entries:
            linked = list(entry.linked_memories or [])[:max_linked]
            if linked:
                entry_to_linked[entry.id] = linked
                all_linked_ids.extend(linked)

        if not all_linked_ids:
            return {}

        # 去重
        unique_ids = list(set(all_linked_ids))

        # 批量查询
        try:
            with get_connection() as conn:
                cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
                cur.execute(
                    """
                    SELECT id, employee, created_at, category, content,
                           source_session, confidence, superseded_by, ttl_days,
                           importance, last_accessed, tags, shared, visibility,
                           trigger_condition, applicability, origin_employee, verified_count,
                           classification, domain,
                           keywords, linked_memories
                    FROM memories
                    WHERE id = ANY(%s) AND tenant_id = %s
                      AND (superseded_by = '' OR superseded_by IS NULL)
                    """,
                    (unique_ids, self._tenant_id),
                )
                rows = cur.fetchall()
        except Exception:
            logger.warning("expand_linked_memories query failed", exc_info=True)
            return {}

        # 建立 id -> MemoryEntry 映射
        id_to_entry: dict[str, MemoryEntry] = {}
        for row in rows:
            id_to_entry[row["id"]] = self._row_to_entry(row)

        # 组装结果
        result: dict[str, list[MemoryEntry]] = {}
        for entry_id, linked_ids in entry_to_linked.items():
            linked_entries = [id_to_entry[lid] for lid in linked_ids if lid in id_to_entry]
            if linked_entries:
                result[entry_id] = linked_entries

        return result

    def _query_hybrid_core(
        self,
        conditions: list[str],
        condition_params: list[Any],
        keywords: list[str],
        limit: int,
    ) -> list[MemoryEntry]:
        """混合检索核心逻辑（关键词 + 向量语义 + 效用评分）.

        供 query_by_keywords 和 query_cross_employee 共用，避免 ~140 行重复代码。

        混合排序公式：final_score = 0.15*kw + 0.40*cosine + 0.15*q_value + 0.15*importance + 0.15*recency
        对低召回记忆（recall_count < 5）使用 Thompson Sampling 重排。
        当 embedding 不可用时，退化为纯关键词匹配（向后兼容）。

        Args:
            conditions: WHERE 子句条件列表（调用方已构建好员工/可见性等过滤条件）
            condition_params: conditions 对应的参数列表
            keywords: 搜索关键词（已截断到 <=10 个）
            limit: 最大返回条数

        Returns:
            匹配到的 MemoryEntry 列表
        """
        # 尝试生成查询向量
        query_embedding = self._get_query_embedding(keywords)

        # 构建 match_count 表达式：对每个搜索关键词累加是否匹配
        match_parts: list[str] = []
        match_params: list[Any] = []

        for kw in keywords:
            match_parts.append(
                "(CASE WHEN EXISTS (SELECT 1 FROM unnest(keywords) AS k WHERE k ILIKE %s) THEN 1 ELSE 0 END)"
            )
            match_params.append(f"%{kw}%")

        match_count_expr = " + ".join(match_parts)
        num_keywords = len(keywords)

        # 深拷贝条件和参数，避免修改调用方数据
        conditions = list(conditions)
        params = list(condition_params)

        if query_embedding is not None:
            # 混合检索：关键词匹配 OR 向量相似度足够高
            or_conditions = []
            for kw in keywords:
                or_conditions.append(
                    "EXISTS (SELECT 1 FROM unnest(keywords) AS k WHERE k ILIKE %s)"
                )
                params.append(f"%{kw}%")
            recall_max_dist = 1.0 - AUTO_LINK_THRESHOLD
            or_conditions.append("(embedding IS NOT NULL AND embedding <=> %s < %s)")
            params.append(str(query_embedding))
            params.append(recall_max_dist)
            conditions.append(f"({' OR '.join(or_conditions)})")

            # 混合评分（SQL 层 4 因子，recency 在 Python 后处理）：
            # 0.15 * keyword_norm + 0.40 * cosine_sim + 0.15 * q_value + 0.15 * importance_norm = 0.85
            # + 0.15 * recency_decay（Python 层）
            score_expr = (
                f"(0.15 * (({match_count_expr}) / {num_keywords}.0)"
                f" + 0.40 * CASE WHEN embedding IS NOT NULL"
                f" THEN 1.0 - (embedding <=> %s) ELSE 0 END"
                f" + 0.15 * COALESCE(q_value, 0.5)"
                f" + 0.15 * (COALESCE(importance, 3) / 5.0))"
            )
            # score_expr 内嵌的参数：match_count 的 N 个 kw + 1 个 embedding
            score_params: list[Any] = list(match_params) + [str(query_embedding)]

            with get_connection() as conn:
                cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
                sql = f"""
                    SELECT id, employee, created_at, category, content,
                           source_session, confidence, superseded_by, ttl_days,
                           importance, last_accessed, tags, shared, visibility,
                           trigger_condition, applicability, origin_employee, verified_count,
                           classification, domain,
                           keywords, linked_memories,
                           recall_count, q_value,
                           {score_expr} AS hybrid_score
                    FROM memories
                    WHERE {" AND ".join(conditions)}
                    ORDER BY hybrid_score DESC, importance DESC, created_at DESC
                    LIMIT %s
                """
                all_params = tuple(score_params + params + [limit])
                cur.execute(sql, all_params)
                rows = cur.fetchall()

            rows = [dict(r) for r in rows]
            rows = _apply_recency(rows)
            rows = _thompson_rescore(rows)
            return [self._row_to_entry(row) for row in rows]

        else:
            # 纯关键词匹配（降级模式）
            or_conditions = []
            for kw in keywords:
                or_conditions.append(
                    "EXISTS (SELECT 1 FROM unnest(keywords) AS k WHERE k ILIKE %s)"
                )
                params.append(f"%{kw}%")
            conditions.append(f"({' OR '.join(or_conditions)})")

            with get_connection() as conn:
                cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
                sql = f"""
                    SELECT id, employee, created_at, category, content,
                           source_session, confidence, superseded_by, ttl_days,
                           importance, last_accessed, tags, shared, visibility,
                           trigger_condition, applicability, origin_employee, verified_count,
                           classification, domain,
                           keywords, linked_memories,
                           ({match_count_expr}) AS match_count
                    FROM memories
                    WHERE {" AND ".join(conditions)}
                    ORDER BY match_count DESC, importance DESC, created_at DESC
                    LIMIT %s
                """
                all_params = tuple(match_params + params + [limit])
                cur.execute(sql, all_params)
                rows = cur.fetchall()

            return [self._row_to_entry(row) for row in rows]

    def query_by_keywords(
        self,
        employee: str,
        keywords: list[str],
        limit: int = 10,
        category: str | None = None,
        visibility: str | None = None,
    ) -> list[MemoryEntry]:
        """按关键词匹配记忆，支持混合检索（关键词 + 向量语义 + 效用评分）.

        混合排序公式：final_score = 0.15*kw + 0.40*cosine + 0.15*q_value + 0.15*importance + 0.15*recency
        对低召回记忆（recall_count < 5）使用 Thompson Sampling 重排。
        当 embedding 不可用时，退化为纯关键词匹配（向后兼容）。
        """
        employee = self._resolve_to_character_name(employee)

        if not keywords:
            return self.query(employee, category=category, limit=limit)

        keywords = keywords[:10]

        conditions = [
            "employee = %s",
            "tenant_id = %s",
            "(superseded_by = '' OR superseded_by IS NULL)",
            "(ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())",
            "(valid_until IS NULL OR valid_until > NOW())",
        ]
        params: list[Any] = [employee, self._tenant_id]

        if category:
            conditions.append("category = %s")
            params.append(category)

        if visibility is not None:
            conditions.append("visibility = %s")
            params.append(visibility)

        return self._query_hybrid_core(conditions, params, keywords, limit)

    def query_cross_employee(
        self,
        keywords: list[str],
        exclude_employee: str = "",
        limit: int = 5,
        category: str | None = None,
    ) -> list[MemoryEntry]:
        """跨员工按关键词匹配记忆，支持混合检索（仅 visibility=open）.

        不限制员工，但排除指定员工（通常是当前员工自身）。
        混合排序公式：final_score = 0.15*kw + 0.40*cosine + 0.15*q_value + 0.15*importance + 0.15*recency
        对低召回记忆（recall_count < 5）使用 Thompson Sampling 重排。
        当 embedding 不可用时，退化为纯关键词匹配。
        """
        if not keywords:
            return []

        exclude_employee = self._resolve_to_character_name(exclude_employee)
        keywords = keywords[:10]

        conditions = [
            "tenant_id = %s",
            "(superseded_by = '' OR superseded_by IS NULL)",
            "(ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())",
            "(valid_until IS NULL OR valid_until > NOW())",
            "visibility = 'open'",
            "classification IN ('public', 'internal')",
        ]
        params: list[Any] = [self._tenant_id]

        if exclude_employee:
            conditions.append("employee != %s")
            params.append(exclude_employee)

        if category:
            conditions.append("category = %s")
            params.append(category)

        return self._query_hybrid_core(conditions, params, keywords, limit)

    def get_knowledge_stats(self) -> dict:
        """返回团队知识结构统计.

        Returns:
            {
                "employee_stats": [{"employee": "xxx", "total": N, "by_category": {...}}],
                "top_keywords": [{"keyword": "xxx", "count": N}],  # Top 30
                "correction_hotspots": [{"keyword": "xxx", "correction_count": N}],
                "knowledge_gaps": [{"keyword": "xxx", "findings": N, "patterns": 0}],
                "weekly_trend": [{"week": "2026-W10", "count": N}],  # 最近 12 周
            }
        """
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)

            # 1. employee_stats: 每个员工的记忆数和分类统计
            cur.execute(
                """
                SELECT employee, category, COUNT(*) as cnt
                FROM memories
                WHERE tenant_id = %s
                  AND (superseded_by = '' OR superseded_by IS NULL)
                GROUP BY employee, category
                ORDER BY employee, category
                """,
                (self._tenant_id,),
            )
            rows = cur.fetchall()
            emp_map: dict[str, dict] = {}
            for row in rows:
                emp = row["employee"]
                if emp not in emp_map:
                    emp_map[emp] = {"employee": emp, "total": 0, "by_category": {}}
                emp_map[emp]["by_category"][row["category"]] = row["cnt"]
                emp_map[emp]["total"] += row["cnt"]
            employee_stats = list(emp_map.values())

            # 2. top_keywords: 展开 keywords 数组统计频次 Top 30
            cur.execute(
                """
                SELECT kw, COUNT(*) as cnt
                FROM memories, unnest(keywords) AS kw
                WHERE tenant_id = %s
                  AND (superseded_by = '' OR superseded_by IS NULL)
                GROUP BY kw
                ORDER BY cnt DESC
                LIMIT 30
                """,
                (self._tenant_id,),
            )
            top_keywords = [{"keyword": r["kw"], "count": r["cnt"]} for r in cur.fetchall()]

            # 3. correction_hotspots: correction 类别中高频关键词
            cur.execute(
                """
                SELECT kw, COUNT(*) as cnt
                FROM memories, unnest(keywords) AS kw
                WHERE tenant_id = %s
                  AND category = 'correction'
                  AND (superseded_by = '' OR superseded_by IS NULL)
                GROUP BY kw
                ORDER BY cnt DESC
                LIMIT 20
                """,
                (self._tenant_id,),
            )
            correction_hotspots = [
                {"keyword": r["kw"], "correction_count": r["cnt"]} for r in cur.fetchall()
            ]

            # 4. knowledge_gaps: 有 finding 但没有 pattern 的关键词
            cur.execute(
                """
                WITH finding_kws AS (
                    SELECT kw, COUNT(*) as findings
                    FROM memories, unnest(keywords) AS kw
                    WHERE tenant_id = %s
                      AND category = 'finding'
                      AND (superseded_by = '' OR superseded_by IS NULL)
                    GROUP BY kw
                ),
                pattern_kws AS (
                    SELECT kw, COUNT(*) as patterns
                    FROM memories, unnest(keywords) AS kw
                    WHERE tenant_id = %s
                      AND category = 'pattern'
                      AND (superseded_by = '' OR superseded_by IS NULL)
                    GROUP BY kw
                )
                SELECT f.kw AS keyword, f.findings, COALESCE(p.patterns, 0) AS patterns
                FROM finding_kws f
                LEFT JOIN pattern_kws p ON f.kw = p.kw
                WHERE COALESCE(p.patterns, 0) = 0
                ORDER BY f.findings DESC
                LIMIT 20
                """,
                (self._tenant_id, self._tenant_id),
            )
            knowledge_gaps = [
                {"keyword": r["keyword"], "findings": r["findings"], "patterns": r["patterns"]}
                for r in cur.fetchall()
            ]

            # 5. weekly_trend: 最近 12 周每周新增记忆数
            cur.execute(
                """
                SELECT to_char(created_at, 'IYYY-"W"IW') AS week, COUNT(*) as cnt
                FROM memories
                WHERE tenant_id = %s
                  AND created_at >= NOW() - INTERVAL '12 weeks'
                GROUP BY week
                ORDER BY week
                """,
                (self._tenant_id,),
            )
            weekly_trend = [{"week": r["week"], "count": r["cnt"]} for r in cur.fetchall()]

            # 6. importance_distribution: 按 importance 分数统计
            cur.execute(
                """
                SELECT importance, COUNT(*) as cnt
                FROM memories
                WHERE tenant_id = %s
                  AND (superseded_by = '' OR superseded_by IS NULL)
                GROUP BY importance
                ORDER BY importance
                """,
                (self._tenant_id,),
            )
            importance_distribution = dict.fromkeys(range(1, 6), 0)
            importance_distribution.update({r["importance"]: r["cnt"] for r in cur.fetchall()})

            # 7. quality_coverage: 字段覆盖率
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE keywords IS NOT NULL AND array_length(keywords, 1) > 0) AS has_keywords,
                    COUNT(*) FILTER (WHERE trigger_condition IS NOT NULL AND trigger_condition != '') AS has_trigger,
                    COUNT(*) FILTER (WHERE applicability IS NOT NULL AND array_length(applicability, 1) > 0) AS has_applicability,
                    COUNT(*) FILTER (WHERE tags IS NOT NULL AND array_length(tags, 1) > 0) AS has_tags
                FROM memories
                WHERE tenant_id = %s
                  AND (superseded_by = '' OR superseded_by IS NULL)
                """,
                (self._tenant_id,),
            )
            row = cur.fetchone()
            total = row["total"]
            if total == 0:
                quality_coverage = {
                    "total": 0,
                    "keywords_pct": 0.0,
                    "trigger_condition_pct": 0.0,
                    "applicability_pct": 0.0,
                    "tags_pct": 0.0,
                }
            else:
                quality_coverage = {
                    "total": total,
                    "keywords_pct": round(row["has_keywords"] / total * 100, 1),
                    "trigger_condition_pct": round(row["has_trigger"] / total * 100, 1),
                    "applicability_pct": round(row["has_applicability"] / total * 100, 1),
                    "tags_pct": round(row["has_tags"] / total * 100, 1),
                }

        return {
            "employee_stats": employee_stats,
            "top_keywords": top_keywords,
            "correction_hotspots": correction_hotspots,
            "knowledge_gaps": knowledge_gaps,
            "weekly_trend": weekly_trend,
            "importance_distribution": importance_distribution,
            "quality_coverage": quality_coverage,
        }

    def record_recall(self, memory_ids: list[str]) -> int:
        """批量增加 recall_count.

        每次这些记忆被召回注入上下文时调用。

        Returns:
            更新的行数
        """
        if not memory_ids:
            return 0
        now = datetime.now(timezone.utc)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE memories
                SET recall_count = recall_count + 1,
                    last_accessed = %s
                WHERE id = ANY(%s) AND tenant_id = %s
                """,
                (now, memory_ids, self._tenant_id),
            )
            return cur.rowcount

    def record_useful(self, memory_ids: list[str], employee: str) -> int:
        """标记这些记忆在任务中被实际使用.

        Returns:
            更新的行数
        """
        if not memory_ids:
            return 0
        employee = self._resolve_to_character_name(employee)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE memories
                SET verified_count = verified_count + 1,
                    q_value = LEAST(1.0, q_value + 0.1 * (1.0 - q_value))
                WHERE id = ANY(%s) AND tenant_id = %s AND employee = %s
                """,
                (memory_ids, self._tenant_id, employee),
            )
            return cur.rowcount

    def decay_unverified_recalls(
        self, recalled_ids: list[str], useful_ids: list[str], employee: str
    ) -> int:
        """衰减被召回但未标记有用的记忆 q_value."""
        unverified = [mid for mid in recalled_ids if mid not in set(useful_ids)]
        if not unverified:
            return 0
        employee = self._resolve_to_character_name(employee)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE memories
                SET q_value = GREATEST(0.0, q_value * 0.95)
                WHERE id = ANY(%s) AND tenant_id = %s AND employee = %s
                """,
                (unverified, self._tenant_id, employee),
            )
            return cur.rowcount

    def decay_stale_memories(
        self,
        mild_days: int = 30,
        strong_days: int = 90,
        mild_factor: float = 0.98,
        strong_factor: float = 0.95,
        floor: float = 0.1,
        dry_run: bool = False,
    ) -> dict:
        """SAGE 遗忘曲线：对长期未召回的记忆做 q_value 时间衰减.

        Args:
            mild_days: 轻度衰减阈值（天数）
            strong_days: 重度衰减阈值（天数）
            mild_factor: 轻度衰减因子
            strong_factor: 重度衰减因子
            floor: q_value 下限（不低于此值）
            dry_run: 只统计不执行

        Returns:
            {"mild_decayed": N, "strong_decayed": N, "already_at_floor": N}
        """
        with get_connection() as conn:
            cur = conn.cursor()

            # 统计已经在 floor 的记忆数
            cur.execute(
                """
                SELECT COUNT(*) FROM memories
                WHERE tenant_id = %s
                  AND q_value <= %s
                  AND (superseded_by = '' OR superseded_by IS NULL)
                  AND (last_accessed < (CURRENT_TIMESTAMP - make_interval(days => %s))
                       OR (last_accessed IS NULL AND created_at < (CURRENT_TIMESTAMP - make_interval(days => %s))))
                """,
                (self._tenant_id, floor, mild_days, mild_days),
            )
            already_at_floor = cur.fetchone()[0]

            if dry_run:
                # Strong: >= strong_days, q_value > floor
                cur.execute(
                    """
                    SELECT COUNT(*) FROM memories
                    WHERE tenant_id = %s
                      AND q_value > %s
                      AND (superseded_by = '' OR superseded_by IS NULL)
                      AND (last_accessed < (CURRENT_TIMESTAMP - make_interval(days => %s))
                           OR (last_accessed IS NULL AND created_at < (CURRENT_TIMESTAMP - make_interval(days => %s))))
                    """,
                    (self._tenant_id, floor, strong_days, strong_days),
                )
                strong_decayed = cur.fetchone()[0]

                # Mild: >= mild_days but < strong_days, q_value > floor
                cur.execute(
                    """
                    SELECT COUNT(*) FROM memories
                    WHERE tenant_id = %s
                      AND q_value > %s
                      AND (superseded_by = '' OR superseded_by IS NULL)
                      AND (last_accessed < (CURRENT_TIMESTAMP - make_interval(days => %s))
                           OR (last_accessed IS NULL AND created_at < (CURRENT_TIMESTAMP - make_interval(days => %s))))
                      AND (last_accessed >= (CURRENT_TIMESTAMP - make_interval(days => %s))
                           OR (last_accessed IS NULL AND created_at >= (CURRENT_TIMESTAMP - make_interval(days => %s))))
                    """,
                    (self._tenant_id, floor, mild_days, mild_days, strong_days, strong_days),
                )
                mild_decayed = cur.fetchone()[0]

                return {
                    "mild_decayed": mild_decayed,
                    "strong_decayed": strong_decayed,
                    "already_at_floor": already_at_floor,
                }

            # Step 1: Strong decay (>= strong_days)
            cur.execute(
                """
                UPDATE memories
                SET q_value = GREATEST(%s, q_value * %s),
                    last_decayed_at = NOW()
                WHERE tenant_id = %s
                  AND q_value > %s
                  AND (superseded_by = '' OR superseded_by IS NULL)
                  AND (last_decayed_at IS NULL OR last_decayed_at < CURRENT_DATE)
                  AND (last_accessed < (CURRENT_TIMESTAMP - make_interval(days => %s))
                       OR (last_accessed IS NULL AND created_at < (CURRENT_TIMESTAMP - make_interval(days => %s))))
                """,
                (floor, strong_factor, self._tenant_id, floor, strong_days, strong_days),
            )
            strong_decayed = cur.rowcount

            # Step 2: Mild decay (>= mild_days but < strong_days)
            cur.execute(
                """
                UPDATE memories
                SET q_value = GREATEST(%s, q_value * %s),
                    last_decayed_at = NOW()
                WHERE tenant_id = %s
                  AND q_value > %s
                  AND (superseded_by = '' OR superseded_by IS NULL)
                  AND (last_decayed_at IS NULL OR last_decayed_at < CURRENT_DATE)
                  AND (last_accessed < (CURRENT_TIMESTAMP - make_interval(days => %s))
                       OR (last_accessed IS NULL AND created_at < (CURRENT_TIMESTAMP - make_interval(days => %s))))
                  AND (last_accessed >= (CURRENT_TIMESTAMP - make_interval(days => %s))
                       OR (last_accessed IS NULL AND created_at >= (CURRENT_TIMESTAMP - make_interval(days => %s))))
                """,
                (
                    floor,
                    mild_factor,
                    self._tenant_id,
                    floor,
                    mild_days,
                    mild_days,
                    strong_days,
                    strong_days,
                ),
            )
            mild_decayed = cur.rowcount

            return {
                "mild_decayed": mild_decayed,
                "strong_decayed": strong_decayed,
                "already_at_floor": already_at_floor,
            }

    def get_recall_stats(self) -> dict:
        """召回效果统计.

        Returns:
            包含 total_recalls, total_useful, hit_rate, top_useful,
            never_recalled 的字典
        """
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)

            # 汇总统计
            cur.execute(
                """
                SELECT COALESCE(SUM(recall_count), 0) AS total_recalls,
                       COALESCE(SUM(verified_count), 0) AS total_useful
                FROM memories
                WHERE tenant_id = %s
                """,
                (self._tenant_id,),
            )
            row = cur.fetchone()
            total_recalls = int(row["total_recalls"])
            total_useful = int(row["total_useful"])
            hit_rate = total_useful / total_recalls if total_recalls > 0 else 0.0

            # Top 10 最有用的记忆
            cur.execute(
                """
                SELECT id, employee, content, category, verified_count, recall_count
                FROM memories
                WHERE tenant_id = %s AND verified_count > 0
                ORDER BY verified_count DESC
                LIMIT 10
                """,
                (self._tenant_id,),
            )
            top_useful = [dict(r) for r in cur.fetchall()]

            # 从未被召回的记忆数
            cur.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM memories
                WHERE tenant_id = %s
                  AND recall_count = 0
                  AND (superseded_by = '' OR superseded_by IS NULL)
                """,
                (self._tenant_id,),
            )
            never_recalled = int(cur.fetchone()["cnt"])

        return {
            "total_recalls": total_recalls,
            "total_useful": total_useful,
            "hit_rate": round(hit_rate, 4),
            "top_useful": top_useful,
            "never_recalled": never_recalled,
        }

    def add_from_session(
        self,
        *,
        employee: str,
        session_id: str,
        summary: str,
        category: Literal["decision", "estimate", "finding", "correction", "pattern"] = "finding",
    ) -> MemoryEntry:
        """根据会话摘要写入记忆."""
        return self.add(
            employee=employee,
            category=category,
            content=summary,
            source_session=session_id,
        )

    # ── 使用记录（仅更新 last_accessed）──

    def record_usage(self, memory_id: str, employee: str) -> None:
        """记录记忆被使用（仅更新 last_accessed 时间戳，不写 feedback 表）.

        Args:
            memory_id: 记忆 ID
            employee: 员工名
        """
        now = datetime.now(timezone.utc)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE memories SET last_accessed = %s WHERE id = %s AND tenant_id = %s",
                (now, memory_id, self._tenant_id),
            )
        logger.debug(
            "记忆使用已记录: memory_id=%s employee=%s last_accessed=%s",
            memory_id,
            employee,
            now.isoformat(),
        )

    # ── 显式反馈（memory_feedback 表）──

    def submit_feedback(
        self,
        memory_id: str,
        employee: str,
        feedback_type: str,
        submitted_by: str,
        context: str = "",
        comment: str = "",
    ) -> dict:
        """提交记忆反馈.

        Args:
            memory_id: 记忆 ID
            employee: 员工名
            feedback_type: helpful / not_helpful / outdated / incorrect
            submitted_by: 提交人
            context: 使用场景
            comment: 反馈评论

        Returns:
            包含 feedback_id 等字段的字典

        Raises:
            ValueError: feedback_type 不在允许值范围内
        """
        VALID_FEEDBACK_TYPES = {"helpful", "not_helpful", "outdated", "incorrect"}
        if feedback_type not in VALID_FEEDBACK_TYPES:
            raise ValueError(
                f"Invalid feedback_type: {feedback_type!r}, "
                f"must be one of {sorted(VALID_FEEDBACK_TYPES)}"
            )

        feedback_id = f"fb-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO memory_feedback
                    (id, memory_id, employee, feedback_type, submitted_by,
                     context, comment, tenant_id, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    feedback_id,
                    memory_id,
                    employee,
                    feedback_type,
                    submitted_by,
                    context,
                    comment,
                    self._tenant_id,
                    now,
                ),
            )

        logger.info(
            "记忆反馈已提交: memory_id=%s type=%s by=%s",
            memory_id,
            feedback_type,
            submitted_by,
        )

        return {
            "feedback_id": feedback_id,
            "memory_id": memory_id,
            "employee": employee,
            "feedback_type": feedback_type,
            "submitted_by": submitted_by,
            "context": context,
            "comment": comment,
            "submitted_at": now.isoformat(),
        }

    def get_feedback(self, memory_id: str, limit: int = 50) -> list[dict]:
        """获取记忆的所有反馈.

        Args:
            memory_id: 记忆 ID
            limit: 最大返回条数

        Returns:
            反馈列表（按时间倒序）
        """
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
            cur.execute(
                """
                SELECT id AS feedback_id, memory_id, employee, feedback_type,
                       submitted_by, context, comment, created_at
                FROM memory_feedback
                WHERE memory_id = %s AND tenant_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (memory_id, self._tenant_id, limit),
            )
            rows = cur.fetchall()

        result = []
        for r in rows:
            d = dict(r)
            # 兼容文件版字段名
            if "created_at" in d and hasattr(d["created_at"], "isoformat"):
                d["submitted_at"] = d.pop("created_at").isoformat()
            elif "created_at" in d:
                d["submitted_at"] = str(d.pop("created_at"))
            result.append(d)
        return result

    def get_feedback_summary(self, employee: str | None = None) -> dict:
        """获取反馈汇总统计.

        Args:
            employee: 按员工过滤（可选）

        Returns:
            与文件版 get_feedback_summary 兼容的字典
        """
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)

            # 按 feedback_type 聚合
            if employee:
                cur.execute(
                    """
                    SELECT feedback_type, COUNT(*) AS cnt
                    FROM memory_feedback
                    WHERE tenant_id = %s AND employee = %s
                    GROUP BY feedback_type
                    """,
                    (self._tenant_id, employee),
                )
            else:
                cur.execute(
                    """
                    SELECT feedback_type, COUNT(*) AS cnt
                    FROM memory_feedback
                    WHERE tenant_id = %s
                    GROUP BY feedback_type
                    """,
                    (self._tenant_id,),
                )
            rows = cur.fetchall()

        feedback_by_type = {
            "helpful": 0,
            "not_helpful": 0,
            "outdated": 0,
            "incorrect": 0,
        }
        total_feedback = 0
        for r in rows:
            ft = r["feedback_type"]
            cnt = int(r["cnt"])
            if ft in feedback_by_type:
                feedback_by_type[ft] = cnt
            total_feedback += cnt

        # 获取涉及的记忆数（去重 memory_id）
        with get_connection() as conn:
            cur = conn.cursor()
            if employee:
                cur.execute(
                    """
                    SELECT COUNT(DISTINCT memory_id)
                    FROM memory_feedback
                    WHERE tenant_id = %s AND employee = %s
                    """,
                    (self._tenant_id, employee),
                )
            else:
                cur.execute(
                    """
                    SELECT COUNT(DISTINCT memory_id)
                    FROM memory_feedback
                    WHERE tenant_id = %s
                    """,
                    (self._tenant_id,),
                )
            total_memories = cur.fetchone()[0]

        return {
            "total_memories": total_memories,
            "total_feedback": total_feedback,
            "feedback_by_type": feedback_by_type,
            "avg_feedback_per_memory": (
                total_feedback / total_memories if total_memories > 0 else 0.0
            ),
        }

    def get_low_quality_memories(
        self,
        employee: str | None = None,
        min_recalls: int = 5,
        max_helpful_ratio: float = 0.3,
    ) -> list[dict]:
        """获取低质量记忆（反馈多但 helpful 比例低）.

        Args:
            employee: 按员工过滤
            min_recalls: 最少反馈次数
            max_helpful_ratio: 最大有帮助比例

        Returns:
            低质量记忆列表，与文件版 get_low_quality_memories 兼容
        """
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)

            sql = """
                SELECT mf.memory_id,
                       mf.employee,
                       COUNT(*) AS total_uses,
                       COUNT(*) FILTER (WHERE mf.feedback_type = 'helpful') AS helpful_count,
                       COUNT(*) FILTER (WHERE mf.feedback_type = 'not_helpful') AS not_helpful_count,
                       COUNT(*) FILTER (WHERE mf.feedback_type = 'outdated') AS outdated_count,
                       COUNT(*) FILTER (WHERE mf.feedback_type = 'incorrect') AS incorrect_count,
                       MAX(mf.created_at) AS last_used
                FROM memory_feedback mf
                WHERE mf.tenant_id = %s
            """
            params: list[Any] = [self._tenant_id]

            if employee:
                sql += " AND mf.employee = %s"
                params.append(employee)

            sql += """
                GROUP BY mf.memory_id, mf.employee
                HAVING COUNT(*) >= %s
            """
            params.append(min_recalls)

            cur.execute(sql, params)
            rows = cur.fetchall()

        result = []
        for r in rows:
            d = dict(r)
            total = int(d["total_uses"])
            helpful = int(d["helpful_count"])
            ratio = helpful / total if total > 0 else 0.0
            if ratio <= max_helpful_ratio:
                d["avg_relevance_score"] = 0.0
                if "last_used" in d and hasattr(d["last_used"], "isoformat"):
                    d["last_used"] = d["last_used"].isoformat()
                result.append(d)

        # 按 helpful 比例升序
        result.sort(
            key=lambda s: (
                int(s["helpful_count"]) / int(s["total_uses"]) if int(s["total_uses"]) > 0 else 0.0
            )
        )
        return result

    def get_popular_memories(
        self,
        employee: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """获取热门记忆（反馈多且 helpful 比例高）.

        Args:
            employee: 按员工过滤
            limit: 最大返回数量

        Returns:
            热门记忆列表，与文件版 get_popular_memories 兼容
        """
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)

            sql = """
                SELECT mf.memory_id,
                       mf.employee,
                       COUNT(*) AS total_uses,
                       COUNT(*) FILTER (WHERE mf.feedback_type = 'helpful') AS helpful_count,
                       COUNT(*) FILTER (WHERE mf.feedback_type = 'not_helpful') AS not_helpful_count,
                       COUNT(*) FILTER (WHERE mf.feedback_type = 'outdated') AS outdated_count,
                       COUNT(*) FILTER (WHERE mf.feedback_type = 'incorrect') AS incorrect_count,
                       MAX(mf.created_at) AS last_used
                FROM memory_feedback mf
                WHERE mf.tenant_id = %s
            """
            params: list[Any] = [self._tenant_id]

            if employee:
                sql += " AND mf.employee = %s"
                params.append(employee)

            sql += """
                GROUP BY mf.memory_id, mf.employee
                HAVING COUNT(*) > 0
                ORDER BY COUNT(*) DESC
            """

            cur.execute(sql, params)
            rows = cur.fetchall()

        result = []
        for r in rows:
            d = dict(r)
            total = int(d["total_uses"])
            helpful = int(d["helpful_count"])
            ratio = helpful / total if total > 0 else 0.0
            if ratio > 0.5:
                d["avg_relevance_score"] = 0.0
                if "last_used" in d and hasattr(d["last_used"], "isoformat"):
                    d["last_used"] = d["last_used"].isoformat()
                result.append(d)

        return result[:limit]
