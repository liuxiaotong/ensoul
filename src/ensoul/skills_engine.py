"""Skills 触发引擎 — 判断和执行 Skills."""

import logging
import time
from typing import Any

from ensoul.skills import Skill, SkillAction, SkillStore, SkillTriggerRecord

logger = logging.getLogger(__name__)


class SkillsEngine:
    """Skills 触发和执行引擎."""

    def __init__(self, skill_store: SkillStore, memory_store: Any):
        self.skill_store = skill_store
        self.memory_store = memory_store

    def check_triggers(
        self, employee: str, task: str, context: dict[str, Any] | None = None
    ) -> list[tuple[Skill, float]]:
        """检查应该触发的 Skills.

        返回: [(skill, match_score), ...]
        """
        context = context or {}
        skills = self.skill_store.list_skills(employee)
        triggered = []

        for skill in skills:
            if not skill.enabled:
                continue

            match_score = self._should_trigger(skill, task, context)
            if match_score > 0:
                triggered.append((skill, match_score))

        # 按优先级和匹配分数排序
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        triggered.sort(
            key=lambda x: (priority_order.get(x[0].metadata.priority, 0), x[1]),
            reverse=True,
        )

        return triggered

    def _should_trigger(self, skill: Skill, task: str, context: dict[str, Any]) -> float:
        """判断是否应该触发 Skill.

        返回: 匹配分数 (0-1)，0 表示不触发
        """
        trigger = skill.trigger

        if trigger.type == "always":
            return 1.0

        elif trigger.type == "keyword":
            return self._keyword_match(trigger.keywords, task)

        elif trigger.type == "semantic":
            # TODO: 实现语义匹配（需要 embedding）
            # 降级：从 trigger.keywords + description 合并关键词
            keywords = list(trigger.keywords or [])  # 保留已有关键词
            if not keywords:
                # 只在没有预设关键词时从 description 提取
                keywords = self._extract_keywords_from_description(skill.description)
            if not keywords:
                logger.warning(f"Semantic trigger has no keywords for {skill.name}")
                return 0.0
            return self._keyword_match(keywords, task)

        return 0.0

    def _keyword_match(self, keywords: list[str], task: str) -> float:
        """关键词匹配."""
        if not keywords:
            return 0.0

        task_lower = task.lower()
        matched = sum(1 for kw in keywords if kw.lower() in task_lower)

        if matched == 0:
            return 0.0

        # 匹配分数 = 匹配的关键词数 / 总关键词数
        return min(matched / len(keywords), 1.0)

    def _extract_keywords_from_description(self, description: str) -> list[str]:
        """从 description 中提取关键词（改进版）."""
        import re

        keywords = []

        # 1. 提取引号中的内容
        keywords.extend(re.findall(r"[\"\'](.*?)[\"\']", description))

        # 2. 提取中文关键词（2-4字的中文词组）
        keywords.extend(re.findall(r"[\u4e00-\u9fff]{2,4}", description))

        # 3. 提取英文关键词（3字母以上，去停用词）
        keywords.extend(
            w
            for w in re.findall(r"[a-zA-Z]{3,}", description)
            if w.lower()
            not in {
                "the",
                "and",
                "for",
                "that",
                "this",
                "with",
                "from",
                "have",
                "not",
            }
        )

        # 去重（保留顺序）
        seen: set[str] = set()
        result: list[str] = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                result.append(kw)

        return result

    def execute_skill(
        self, skill: Skill, employee: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """执行 Skill.

        返回: {
            "executed_actions": [...],
            "enhanced_context": {...}
        }
        """
        context = context or {}
        start_time = time.time()

        executed_actions = []
        enhanced_context = {}

        for action in skill.actions:
            try:
                result = self._execute_action(action, employee, context)
                executed_actions.append(
                    {"type": action.type, "status": "success", "result": result}
                )

                # 合并到 enhanced_context
                if action.type == "query_memory":
                    enhanced_context.setdefault("memories", []).extend(result.get("memories", []))
                elif action.type == "load_checklist":
                    enhanced_context.setdefault("checklist_items", []).extend(
                        result.get("items", [])
                    )
                elif action.type == "read_wiki":
                    wiki_content = result.get("content", "")
                    if wiki_content:
                        enhanced_context.setdefault("wiki_docs", []).append(
                            {
                                "title": result.get("title", ""),
                                "content": wiki_content,
                                "doc_id": result.get("doc_id"),
                            }
                        )

            except Exception as e:
                logger.error(f"Failed to execute action {action.type}: {e}")
                executed_actions.append({"type": action.type, "status": "failure", "error": str(e)})

        execution_time_ms = int((time.time() - start_time) * 1000)

        return {
            "executed_actions": executed_actions,
            "enhanced_context": enhanced_context,
            "execution_time_ms": execution_time_ms,
        }

    def _execute_action(
        self, action: SkillAction, employee: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """执行单个 Action."""
        if action.type == "query_memory":
            return self._execute_query_memory(action, employee, context)
        elif action.type == "load_checklist":
            return self._execute_load_checklist(action, employee, context)
        elif action.type == "read_wiki":
            return self._execute_read_wiki(action, employee, context)
        else:
            raise ValueError(f"Unknown action type: {action.type}")

    def _execute_query_memory(
        self, action: SkillAction, employee: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """执行 query_memory 动作."""
        params = action.params

        # 从 params 中提取参数
        # 注意：MemoryStore.query 不支持自由文本查询，只支持 category 过滤
        # 这里简化实现，使用 category 参数
        category = params.get("category")
        limit = params.get("limit", 10)

        # 信息分级过滤：从 context 中读取 channel + sender_type，计算有效许可
        clearance_kwargs: dict[str, Any] = {}
        channel = context.get("channel", "")
        if channel:
            from ensoul.classification import get_effective_clearance

            _sender_type = context.get("sender_type", "")
            clearance = get_effective_clearance(employee, channel, sender_type=_sender_type)
            clearance_kwargs["classification_max"] = clearance["classification_max"]
            clearance_kwargs["allowed_domains"] = clearance["allowed_domains"]
            clearance_kwargs["include_confidential"] = clearance["include_confidential"]

        # 查询记忆
        memories = self.memory_store.query(
            employee=employee,
            category=category,
            limit=limit,
            **clearance_kwargs,
        )

        # Phase 4：审计日志（skills_engine 层有 channel 信息）
        from ensoul.classification import CHANNEL_SOURCE_TYPE

        source_type = CHANNEL_SOURCE_TYPE.get(channel, "external") if channel else "internal"
        logger.info(
            "memory_audit: employee=%s channel=%s source_type=%s "
            "classification_max=%s allowed_domains=%s returned=%d",
            employee,
            channel,
            source_type,
            clearance_kwargs.get("classification_max", "none"),
            clearance_kwargs.get("allowed_domains", "none"),
            len(memories),
        )

        # 兼容 MemoryStore（文件版返回 MemoryEntry）和 MemoryStoreDB（返回 dict）
        result_list = []
        for m in memories:
            if isinstance(m, dict):
                result_list.append(m)
            else:
                result_list.append(m.model_dump())

        return {"memories": result_list, "count": len(result_list)}

    def _execute_load_checklist(
        self, action: SkillAction, employee: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """执行 load_checklist 动作 - 从员工 soul 中提取检查清单."""
        params = action.params
        section = params.get("section", "工作检查清单")

        try:
            # 使用 config_store 获取 soul
            from ensoul.config_store import get_soul

            soul_data = get_soul(employee)
            if not soul_data:
                logger.warning(f"Soul not found for {employee}")
                return {"items": [], "section": section}

            soul_content = soul_data.get("content", "")

            if not soul_content or len(soul_content) < 100:
                logger.warning(f"Empty or invalid soul content for {employee}")
                return {"items": [], "section": section}

            # 解析 Markdown 提取检查清单
            items = self._parse_checklist_from_markdown(soul_content, section)

            return {"items": items, "section": section, "count": len(items)}

        except Exception as e:
            logger.warning(f"load_checklist failed: {e}")
            return {"items": [], "section": section}

    def _parse_checklist_from_markdown(self, content: str, section: str) -> list[str]:
        """从 Markdown 内容中提取检查清单."""
        lines = content.split("\n")
        items = []
        in_section = False
        current_subsection = ""

        for line in lines:
            # 检测章节标题（## 开头，不是 ###）
            if line.startswith("## ") and section in line:
                in_section = True
                continue

            # 遇到下一个同级标题（## 开头，不是 ###），退出
            if in_section and line.startswith("## ") and section not in line:
                break

            # 在目标章节内
            if in_section:
                # 检测子章节（### 开头）
                if line.startswith("### "):
                    current_subsection = line.strip("# ").strip()
                    continue

                # 提取检查项（- [ ] 格式）
                stripped = line.strip()
                if stripped.startswith("- [ ]"):
                    item_text = stripped[5:].strip()
                    if current_subsection:
                        items.append(f"[{current_subsection}] {item_text}")
                    else:
                        items.append(item_text)

        return items

    def _execute_read_wiki(
        self, action: SkillAction, employee: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """执行 read_wiki 动作."""
        import os

        import httpx

        params = action.params
        doc_id = params.get("doc_id", "")
        space_slug = params.get("space_slug", "")
        doc_slug = params.get("doc_slug", "")

        wiki_url = os.environ.get("WIKI_API_URL", "").rstrip("/")
        wiki_token = os.environ.get("WIKI_ADMIN_TOKEN", "")
        if not wiki_url or not wiki_token:
            logger.warning("read_wiki: WIKI_API_URL or WIKI_ADMIN_TOKEN not configured")
            return {"content": "", "error": "wiki not configured"}

        headers = {"Authorization": f"Bearer {wiki_token}"}

        if not doc_id and space_slug and doc_slug:
            doc_id = self._resolve_wiki_doc_id(wiki_url, headers, space_slug, doc_slug)
            if not doc_id:
                return {"content": "", "error": f"doc not found: {space_slug}/{doc_slug}"}

        if not doc_id:
            logger.warning("read_wiki: no doc_id or space_slug/doc_slug provided")
            return {"content": "", "error": "no doc_id or slug provided"}

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(
                    f"{wiki_url}/api/docs/{doc_id}",
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                return {
                    "content": data.get("content", ""),
                    "title": data.get("title", ""),
                    "doc_id": doc_id,
                }
        except Exception as e:
            logger.error(f"read_wiki failed for doc_id={doc_id}: {e}")
            return {"content": "", "doc_id": doc_id, "error": str(e)}

    def _resolve_wiki_doc_id(
        self, wiki_url: str, headers: dict, space_slug: str, doc_slug: str
    ) -> str | None:
        """通过 space_slug + doc_slug 解析 doc_id."""
        import httpx

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{wiki_url}/api/spaces/{space_slug}/docs",
                    headers=headers,
                )
                resp.raise_for_status()
                docs = resp.json().get("docs", [])
                for doc in docs:
                    if doc.get("slug") == doc_slug:
                        return doc.get("id", "")
        except Exception as e:
            logger.error(f"resolve_wiki_doc_id failed: {e}")
        return None

    def _replace_template_vars(self, template: str, context: dict[str, Any]) -> str:
        """替换模板变量."""
        # 替换 {{task_keywords}}
        if "{{task_keywords}}" in template:
            keywords = context.get("task_keywords", [])
            template = template.replace("{{task_keywords}}", " ".join(keywords))

        return template

    def record_trigger(
        self,
        skill: Skill,
        employee: str,
        task: str,
        match_score: float,
        execution_result: dict[str, Any],
    ) -> None:
        """记录触发历史."""
        record = SkillTriggerRecord(
            skill_id=skill.skill_id,
            employee=employee,
            task=task,
            match_score=match_score,
            execution_status="success"
            if all(a["status"] == "success" for a in execution_result["executed_actions"])
            else "failure",
            execution_time_ms=execution_result.get("execution_time_ms", 0),
            actions_executed=len(execution_result["executed_actions"]),
            result=execution_result,
        )

        self.skill_store.record_trigger(record)
