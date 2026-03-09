"""外部讨论导入 — 将 Claude Code 等外部对话中的员工讨论写入记忆和会议记录."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ensoul.memory import get_memory_store
from ensoul.paths import resolve_project_dir

logger = logging.getLogger(__name__)


class ParticipantInput(BaseModel):
    """单个参与者的讨论输入."""

    name: str = Field(description="员工角色名")
    slug: str = Field(default="", description="员工 slug，为空时自动解析")
    contributions: list[str] = Field(default_factory=list, description="该员工的发言/观点")
    action_items: list[str] = Field(default_factory=list, description="该员工的待办事项")
    native_model: str = Field(default="", description="员工自身配置的模型，如 kimi-k2.5")


class DiscussionInput(BaseModel):
    """一次外部讨论的完整输入."""

    topic: str = Field(description="讨论主题")
    date: str = Field(default="", description="日期 (YYYY-MM-DD)，默认今天")
    source: str = Field(default="claude-code", description="讨论来源")
    context: str = Field(default="", description="讨论背景")
    runtime_model: str = Field(
        default="", description="实际推理模型（讨论环境的模型，如 claude-opus-4-6）"
    )
    participants: list[ParticipantInput] = Field(default_factory=list)
    shared_conclusions: list[str] = Field(default_factory=list, description="团队共识")


def sync_to_remote_server(data: dict) -> bool:
    """将讨论数据同步到远程服务端.

    通过 POST /api/memory/ingest 写入服务端记忆和会议记录。
    需要 ENSOUL_REMOTE_URL 和 ENSOUL_API_TOKEN 环境变量。

    Returns:
        True 成功, False 失败（不抛异常）
    """
    remote_url = os.environ.get("ENSOUL_REMOTE_URL", "")
    api_token = os.environ.get("ENSOUL_API_TOKEN", "")
    if not remote_url or not api_token:
        logger.info("ENSOUL_REMOTE_URL 或 ENSOUL_API_TOKEN 未设置，跳过线上同步")
        return False

    try:
        import httpx
    except ImportError:
        logger.warning("httpx 未安装，无法同步到远程服务端")
        return False

    url = f"{remote_url.rstrip('/')}/api/memory/ingest"
    try:
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_token}"},
            json=data,
            timeout=10.0,
        )
        resp.raise_for_status()
        result = resp.json()
        logger.info("线上同步成功: %d 条记忆", result.get("memories_written", 0))
        return True
    except Exception as e:
        logger.warning("同步到远程服务端失败: %s", e)
        return False


class DiscussionIngestor:
    """将外部讨论数据写入记忆系统."""

    def __init__(self, project_dir: Path | None = None, tenant_id: str | None = None):
        self.project_dir = resolve_project_dir(project_dir)
        self.store = get_memory_store(project_dir=project_dir, tenant_id=tenant_id)
        self._name_to_slug: dict[str, str] = {}
        self._slug_to_name: dict[str, str] = {}
        self._load_name_map()

    def _load_name_map(self) -> None:
        """从 global 员工目录构建 character_name <-> slug 双向映射."""
        global_dir = self.project_dir / ".crew" / "global"
        if not global_dir.is_dir():
            return
        for emp_dir in global_dir.iterdir():
            if not emp_dir.is_dir():
                continue
            yaml_path = emp_dir / "employee.yaml"
            if not yaml_path.exists():
                continue
            try:
                config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(config, dict):
                continue
            name = config.get("name", "")
            character_name = config.get("character_name", "")
            if character_name and name:
                self._name_to_slug[character_name] = name
                self._slug_to_name[name] = character_name

    def resolve_slug(self, name: str) -> str:
        """将角色名解析为 slug，找不到则原样返回.

        保留向后兼容，新代码请用 resolve_character_name()。
        """
        return self._name_to_slug.get(name, name)

    def resolve_character_name(self, name_or_slug: str) -> str:
        """将 slug 或角色名统一解析为花名（character_name）.

        如果输入已经是花名则直接返回；如果是 slug 则转为花名；找不到则原样返回。
        """
        # 已经是花名（在 _name_to_slug 的 key 中）
        if name_or_slug in self._name_to_slug:
            return name_or_slug
        # 是 slug，转为花名
        if name_or_slug in self._slug_to_name:
            return self._slug_to_name[name_or_slug]
        return name_or_slug

    def ingest(self, data: DiscussionInput) -> dict:
        """导入一次讨论，返回写入结果摘要."""
        date = data.date or datetime.now().strftime("%Y-%m-%d")
        session_id = f"external-{date}-{uuid.uuid4().hex[:8]}"

        results: dict = {
            "memories_written": 0,
            "synced_to_crew": False,
            "meeting_saved": False,
            "participants": [],
        }

        # 为每位参与者写入本地记忆
        for p in data.participants:
            char_name = self.resolve_character_name(p.slug or p.name)
            if not char_name:
                logger.warning("无法解析员工: %s，跳过", p.name)
                continue

            parts = [f"参与讨论：{data.topic}"]
            if data.context:
                parts.append(f"背景：{data.context}")
            if p.contributions:
                parts.append("我的观点：")
                for c in p.contributions:
                    parts.append(f"- {c}")
            if data.shared_conclusions:
                parts.append("团队共识：")
                for c in data.shared_conclusions:
                    parts.append(f"- {c}")
            if p.action_items:
                parts.append("我的待办：")
                for a in p.action_items:
                    parts.append(f"- {a}")

            # 标注推理环境：当 runtime_model 与 native_model 不同时标记为代理推理
            is_proxied = bool(
                data.runtime_model and p.native_model and data.runtime_model != p.native_model
            )
            if is_proxied:
                parts.append(
                    f"[推理环境：由 {data.runtime_model} 代理推理，本人模型为 {p.native_model}]"
                )
            elif data.runtime_model:
                parts.append(f"[推理环境：{data.runtime_model}]")

            content = "\n".join(parts)

            tags = ["discussion", data.source]
            if data.runtime_model:
                tags.append(f"runtime:{data.runtime_model}")
            if is_proxied:
                tags.append("proxied")

            from ensoul.memory_pipeline import process_memory

            process_memory(
                raw_text=content,
                employee=char_name,
                store=self.store,
                skip_reflect=True,
                source_session=session_id,
                category="finding",
                tags=tags,
                shared=True,
            )
            results["memories_written"] += 1
            results["participants"].append({"name": p.name, "character_name": char_name})

        # 保存本地会议记录
        meetings_dir = self.project_dir / ".crew" / "meetings"
        meetings_dir.mkdir(parents=True, exist_ok=True)

        meeting_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

        index_entry = {
            "meeting_id": meeting_id,
            "name": f"external-{data.source}",
            "topic": data.topic,
            "participants": [
                self.resolve_character_name(p.slug or p.name) for p in data.participants
            ],
            "mode": "external-ingest",
            "rounds": 1,
            "output_format": "memory",
            "started_at": datetime.now().isoformat(),
            "args": {"source": data.source, "date": date, "runtime_model": data.runtime_model},
        }

        index_path = meetings_dir / "index.jsonl"
        with index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

        # 写入完整讨论内容
        content_parts = [
            f"# {data.topic}\n",
            f"日期: {date}",
            f"来源: {data.source}",
        ]
        if data.runtime_model:
            content_parts.append(f"推理环境: {data.runtime_model}")
        content_parts.append("")
        if data.context:
            content_parts.append(f"## 背景\n\n{data.context}\n")
        for p in data.participants:
            model_note = ""
            if p.native_model and data.runtime_model and data.runtime_model != p.native_model:
                model_note = f" _(代理推理，本人模型: {p.native_model})_"
            content_parts.append(f"## {p.name}{model_note}\n")
            for c in p.contributions:
                content_parts.append(f"- {c}")
            if p.action_items:
                content_parts.append("\n待办：")
                for a in p.action_items:
                    content_parts.append(f"- {a}")
            content_parts.append("")
        if data.shared_conclusions:
            content_parts.append("## 共识\n")
            for c in data.shared_conclusions:
                content_parts.append(f"- {c}")

        prompt_path = meetings_dir / f"{meeting_id}.md"
        prompt_path.write_text("\n".join(content_parts), encoding="utf-8")

        results["meeting_saved"] = True
        results["meeting_id"] = meeting_id

        # 同步到远程服务端
        results["synced_to_crew"] = sync_to_remote_server(data.model_dump())

        return results


__all__ = ["DiscussionInput", "DiscussionIngestor", "ParticipantInput", "sync_to_remote_server"]
