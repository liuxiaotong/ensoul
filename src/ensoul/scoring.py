"""评分引擎公共模块 -- 供 daily_eval.py 和 run_eval_testset.py 共用.

提供:
- score_trajectory(): 评分单条轨迹（先尝试 gym，失败则 fallback）
- check_behavior_match(): 行为匹配检查（加权 + 同义词 + 短语匹配）
- 以及各种辅助常量和函数
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# ── 员工-评估域映射 ────────────────────────────────────────────────────

EMPLOYEE_DOMAIN_MAP: dict[str, str] = {
    # slug-based mapping (extend with your own employees)
    "ceo-assistant": "conversation",
    "customer-success": "conversation",
    "community-operator": "conversation",
    "code-reviewer": "engineering",
    "backend-engineer": "engineering",
    "frontend-engineer": "engineering",
    "test-engineer": "engineering",
    "e2e-tester": "engineering",
    "devops-engineer": "engineering",
    "dba": "engineering",
    "security-auditor": "engineering",
    "debug-expert": "engineering",
    "performance-optimizer": "engineering",
    "refactor-guide": "engineering",
    "pr-creator": "engineering",
    "doc-writer": "engineering",
    "data-engineer": "engineering",
    "mlops-engineer": "engineering",
    "i18n-expert": "engineering",
    "product-manager": "advisory",
    "hr-manager": "advisory",
    "finance-expert": "advisory",
    "legal-counsel": "advisory",
    "bd-manager": "advisory",
    "ux-designer": "advisory",
    "api-designer": "advisory",
    "solutions-architect": "advisory",
    "algorithm-researcher": "advisory",
    "nlp-researcher": "advisory",
    "sociology-researcher": "advisory",
    "economics-researcher": "advisory",
    "data-quality-expert": "advisory",
    "benchmark-specialist": "advisory",
}

# 各域的参考步数默认值（用于 efficiency 计算）
DOMAIN_REFERENCE_STEPS: dict[str, int] = {
    "conversation": 3,  # 对话类：回复+偶尔查资料
    "engineering": 15,  # 工程类：读代码+改代码+跑测试
    "advisory": 5,  # 顾问类：分析+建议
    "discussion": 3,  # 讨论类：发言
}

EXEMPLAR_THRESHOLD = 0.7

# 匹配 soul prompt 开头模式（"你是XXX" 后面跟角色描述）
_SOUL_PROMPT_PREFIX = "你是"
_SOUL_PROMPT_MIN_LEN = 200


# ── 辅助函数 ─────────────────────────────────────────────────────────


def _extract_task_from_soul_prompt(text: str) -> str | None:
    """尝试从 soul prompt 中提取 ## 任务 之后的实际任务描述.

    委托给 ensoul.trajectory.extract_task_from_soul_prompt 公共实现。
    """
    from ensoul.trajectory import extract_task_from_soul_prompt

    return extract_task_from_soul_prompt(text)


def _sanitize_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    """评分前防御性清洗轨迹数据.

    处理三类脏数据:
    1. task 是 dict -> 提取 description
    2. task 以"你是"开头且超 200 字 -> 尝试提取 ## 任务 后的内容
    3. steps 中字段名不统一 -> 标准化 tool/params/output
    """
    # Intentional in-place mutation for performance — 避免大量深拷贝开销
    traj = trajectory

    # ── task 清洗 ──
    task = traj.get("task")
    if isinstance(task, dict):
        # 格式 B: agentrecorder Trajectory 对象的 task 字段
        traj["task"] = task.get("description") or task.get("task_id", "")

    task = traj.get("task", "")
    if (
        isinstance(task, str)
        and task.startswith(_SOUL_PROMPT_PREFIX)
        and len(task) > _SOUL_PROMPT_MIN_LEN
    ):
        extracted = _extract_task_from_soul_prompt(task)
        if extracted:
            traj["task"] = extracted
        # 提取不到的不改，让评分正常走（评分结果可能偏低但不会崩）

    # ── steps 字段标准化 ──
    for step in traj.get("steps", []):
        # tool_call.name -> tool
        if "tool" not in step:
            tc = step.get("tool_call", {})
            if isinstance(tc, dict) and tc.get("name"):
                step["tool"] = tc["name"]
            elif step.get("tool_name"):
                step["tool"] = step["tool_name"]
            else:
                step["tool"] = "unknown"

        # tool_call.parameters -> params
        if "params" not in step:
            tc = step.get("tool_call", {})
            if isinstance(tc, dict) and "parameters" in tc:
                step["params"] = tc["parameters"]
            elif "tool_params" in step:
                step["params"] = step["tool_params"]
            else:
                step["params"] = {}

        # tool_result.output -> output
        if "output" not in step:
            tr = step.get("tool_result", {})
            if isinstance(tr, dict) and "output" in tr:
                step["output"] = tr["output"]
            elif "tool_output" in step:
                step["output"] = step["tool_output"]
            else:
                step["output"] = ""

    # ── reference_steps 默认值（解决 efficiency 恒=1.0） ──
    if traj.get("reference_steps") is None:
        employee = traj.get("metadata", {}).get("employee", "")
        domain = _get_domain(employee)
        traj["reference_steps"] = DOMAIN_REFERENCE_STEPS.get(domain, 10)

    return traj


def _get_domain(employee: str, *, project_dir: Path | None = None) -> str:
    """获取员工所属评估域.

    Args:
        employee: 员工名（中文名或 slug）
        project_dir: 项目根目录，用于 discover_employees fallback。
                     None 时自动检测。
    """
    domain = EMPLOYEE_DOMAIN_MAP.get(employee)
    if domain:
        return domain
    # fallback: 尝试从员工角色推断（新增员工无需手动加 MAP）
    try:
        from ensoul.discovery import discover_employees

        disc = discover_employees(project_dir=project_dir)
        emp = disc.get(employee)
        if emp:
            role = (emp.description or "").lower()
            if any(
                k in role for k in ("工程", "engineer", "开发", "测试", "test", "运维", "devops")
            ):
                return "engineering"
            if any(k in role for k in ("研究", "设计", "经理", "顾问", "法务", "财务", "HR", "hr")):
                return "advisory"
    except (ImportError, Exception):
        pass
    return "conversation"  # 默认


def traj_employee(traj: dict[str, Any]) -> str:
    """从标准化轨迹中提取员工名."""
    return traj.get("metadata", {}).get("employee", "unknown")


# ── 评分 ─────────────────────────────────────────────────────────────


def _try_gym_score(
    trajectory: dict[str, Any],
    domain: str,
    use_judge: bool = False,
    model_name: str = "",
    provider: str = "openai",
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """尝试用 gym RewardEngine 评分. 返回 None 表示 import 失败."""
    try:
        from agentreward.config import RewardConfig
        from agentreward.reward import RewardEngine
    except ImportError:
        return None

    # ensoul 轨迹的 outcome 全是 success=true，无实际意义
    # 因此 outcome_weight=0，全靠 process(LLM judge) + efficiency 打分
    if use_judge and model_name:
        config = RewardConfig(
            rule_weight=0.3,
            model_weight=0.7,
            domain=domain,
            model_name=model_name,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            outcome_weight=0.0,
            process_weight=0.9,
            efficiency_weight=0.1,
        )
    else:
        config = RewardConfig(
            rule_weight=1.0,
            model_weight=0.0,
            domain=domain,
        )

    engine = RewardEngine(config=config)
    reward = engine.score(trajectory)

    rubric_scores = {}
    if reward.step_rewards:
        rubric_scores = reward.step_rewards[0].rubric_scores

    return {
        "total_score": round(reward.total_score, 4),
        "outcome_score": round(reward.outcome_score, 4),
        "process_score": round(reward.process_score, 4),
        "efficiency_score": round(reward.efficiency_score, 4),
        "rubric_scores": {k: round(v, 4) for k, v in rubric_scores.items()},
        "engine": "gym",
    }


def _fallback_score(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Fallback 评分：纯规则，不依赖 gym 包."""
    steps = trajectory.get("steps", [])
    outcome = trajectory.get("outcome", {})

    score = 0.0

    # 基础分：成功 = 0.6
    if outcome.get("success", False):
        score += 0.6

    # 步数效率加分
    if len(steps) < 10:
        score += 0.2

    # thought 字段加分（检查原始轨迹数据中是否有 thought）
    has_thought = any(s.get("thought") or s.get("output", "") for s in steps)
    if has_thought:
        score += 0.1

    # 无错误加分
    has_error = any(
        s.get("error") or (s.get("output", "") and "error" in s.get("output", "").lower()[:100])
        for s in steps
    )
    if not has_error:
        score += 0.1

    efficiency = 1.0 if len(steps) < 10 else max(0.0, 10.0 / len(steps))
    return {
        "total_score": round(min(score, 1.0), 4),
        "outcome_score": 0.6 if outcome.get("success") else 0.0,
        "process_score": round(score - (0.6 if outcome.get("success") else 0.0), 4),
        "efficiency_score": round(efficiency, 4),
        "rubric_scores": {},
        "engine": "fallback",
    }


def score_trajectory(
    trajectory: dict[str, Any],
    employee: str,
    *,
    use_judge: bool = False,
    model_name: str = "",
    provider: str = "openai",
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """评分单条轨迹. 先尝试 gym，失败则 fallback.

    hr-manager 的轨迹强制走规则层（不用 LLM Judge）。
    """
    # 防御性清洗：处理 task dict/soul prompt/字段名不统一
    trajectory = _sanitize_trajectory(trajectory)

    domain = _get_domain(employee)

    # hr-manager 自己的轨迹只走规则层
    employee_use_judge = use_judge and employee not in ("hr-manager",)

    result = _try_gym_score(
        trajectory,
        domain,
        use_judge=employee_use_judge,
        model_name=model_name,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
    )

    if result is None:
        result = _fallback_score(trajectory)

    return result


def _build_exemplar_content(result: dict[str, Any]) -> str:
    """从评分结果构建范例记忆内容."""
    score = result["total_score"]
    task = (result.get("task") or "")[:100]
    # 从 rubric_scores 提取亮点（取分数最高的维度）
    rubric = result.get("rubric_scores", {})
    highlights = ""
    if rubric:
        top_dims = sorted(rubric.items(), key=lambda x: x[1], reverse=True)[:3]
        highlights = "；".join(f"{k}={v:.2f}" for k, v in top_dims)
    if highlights:
        return f"[高分范例 {score:.2f}分] 任务：{task}。表现亮点：{highlights}"
    return f"[高分范例 {score:.2f}分] 任务：{task}"


# ── 行为匹配（S7 改进版） ───────────────────────────────────────────────

# 中英文常见同义词/近义词映射表（双向）
_SYNONYM_MAP: dict[str, list[str]] = {
    "错误": ["error", "异常", "exception", "bug", "故障", "fault"],
    "error": ["错误", "异常", "exception", "bug", "故障", "fault"],
    "处理": ["handle", "handling", "处置", "应对", "解决"],
    "handle": ["处理", "handling", "处置", "应对", "解决"],
    "handling": ["处理", "handle", "处置", "应对", "解决"],
    "测试": ["test", "testing", "验证", "检验", "校验"],
    "test": ["测试", "testing", "验证", "检验", "校验"],
    "testing": ["测试", "test", "验证", "检验", "校验"],
    "代码": ["code", "源码", "源代码", "程序"],
    "code": ["代码", "源码", "源代码", "程序"],
    "优化": ["optimize", "optimization", "改善", "提升", "改进"],
    "optimize": ["优化", "optimization", "改善", "提升", "改进"],
    "安全": ["security", "safe", "safety", "防护"],
    "security": ["安全", "safe", "safety", "防护"],
    "性能": ["performance", "效率", "速度"],
    "performance": ["性能", "效率", "速度"],
    "部署": ["deploy", "deployment", "发布", "上线"],
    "deploy": ["部署", "deployment", "发布", "上线"],
    "deployment": ["部署", "deploy", "发布", "上线"],
    "数据": ["data", "数据集", "dataset"],
    "data": ["数据", "数据集", "dataset"],
    "配置": ["config", "configuration", "设置", "setting"],
    "config": ["配置", "configuration", "设置", "setting"],
    "分析": ["analyze", "analysis", "解析"],
    "analyze": ["分析", "analysis", "解析"],
    "analysis": ["分析", "analyze", "解析"],
    "建议": ["suggest", "suggestion", "recommend", "recommendation", "推荐"],
    "suggest": ["建议", "suggestion", "recommend", "recommendation", "推荐"],
    "文档": ["document", "documentation", "doc", "文件"],
    "document": ["文档", "documentation", "doc", "文件"],
    "检查": ["check", "inspect", "review", "审查"],
    "check": ["检查", "inspect", "review", "审查"],
    "review": ["检查", "inspect", "审查", "评审", "复查"],
    "修复": ["fix", "repair", "修正", "修改"],
    "fix": ["修复", "repair", "修正", "修改"],
    "创建": ["create", "新建", "生成", "build"],
    "create": ["创建", "新建", "生成", "build"],
    "删除": ["delete", "remove", "移除", "清除"],
    "delete": ["删除", "remove", "移除", "清除"],
    "日志": ["log", "logging", "记录"],
    "log": ["日志", "logging", "记录"],
    "接口": ["api", "interface", "端口"],
    "api": ["接口", "interface"],
    "重构": ["refactor", "refactoring", "重写"],
    "refactor": ["重构", "refactoring", "重写"],
    "异常": ["exception", "error", "错误"],
    "exception": ["异常", "error", "错误"],
    "验证": ["validate", "validation", "校验", "检验"],
    "validate": ["验证", "validation", "校验", "检验"],
    "回复": ["reply", "respond", "response", "回答", "答复"],
    "respond": ["回复", "reply", "response", "回答", "答复"],
    "response": ["回复", "reply", "respond", "回答", "答复"],
}

# 中英文短语对照（常见的跨语言短语）
_PHRASE_MAP: dict[str, list[str]] = {
    "错误处理": ["error handling", "error handle", "异常处理", "exception handling"],
    "error handling": ["错误处理", "异常处理", "exception handling"],
    "代码审查": ["code review", "代码检查", "代码评审"],
    "code review": ["代码审查", "代码检查", "代码评审"],
    "单元测试": ["unit test", "unit testing"],
    "unit test": ["单元测试", "unit testing"],
    "性能优化": ["performance optimization", "性能改善", "性能提升"],
    "performance optimization": ["性能优化", "性能改善", "性能提升"],
    "数据库": ["database", "db"],
    "database": ["数据库", "db"],
    "版本控制": ["version control", "git"],
    "version control": ["版本控制", "git"],
}

# 核心词性标记：动词和核心名词权重更高
# 中文核心动词（在行为描述中出现时权重加倍）
_CORE_VERBS: set[str] = {
    "检查",
    "审查",
    "分析",
    "测试",
    "验证",
    "修复",
    "优化",
    "部署",
    "创建",
    "删除",
    "配置",
    "处理",
    "建议",
    "评估",
    "实现",
    "设计",
    "重构",
    "编写",
    "生成",
    "监控",
    "诊断",
    "排查",
    "解决",
    "check",
    "review",
    "analyze",
    "test",
    "validate",
    "fix",
    "optimize",
    "deploy",
    "create",
    "delete",
    "configure",
    "handle",
    "suggest",
    "evaluate",
    "implement",
    "design",
    "refactor",
    "write",
    "generate",
    "monitor",
    "diagnose",
    "debug",
    "resolve",
}


def _tokenize_behavior(text: str) -> list[str]:
    """将行为描述拆分为关键词列表（中文词 + 英文词）."""
    return re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z_]+", text.lower())


def _match_token_in_output(token: str, output_lower: str) -> bool:
    """检查单个 token 是否在输出中匹配（含同义词扩展）."""
    # 直接匹配
    if token in output_lower:
        return True
    # 同义词匹配
    synonyms = _SYNONYM_MAP.get(token, [])
    return any(syn.lower() in output_lower for syn in synonyms)


def _match_phrase_in_output(phrase: str, output_lower: str) -> bool:
    """检查短语是否在输出中匹配（含跨语言短语映射）."""
    phrase_lower = phrase.lower()
    if phrase_lower in output_lower:
        return True
    equivalents = _PHRASE_MAP.get(phrase_lower, [])
    return any(eq.lower() in output_lower for eq in equivalents)


def check_behavior_match(
    output: str,
    expected_behaviors: list[str],
    *,
    threshold: float = 0.5,
) -> dict[str, bool]:
    """检查输出是否包含期望行为（加权匹配 + 同义词 + 短语匹配）.

    改进点（相对旧版 _check_behaviors）：
    1. 支持同义词/近义词匹配（中英文常见同义词表）
    2. 支持短语匹配（如 "错误处理" 匹配 "error handling"）
    3. 匹配阈值可配（默认 50%）
    4. 加权匹配：核心动词/名词权重更高

    Args:
        output: 员工输出的文本
        expected_behaviors: 期望行为描述列表
        threshold: 匹配阈值（0.0-1.0），加权得分达到此比例算通过

    Returns:
        {behavior_description: matched_bool} 的字典
    """
    output_lower = output.lower()
    results: dict[str, bool] = {}

    for behavior in expected_behaviors:
        # Step 1: 尝试整个行为描述的短语匹配
        if _match_phrase_in_output(behavior, output_lower):
            results[behavior] = True
            continue

        # Step 2: 检查行为描述中的子短语（连续的中文词组或英文词组）
        # 提取可能的短语（2-4个字的中文组合，或英文多词组合）
        phrase_matched = False
        for phrase in _PHRASE_MAP:
            if phrase.lower() in behavior.lower():
                if _match_phrase_in_output(phrase, output_lower):
                    phrase_matched = True
                    break
        if phrase_matched:
            results[behavior] = True
            continue

        # Step 3: 逐 token 加权匹配
        tokens = _tokenize_behavior(behavior)
        if not tokens:
            results[behavior] = False
            continue

        total_weight = 0.0
        matched_weight = 0.0
        for token in tokens:
            # 核心动词/名词权重 = 2.0，普通词 = 1.0
            weight = 2.0 if token in _CORE_VERBS else 1.0
            total_weight += weight
            if _match_token_in_output(token, output_lower):
                matched_weight += weight

        results[behavior] = matched_weight >= max(total_weight * threshold, 1.0)

    return results
