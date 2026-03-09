<div align="right">

[English](README.md) | **中文**

</div>

<div align="center">

<h1>ensoul</h1>

<h3>给你的 AI 赋予灵魂。</h3>

<p><strong>数字员工的 身份 + 记忆 + 协商 框架。</strong><br/>
<em>用 Markdown 定义你的 AI 是谁，ensoul 搞定剩下的。</em></p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
<br/>
[![Memory Modules](https://img.shields.io/badge/记忆模块-9-purple.svg)](#记忆管线)
[![Providers](https://img.shields.io/badge/LLM_供应商-7-orange.svg)](#多供应商执行)
[![Status](https://img.shields.io/badge/状态-Alpha-yellow.svg)](#状态)

[为什么选 ensoul？](#为什么选-ensoul) · [快速开始](#快速开始) · [架构](#核心架构) · [记忆管线](#记忆管线) · [灵魂进化](#灵魂进化) · [与其他框架对比](#与其他框架对比) · [Crew 生态](#crew-生态)

</div>

---

## 为什么选 ensoul？

大多数 AI Agent 框架关注的是 Agent **做什么**（任务编排、工具调用）。ensoul 关注的是 Agent **是谁**。

| 概念 | 含义 |
|------|------|
| **灵魂 (Soul)** | 一个 Markdown 文件，定义人格、价值观、沟通风格和行为准则 |
| **记忆 (Memory)** | 持久化语义记忆，Reflect → Connect → Store 三阶段管线——不是聊天记录 |
| **协商 (Negotiation)** | 多 Agent 结构化讨论，可配置角色、立场和强制分歧 |

---

## 快速开始

```bash
pip install ensoul
```

### 定义一个员工

创建 `employees/code-reviewer/employee.md`：

```markdown
# 代码审查员

## 身份
你是一位资深代码审查员，重视清晰、安全和简洁。
你会抵制不必要的复杂性。

## 沟通风格
- 直接且具体——总是引用行号
- 表扬好的模式，不只是找问题
- "可以合并"或"需要修改"——不会不看就说"没问题"

## 行为准则
- 安全问题永远是阻塞项
- 风格偏好永远不是阻塞项
- 不确定时问清楚，不要假设意图
```

### 通过 MCP 加载

添加到 Claude Code 的 MCP 配置：

```json
{
  "mcpServers": {
    "ensoul": {
      "command": "uvx",
      "args": ["ensoul"],
      "env": {
        "ENSOUL_EMPLOYEES_DIR": "./employees"
      }
    }
  }
}
```

Claude Code 就可以加载你的代码审查员的灵魂，调取过往审查的相关记忆，在不同会话间保持一致的行为。

---

## 核心架构

```
┌─────────────────────────────────────────┐
│              ensoul                      │
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │   身份   │  │   记忆   │  │  协商  │ │
│  │          │  │          │  │        │ │
│  │ Soul.md  │  │ Reflect  │  │  角色  │ │
│  │  解析器  │  │ Connect  │  │  轮次  │ │
│  │  引擎    │  │ Store    │  │  立场  │ │
│  │  进化    │  │  搜索    │  │  焦点  │ │
│  └──────────┘  └──────────┘  └────────┘ │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │      多供应商 LLM 执行引擎       │   │
│  │  Anthropic · OpenAI · DeepSeek  │   │
│  │  Moonshot · Gemini · Qwen       │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

---

## 记忆管线

ensoul 的记忆系统不是在聊天记录上拼一个向量数据库。它是一个三阶段管线，模拟人类整合经验的过程：

| 阶段 | 做什么 |
|------|--------|
| **Reflect（反思）** | LLM 从原始交互中提取结构化笔记，决定哪些值得记住 |
| **Connect（关联）** | 关键词 + 语义匹配找到相关记忆，决定合并、链接还是新建 |
| **Store（存储）** | 写入 PostgreSQL，带向量嵌入、标签、置信度和时间有效性 |

9 个记忆模块负责质量评分、语义搜索、去重、标签管理、缓存，以及将零散发现自动聚合为可复用的模式。

---

## 灵魂进化

行为准则不是一成不变的。ensoul 监控积累的记忆，寻找模式：

- 高频 **pattern** → 自动提名为灵魂准则候选
- 反复出现的 **correction** → 标记过时规则待归档

所有候选都需要人类审批后才真正更新灵魂。AI 提议，人类决定。

---

## 多供应商执行

ensoul 内置统一执行器，支持 7 家 LLM 供应商：

| 供应商 | 模型 |
|--------|------|
| Anthropic | Claude Opus, Sonnet, Haiku |
| OpenAI | GPT-4o, GPT-4o-mini |
| DeepSeek | DeepSeek Chat |
| Moonshot | Kimi K2.5 |
| Google | Gemini 2.0 Flash |
| 智谱 | GLM-4 |
| 阿里云 | Qwen |

自带自动降级、指数退避重试和代理支持。

---

## 与其他框架对比

| | ensoul | crewAI / AutoGen |
|---|---|---|
| 关注点 | Agent **是谁** | Agent **做什么** |
| 身份 | 丰富的 Markdown 灵魂：人格、价值观、风格 | 角色字符串或系统提示词 |
| 记忆 | 语义管线：反思/关联/存储 | 聊天记录或简单 RAG |
| 进化 | 灵魂从经验中自动进化 | 静态配置 |
| 协商 | 结构化多轮讨论，带立场和强制分歧 | 顺序任务传递 |

---

## Crew 生态

ensoul 是从 [Crew](https://github.com/liuxiaotong/knowlyr-crew) 中抽取的核心框架。Crew 是一个数字员工管理系统，自 2025 年初起在生产环境运行 30+ AI 员工。

```
┌─────────────────────────────────┐
│  Crew（私有）                    │
│  30+ 员工 · 飞书 · 企业微信      │
│  ┌───────────────────────────┐  │
│  │  ensoul（开源）            │  │
│  │  身份 · 记忆 · 协商        │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

经过生产验证的模块从 Crew 代码库中增量抽取到 ensoul。

---

## 状态

🚧 **Alpha** — 核心模块（身份、记忆、执行）已可用。讨论引擎和 MCP Server 即将到来。API 可能变化。

## 许可证

MIT
