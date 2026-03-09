<div align="right">

**English** | [中文](README.zh-CN.md)

</div>

<div align="center">

<h1>ensoul</h1>

<h3>Give your AI a soul.</h3>

<p><strong>Identity + Memory + Negotiation framework for digital employees.</strong><br/>
<em>Define who your AI is in Markdown. ensoul handles the rest.</em></p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
<br/>
[![Memory Modules](https://img.shields.io/badge/Memory_Modules-9-purple.svg)](#memory-pipeline)
[![Providers](https://img.shields.io/badge/LLM_Providers-7-orange.svg)](#multi-provider-execution)
[![Status](https://img.shields.io/badge/Status-Alpha-yellow.svg)](#status)

[Why ensoul?](#why-ensoul) · [Quick Start](#quick-start) · [Architecture](#core-architecture) · [Memory Pipeline](#memory-pipeline) · [Soul Evolution](#soul-evolution) · [vs Other Frameworks](#how-it-differs-from-other-frameworks) · [Crew Ecosystem](#crew-ecosystem)

</div>

---

## Why ensoul?

Most AI agent frameworks focus on **what** agents do (task orchestration, tool calling). ensoul focuses on **who** agents are.

| Concept | What it means |
|---------|--------------|
| **Soul** | A Markdown file that defines personality, values, communication style, and behavioral guidelines |
| **Memory** | Persistent, semantic memory with a Reflect → Connect → Store pipeline — not just chat history |
| **Negotiation** | Multi-agent discussions with configurable roles, stances, and structured disagreement |

---

## Quick Start

```bash
pip install ensoul
```

### Define an employee

Create `employees/code-reviewer/employee.md`:

```markdown
# Code Reviewer

## Identity
You are a senior code reviewer who values clarity, safety, and simplicity.
You push back on unnecessary complexity.

## Communication style
- Direct and specific — always reference line numbers
- Praise good patterns, don't just find problems
- "Ship it" or "Needs work" — never "LGTM" without reading

## Behavioral guidelines
- Security issues are always blocking
- Style preferences are never blocking
- When unsure, ask — don't assume intent
```

### Load via MCP

Add to your Claude Code MCP config:

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

Now Claude Code can load your code reviewer's soul, recall relevant memories from past reviews, and behave consistently across sessions.

---

## Core Architecture

```
┌─────────────────────────────────────────┐
│              ensoul                      │
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Identity │  │  Memory  │  │  Nego  │ │
│  │          │  │          │  │        │ │
│  │ Soul.md  │  │ Reflect  │  │ Roles  │ │
│  │ Parser   │  │ Connect  │  │ Rounds │ │
│  │ Engine   │  │ Store    │  │ Stance │ │
│  │ Evolve   │  │ Search   │  │ Focus  │ │
│  └──────────┘  └──────────┘  └────────┘ │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  Multi-provider LLM execution   │   │
│  │  Anthropic · OpenAI · DeepSeek  │   │
│  │  Moonshot · Gemini · Qwen       │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

---

## Memory Pipeline

ensoul's memory system is not a vector database bolted onto chat history. It's a three-stage pipeline that mirrors how humans consolidate experience:

| Stage | What happens |
|-------|-------------|
| **Reflect** | LLM extracts structured notes from raw interaction, deciding what's worth remembering |
| **Connect** | Keyword + semantic matching finds related memories, deciding whether to merge, link, or create new |
| **Store** | Writes to PostgreSQL with embeddings, tags, confidence scores, and temporal validity |

9 memory modules handle quality scoring, semantic search, deduplication, tag management, caching, and automatic consolidation of scattered findings into reusable patterns.

---

## Soul Evolution

Behavioral guidelines aren't static. ensoul watches for patterns in accumulated memories:

- High-frequency **patterns** → auto-promoted as soul guideline candidates
- Repeated **corrections** → flag outdated rules for archival

All candidates require human approval before updating the soul. The AI proposes; the human disposes.

---

## Multi-provider Execution

ensoul ships with a unified executor supporting 7 LLM providers:

| Provider | Models |
|----------|--------|
| Anthropic | Claude Opus, Sonnet, Haiku |
| OpenAI | GPT-4o, GPT-4o-mini |
| DeepSeek | DeepSeek Chat |
| Moonshot | Kimi K2.5 |
| Google | Gemini 2.0 Flash |
| Zhipu | GLM-4 |
| Alibaba | Qwen |

Automatic fallback, retry with exponential backoff, and proxy support included.

---

## How it differs from other frameworks

| | ensoul | crewAI / AutoGen |
|---|---|---|
| Focus | **Who** the agent is | **What** the agent does |
| Identity | Rich Markdown souls with personality, values, style | Role string or system prompt |
| Memory | Semantic pipeline with reflect/connect/store | Chat history or simple RAG |
| Evolution | Soul auto-evolves from experience | Static configuration |
| Negotiation | Structured multi-round discussions with stances | Sequential task handoff |

---

## Crew Ecosystem

ensoul is the core framework extracted from [Crew](https://github.com/liuxiaotong/knowlyr-crew) — a digital employee management system running 30+ AI employees in production since early 2025.

```
┌─────────────────────────────────┐
│  Crew (private)                 │
│  30+ employees · Feishu · WeCom │
│  ┌───────────────────────────┐  │
│  │  ensoul (open source)     │  │
│  │  Identity · Memory · Nego │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

Battle-tested modules are incrementally extracted from the production Crew codebase into ensoul as they mature.

---

## Status

🚧 **Alpha** — Core modules (identity, memory, execution) are available. Discussion engine and MCP server coming next. APIs may change.

## License

MIT
