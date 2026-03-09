# ensoul

**Give your AI a soul.**

ensoul is a framework for building **digital employees** — AI agents with persistent identity, long-term memory, and multi-agent negotiation. Define who your AI *is* in Markdown, and ensoul handles the rest.

> Part of the [Crew](https://github.com/liuxiaotong/knowlyr-crew) ecosystem — a digital employee management system used in production at [Knowlyr](https://knowlyr.com).

## Why ensoul?

Most AI agent frameworks focus on **what** agents do (task orchestration, tool calling). ensoul focuses on **who** agents are:

| Concept | What it means |
|---------|--------------|
| **Soul** | A Markdown file that defines personality, values, communication style, and behavioral guidelines |
| **Memory** | Persistent, semantic memory with a Reflect → Connect → Store pipeline — not just chat history |
| **Negotiation** | Multi-agent discussions with configurable roles, stances, and structured disagreement |

## Quick start

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

## Core architecture

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
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  MCP Server · CLI · Webhook     │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Key features

- **Markdown-native** — Define employees in plain Markdown. No YAML/JSON config hell.
- **Memory pipeline** — Automatic extraction, deduplication, and consolidation of knowledge across sessions.
- **Soul evolution** — Behavioral guidelines auto-evolve from accumulated experience (with human approval).
- **Multi-agent discussions** — Structured debates with configurable stances, mandatory challenges, and agreement limits.
- **Multi-provider** — Works with Anthropic, OpenAI, DeepSeek, Moonshot, Google Gemini, and more.
- **MCP-native** — First-class Model Context Protocol support for Claude Code, Cursor, and other AI IDEs.
- **Channel integrations** — Built-in support for Feishu (Lark) and WeCom bots.

## How it differs from other frameworks

| | ensoul | crewAI / AutoGen |
|---|---|---|
| Focus | **Who** the agent is | **What** the agent does |
| Identity | Rich Markdown souls with personality, values, style | Role string or system prompt |
| Memory | Semantic pipeline with reflect/connect/store | Chat history or simple RAG |
| Evolution | Soul auto-evolves from experience | Static configuration |
| Negotiation | Structured multi-round discussions with stances | Sequential task handoff |

## Production usage

ensoul powers the digital employee system at Knowlyr, managing 30+ AI employees across engineering, product, and operations — running in production since early 2025.

## Status

🚧 **Alpha** — Core modules are being extracted from the production [Crew](https://github.com/liuxiaotong/knowlyr-crew) codebase. APIs may change.

## License

MIT
