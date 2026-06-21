# Chapter 12 — Claude Code: Skills, Rules, Hooks & Subagents

> **You will master Claude Code's extensibility model — the same configuration
> system that governs how Claude Code behaves in *this very repository*.**

---

## Prerequisites

- [Chapter 10 — MCP](../10_mcp/README.md) *(tool/protocol thinking)*
- [Chapter 08 — AI Applications](../08_ai_apps/README.md) *(agents, tool use, multi-agent patterns)*
- Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code` or the native installer)
- No Python packages required — every artifact in this chapter is Markdown, JSON, or a small script

---

## Why This Chapter?

Claude Code is configured entirely through files, not a GUI:

```
CLAUDE.md                  → persistent project "memory" / rules
.claude/settings.json      → permissions, hooks, model, env
.claude/agents/*.md        → custom subagents
.claude/skills/*/SKILL.md  → custom skills
.claude/commands/*.md      → custom slash commands
```

Every concept in this chapter is something you can point at the root
[CLAUDE.md](../CLAUDE.md) of *this repo* and see in action.

---

## Learning Path

```
00_concepts/                  ← Config hierarchy: where each file lives and who wins
01_rules_and_memory/          ← CLAUDE.md — project, user, and import syntax
02_subagents/                 ← .claude/agents/*.md — specialised sub-assistants
03_skills/                    ← .claude/skills/*/SKILL.md — reusable procedures
04_hooks/                     ← .claude/settings.json hooks — deterministic guardrails
05_slash_commands/            ← .claude/commands/*.md — reusable prompts
06_settings_and_permissions/  ← settings.json schema, allow/deny/ask rules
07_labs/                      ← Wire all of the above into this repo for real
```

---

## Sub-chapter Breakdown

### `00_concepts/` — The Configuration Model
Read this first — the four mechanisms (memory, settings, capabilities, hooks) and
how they layer.

### `01_rules_and_memory/` — CLAUDE.md
| File | What it teaches |
|------|----------------|
| `examples/CLAUDE.md.sample` | Annotated project memory file anatomy |
| `examples/user-CLAUDE.md.sample` | User-level memory (`~/.claude/CLAUDE.md`) — preferences that apply across *all* projects |

### `02_subagents/` — Custom Subagents
| File | What it teaches |
|------|----------------|
| `agents/code-reviewer.md` | A read-only review subagent with a restricted toolset |
| `agents/notebook-doctor.md` | A subagent scoped to one recurring problem in *this* repo |

### `03_skills/` — Custom Skills
| File | What it teaches |
|------|----------------|
| `skills/new-chapter/SKILL.md` | A skill that scaffolds a new numbered top-level chapter following this repo's own conventions |

### `04_hooks/` — Deterministic Guardrails
| File | What it teaches |
|------|----------------|
| `hooks/do_not_touch_guard.py` | A `PreToolUse` hook that mechanically enforces this repo's "Do Not Touch" list |
| `hooks/settings.example.json` | How to wire that script into `settings.json` |

### `05_slash_commands/` — Reusable Prompts
| File | What it teaches |
|------|----------------|
| `commands/add-app.md` | A `/add-app` command that scaffolds a new `08_ai_apps/NN_name/` app |

### `06_settings_and_permissions/` — The settings.json Schema
| File | What it teaches |
|------|----------------|
| `settings/settings.example.json` | Permission `allow`/`deny`/`ask` rules, env vars, model pin |

### `07_labs/` — Put It All Together
Install the guard hook, the subagent, and the skill into this repo's own
`.claude/` folder and watch them fire for real.

---

## Key Concepts at a Glance

```markdown
<!-- CLAUDE.md — loaded automatically every session -->
## Do Not Touch
- `reference/` — external cloned repo, keep intact
```

```yaml
# .claude/agents/code-reviewer.md
---
name: code-reviewer
description: Use proactively after any non-trivial code change to review for bugs.
tools: Read, Grep, Glob
model: sonnet
---
You are a meticulous reviewer. Read the diff, check correctness...
```

```json
// .claude/settings.json — a PreToolUse hook
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [{ "type": "command", "command": "python3 .claude/hooks/do_not_touch_guard.py" }]
      }
    ]
  }
}
```

---

## Next Step

```
... → MCP (10) → Azure AI Foundry (11) → Claude Code (12)  ← You are here
```

This is the last chapter — go build something with the full stack.
