# 02 — Subagents: Specialised Sub-Assistants

> A subagent is a separate Claude instance with its own system prompt, its own
> tool allowlist, and (optionally) its own model — invoked by the main agent for a
> narrow job, then it reports back and disappears.

---

## Why Subagents Exist

The main conversation accumulates context: your question, files read, dead ends
explored. A subagent starts clean, does one job with a tight toolset, and returns
only its conclusion — protecting the main context window and (because it can't
use tools outside its allowlist) limiting blast radius.

Use a subagent when a task is:
- **Self-contained** — doesn't need the back-and-forth of the main conversation
- **Narrow** — "review this diff," not "build this feature"
- **Repeated** — worth defining once instead of re-explaining every time

Don't reach for a subagent for a single grep or a quick file read — that's
overhead for no benefit. The threshold question: would a smart colleague need more
than a couple of tool calls to do this independently?

---

## Anatomy of `.claude/agents/<name>.md`

```markdown
---
name: code-reviewer
description: Use proactively after any non-trivial code change to review for bugs.
tools: Read, Grep, Glob, Bash
model: sonnet
---

<system prompt body — the subagent's entire briefing, nothing from the main
conversation carries over except what's explicitly handed to it at invocation>
```

| Field | Required? | Notes |
|-------|-----------|-------|
| `name` | Yes | How it's invoked — the subagent type |
| `description` | Yes | The *only* signal the main agent uses to decide when to delegate here — write it like a trigger condition, not a summary |
| `tools` | No | Omit to inherit every tool; restrict to the minimum the job needs |
| `model` | No | Pin to a cheaper/faster model for high-volume narrow jobs |

---

## Examples in This Folder

| File | Pattern |
|------|---------|
| [`agents/code-reviewer.md`](agents/code-reviewer.md) | Read-only review — `tools` excludes `Edit`/`Write` entirely, so it physically cannot "fix" anything it finds, only report it |
| [`agents/notebook-doctor.md`](agents/notebook-doctor.md) | Scoped to one narrow, recurring problem in *this* repo (broken notebook kernel metadata) rather than "notebooks" in general |

Both are reference copies here — see [07_labs/](../07_labs/README.md) to actually
install one into this repo's `.claude/agents/`.

---

## Writing a Good `description`

The main agent reads only the `description` field to decide whether to delegate —
not the body. Compare:

```yaml
description: Reviews code.                    # too vague — when, exactly?
description: Use proactively after writing or  # names the trigger condition
  modifying non-trivial code, to review for
  correctness bugs before the user sees the diff.
```

---

## Common Mistakes

| Mistake | Why it's a problem |
|---------|--------------------|
| Giving every subagent full tool access | Defeats the blast-radius limit that's half the point |
| Vague `description` | Main agent won't know when to delegate, or will over/under-delegate |
| One giant "do everything" subagent | No better than the main agent — narrow it |
| Expecting it to remember prior invocations | Each invocation is stateless unless you resume it by name/ID |

---

## Ready to Build?

Move to **[03_skills/](../03_skills/README.md)** →
