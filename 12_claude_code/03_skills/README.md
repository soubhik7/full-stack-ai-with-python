# 03 — Skills: Reusable, Discoverable Procedures

> A skill is a packaged "how-to" — a `SKILL.md` plus, optionally, supporting
> scripts/templates in the same folder — that Claude can discover and invoke when
> the task matches, without you having to re-explain the procedure every time.

---

## Skills vs. Subagents vs. Slash Commands

These three get confused because they're all "files Claude reads to do something
custom." The difference is *who decides to use them, and how much they carry*:

| Mechanism | Who decides to invoke it | Carries... |
|-----------|--------------------------|-----------|
| **Subagent** | Main agent delegates | A fresh context + restricted tools |
| **Skill** | Main agent matches `description` to the task, invoked in-context | Instructions, executed by the *same* agent — no context switch |
| **Slash command** | You, by typing `/name` explicitly | A literal prompt template, no autonomous matching |

A skill is closer to "an instruction manual the model can pull off the shelf"
than a separate worker. It doesn't fork context the way a subagent does.

---

## Anatomy of `.claude/skills/<name>/SKILL.md`

```markdown
---
name: new-chapter
description: Scaffold a new top-level numbered chapter in this curriculum,
  following the repo's existing README + numbered-subfolder conventions.
---

# Body: the actual procedure, written for an agent with no other context
```

The `description` is everything — it's matched against the current task to
decide relevance, the same way a subagent's `description` is. Be specific about
*when* to use it, not just *what* it does.

A skill's folder can hold more than the one file — helper scripts, templates,
reference data — anything the procedure needs that's tedious to inline as prose.

---

## Example in This Folder

[`skills/new-chapter/SKILL.md`](skills/new-chapter/SKILL.md) — codifies the exact
procedure used to build *this very chapter*: number it, scaffold the README
tree, update the root `CLAUDE.md`'s tables, respect "Do Not Touch." Without this
skill, that procedure lives only in a Claude Code transcript and gets
re-derived (slightly differently) every time someone adds a chapter.

---

## When to Write a Skill vs. Just... Asking

Don't write a skill for a one-off. Write one when you'd otherwise explain the
same multi-step procedure more than once — scaffolding conventions, a deploy
checklist, a recurring data migration shape.

---

## Common Mistakes

| Mistake | Why it's a problem |
|---------|--------------------|
| Description that just restates the name | Gives the matcher nothing to match on |
| Procedure that assumes context from "this conversation" | Skills get invoked cold, like subagents — write for a stranger |
| Skill that's really just a single shell command | That's a slash command or a `tools/` script, not a skill |

---

## Ready to Build?

Move to **[04_hooks/](../04_hooks/README.md)** →
