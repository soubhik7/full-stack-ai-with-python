# 05 — Slash Commands: Reusable Prompts

> A slash command is a prompt template saved to a file. You invoke it explicitly
> by typing `/name`; nothing about *whether* to use it is left to the model.

---

## Slash Commands vs. Skills

In current Claude Code, typing `/something` is routed through the same
mechanism that powers skills — so the line between "slash command" and "skill"
has blurred. The distinction that still matters conceptually:

- A **command** is invoked *by name, by you* — deterministic, no discovery
- A **skill** is invoked *by the model*, matched against the task — autonomous

If you want something to fire only when you explicitly ask, write it as a
command. If you want Claude to reach for it on its own when relevant, write it
as a skill (see [03_skills/](../03_skills/README.md)).

---

## Anatomy of `.claude/commands/<name>.md`

```markdown
---
description: One-line description shown in command lists
argument-hint: <app-number> <app-name> "<description>"
allowed-tools: Bash(git add:*), Bash(git commit:*)
---

Prompt body. Use $1, $2, $3 for positional arguments, or $ARGUMENTS for the
whole string. Use @path/to/file to inline a file's contents, or !`command` to
run a shell command and inline its output.
```

`allowed-tools` scopes what the command can do without per-call confirmation —
same idea as a subagent's `tools` field, applied to a prompt template instead
of a sub-conversation.

---

## Example in This Folder

[`commands/add-app.md`](commands/add-app.md) — `/add-app 18 my_demo "shows X"`
scaffolds `08_ai_apps/18_my_demo/` following this repo's existing app
conventions (standalone, runnable `main.py`, README, no Docker unless actually
needed) and updates the chapter table in the root `CLAUDE.md`.

---

## Common Mistakes

| Mistake | Why it's a problem |
|---------|--------------------|
| No `argument-hint` | You'll forget the expected argument order three uses from now |
| Command body that re-derives conventions instead of pointing at them | Drifts from the real conventions the moment they change; link to the source of truth instead |
| Using a command for something that should auto-trigger | That's a skill — see [03_skills/](../03_skills/README.md) |

---

## Ready to Build?

Move to **[06_settings_and_permissions/](../06_settings_and_permissions/README.md)** →
