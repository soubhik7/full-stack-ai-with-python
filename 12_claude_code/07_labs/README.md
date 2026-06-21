# 07 — Labs: Wire It Into This Repo

> Everything in `02_subagents/` through `06_settings_and_permissions/` is a
> reference copy. This lab installs three of them into this repo's *actual*
> `.claude/` directory and confirms they fire for real.

This repo doesn't have a `.claude/` directory yet — these labs create one.
Treat that as a real, standing change to how Claude Code behaves in this repo
from then on (not something this chapter does automatically) — review each
step before copying files in.

---

## Lab 1 — Install the Do-Not-Touch Guard Hook

1. Create `.claude/hooks/` and copy in
   [`../04_hooks/hooks/do_not_touch_guard.py`](../04_hooks/hooks/do_not_touch_guard.py).
2. Create or extend `.claude/settings.json` with the `hooks.PreToolUse` block
   from
   [`../04_hooks/hooks/settings.example.json`](../04_hooks/hooks/settings.example.json).
3. Ask Claude Code to edit a file under `reference/` — confirm the edit is
   blocked and the message names the right "Do Not Touch" entry from
   `CLAUDE.md`.
4. Ask it to edit a file under `08_ai_apps/` — confirm that one goes through
   normally. (If both get blocked, the path-prefix check in the script is
   wrong; if neither does, the hook isn't wired up — check `matcher` against
   the tool name you actually used.)

---

## Lab 2 — Install the Notebook Doctor Subagent

1. Create `.claude/agents/` and copy in
   [`../02_subagents/agents/notebook-doctor.md`](../02_subagents/agents/notebook-doctor.md).
2. Find or create a notebook with a deliberately wrong `kernelspec` (swap
   `kernelsoubhik` for some other kernel name) and ask Claude Code to fix it.
3. Confirm the main agent delegates to `notebook-doctor` rather than
   hand-editing the JSON itself — if it doesn't delegate, sharpen the
   `description` field (see [02_subagents/](../02_subagents/README.md) →
   "Writing a Good `description`").

---

## Lab 3 — Install the New-Chapter Skill

1. Create `.claude/skills/new-chapter/` and copy in
   [`../03_skills/skills/new-chapter/SKILL.md`](../03_skills/skills/new-chapter/SKILL.md).
2. Ask: *"I want to add a chapter on vector databases"* — without mentioning
   skills, hooks, or scaffolding by name — and confirm Claude reaches for the
   skill on its own rather than improvising a one-off structure.
3. Throw it away afterward (`git checkout -- .` on whatever it scaffolded)
   unless you actually want that chapter — this is a drill, not a request to
   add one.

---

## Optional: the `/add-app` Command

Copy
[`../05_slash_commands/commands/add-app.md`](../05_slash_commands/commands/add-app.md)
into `.claude/commands/add-app.md` and run
`/add-app 18 my_demo "shows X"` to scaffold a new app the same way Lab 3
scaffolds a chapter, but deterministically (by name) instead of by autonomous
matching.

---

## What You Should Walk Away With

The same fact, demonstrated four ways: **rules are advisory, hooks are
enforced, subagents/skills are capabilities you can hand off to.** Knowing
which one a given problem needs is the actual skill this chapter teaches — the
files are just the vocabulary.

---

## Chapter Complete

```
... → MCP (10) → Azure AI Foundry (11) → Claude Code (12)  ← You finished here
```
