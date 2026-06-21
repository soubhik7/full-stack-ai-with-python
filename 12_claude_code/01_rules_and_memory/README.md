# 01 — Rules & Memory: CLAUDE.md

> The file you're reading this curriculum through right now was itself loaded the
> same way Claude Code loads every `CLAUDE.md` in every project.

---

## What Goes In It

`CLAUDE.md` is **persistent context**, not a prompt. Good candidates:

- Repo structure that isn't obvious from `ls` (this repo's chapter table)
- Commands that aren't discoverable (`make style`, custom test invocations)
- Conventions a linter can't enforce (naming, "do not touch" zones)
- Decisions that would otherwise get re-litigated every session

Bad candidates — things that belong in code/comments instead:
- Anything a linter or type checker already enforces
- Implementation detail that changes every time the code changes
- Long prose explanations — Claude can read the code

---

## Where Claude Code Looks

| Location | Scope | Checked into git? |
|----------|-------|-------------------|
| `~/.claude/CLAUDE.md` | Every project, this user | No (lives outside any repo) |
| `<repo-root>/CLAUDE.md` | This project, every user | **Yes** — this is the one in this repo's root |
| `<repo-root>/CLAUDE.local.md` | This project, this user only | No (gitignored; legacy — prefer imports) |
| Any `CLAUDE.md` in a parent directory of your cwd | Everything under that parent | Depends |

All of these **stack** — Claude Code loads every one it finds walking from cwd up
to `/`, plus the user-level file. There's no "wins" here, only "all of the above."

---

## Import Syntax

A `CLAUDE.md` can pull in other files instead of inlining everything:

```markdown
See @docs/architecture.md for the full system diagram.
Coding conventions: @.agent/docs/style-guide.md
```

Imports resolve relative to the file that contains them, and can chain up to 5 hops
deep. This repo uses a flat single file rather than imports because the curriculum
itself *is* the documentation — but for a large monorepo, splitting by area
(`@frontend/CLAUDE.md`, `@backend/CLAUDE.md`) keeps each file focused.

---

## Annotated Example

See [`examples/CLAUDE.md.sample`](examples/CLAUDE.md.sample) for a minimal,
fully-annotated `CLAUDE.md` anatomy. Compare it against the real, much larger
[root `CLAUDE.md`](../../CLAUDE.md) of this repo — same structure, more content.

User-level memory is a different document with a different job — see
[`examples/user-CLAUDE.md.sample`](examples/user-CLAUDE.md.sample): preferences
about *how you work*, not facts about *this project*.

---

## The `#` Shortcut

Typing `# <something>` at the start of a Claude Code prompt appends it to the
nearest `CLAUDE.md` immediately — no file editing required. Use `/memory` to open
the relevant memory file directly in your editor.

---

## Common Mistakes

| Mistake | Why it's a problem |
|---------|--------------------|
| Pasting whole files into `CLAUDE.md` | Claude can already read files — this just wastes context every session |
| Writing it once and never updating | Stale rules are worse than no rules — they actively mislead |
| Putting secrets in it | It's loaded into every session's context and often committed to git |
| Using it for one-off task state | That's what a plan or todo list is for, not memory |

---

## Ready to Build?

Move to **[02_subagents/](../02_subagents/README.md)** →
