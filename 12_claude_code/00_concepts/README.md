# 00 — The Configuration Model

> Four different file types, four different jobs. Understand which one to reach
> for before writing any of them.

---

## Three Questions, Three Mechanisms

| Question | Mechanism | File |
|----------|-----------|------|
| "What should Claude always know about this project?" | **Memory** | `CLAUDE.md` |
| "What is Claude allowed to do, automatically, without asking?" | **Settings** | `.claude/settings.json` |
| "What new capability should Claude have?" | **Agents / Skills / Commands** | `.claude/agents/`, `.claude/skills/`, `.claude/commands/` |

This chapter has one sub-folder per row of that table, plus hooks — a fourth
mechanism: deterministic code that runs on specific events, not LLM judgment.

---

## The `.claude/` Directory Layout

```
your-project/
├── CLAUDE.md                  ← project memory (checked into git)
├── CLAUDE.local.md            ← personal project memory (gitignored, legacy)
└── .claude/
    ├── settings.json          ← shared project settings (checked into git)
    ├── settings.local.json    ← personal project settings (gitignored)
    ├── agents/
    │   └── code-reviewer.md
    ├── skills/
    │   └── my-skill/SKILL.md
    └── commands/
        └── my-command.md

~/.claude/                      ← user-level, applies to every project
├── CLAUDE.md
├── settings.json
├── agents/
├── skills/
└── commands/
```

---

## Precedence: Settings

When the same setting is defined in more than one place, this is the order Claude
Code resolves it (highest wins):

```
1. Enterprise managed policy   (admin-controlled, cannot be overridden)
2. Command-line flags          (--permission-mode, etc.)
3. .claude/settings.local.json (personal, this project, gitignored)
4. .claude/settings.json       (shared, this project, checked into git)
5. ~/.claude/settings.json     (personal, every project)
```

`permissions.deny` rules are the exception — a deny at *any* level blocks the
action; it can't be re-allowed by a lower-precedence file.

---

## Precedence: Memory (`CLAUDE.md`)

Unlike settings, memory files don't override each other — they **stack**. Claude
Code walks from your current directory up to the filesystem root, loading every
`CLAUDE.md` it finds, then adds `~/.claude/CLAUDE.md` on top. All of it ends up in
context simultaneously. See [01_rules_and_memory/](../01_rules_and_memory/README.md)
for the import syntax (`@path/to/file`) that lets a `CLAUDE.md` pull in other docs.

---

## Where This Repo's Own Config Lives

This repo doesn't currently have a `.claude/` directory — only the root
[CLAUDE.md](../../CLAUDE.md). Everything else in this chapter (`02_subagents/`
through `06_settings_and_permissions/`) ships as **reference copies** you can
study, not live config — [07_labs/](../07_labs/README.md) is where you actually
install something into `.claude/` and watch it take effect.

---

## Ready to Build?

Move to **[01_rules_and_memory/](../01_rules_and_memory/README.md)** →
