# 06 — settings.json: Permissions, Env, and Model

> Hooks ([04_hooks/](../04_hooks/README.md)) are *code* that runs on events.
> `settings.json` is *configuration* — what's allowed without asking, what's
> denied outright, what env vars are set, which model to use.

---

## The Permission Model

Every tool call is classified `allow`, `ask`, or `deny`:

```json
{
  "permissions": {
    "allow": ["Bash(git status)", "Bash(git diff:*)", "Read(./**)"],
    "ask":   ["Bash(git push:*)"],
    "deny":  ["Bash(rm -rf:*)", "Read(./.env)"]
  }
}
```

- **`allow`** — runs without a prompt
- **`ask`** — prompts you every time, even in less-interactive permission modes
- **`deny`** — blocked outright, takes precedence over `allow` at *any* config level

Patterns follow `ToolName(specifier)` — `Bash(git diff:*)` matches any
`git diff` invocation; bare `Bash(git status)` matches that exact command only.
Omitting the specifier (`"Edit"`) matches the tool unconditionally.

---

## Other Top-Level Keys

| Key | Purpose |
|-----|---------|
| `env` | Environment variables injected into every session |
| `model` | Pin a specific model instead of using the default |
| `hooks` | See [04_hooks/](../04_hooks/README.md) |
| `includeCoAuthoredBy` | Whether commits get a `Co-Authored-By` trailer |
| `cleanupPeriodDays` | How long local session transcripts are retained |

---

## File Precedence (Recap from `00_concepts/`)

```
enterprise managed policy  >  CLI flags  >  settings.local.json  >  settings.json  >  ~/.claude/settings.json
```

`settings.json` is checked into git — team-wide rules. `settings.local.json` is
gitignored — your personal overrides (e.g. a model preference) on top of the
team baseline. `deny` rules are the one thing that can't be overridden downward.

---

## Example in This Folder

[`settings/settings.example.json`](settings/settings.example.json) — a starter
project `settings.json`: a permissive allowlist for read-only/common commands,
an `ask` gate on `git push`, and a `deny` on `rm -rf` and reading `.env`.

---

## Common Mistakes

| Mistake | Why it's a problem |
|---------|--------------------|
| `"allow": ["Bash"]` with no specifier | Allows *every* shell command unattended — scope it |
| Putting secrets in `env` inside `settings.json` | That file is typically committed to git |
| Relying on `ask` for something that should be `deny` | `ask` can still be approved in the moment — use `deny` for hard lines |

---

## Ready to Build?

Move to **[07_labs/](../07_labs/README.md)** →
