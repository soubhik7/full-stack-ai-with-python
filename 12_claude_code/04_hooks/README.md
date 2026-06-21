# 04 — Hooks: Deterministic Guardrails

> Everything so far (memory, subagents, skills) relies on the model *choosing* to
> follow it. A hook is the opposite: plain code that runs automatically on a
> specific event, with no model judgment involved — it can't be talked out of it.

---

## Why Hooks Exist

`CLAUDE.md` says "don't touch `reference/`." A well-behaved model respects that.
But "well-behaved" isn't "guaranteed" — instructions compete with the rest of the
prompt, and a long enough session can drift. A hook that mechanically blocks the
`Edit` tool from touching `reference/` doesn't have that failure mode: it's a
shell exit code, not a suggestion.

Use a hook when a rule must hold **every time**, not just "usually" — security
boundaries, "do not touch" zones, formatting-on-save, audit logging.

---

## Hook Events

| Event | Fires when | Typical use |
|-------|-----------|-------------|
| `PreToolUse` | Before a tool call executes | Block/allow based on the tool input |
| `PostToolUse` | After a tool call completes | Auto-format, lint, log what changed |
| `UserPromptSubmit` | You submit a prompt | Inject extra context, validate input |
| `Notification` | Claude Code sends a notification | Route to Slack/desktop notifications |
| `Stop` / `SubagentStop` | Main agent / a subagent finishes | Run a final check, send a completion ping |
| `SessionStart` | A session begins | Warm a cache, print a project status banner |

---

## Wiring a Hook (`.claude/settings.json`)

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          { "type": "command", "command": "python3 .claude/hooks/do_not_touch_guard.py" }
        ]
      }
    ]
  }
}
```

`matcher` is matched against the tool name (regex-capable — `"Edit|Write"`
matches either). Omit `matcher` for events that aren't tool-scoped
(`UserPromptSubmit`, `Stop`, etc.).

---

## What a Hook Receives and Returns

The command receives a JSON payload on **stdin** — for `PreToolUse`, that
includes `tool_name` and `tool_input`. It communicates back via **exit code**:

| Exit code | Effect |
|-----------|--------|
| `0` | Allow / continue, stdout shown to the user (not fed back to the model) |
| `2` | Block the action; **stderr** is fed back to Claude as the reason |
| anything else | Treated as a non-blocking error |

More recent Claude Code versions also support a richer structured-JSON response
on stdout for fine-grained control (e.g. allow-with-modification) — check
`claude code docs hooks` for the current exact schema, this surface moves fast.

---

## Example in This Folder

[`hooks/do_not_touch_guard.py`](hooks/do_not_touch_guard.py) — a `PreToolUse`
hook that mechanically enforces *this repo's own* `CLAUDE.md` → "Do Not Touch"
list. It reads `tool_input.file_path`, checks it against the guarded prefixes,
and exits `2` (blocking, with an explanation on stderr) if the model tries to
edit inside one of them — turning a written rule into something that can't be
drifted past.

[`hooks/settings.example.json`](hooks/settings.example.json) shows how to wire
it in. See [07_labs/](../07_labs/README.md) to actually install it.

---

## Common Mistakes

| Mistake | Why it's a problem |
|---------|--------------------|
| Slow hooks (network calls, big subprocess) | Every matching tool call now waits on it |
| Hooks that crash on unexpected input | Read the payload defensively — Claude Code will call this constantly |
| Using a hook where a `CLAUDE.md` rule would do | Reach for hooks only when "usually" isn't good enough |
| Forgetting hooks apply to *every* session, including future ones | Unlike a one-off instruction, this is a standing behavior change |

---

## Ready to Build?

Move to **[05_slash_commands/](../05_slash_commands/README.md)** →
