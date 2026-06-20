# Lab 05 — Hosted Agent

> **Pattern:** Azure AI Foundry Agent Service — a persistent agent + thread that remembers prior turns.

---

## What Makes This "Hosted"?

`create_agent()` registers the agent's model + instructions on Azure, separately from any one conversation. `threads.create()` opens a conversation that Azure stores server-side. Every `messages.create()` + `runs.create_and_process()` against the same `thread_id` sees the full prior history automatically — you never resend earlier turns yourself.

---

## Files

| File | What it shows |
|------|--------------|
| `main.py` | Two turns in one thread; the second turn correctly builds on the first |

---

## Run It

```bash
cd 11_azure_ai_foundry/05_hosted_agent
python main.py
```

---

## Contrast with Lab 04

[04_prompt_agent](../04_prompt_agent/) asks a structurally similar follow-up and gets a blank — no thread, no memory. Run both and compare.

---

## Previous / Next

← [04 — Prompt Agent](../04_prompt_agent/) · [06 — Connected Agents](../06_connected_agents/) →
