# Lab 04 — Prompt Agent

> **Pattern:** Lightweight and stateless — a system prompt + chat completion, no Azure-hosted Agent resource at all.

---

## What Makes This a "Prompt Agent"?

No `create_agent()` call anywhere in this file. No `agent_id`, no `thread_id`. Just a system prompt and `chat.completions.create()` — the same shape as any one-off LLM API call, via `AIProjectClient.get_openai_client()`. Function calling still works, but nothing executes it automatically: this file manually checks for `message.tool_calls`, runs the Python function, and appends the result before asking the model to finish.

Use this when you don't need multi-turn memory and want the cheapest, simplest integration.

---

## Files

| File | What it shows |
|------|--------------|
| `main.py` | A manual function-calling loop with no persisted agent or thread |

---

## Run It

```bash
cd 11_azure_ai_foundry/04_prompt_agent
python main.py
```

The script asks two questions back to back. The second ("What did I just ask about?") gets no useful answer — proof there's no memory between calls.

---

## Contrast with Lab 05

[05_hosted_agent](../05_hosted_agent/) asks the same *kind* of follow-up question and answers it correctly, because the thread (not the caller) carries history there.

---

## Previous / Next

← [03 — Toolbox](../03_toolbox/) · [05 — Hosted Agent](../05_hosted_agent/) →
