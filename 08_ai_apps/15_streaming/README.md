# App 15 — Streaming AI Responses

> **Pattern:** Stream tokens as they're generated — no waiting for the full response.

---

## Why Streaming?

Without streaming: User waits 5 seconds → sees the full response at once.  
With streaming: User sees words appearing immediately, like a real conversation.

This dramatically improves **perceived performance** and is essential for:
- Chat applications
- Long-form content generation
- Real-time code generation
- Voice pipelines (read first N tokens while generating the rest)

---

## Files

| File | What it shows |
|------|--------------|
| `main.py` | All streaming patterns in one script |

---

## Streaming Patterns Covered

| Pattern | Use case |
|---------|---------|
| Basic streaming | Any chat response |
| Stream with callback | Progress tracking, logging |
| Streaming to file | Long document generation |
| Parallel streaming | Generate multiple responses simultaneously |
| Streaming + early stop | Stop when you've seen enough |

---

## Run It

```bash
cd 08_ai_apps/15_streaming
python main.py
```

Requires: `OPENAI_API_KEY` in `.env`

---

## Key Concept: How Streaming Works

```python
# Without streaming (blocks until complete)
response = client.chat.completions.create(model="gpt-4o", messages=[...])
print(response.choices[0].message.content)  # prints after 5+ seconds

# With streaming (tokens arrive immediately)
stream = client.chat.completions.create(model="gpt-4o", messages=[...], stream=True)
for chunk in stream:
    token = chunk.choices[0].delta.content or ""
    print(token, end="", flush=True)  # prints each token as it arrives
```

---

## Previous App

← [14 — MCP Server](../14_mcp_server/)

## Next App

→ [16 — Structured Outputs](../16_structured_outputs/)
