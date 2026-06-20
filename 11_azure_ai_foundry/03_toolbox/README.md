# Lab 03 — Toolbox

> **Pattern:** Bundle several tools into one `ToolSet` so an agent can pull from all of them in a single turn.

---

## Why a Toolbox?

`02_tools/` wired one tool to one agent at a time. Real agents usually need more than one capability at once — a fact lookup *and* a calculation, say — without hand-assembling a tools list everywhere the agent is created. A toolbox is that assembly done once, in `toolbox.py`, then reused.

This is also a deliberate echo of how the study notes describe Azure AI Foundry itself: "a toolbox that simplifies the creation and deployment of AI agents" (§7.1).

---

## Files

| File | What it shows |
|------|--------------|
| `toolbox.py` | `build_toolbox()` — a `ToolSet` combining two function tools + the code interpreter |
| `main.py` | One agent, one question, answered using *both* a custom function and the code interpreter |

---

## Run It

```bash
cd 11_azure_ai_foundry/03_toolbox
python main.py
```

---

## Key Concept: One Bundle, Many Tools

```python
toolset = ToolSet()
toolset.add(FunctionTool(functions={get_checkout_time, get_nightly_rate}))
toolset.add(CodeInterpreterTool())

agent = agents_client.create_agent(model=DEPLOYMENT, name="...", toolset=toolset)
```

---

## Previous / Next

← [02 — Tools](../02_tools/) · [04 — Prompt Agent](../04_prompt_agent/) →
