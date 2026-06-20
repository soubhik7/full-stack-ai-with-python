# Lab 02 — Tools

> **Pattern:** Two distinct tool types, each wired to its own one-off agent so the mechanics stay isolated.

---

## Tool Types Covered

| Tool | What it is | File |
|------|-----------|------|
| **Function tool** | A custom Python function you write; the model decides when to call it, you (or `enable_auto_function_calls`) execute it | `function_tool.py` |
| **Code interpreter** | A built-in sandbox where the agent writes *and runs* real Python — for exact computation, not predicted numbers | `code_interpreter_tool.py` |

---

## Run It

```bash
cd 11_azure_ai_foundry/02_tools
python function_tool.py
python code_interpreter_tool.py
```

---

## Key Concept: Custom vs. Built-in

```python
# Function tool — you wrote check_room_availability(); Azure calls it for you
function_tool = FunctionTool(functions={check_room_availability})
agents_client.enable_auto_function_calls({check_room_availability})

# Code interpreter — no function to write; the agent writes its own code
code_interpreter = CodeInterpreterTool()
```

`03_toolbox/` combines both of these into a single `ToolSet` attached to one agent.

---

## Previous / Next

← [01 — Workflow](../01_workflow/) · [03 — Toolbox](../03_toolbox/) →
