# Lab 00 — Setup & Connection Check

> **Pattern:** Sanity-check auth and endpoint before running anything else in this chapter.

---

## Run It

```bash
az login                       # if you haven't already
cd 11_azure_ai_foundry/00_setup
python verify_connection.py
```

Expected output:
```
Project endpoint: https://<resource>.services.ai.azure.com/api/projects/<project>
Model deployment: <your-deployment>
Model response:   connection ok
```

If this fails, every other lab in `11_azure_ai_foundry/` will fail the same way — fix it here first.

---

## Files

| File | What it shows |
|------|--------------|
| `verify_connection.py` | `AIProjectClient` + `DefaultAzureCredential` + one stateless chat call |

---

## Next Lab

→ [01 — Workflow](../01_workflow/)
