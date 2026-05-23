# CLAUDE.md — Project Intelligence

## What This Repo Is

A mixed learning + project + demos repository for AI/ML work with Python.
Four top-level sections with distinct purposes — do not conflate them.

| Section | Purpose |
|---------|---------|
| `01_python/` | Structured curriculum — don't reorganise |
| `research/` | Notebooks & exploration — no production code |
| `projects/` | End-to-end self-directed ML projects |
| `apps/` | Runnable applications (FastAPI, LangGraph, voice, etc.) |
| `tools/` | Dev utility scripts (notebook fixes, SSL, image patching) |

---

## Python Environment

- **venv**: `venv/` at repo root — `source venv/bin/activate`
- **Python**: 3.11.15 (Homebrew `/opt/homebrew/bin/python3.11`)
- **Jupyter kernel**: `KernelSoubhik` — registered at `~/Library/Jupyter/kernels/kernelsoubhik`
- **Requirements**: `requirements.txt` at root covers all notebooks and apps

To install all packages:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## Naming Conventions

- All directories: `lowercase_underscore` (no spaces, no `&`, no CamelCase, no dots)
- Python files: `lowercase_underscore.py`
- Notebooks: descriptive names are fine; avoid `&`, commas, and leading spaces
- Each app in `apps/` is **independent** — don't share state across apps

---

## Do Not Touch

- `research/comprehensive/` — external cloned reference repo; keep intact as-is
- `01_python/` — well-structured curriculum with its own README; respect the numbering
- `venv/` — not committed (covered by `.gitignore`)

---

## Running Apps

```bash
# Apps with Docker dependencies
cd apps/<name>
docker-compose up -d
python main.py          # or server.py

# Standalone apps (no Docker)
cd apps/<name>
python main.py
```

Apps that use Docker: `rag/`, `rag_queue/`, `langraph/`, `mem_agent/`

---

## Running Notebooks

All 187 notebooks are pre-configured to use `KernelSoubhik`.
Open any notebook in JupyterLab — the kernel is already selected.

```bash
jupyter lab        # then navigate to any .ipynb
```

---

## Key Paths

| What | Where |
|------|-------|
| All notebooks | `research/**/*.ipynb`, `projects/**/*.ipynb` |
| ML basics (sklearn) | `research/machine_learning/` |
| Deep learning from scratch | `research/deep_learning/` |
| LLM from scratch (PyTorch) | `research/llm/01_llm_from_scratch_pytorch/` |
| RAG learning path | `research/rag/` (4 approaches, 01–04) |
| Complete reference curriculum | `research/comprehensive/` |
| Coursera assignments | `research/deep_learning/coursera/`, `research/coursework/math_for_ml/` |
| Gmail AI assistant | `projects/gmail_assistant/` |
| Made With ML (MLOps) | `projects/made_with_ml/` |
| Prompt examples | `apps/prompts/` |
| Notebook tools | `tools/` |

---

## Git Conventions

- Always use `git mv` (or OS rename + `git add -A`) to move files — never `cp` + `rm`
- Branch from `main` for new features; use `feature/<name>` or `restructure/<name>`
- Commit message format: `type: short description` (feat/fix/refactor/chore/docs)

---

## Notebook Maintenance (`tools/`)

| Script | Purpose |
|--------|---------|
| `fix_notebook_metadata.py` | Fix Jupyter kernel metadata in notebooks |
| `fix_ssl.py` / `ssl_config.py` | macOS SSL certificate fixes |
| `add_missing_images_to_notebook.py` | Patch missing image cells |
| `compare_article_notebook.py` | Diff article vs notebook content |
| `create_model_notebook.py` | Scaffold a new model notebook |
