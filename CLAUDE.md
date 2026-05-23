# CLAUDE.md — Project Intelligence

## What This Repo Is

A **sequential learning curriculum** for AI/ML with Python.
Ten numbered top-level chapters — students work through them 00 → 09 in order.

| Chapter | Content |
|---------|---------|
| `00_setup/` | Environment setup guide |
| `01_python/` | Python fundamentals — chai-shop themed, 00–14 |
| `02_math_for_ml/` | Coursera Math for ML (linear algebra, calculus, probability) |
| `03_machine_learning/` | Regression → Classification → Clustering → RecSys |
| `04_deep_learning/` | Perceptron → MLP → Libraries → NN from scratch → Coursera |
| `05_nlp/` | Preprocessing → Tagging → Practice |
| `06_large_language_models/` | HuggingFace → LLM from scratch → Fine-tuning |
| `07_rag/` | 4-approach RAG learning path |
| `08_ai_apps/` | 13 numbered runnable apps (01–13) |
| `09_projects/` | 5 numbered capstone projects (01–05) |
| `reference/` | External cloned reference repo — keep intact |
| `tools/` | Dev utility scripts |

---

## Python Environment

- **venv**: `venv/` at repo root — `source venv/bin/activate`
- **Python**: 3.11.15 (Homebrew `/opt/homebrew/bin/python3.11`)
- **Jupyter kernel**: `KernelSoubhik` — registered at `~/Library/Jupyter/kernels/kernelsoubhik`
- **Requirements**: `requirements.txt` at root covers all notebooks and apps

```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## Naming Conventions

- All directories: `lowercase_underscore` (no spaces, no `&`, no CamelCase, no dots)
- Python files: `lowercase_underscore.py`
- Notebooks: `NN_descriptive_name.ipynb` (zero-padded number prefix)
- Each app in `08_ai_apps/` is **independent** — don't share state across apps

---

## Do Not Touch

- `reference/` — external cloned reference repo (FraidoonOmarzai); keep intact
- `01_python/` — well-structured curriculum; respect the numbering
- `02_math_for_ml/` — Coursera official materials; do NOT rename internal files/folders
- `04_deep_learning/05_coursera_assignments/W*/` — utility `.py` files imported by notebooks
- `09_projects/04_made_with_ml/` — standalone sub-repo with own `pyproject.toml`
- `venv/` — not committed (covered by `.gitignore`)

---

## Running Apps

```bash
# Apps with Docker (05_langraph, 06_rag, 07_rag_queue, 09_mem_agent)
cd 08_ai_apps/<name>
docker-compose up -d
python main.py          # or mem.py / server.py

# Standalone apps
cd 08_ai_apps/<name>
python main.py
```

---

## Running Notebooks

All notebooks are pre-configured to use `KernelSoubhik`.

```bash
source venv/bin/activate
jupyter lab        # then navigate to any .ipynb
```

---

## Key Paths

| What | Where |
|------|-------|
| Setup guide | `00_setup/README.md` |
| Python curriculum | `01_python/` |
| Math (Coursera) | `02_math_for_ml/` |
| ML notebooks (sklearn) | `03_machine_learning/` |
| Deep learning | `04_deep_learning/` |
| NLP | `05_nlp/` |
| LLMs + HuggingFace | `06_large_language_models/` |
| RAG (4 approaches) | `07_rag/` |
| Runnable AI apps | `08_ai_apps/` |
| End-to-end projects | `09_projects/` |
| Comprehensive reference | `reference/` |
| Notebook tools | `tools/` |

---

## Git Conventions

- Always use `os.rename()` + `git add -A` to move files — `git mv` fails on `&` in filenames
- Branch from `main` for new features: `feature/<name>` or `restructure/<name>`
- Commit format: `type: short description` (feat/fix/refactor/chore/docs)

---

## Notebook Maintenance (`tools/`)

| Script | Purpose |
|--------|---------|
| `fix_notebook_metadata.py` | Fix Jupyter kernel metadata in notebooks |
| `fix_ssl.py` / `ssl_config.py` | macOS SSL certificate fixes |
| `add_missing_images_to_notebook.py` | Patch missing image cells |
| `compare_article_notebook.py` | Diff article vs notebook content |
| `create_model_notebook.py` | Scaffold a new model notebook |
