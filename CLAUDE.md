# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

A **sequential learning curriculum** for AI/ML with Python.
Numbered top-level chapters — students work through them 00 → 11 in order.

| Chapter | Content |
|---------|---------|
| `00_setup/` | Environment setup guide |
| `01_python/` | Python fundamentals — chai-shop themed, 00–14 |
| `02_math_for_ml/` | Coursera Math for ML (linear algebra, calculus, probability) |
| `03_machine_learning/` | Regression → Classification → Clustering → RecSys |
| `04_deep_learning/` | Perceptron → MLP → Libraries → NN from scratch → Coursera |
| `05_nlp/` | Preprocessing → Tagging → Practice |
| `06_large_language_models/` | HuggingFace → LLM from scratch → Fine-tuning → Prompt Engineering → Evaluation |
| `07_rag/` | 4-approach RAG learning path |
| `08_ai_apps/` | 17 numbered runnable apps (01–17): hello-world → MCP server → streaming → structured outputs → multi-agent |
| `09_projects/` | 5 numbered capstone projects (01–05) |
| `10_mcp/` | Model Context Protocol — concepts → server → client → Claude integration → labs |
| `11_azure_ai_foundry/` | Azure AI Foundry study notes (PDF) + 8 runnable agent-service labs (00–07) |
| `reference/` | External cloned reference repo — keep intact |
| `tools/` | Dev utility scripts |

### Chapter 06 Sub-chapters
| Sub-chapter | Content |
|-------------|---------|
| `01_transformers_and_huggingface/` | Pre-trained pipelines, fine-tuning BERT |
| `02_llm_from_scratch/` | GPT architecture in PyTorch |
| `03_llm_fine_tuning/` | Adapt a model on custom data |
| `04_prompt_engineering/` | System prompts, few-shot, CoT, structured output, chaining |
| `05_llm_evaluation/` | BLEU, ROUGE, BERTScore, LLM-as-judge, eval pipelines |

### Chapter 08 Apps
| App | Pattern |
|-----|---------|
| 01–13 | Original apps (hello-world → tokenization → prompts → Ollama → LangGraph → RAG → RAG+queue → tool-use agent → memory agent → voice agent → HuggingFace → vision → frontend todo) |
| `14_mcp_server/` | Task manager MCP server + GPT-4o client |
| `15_streaming/` | All streaming patterns (basic, callbacks, parallel, early-stop) |
| `16_structured_outputs/` | Pydantic extraction, enums, nested models, pipelines |
| `17_multi_agent/` | Orchestrator+workers, sequential pipeline, debate/critic loop |

Apps requiring Docker: `05_langraph`, `06_rag`, `07_rag_queue`, `09_mem_agent`. The rest are standalone — see "Running Apps" below.

### Chapter 09 Projects
| Project | Stack |
|---------|-------|
| `01_house_price_ml/` | Linear/polynomial regression, K-Means, hierarchical clustering |
| `02_sentiment_analysis/` | sklearn pipeline + custom tokenizer |
| `03_gmail_assistant/` | `builtin_model/` (HF summarizer) + `custom_model/` (own-trained PyTorch model, FastAPI) |
| `04_made_with_ml/` | Standalone MLOps sub-repo (Ray, MLflow) — own `pyproject.toml`/`Makefile`, see Testing & Linting below |
| `05_great_learning_certification/` | 8 AIML certification projects (stats → CV → NLP) |

### Chapter 10 Sub-chapters
| Sub-chapter | Content |
|-------------|---------|
| `01_mcp_concepts/` | Protocol architecture, transports, primitives |
| `02_mcp_server/` | FastMCP servers: hello, calculator, resources+prompts |
| `03_mcp_client/` | Basic client, OpenAI agentic loop client |
| `04_mcp_with_claude/` | Claude Desktop config, OpenAI+MCP full app |
| `05_labs/` | Weather, SQLite, Filesystem servers + multi-tool capstone |

### Chapter 11 Labs (`11_azure_ai_foundry/`)
Companion to `Azure_AI_Foundry_Study_Notes.pdf`. All 8 labs run against a real Azure AI Foundry project using `azure-ai-projects` + `azure-ai-agents` + `azure-identity` (Entra ID via `DefaultAzureCredential`, no API keys — auth is `az login`). Share one running example (a "Crystal Hotels" booking assistant).

| Lab | Concept |
|-----|---------|
| `00_setup/` | Verify `.env` (`AZURE_AI_PROJECT_ENDPOINT`, `AZURE_AI_MODEL_DEPLOYMENT`) + auth |
| `01_workflow/` | Foundry's 5-component workflow (Model Catalog → Playground → Agents → Projects → Deployments), made programmatic |
| `02_tools/` | Function tool + built-in code interpreter, each in isolation |
| `03_toolbox/` | Bundling multiple tools into one reusable `ToolSet` |
| `04_prompt_agent/` | Stateless prompt-only agent — no persisted Agent resource |
| `05_hosted_agent/` | Persistent Agent Service — agent + thread with memory |
| `06_connected_agents/` | Multi-agent orchestration via `ConnectedAgentTool` |
| `07_knowledge_rag/` | Grounding answers in an uploaded doc via file search + vector store |

Every lab deletes the agents/threads/vector stores/files it creates — nothing accumulates in the Foundry project across repeated runs. To find your own project's endpoint: `az rest --method get --url "https://management.azure.com/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<account>/projects?api-version=2025-04-01-preview"`.

---

## Python Environment

- **venv**: `venv/` at repo root — `source venv/bin/activate`
- **Python**: 3.11.15 (Homebrew `/opt/homebrew/bin/python3.11`)
- **Jupyter kernel**: `KernelSoubhik` — registered at `~/Library/Jupyter/kernels/kernelsoubhik`
- **Requirements**: `requirements.txt` at root covers all notebooks and apps (includes `pytest`, `black`, `flake8`, `isort` for ad-hoc use)
- **API keys**: `.env` at repo root — `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`, `LANGCHAIN_API_KEY`, `GROQ_API_KEY`

```bash
source venv/bin/activate
pip install -r requirements.txt
```

There is no root-level test/lint config — most chapters are notebooks or one-off scripts with no automated tests. `09_projects/04_made_with_ml/` is the one exception with real tooling (see Testing & Linting).

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

## Testing & Linting

Only `09_projects/04_made_with_ml/` has a real test/lint setup (own `pyproject.toml` + `Makefile`, configured for `black`, `isort`, `flake8`, `pytest`, `pytest-cov`). Run these from inside that directory:

```bash
cd 09_projects/04_made_with_ml

# Style: black + flake8 + isort + pyupgrade
make style

# Tests
python3 -m pytest tests/code --verbose --disable-warnings
pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings
pytest --run-id=$RUN_ID tests/model --verbose --disable-warnings

# Single test
python3 -m pytest tests/code/test_data.py --verbose --disable-warnings

# Coverage
python3 -m pytest tests/code --cov madewithml --cov-report term --disable-warnings
```

Files elsewhere named `test_*.py` (e.g. `01_python/00_python/test_python.py`, the Coursera `W*/test_utils.py` helpers) are curriculum scripts/grader utilities, not a pytest suite — run them directly with `python <file>.py`, not `pytest`.

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
| Azure AI Foundry labs | `11_azure_ai_foundry/` |
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
| `add_missing_images_to_notebook.py` / `list_and_add_missing_images.py` | Patch missing image cells |
| `download_images.py` | Fetch images referenced by `article_image_urls.txt` |
| `compare_article_notebook.py` / `generate_parity_detailed.py` | Diff article vs notebook content |
| `patch_notebook_from_article.py` / `add_missing_code_blocks.py` | Backfill notebook cells from source article |
| `create_model_notebook.py` | Scaffold a new model notebook |

---

## Known Gotchas

- `.github/workflows/gmail-ai.yml` and `gmail-ai-custom.yml` still reference the pre-restructure path `Scikit_ML/SelfProject/Gmail_AI_Assistant_*`, which no longer exists — the code now lives at `09_projects/03_gmail_assistant/{builtin_model,custom_model}/`. These workflows are stale; don't use them as a template for current paths, and fix the `working-directory` if re-enabling them.
