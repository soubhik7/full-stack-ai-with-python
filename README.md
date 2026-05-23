# Full-Stack AI with Python
### A Complete, Sequential Curriculum — From Python Basics to Production AI Systems

---

## 🗺️ Learning Path

This repository is structured as a **step-by-step curriculum**. Work through the chapters in order — each chapter builds on the previous one.

```
00_setup/                   ← Start here — environment & prerequisites
01_python/                  ← Python fundamentals (14 topics, ~100 exercises)
02_math_for_ml/             ← Linear algebra, calculus, probability & statistics
03_machine_learning/        ← Regression → Classification → Clustering → RecSys
04_deep_learning/           ← Perceptron → MLP → Libraries → From-scratch → Coursera
05_nlp/                     ← Text preprocessing → POS tagging → NER
06_large_language_models/   ← HuggingFace → LLM from scratch → Fine-tuning
07_rag/                     ← Retrieval-Augmented Generation (4 approaches)
08_ai_apps/                 ← 13 runnable AI applications
09_projects/                ← 5 end-to-end capstone projects
reference/                  ← Comprehensive AI/ML reference notebooks
tools/                      ← Developer utilities
```

---

## 📚 Chapter Guide

### `00_setup` — Environment Setup
> **Time:** 15 min · **Prerequisite:** None

Get your Python environment, Jupyter kernel, and API keys configured before anything else.

---

### `01_python` — Python Fundamentals
> **Time:** 2–4 weeks · **Prerequisite:** None

Chai-shop themed Python curriculum covering the language from the ground up.

| Module | Topic |
|--------|-------|
| `00_python/` | Hello Python |
| `02_datatypes/` | Strings, numbers, lists, dicts, sets, tuples |
| `03_conditionals/` | if / elif / else, match |
| `04_loops/` | for, while, walrus, dict comprehensions |
| `05_functions/` | Scopes, closures, lambdas, builtins |
| `06_chai_business/` | Modules, packages, project structure |
| `07_comprehensions/` | List, set, dict, generator comprehensions |
| `08_generators/` | yield, infinite generators, send/close |
| `09_decorators/` | Closures, logging, auth decorators |
| `10_oop/` | Classes, inheritance, MRO, static/class methods |
| `11_exceptions/` | try/except/finally, custom exceptions |
| `12_threads_concurrency/` | Threading, multiprocessing, GIL |
| `13_async_python/` | asyncio, race conditions, deadlocks |
| `14_pydantic/` | Data validation, serialisation |
| `challenges/` | 6 coding challenge sets (utilities → web → data science) |

---

### `02_math_for_ml` — Mathematics for Machine Learning
> **Time:** 3–5 weeks · **Prerequisite:** `01_python`

Coursera **Mathematics for Machine Learning** specialisation — graded assignments included.

| Sub-chapter | Content |
|-------------|---------|
| `01. Linear Algebra/` | Vectors, matrices, eigenvalues — 4 weeks |
| `02. Calculus/` | Derivatives, gradient descent — 3 weeks |
| `03. Probability and Statistics/` | Distributions, hypothesis testing — 4 weeks |

---

### `03_machine_learning` — Classical Machine Learning
> **Time:** 2–3 weeks · **Prerequisite:** `02_math_for_ml`

Hands-on scikit-learn notebooks progressing from simple to complex.

| Sub-chapter | Algorithms |
|-------------|-----------|
| `01_regression/` | Simple linear → Multiple linear → Polynomial → Nonlinear |
| `02_classification/` | Logistic regression → KNN → Decision trees → SVM → MNIST |
| `03_clustering/` | K-Means → Hierarchical → DBSCAN |
| `04_recommendation/` | Content-based → Collaborative filtering |

---

### `04_deep_learning` — Deep Learning
> **Time:** 4–6 weeks · **Prerequisite:** `03_machine_learning`

From a single neuron to multi-layer networks, with three major frameworks and Coursera assignments.

| Sub-chapter | Content |
|-------------|---------|
| `01_perceptron/` | Neuron model, AND/OR/XOR operators (6 notebooks) |
| `02_multilayer_perceptron/` | MLP, Iris, credit data, 3-layer network (5 notebooks) |
| `03_neural_network_libraries/` | PyBrain → sklearn → TensorFlow → PyTorch (10 notebooks) |
| `04_nn_from_scratch/` | Full neural network implementation without libraries |
| `05_coursera_assignments/` | Andrew Ng Deep Learning Specialisation (W2–W5) |

---

### `05_nlp` — Natural Language Processing
> **Time:** 1–2 weeks · **Prerequisite:** `03_machine_learning`

Classic NLP pipeline — from raw text to structured linguistic features.

| Sub-chapter | Content |
|-------------|---------|
| `01_preprocessing/` | Cleaning → Tokenisation → Stemming → Lemmatisation → N-grams |
| `02_tagging/` | POS tagging, Named Entity Recognition |
| `03_practice/` | Combined preprocessing + tagging practice |

---

### `06_large_language_models` — Large Language Models
> **Time:** 3–4 weeks · **Prerequisite:** `04_deep_learning`, `05_nlp`

Understand transformers, build an LLM from scratch, and fine-tune on custom data.

| Sub-chapter | Content |
|-------------|---------|
| `01_transformers_and_huggingface/` | Pipelines (text, image), PyTorch vs TensorFlow, fine-tuning |
| `02_llm_from_scratch/` | GPT-style LLM built with PyTorch — no shortcuts |
| `03_llm_fine_tuning/` | Data collection, training, evaluation pipeline |

---

### `07_rag` — Retrieval-Augmented Generation
> **Time:** 2–3 weeks · **Prerequisite:** `06_large_language_models`

Four progressively complex approaches to building RAG systems.

| Sub-chapter | Content |
|-------------|---------|
| `01_rag_from_scratch/` | RAG built from first principles (5 notebooks, parts 1–18) |
| `02_langchain_guide/` | Data ingestion → splitting → embedding → vector DB → chatbot |
| `03_llm_retrieval_qa/` | LLM-backed retrieval Q&A (FastAPI + LangChain) |
| `04_pure_retrieval_local/` | Local pure-retrieval without an LLM |

---

### `08_ai_apps` — AI Applications
> **Time:** As needed · **Prerequisite:** `07_rag`

13 runnable applications demonstrating real-world AI patterns.

| App | What it demonstrates |
|-----|---------------------|
| `01_hello_world/` | Gemini API — first steps |
| `02_tokenization/` | tiktoken tokenisation |
| `03_prompts/` | Zero-shot, few-shot, Chain-of-Thought, persona |
| `04_ollama/` | Local model inference with FastAPI |
| `05_langraph/` | LangGraph stateful chat + memory checkpoints |
| `06_rag/` | Full RAG app (FastAPI + Qdrant + Docker) |
| `07_rag_queue/` | Async RAG with Redis queue |
| `08_weather_agent/` | Tool-use / function-calling agent |
| `09_mem_agent/` | Redis-backed memory agent |
| `10_voice_agent/` | Voice agent with browser frontend |
| `11_huggingface_basic/` | HuggingFace image-text-to-text pipeline |
| `12_image/` | OpenAI vision / image captioning |
| `13_todo/` | Frontend todo app (HTML/JS) |

```bash
# Apps with Docker (06, 07, 05, 09)
cd 08_ai_apps/<app>
docker-compose up -d && python main.py

# Standalone apps
cd 08_ai_apps/<app>
python main.py
```

---

### `09_projects` — Capstone Projects
> **Time:** 2–4 weeks · **Prerequisite:** All chapters

Five end-to-end projects covering the full ML lifecycle.

| Project | Stack |
|---------|-------|
| `01_house_price_ml/` | Linear regression, polynomial, K-Means, hierarchical clustering |
| `02_sentiment_analysis/` | Amazon Alexa reviews — sklearn pipeline + custom tokeniser |
| `03_gmail_assistant/` | Gmail summariser with built-in and custom HuggingFace models |
| `04_made_with_ml/` | End-to-end MLOps — Ray, MLflow, Anyscale |
| `05_great_learning_certification/` | 8 AIML certification projects (stats → CV → NLP) |

---

### `reference/` — Comprehensive Reference
External notebook collection covering the full AI/ML spectrum. Use as a reference, not as a sequential lesson.

---

## ⚡ Quick Start

```bash
# 1. Read the setup guide
cat 00_setup/README.md

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Launch JupyterLab
jupyter lab

# 4. Start with Chapter 01
# Navigate to 01_python/ and open README.md
```

---

## 🔧 Environment

| Item | Value |
|------|-------|
| **Python** | 3.11.15 (Homebrew) |
| **Virtual env** | `venv/` at repo root |
| **Jupyter kernel** | `KernelSoubhik` |
| **Kernel location** | `~/Library/Jupyter/kernels/kernelsoubhik/` |

---

## 🔑 Required API Keys (`.env` at repo root)

```
OPENAI_API_KEY=
GOOGLE_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
LANGCHAIN_API_KEY=
GROQ_API_KEY=
```
