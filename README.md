# Full-Stack AI with Python

A personal learning and project repository covering the full AI/ML spectrum —
from Python fundamentals and classical machine learning through deep learning,
NLP, LLMs, RAG systems, and agentic AI.

---

## Repository Layout

```
full-stack-ai-with-python/
│
├── 01_python/                    Python fundamentals curriculum (chai-shop themed, 00–14)
│
├── research/                     Learning notebooks, coursework & exploration
│   ├── machine_learning/
│   │   ├── basics/               scikit-learn hands-on (classification, clustering, regression)
│   │   └── classification/       MNIST classification notebooks
│   ├── deep_learning/
│   │   ├── 01_perceptron/        Single-layer perceptron from scratch
│   │   ├── 02_multilayer_perceptron/
│   │   ├── 03_libraries/         sklearn / TensorFlow / PyTorch comparisons
│   │   ├── coursera/             Coursera Deep Learning Specialisation assignments
│   │   └── nn_from_scratch/      Neural network implementation from scratch
│   ├── nlp/
│   │   ├── 01_preprocessing/     Cleaning, tokenisation, stemming, lemmatisation
│   │   ├── 02_tagging/           POS tagging, NER
│   │   └── practice/             Combined NLP practice notebooks
│   ├── llm/
│   │   ├── 01_llm_from_scratch_pytorch/   GPT-style LLM built with PyTorch
│   │   └── 02_llm_model/         Fine-tuned model artefacts
│   ├── rag/
│   │   ├── 01_rag_from_scratch/  RAG built from first principles
│   │   ├── 02_langchain_guide/   Step-by-step LangChain RAG implementation
│   │   ├── 03_llm_retrieval_qa/  LLM-model-based retrieval Q&A
│   │   └── 04_pure_retrieval_local/ Local pure-retrieval Q&A
│   ├── huggingface/              HuggingFace pipelines & fine-tuning notebooks
│   ├── coursework/
│   │   ├── great_learning/       8 Great Learning AIML certification projects
│   │   └── math_for_ml/          Coursera Math for ML (linear algebra, calculus, probability)
│   └── comprehensive/            Comprehensive AI/ML reference notebooks (external — kept intact)
│
├── projects/                     Self-directed end-to-end ML projects
│   ├── house_price_ml/
│   │   ├── 01_linear_regression/
│   │   ├── 02_nonlinear_regression/
│   │   ├── 03_kmeans/
│   │   └── 04_hierarchical/
│   ├── gmail_assistant/
│   │   ├── builtin_model/        Gmail summariser using built-in HuggingFace model
│   │   └── custom_model/         Gmail summariser with custom-trained model
│   ├── made_with_ml/             End-to-end MLOps project (Ray, MLflow, Anyscale)
│   └── sentiment_analysis/       Amazon Alexa reviews sentiment classifier
│
├── apps/                         Runnable applications & demos
│   ├── hello_world/              Gemini API examples
│   ├── langraph/                 LangGraph chat + memory checkpoint demos
│   ├── rag/                      RAG app (FastAPI + Qdrant + Docker)
│   ├── rag_queue/                RAG with Redis queue (async processing)
│   ├── voice_agent/              Voice agent with browser frontend
│   ├── weather_agent/            Weather agent using tool-use
│   ├── mem_agent/                Memory agent (Redis-backed)
│   ├── ollama/                   Ollama local-model FastAPI server
│   ├── prompts/                  Prompt engineering examples (zero/few-shot, CoT, persona)
│   ├── tokenization/             tiktoken tokenisation demo
│   ├── huggingface_basic/        HuggingFace image-text-to-text pipeline
│   ├── image/                    OpenAI vision / image captioning
│   └── todo/                     Frontend todo app (HTML/JS)
│
└── tools/                        Developer utilities (notebook maintenance, SSL fix, etc.)
```

---

## Quick Start

```bash
# 1. Activate the project virtual environment
source venv/bin/activate          # or: venv\Scripts\activate on Windows

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Copy and populate environment variables
cp .env .env.local                # edit with your API keys

# 4. Launch JupyterLab and select "KernelSoubhik"
jupyter lab
```

### Running an App

Most apps in `apps/` with a `docker-compose.yml` need Docker first:

```bash
cd apps/<app_name>
docker-compose up -d
python main.py        # or server.py
```

---

## Environment

| Item | Value |
|------|-------|
| **Python** | 3.11.15 (Homebrew) |
| **Kernel name** | `KernelSoubhik` |
| **Kernel location** | `~/Library/Jupyter/kernels/kernelsoubhik` |
| **venv** | `venv/` at repo root |

---

## Required API Keys (`.env`)

```
OPENAI_API_KEY=
GOOGLE_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
LANGCHAIN_API_KEY=
GROQ_API_KEY=
```
