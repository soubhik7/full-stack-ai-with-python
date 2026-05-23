# Chapter 08 — AI Applications

> **You will run 13 progressively complex AI applications — from a "Hello World" API call to a full voice agent with browser frontend.**

---

## Prerequisites

- [Chapter 07 — RAG](../07_rag/README.md)
- API keys in `.env` (see `00_setup/README.md`)
- Docker Desktop (for apps 05, 06, 07, 09)

---

## Applications (01 → 13)

Work through them in order — each app introduces a new pattern or capability.

---

### `01_hello_world/` — Your First API Call
**Pattern:** Direct LLM invocation

```bash
cd 08_ai_apps/01_hello_world
python main.py
```

| File | What it shows |
|------|--------------|
| `main.py` | Gemini API — simple text generation |
| `gemini_hello.py` | Gemini `GenerativeModel` |
| `gemini_openai.py` | Same Gemini model via OpenAI-compatible API |

---

### `02_tokenization/` — How Text Becomes Numbers
**Pattern:** Tokenisation / encoding

```bash
python 08_ai_apps/02_tokenization/main.py
```

Uses `tiktoken` to show BPE tokenisation — explains why "ChatGPT" ≠ one token.

---

### `03_prompts/` — Prompt Engineering Patterns
**Pattern:** Prompt techniques

| File | Pattern |
|------|---------|
| `zero.py` | Zero-shot prompting |
| `few.py` | Few-shot with examples |
| `cot.py` | Chain-of-Thought reasoning |
| `persona.py` | Persona / role assignment |

---

### `04_ollama/` — Local Models
**Pattern:** Local inference (no API key needed)

Requires [Ollama](https://ollama.com) installed and a model pulled (`ollama pull llama3.2`).

```bash
python 08_ai_apps/04_ollama/server.py
```

---

### `05_langraph/` — Stateful Conversations
**Pattern:** LangGraph stateful agent · **Requires Docker**

```bash
cd 08_ai_apps/05_langraph
docker-compose up -d
python chat.py
```

| File | What it shows |
|------|--------------|
| `chat.py` | Basic LangGraph chat |
| `chat_2.py` | Multi-turn with tool use |
| `chat_checkpoint.py` | Persistent memory across sessions |

---

### `06_rag/` — Full RAG Application
**Pattern:** Production RAG · **Requires Docker (Qdrant)**

```bash
cd 08_ai_apps/06_rag
docker-compose up -d
python index.py    # ingest documents
python chat.py     # query the index
```

---

### `07_rag_queue/` — Async RAG with Redis
**Pattern:** Async processing queue · **Requires Docker (Redis)**

```bash
cd 08_ai_apps/07_rag_queue
docker-compose up -d
python server.py &
python main.py
```

Decouples document ingestion from query handling via a Redis job queue.

---

### `08_weather_agent/` — Tool-Use Agent
**Pattern:** Function calling / tools

```bash
python 08_ai_apps/08_weather_agent/main.py
```

The agent decides when to call a `get_weather(city)` tool and when to answer directly.

---

### `09_mem_agent/` — Memory Agent
**Pattern:** Redis-backed persistent memory · **Requires Docker (Redis)**

```bash
cd 08_ai_apps/09_mem_agent
docker-compose up -d
python mem.py
```

The agent remembers conversation context across sessions using Redis.

---

### `10_voice_agent/` — Voice Interface
**Pattern:** Speech-to-text → LLM → text-to-speech

```bash
python 08_ai_apps/10_voice_agent/main.py
```

Browser-based frontend (`cusor.py` serves the UI). Speak to an AI and hear its response.

---

### `11_huggingface_basic/` — HuggingFace Inference
**Pattern:** Image-text-to-text pipeline

```bash
python 08_ai_apps/11_huggingface_basic/main.py
```

Uses a vision-language model to answer questions about images — no API key required.

---

### `12_image/` — Vision & Image Captioning
**Pattern:** OpenAI GPT-4 Vision

```bash
python 08_ai_apps/12_image/main.py
```

Send an image URL → get a detailed text description.

---

### `13_todo/` — Frontend + AI Backend
**Pattern:** HTML/JS frontend, Python backend

A simple todo app demonstrating how a classic web app is structured alongside AI components.

---

## Environment Summary

| App | Docker service | Env vars needed |
|-----|---------------|-----------------|
| 01 | — | `GOOGLE_API_KEY` |
| 02 | — | — |
| 03 | — | `OPENAI_API_KEY` or `GOOGLE_API_KEY` |
| 04 | — | — (Ollama local) |
| 05 | Redis | `OPENAI_API_KEY` |
| 06 | Qdrant | `OPENAI_API_KEY` |
| 07 | Redis + Qdrant | `OPENAI_API_KEY` |
| 08 | — | `OPENAI_API_KEY` |
| 09 | Redis | `OPENAI_API_KEY` |
| 10 | — | `OPENAI_API_KEY` |
| 11 | — | — (HuggingFace local) |
| 12 | — | `OPENAI_API_KEY` |
| 13 | — | — |

---

## Next Step

Head to **[Chapter 09 — Projects](../09_projects/README.md)** →
