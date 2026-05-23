# Chapter 07 — Retrieval-Augmented Generation (RAG)

> **You will build RAG systems four ways — from pure scratch to production-ready — so you understand what every framework is doing under the hood.**

---

## Prerequisites

- [Chapter 06 — Large Language Models](../06_large_language_models/README.md)
- Packages: `langchain`, `langchain-openai`, `langchain-community`, `qdrant-client`, `faiss-cpu`, `pypdf`, `sentence-transformers`
- (Optional) Docker — for `03_llm_retrieval_qa/` and `04_pure_retrieval_local/`

---

## What is RAG?

RAG = Retrieval + Generation. Instead of relying solely on a model's training data, you:

1. **Ingest** documents → split into chunks → embed into vectors
2. **Retrieve** the most relevant chunks for a given query
3. **Generate** a grounded answer by passing chunks + query to an LLM

```
User query
  → Embed query
  → Search vector store  →  Top-k chunks
  → [chunks + query]  →  LLM  →  Answer
```

---

## Learning Path

Work through the approaches **in order** — each one adds complexity and abstraction.

```
01_rag_from_scratch/     ← No framework. Pure Python + NumPy.
02_langchain_guide/      ← LangChain building blocks, step by step
03_llm_retrieval_qa/     ← Production RAG app (FastAPI + LangChain)
04_pure_retrieval_local/ ← Retrieval only, no LLM (local embeddings)
```

---

## Sub-chapter Breakdown

### `01_rag_from_scratch/` — First Principles

Build every component manually so you understand what LangChain abstracts away.

| Notebook | RAG parts covered |
|----------|------------------|
| `rag_from_scratch_1_to_4.ipynb` | Chunking, embedding, cosine similarity, vector search |
| `rag_from_scratch_5_to_9.ipynb` | Retrieval strategies — multi-query, RAG fusion, decomposition |
| `rag_from_scratch_10_and_11.ipynb` | Step-back prompting, HyDE |
| `rag_from_scratch_12_to_14.ipynb` | Re-ranking, CRAG (corrective RAG) |
| `rag_from_scratch_15_to_18.ipynb` | Self-RAG, adaptive retrieval |

---

### `02_langchain_guide/` — LangChain Step by Step

A nine-step tutorial building a full conversational RAG chatbot.

| # | File | Topic |
|---|------|-------|
| 01 | `01_data_ingestion.ipynb` | `PyPDFLoader`, `WebBaseLoader`, `DirectoryLoader` |
| 02 | `02_text_splitting.ipynb` | `RecursiveCharacterTextSplitter`, chunk size/overlap |
| 03 | `03_embedding.ipynb` | OpenAI and HuggingFace embeddings |
| 04 | `04_vector_db.ipynb` | FAISS and Qdrant vector stores |
| 05 | `05_langchain_openai.ipynb` | LCEL chains: retriever + prompt + LLM |
| 05.1 | `05_1_langchain_ollama_app.py` | Same chain with local Ollama model |
| 06 | `06_lcel_groq.ipynb` | LCEL with Groq (fast inference) |
| 06.1 | `06_1_langsmith_fastapi_groq/` | LangServe + FastAPI deployment |
| 07 | `07_1_vector_store_retrievers.ipynb` | Retriever types and configuration |
| 07 | `07_chatbot_message_history.ipynb` | `RunnableWithMessageHistory` |
| 08 | `08_conversation_qa_chatbot_history.ipynb` | Full conversational RAG with history |

---

### `03_llm_retrieval_qa/` — Production App

A modular FastAPI application with clean separation of concerns.

```
app/
├── document_loader.py   ← Load and chunk documents
├── vector_store.py      ← Qdrant client wrapper
├── rag_chain.py         ← LangChain retrieval chain
└── main.py              ← FastAPI routes
```

---

### `04_pure_retrieval_local/` — No LLM Required

Semantic search without an LLM — useful when you just need to find relevant documents.

```
app/
├── document_loader.py   ← Load documents
├── vector_store.py      ← FAISS index
├── retriever.py         ← Similarity search
└── main.py              ← FastAPI routes
```

---

## Key Concepts Glossary

| Term | Meaning |
|------|---------|
| **Chunk** | A fixed-size piece of a document |
| **Embedding** | Vector representation of text |
| **Vector store** | Database optimised for similarity search |
| **Retriever** | Component that finds relevant chunks for a query |
| **LCEL** | LangChain Expression Language — composable chain syntax |
| **HyDE** | Hypothetical Document Embeddings — generate a fake answer, embed it, retrieve with that |
| **CRAG** | Corrective RAG — verify retrieved docs before generating |

---

## Next Step

Head to **[Chapter 08 — AI Applications](../08_ai_apps/README.md)** →
