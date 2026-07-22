# 🧠 Step-by-Step RAG Implementation Guide with LangChain

This repository presents a comprehensive, modular walkthrough of building a **Retrieval-Augmented Generation (RAG)** system using **LangChain**, supporting various LLM backends (OpenAI, Groq, Ollama) and embedding/vector DB options.

Each stage of the pipeline is separated into its own notebook or app file — making this a great educational resource or boilerplate for production RAG systems.

---

## 📁 Project Structure

| Stage | File | Description |
|-------|------|-------------|
| 01 | `01_DataIngestion.ipynb` | Ingest data from multiple sources (PDFs, text files, web, Wikipedia, Arxiv) |
| 02 | `02_Text_Splitting.ipynb` | Split raw content using various splitter strategies |
| 03 | `03_Embedding.ipynb` | Generate embeddings using OpenAI, Ollama, or HuggingFace |
| 04 | `04_VectorDB.ipynb` | Store and retrieve documents using FAISS or Chroma |
| 05 | `05_Langchain_OpenAI.ipynb` | RAG using OpenAI models |
| 05.1 | `05.1_Langchain_Ollama_app.py` | Streamlit app using local Ollama LLM |
| 06 | `06_LCEL_Groq.ipynb` | RAG with LCEL chaining and Groq-hosted models |
| 07 | `07_Chatbot_MessageHistory.ipynb` | Add persistent memory across chat turns |
| 07.1 | `07.1VectorStore&Retrivers.ipynb` | Custom retriever configurations |
| 08 | `08_ConversationQAchatBotHist.ipynb` | Final RAG chatbot with full conversational capability |

---

## 🧭 Step-by-Step RAG Pipeline

### 🔹 Step 1: Data Ingestion
**File**: `01_DataIngestion.ipynb`

Supported loaders:
- 📄 `TextLoader`: Load `.txt` or `.md` files.
- 📕 `PyPDFLoader`: Load academic papers or documents.
- 🌐 `WebBaseLoader`: Extract content from URLs.
- 📰 `ArxivLoader`: Retrieve scientific abstracts from arXiv.
- 📚 `WikipediaLoader`: Pull articles directly from Wikipedia.

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, ArxivLoader, WikipediaLoader
```

---

### 🔹 Step 2: Text Splitting
**File**: `02_Text_Splitting.ipynb`

Splitters used:
- 🪓 `RecursiveCharacterTextSplitter`: Ideal for structured documents.
- ✂️ `CharacterTextSplitter`: Basic splitting by characters.
- 🧩 `HTMLHeaderTextSplitter`: Preserve sections in HTML/XML files.
- 🔄 `RecursiveJsonSplitter`: Token-aware JSON content splitter.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, HTMLHeaderTextSplitter, RecursiveJsonSplitter
```

---

### 🔹 Step 3: Embedding Generation
**File**: `03_Embedding.ipynb`

Supported embedding backends:
- 🔐 `OpenAIEmbeddings` – Use with OpenAI models like `text-embedding-3-small`.
- 🧠 `HuggingFaceEmbeddings` – Open-source models like `all-MiniLM-L6-v2`.
- 🖥️ `OllamaEmbeddings` – Use locally hosted models via Ollama.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
```

---

### 🔹 Step 4: Vector Store Creation
**File**: `04_VectorDB.ipynb`

Supported vector databases:
- 🧭 `FAISS`: Fast and efficient for local use.
- 📦 `Chroma`: More flexible; supports metadata filtering.

```python
from langchain_community.vectorstores import FAISS, Chroma
```

---

### 🔹 Step 5: QA over Vector Context (OpenAI)
**File**: `05_Langchain_OpenAI.ipynb`

- Uses `ChatOpenAI` for question answering.
- Combines prompt + retrieval + model in a LangChain chain.

---

### 🔹 Step 5.1: Ollama Local Chatbot
**File**: `05.1_Langchain_Ollama_app.py`

- Streamlit app using `gemma:2b` or other Ollama models.
- Fully local inference pipeline with custom prompts.

```bash
streamlit run 05.1_Langchain_Ollama_app.py
```

---

### 🔹 Step 6: LCEL RAG with Groq
**File**: `06_LCEL_Groq.ipynb`

- Uses `ChatGroq` with Llama3 or Mixtral.
- Chain is constructed using LCEL syntax: `prompt | llm | parser`.

```python
from langchain_groq import ChatGroq
llm = ChatGroq(model_name="llama3-8b-8192")
```

---

### 🔹 Step 7: Add Memory to Chatbot
**File**: `07_Chatbot_MessageHistory.ipynb`

- Implements session-based memory using:
  - `ChatMessageHistory`
  - `RunnableWithMessageHistory`

---

### 🔹 Step 7.1: Advanced Retriever Config
**File**: `07.1VectorStore&Retrivers.ipynb`

- Custom retrievers using `as_retriever()`, search filters, and top-k tweaking.

---

### 🔹 Step 8: Full Conversational RAG
**File**: `08_ConversationQAchatBotHist.ipynb`

- Combines everything:
  - Message history
  - Contextual question rewriting
  - Retrieval + prompt + LLM
  - Final QA delivery

---

## ⚙️ Tech Stack

| Component | Options |
|----------|---------|
| Loaders | Text, PDF, Web, Wikipedia, Arxiv |
| Splitters | RecursiveChar, Char, HTML, JSON |
| Embeddings | OpenAI, HuggingFace, Ollama |
| Vector DB | FAISS, Chroma |
| LLMs | OpenAI, Groq (via `ChatGroq`), Ollama |
| Chain Framework | LangChain, LCEL |
| UI | Streamlit |

---

## 🚀 Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/langchain-rag-pipeline.git
   cd langchain-rag-pipeline
   ```

2. Install requirements:
   ```bash
   pip3 install -r requirements.rag_langchain.txt
   ```

3. Add `.env` file:
   ```env
   OPENAI_API_KEY=your-key
   Groq_API_KEY=your-key
   LANGCHAIN_API_KEY=your-key
   ```

4. Launch:
   ```bash
   streamlit run 05.1_Langchain_Ollama_app.py
   ```

---

## 📈 Future Enhancements

- 🧩 Add hybrid retrievers (keyword + vector)
- 📤 Upload interface for web PDF ingestion
- 🧠 Multi-document summarization
- 🧪 Response scoring and evaluation UI

---
