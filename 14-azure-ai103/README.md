# Chapter 14 — Azure AI-103: Building AI Solutions on Azure

> **53 course scripts, each paired with a beginner-to-advanced Jupyter notebook** — line-by-line explanations, prerequisites, Microsoft **AI-102** exam tips, and real alternative approaches for every file.

This chapter is a Udemy-style course ("Azure AI 103") on building production AI solutions on Azure: Azure OpenAI / AI Foundry agents, Azure AI Search + RAG, LangChain/LangGraph orchestration, image generation & editing, Content Safety, Azure AI Language, Azure AI Speech, and Azure AI Document Intelligence. The original `.py` scripts (companion to `Slides.pdf`) are untouched — every script now has a sibling `.ipynb` notebook that teaches it.

**AI-103 is now a real Microsoft exam**: *AI-103 — Developing AI Apps and Agents on Azure* (skills measured as of April 16, 2026), which replaced the retired AI-102 (retired June 30, 2026) and awards the **Azure AI Apps and Agents Developer Associate** certification. The notebooks' inline "exam tips" were written against AI-102 (the current exam at the time of conversion); the concepts carry over directly.

📝 **Exam prep:** [`EXAM_NOTES.md`](EXAM_NOTES.md) is a full AI-103 quick-revision guide: all 151 notebook exam tips condensed and reorganized under the official 5-domain AI-103 blueprint, plus every syllabus topic the course doesn't cover (Foundry IQ, Microsoft Agent Framework, A2A protocol, M365/Work IQ agent publishing, Voice Live, Sora 2 video generation, Language/Speech MCP servers, monitoring/security, responsible multimodal AI), with recap tables, mnemonics, and 55 flash facts. [`EXAM_PRACTICE_QUESTIONS.md`](EXAM_PRACTICE_QUESTIONS.md) adds 65 exam-style scenario questions (blueprint-weighted, with answers + explanations) for self-testing.

---

## Prerequisites

- **All Azure credentials, project details, and model deployment names for this entire chapter are centralized in [`azure_config.py`](azure_config.py).** Every `.py` script and `.ipynb` notebook — across all 8 Section Code folders and `Labs/` — reads its configuration through a single `config` object (`from azure_config import config`) instead of each file duplicating its own `load_dotenv()` + `os.getenv("...")` calls with a different variable name. Add a value **once** to the repo-root `.env` and every script/notebook that needs it finds it the same way. See the module's docstring and class for the full list of canonical env var names (`AZURE_AI_PROJECT_ENDPOINT`, `AZURE_AI_MODEL_DEPLOYMENT`, `AZURE_CONTENT_SAFETY_ENDPOINT`/`KEY`, `AZURE_LANGUAGE_ENDPOINT`/`KEY`, `AZURE_SPEECH_ENDPOINT`/`KEY`/`REGION`, `AZURE_CONTENT_UNDERSTANDING_ENDPOINT`/`KEY`, `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`/`KEY`, `AZURE_SEARCH_ENDPOINT`, and more). Each notebook's **Prerequisites** cell still lists exactly which of these it needs.
- The root `.env` already has `AZURE_AI_PROJECT_ENDPOINT` / `AZURE_AI_MODEL_DEPLOYMENT` (see `11_azure_ai_foundry/README.md`, whose Foundry resource this chapter reuses) plus the direct Azure OpenAI / image-generation / MCP values. Blank placeholders for the other services (Search, Content Safety, Language, Translator, Speech, Content Understanding, Document Intelligence, Voice Live) are already present in root `.env` — fill in only the ones for services you actually have Azure resources for.
- Several notebooks depend on Azure SDK packages **not yet in the root `requirements.txt`** (`azure-ai-textanalytics`, `azure-cognitiveservices-speech`, `azure-ai-transcription`, `azure-search-documents`, `azure-ai-contentsafety`, `azure-ai-documentintelligence`, `azure-functions`, `langgraph`, `langchain-azure-ai`, `faiss-cpu`, `pypdf`, `openai-agents`). Each notebook that needs one says so explicitly with the `pip3 install` line — install on demand rather than bulk-adding to root `requirements.txt`.
- Most notebooks call a **live Azure service** and cannot be executed end-to-end without real credentials in `.env`. They are correct, validated Python (every `.py` passes `py_compile`; every `.ipynb` cell passes `ast.parse`), written to run as-is once credentials are supplied — this repo does not currently have keys configured for every service used across the chapter.
- The Application Insights connection string previously hardcoded in plaintext in `04_agent_frameworks/01-langchain/lanchain_telemetry.py` has been removed and is now read via `config.appinsights_connection_string` (`AZURE_APPINSIGHTS_CONNECTION_STRING` in `.env`) — if that instrumentation key was ever real, rotate it in the Azure portal regardless, since it may still be present in older git history.

---

## Sections

| Section | Files | Topic |
|---|---|---|
| `01_responses_api_basics/` | 6 | Azure OpenAI Responses API basics — first call, model behavior, reasoning effort, multimodal input, web search tool, code interpreter tool |
| `02_foundry_agent_service/` | 12 | Azure AI Foundry Agent Service — prompt agents, function tools, IT helpdesk agent, Azure AI Search, RAG agent, two Azure Functions apps as agent tool backends |
| `03_conversations_and_evaluation/` | 2 | Foundry conversations API + LLM-as-judge response-completeness evaluation (companion no-code labs: CloudXeus agents, structured output, conditional workflow) |
| `04_agent_frameworks/` | 5 | Agent frameworks on Azure — LangChain, LangChain + OpenTelemetry tracing, LangGraph, an MCP-backed Azure Function |
| `05_image_generation_and_safety/` | 7 | Azure OpenAI image generation & editing (prompt edit, masked inpainting) + Content Safety (text/image moderation) |
| `06_language_and_speech/` | 13 | Azure AI Language (NER, PII, sentiment, language detection, translation) + Azure AI Speech (speech-to-text, text-to-speech, real-time & translated speech) — includes both LLM-prompted and dedicated-service approaches to the same tasks |
| `07_content_understanding/` | 3 | Document/content understanding agent — invoice analysis, image analysis, content agent |
| `08_document_intelligence_capstone/` | 5 | Azure AI Document Intelligence + Foundry agent via MCP, applied to invoice processing end-to-end |

Each `NN_name.py` has a sibling `NN_name.ipynb` in the same folder. Notebook cell order: **Title & difficulty → Prerequisites (packages / Azure resources / env vars) → What You'll Learn → annotated code (💡 exam tip, 🔄 alternatives per chunk) → Summary → Try It Yourself.**

---

## Hands-On Labs (`Labs/`)

A second, complementary set of materials from the same course — 21 lab exercise scripts (not notebooks) covering the same ground as the numbered Section Code folders (`01_responses_api_basics/` … `08_document_intelligence_capstone/`): Azure OpenAI chat basics, Foundry Agent Service (function calling, MCP tools, Microsoft Agent Framework), Azure AI Language, and Azure AI Speech (including a real-time voice agent). Unlike Section Code, a few of these are intentionally unfinished "fill in the blank" exercises and several expect local sample data not included in this repo. See [`Labs/README.md`](Labs/README.md) for a full beginner-friendly breakdown of every file.

---

## Previous Chapter

← [Chapter 13 — Agentic SDLC](../13_agentic_sdlc/)
