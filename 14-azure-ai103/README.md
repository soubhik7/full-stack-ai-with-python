# Chapter 14 — Azure AI-103: Building AI Solutions on Azure

> **53 course scripts, each paired with a beginner-to-advanced Jupyter notebook** — line-by-line explanations, prerequisites, Microsoft **AI-102** exam tips, and real alternative approaches for every file.

This chapter is a Udemy-style course ("Azure AI 103") on building production AI solutions on Azure: Azure OpenAI / AI Foundry agents, Azure AI Search + RAG, LangChain/LangGraph orchestration, image generation & editing, Content Safety, Azure AI Language, Azure AI Speech, and Azure AI Document Intelligence. The original `.py` scripts (companion to `Slides.pdf`) are untouched — every script now has a sibling `.ipynb` notebook that teaches it.

**AI-103 is now a real Microsoft exam**: *AI-103 — Developing AI Apps and Agents on Azure* (skills measured as of April 16, 2026), which replaced the retired AI-102 (retired June 30, 2026) and awards the **Azure AI Apps and Agents Developer Associate** certification. The notebooks' inline "exam tips" were written against AI-102 (the current exam at the time of conversion); the concepts carry over directly.

📝 **Exam prep:** [`EXAM_NOTES.md`](EXAM_NOTES.md) is a full AI-103 quick-revision guide: all 151 notebook exam tips condensed and reorganized under the official 5-domain AI-103 blueprint, plus every syllabus topic the course doesn't cover (Foundry IQ, Microsoft Agent Framework, A2A protocol, M365/Work IQ agent publishing, Voice Live, Sora 2 video generation, Language/Speech MCP servers, monitoring/security, responsible multimodal AI), with recap tables, mnemonics, and 55 flash facts. [`EXAM_PRACTICE_QUESTIONS.md`](EXAM_PRACTICE_QUESTIONS.md) adds 65 exam-style scenario questions (blueprint-weighted, with answers + explanations) for self-testing.

---

## Prerequisites

- `.env` at the repo root needs, at minimum, the existing `AZURE_AI_PROJECT_ENDPOINT` / `AZURE_AI_MODEL_DEPLOYMENT` (see `11_azure_ai_foundry/README.md`). Individual notebooks add their own service-specific variables (e.g. `AZURE_LANGUAGE_KEY`, `AZURE_SPEECH_KEY`, `AZURE_SEARCH_ENDPOINT`, `AZURE_CONTENTSAFETY_KEY`, `AZURE_DOCUMENTINTELLIGENCE_KEY`) — each notebook's **Prerequisites** cell lists exactly which ones it needs.
- Several notebooks depend on Azure SDK packages **not yet in the root `requirements.txt`** (`azure-ai-textanalytics`, `azure-cognitiveservices-speech`, `azure-ai-transcription`, `azure-search-documents`, `azure-ai-contentsafety`, `azure-ai-documentintelligence`, `azure-functions`, `langgraph`, `langchain-azure-ai`, `faiss-cpu`, `pypdf`, `openai-agents`). Each notebook that needs one says so explicitly with the `pip3 install` line — install on demand rather than bulk-adding to root `requirements.txt`.
- Most notebooks call a **live Azure service** and cannot be executed end-to-end without real credentials in `.env`. They are correct, validated Python (structure + syntax checked), written to run as-is once credentials are supplied — this repo does not currently have keys configured for every service used across the chapter.
- ⚠️ **`04. Section Code/01-langchain/lanchain_telemetry.py`** has a real-looking Application Insights connection string hardcoded in plaintext (already committed to git history prior to this chapter's notebook conversion). If that resource is real, rotate/regenerate the instrumentation key in the Azure portal. The companion notebook reads it from `AZURE_APPINSIGHTS_CONNECTION_STRING` in `.env` instead.

---

## Sections

| Section | Files | Topic |
|---|---|---|
| `01. Section Code/` | 6 | Azure OpenAI Responses API basics — first call, model behavior, reasoning effort, multimodal input, web search tool, code interpreter tool |
| `02. Section Code/` | 12 | Azure AI Foundry Agent Service — prompt agents, function tools, IT helpdesk agent, Azure AI Search, RAG agent, two Azure Functions apps as agent tool backends |
| `03. Section Code/` | 2 | Foundry conversations API + LLM-as-judge response-completeness evaluation (companion no-code labs: CloudXeus agents, structured output, conditional workflow) |
| `04. Section Code/` | 5 | Agent frameworks on Azure — LangChain, LangChain + OpenTelemetry tracing, LangGraph, an MCP-backed Azure Function |
| `05. Section Code/` | 7 | Azure OpenAI image generation & editing (prompt edit, masked inpainting) + Content Safety (text/image moderation) |
| `06. Section Code/` | 13 | Azure AI Language (NER, PII, sentiment, language detection, translation) + Azure AI Speech (speech-to-text, text-to-speech, real-time & translated speech) — includes both LLM-prompted and dedicated-service approaches to the same tasks |
| `07. Section Code/` | 3 | Document/content understanding agent — invoice analysis, image analysis, content agent |
| `08. Section Code/` | 5 | Azure AI Document Intelligence + Foundry agent via MCP, applied to invoice processing end-to-end |

Each `NN_name.py` has a sibling `NN_name.ipynb` in the same folder. Notebook cell order: **Title & difficulty → Prerequisites (packages / Azure resources / env vars) → What You'll Learn → annotated code (💡 exam tip, 🔄 alternatives per chunk) → Summary → Try It Yourself.**

---

## Previous Chapter

← [Chapter 13 — Agentic SDLC](../13_agentic_sdlc/)
