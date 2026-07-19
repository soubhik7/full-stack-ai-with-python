# 🎯 AI-103 Exam Short Notes — Developing AI Apps and Agents on Azure

> **AI-103 is the real, current exam** (skills measured as of **April 16, 2026**) — it replaced the retired AI-102 (retired June 30, 2026) and awards the certification **Microsoft Certified: Azure AI Apps and Agents Developer Associate**. Pass mark: **700/1000**. Mostly GA features; commonly-used Preview features may appear. Official companion course: **AI-103T00 "Develop AI apps and agents on Azure"** (4 days, 4 learning paths, 30 modules).
>
> These notes condense all 8 sections of this course chapter (53 scripts + notebooks + labs) **plus** every official AI-103 syllabus topic and course-syllabus module, organized by the real exam domains. Read top-to-bottom once, then use the recap tables for last-day revision.

### Official exam blueprint

| # | Domain | Weight |
|---|---|---|
| 1 | Plan and manage an Azure AI solution | **25–30%** |
| 2 | Implement generative AI and agentic solutions | **30–35%** ⭐ biggest |
| 3 | Implement computer vision solutions | 10–15% |
| 4 | Implement text analysis solutions | 10–15% |
| 5 | Implement information extraction solutions | 10–15% |

**🧠 Weight hook:** *"Plan a quarter, generate a third, see-say-extract a sixth each."* Domains 1+2 ≈ 60% of the exam — agents and Foundry management are where you win.

### Official course AI-103T00 syllabus map (4 learning paths)

| Learning path | Modules | Maps to domain |
|---|---|---|
| **Develop generative AI apps in Azure** (AI-3016) | Plan/prepare AI development → model catalog: select, deploy, evaluate → chat app with Foundry SDK + Responses API → apps that use tools → optimize (prompt eng / RAG / fine-tune) → responsible generative AI | 1, 2 |
| **Develop AI agents on Azure** (AI-3026) | Foundry Agent Service + VS Code → custom tools → MCP tools → **Foundry IQ** knowledge → **Microsoft 365 / Work IQ integration** → agent workflows → **Microsoft Agent Framework** → multi-agent orchestration → **A2A protocol** | 2 |
| **Develop natural language solutions in Azure** | Analyze text with Azure Language → text-analysis agent with **Language MCP server** → speech-capable generative AI app → speech-enabled apps (STT/TTS) → speech agent with **Speech MCP server** → **Voice Live agent** → translate text & speech | 4 |
| **Extract insights from visual data on Azure** (AI-3008) | Vision-enabled generative AI app → **image generation** → **video generation (Sora 2)** → analyze images with **Content Understanding** → multimodal analysis solution → Content Understanding client app/API → **Document Intelligence** → knowledge mining with **Azure AI Search** | 3, 5 |

> Terminology note: Microsoft renamed things — **"Microsoft Foundry"** = Azure AI Foundry; **"Foundry Tools"** = the Azure AI services (Language, Speech, Vision, Translator, Document Intelligence, Content Understanding…); **"Foundry Models"** includes Azure OpenAI models. Expect both old and new names in questions.

---

# DOMAIN 1 — Plan & Manage an Azure AI Solution (25–30%)

## 1. Choosing services and models

| Need | Pick |
|---|---|
| General reasoning/chat/generation | **LLM** (GPT-family) |
| Low cost / low latency / edge | **Small language model (SLM)** (e.g. Phi family) |
| Code generation / completion | **Code model** (code-optimized deployments, e.g. Codex-class) |
| Images/audio/video input or output | **Multimodal model** |
| Purpose-built skill (OCR, translation, speech, PII…) | **Foundry Tools** (dedicated Azure AI service) |
| Generative tasks / grounding / vector search / agent workflows | Foundry Models / Foundry IQ + AI Search / Agent Service |

- Model selection in the **model catalog**: compare by **benchmarks**, cost, context window, modality, region availability; evaluate manually (playground) and with automated evaluations before committing.
- Retrieval/indexing choice: keyword vs vector vs hybrid vs semantic (see Domain 5).
- Agent solutions additionally need: **memory** (threads/conversations), **tools** (function/OpenAPI/MCP), **knowledge** (Foundry IQ, AI Search, File Search).

### Scenario → service cheat sheet

| Scenario keyword | Pick |
|---|---|
| "chat / generate text or code" | Foundry Models (Azure OpenAI) |
| "ground answers in company data" | RAG: Foundry IQ / Azure AI Search + model |
| "agent that takes actions / calls tools" | Foundry Agent Service |
| "complex / multi-agent code-first orchestration" | Microsoft Agent Framework |
| "agents discover & talk to each other across systems" | A2A protocol |
| "agent inside Teams / Microsoft 365 Copilot" | M365 integration + Work IQ |
| "caption / describe / Q&A over an image" | Multimodal model (vision-enabled chat) |
| "create image from text" | Image generation model (gpt-image / DALL-E) |
| "create video from text" | **Sora 2** video generation |
| "schema-driven extraction from docs/images/audio/video" | Content Understanding |
| "invoice / receipt / ID field extraction" | Document Intelligence prebuilt |
| "search over millions of docs, facets, filters" | Azure AI Search |
| "sentiment / NER / PII / summarize" | Azure Language (or LLM prompting — know the trade-off) |
| "translate text/documents" | Azure Translator (documents keep formatting) |
| "transcribe / synthesize / translate speech" | Azure Speech |
| "live voice conversation with an agent" | **Voice Live API** |
| "moderate content with thresholds" | Content Safety |
| "block jailbreaks / hidden prompt attacks" | Prompt Shields |

## 2. Setting up — infrastructure, deployment, CI/CD

- **Hub → project** structure (newer: Foundry-resource-based projects). Project = where you build; connections wire in external resources (search, storage, MCP servers) by name — the credential/endpoint indirection layer.
- Endpoint format: `https://<resource>.cognitiveservices.azure.com/` or `<resource>.services.ai.azure.com`; `AIProjectClient` is the project-level SDK entry; `get_openai_client()` bridges to the OpenAI-compatible surface.

### Model deployment types (frequent question)

| Type | Billing | Use when |
|---|---|---|
| **Standard** | Pay-per-token, regional | Default; variable workloads |
| **Global Standard** | Pay-per-token, global routing | Throughput/availability, no residency constraint |
| **Global Batch** | ~50% cheaper, async (24 h) | Large offline jobs |
| **Provisioned (PTU)** | Reserved throughput units | Predictable latency, steady high volume |
| **Serverless / MaaS** | Pay-per-token, no infra | Catalog models (Llama, Mistral…) |

- **CI/CD:** provision via Bicep/ARM/Terraform; secrets from Key Vault or keyless auth; automated evaluations as pipeline quality gates before promoting a model/agent version; pin agent **versions** for reproducibility.
- Multi-service resource (one endpoint/key, many services) vs single-service (isolation + **F0 free tier**). Containers exist for edge/on-prem: run with **`Eula=accept`, `Billing=<uri>`, `ApiKey=<key>`** — billing telemetry goes to Azure, **your data doesn't**.

## 3. Authentication & security (asked on EVERY exam)

| | API Key | Microsoft Entra ID (keyless) |
|---|---|---|
| Mechanism | `api-key` / `Ocp-Apim-Subscription-Key` header, `AzureKeyCredential` | OAuth bearer tokens via `DefaultAzureCredential` |
| Secret | Long-lived shared secret | Short-lived scoped tokens, **no stored secret** |
| Access control | Anyone with the key | **RBAC** role assignments |
| Verdict | Quick dev/test | **Production standard** |

- Credential classes: `DefaultAzureCredential` (tries a chain — env vars → managed identity → CLI…), `AzureCliCredential` (local dev, fails fast), `ClientSecretCredential` (service principal), **managed identity** (Azure-hosted, no secret at all).
- Token scope for Foundry: `https://ai.azure.com/.default` via `get_bearer_token_provider`.
- **RBAC least privilege:** *Azure AI User / Developer* can **call** agents; managing needs broader roles; multi-resource apps (agent + search) need roles **per resource**.
- Two keys per resource → zero-downtime rotation; store keys in **Key Vault**; prefer **keyless (Entra ID)** everywhere it's supported.
- **Private networking:** private endpoints / VNet integration keep traffic off the public internet — pair with managed identity for the exam's "most secure" answer.
- Resilient pattern: *key if present, else `DefaultAzureCredential`* — same code local and in production.

**🧠 Memory hook:** *"Key = quick, Entra = enterprise, keyless + private = production."*

## 4. Quotas, monitoring, and cost

- Azure OpenAI quota = **TPM (tokens per minute)** per deployment (RPM scales with it); handle 429s with retry/backoff; scale = raise quota, add deployments, or move to PTU.
- **Azure Monitor metrics** (calls, errors, latency, tokens) → alerts; **diagnostic settings** stream logs to **Log Analytics** (KQL) / Event Hub / Storage.
- AI-103 adds AI-specific monitoring: **model performance & drift**, **safety events** (filter triggers, jailbreak attempts), **grounding quality**, plus data-side health: **ingestion quality, search index health, relevance performance**.
- Cost management: pricing tier + per-token/per-transaction; batch for bulk, SLMs where they suffice, cache/reuse, right-size deployments.

## 5. Responsible AI (both a Domain-1 objective and a theme everywhere)

### The six Microsoft principles — memorize

**F**airness, **R**eliability & safety, **P**rivacy & security, **I**nclusiveness, **T**ransparency, **A**ccountability.
**🧠 Mnemonic: F-R-P-I-T-A — "Fair Robots Protect Individuals Through Accountability."**

### Harm-prevention toolbox

| Layer | What it does |
|---|---|
| **Content filters** | Per-deployment severity thresholds on the 4 harm categories; jailbreak + protected-material detections included |
| **Content Safety service** | Own resource; granular severity scores: **Hate, SelfHarm, Sexual, Violence** (🧠 *HSSV*), scale **0/2/4/6** (0–7 via `output_type`); custom **blocklists (text-only)**; `analyze_text`/`analyze_image` are **synchronous** |
| **Prompt Shields** | **Jailbreak** (direct) + **indirect** prompt-injection attacks (instructions hidden in documents **or images**) |
| **Groundedness detection** | Flags claims unsupported by the provided sources |
| **Evaluators** | `azure-ai-evaluation` / portal: Completeness, Groundedness, Relevance, safety evaluations — run pre-deploy and continuously |
| **Explanation tooling** | Transparency in practice: surface *why* an answer was produced — citations/grounding sources, trace visualizations, evaluator score breakdowns — so outputs are explainable to users and auditors |
| **Auditing** | **Trace logging, provenance metadata, approval workflows** — who/what/why for every agent action |
| **Agent governance** | **Oversight modes** (autonomous ↔ human-approval), constraints, **tool-access controls** (`allowed_tools`, `require_approval`) |

- The classic gate pattern: run Content Safety on user input, **block severity ≥ 4 (Medium+) before it reaches the model**.
- Trace **metadata** vs trace **content** are separable settings (`enable_content_recording`) — compliance/data-residency implication.
- The 4-stage responsible-GenAI process from the course module: **Identify → Measure → Mitigate → Operate** (map harms, evaluate, layer mitigations at model/safety-system/system-prompt/UX levels, then monitor in production).

---

# DOMAIN 2 — Generative AI & Agentic Solutions (30–35%, the biggest domain)

## 6. Clients, Responses API, and generation parameters

| Client | Signature |
|---|---|
| Classic | `AzureOpenAI(azure_endpoint=…, api_key=…, api_version=…)` |
| v1 surface | `OpenAI(base_url="<resource>/openai/v1", api_key=…)` — no `api_version` |
| Via Foundry | `AIProjectClient(...).get_openai_client()` |

- `model=` on Azure = your **deployment name**, not necessarily the model family name.
- **Responses API** = the unified, forward-path API (supersedes Assistants): multi-turn state, built-in tools, reasoning models, one call shape.
- **🧠 Spot-the-API:** `"type": "image_url"` in `messages` → Chat Completions; `"type": "input_image"` in `input` → Responses API (blocks: `input_text`, `input_image`, `input_file`).
- `extra_body` = SDK escape hatch for Azure-specific fields (e.g. `agent_reference`).
- Parameters: `temperature` 0–2 (low = deterministic; prefer **structured outputs/JSON schema** over low temperature for parseable output), `max_output_tokens` (truncation is signalled), reasoning `effort` (quality vs latency + billed reasoning tokens), `tool_choice` (`auto`/`required`/`none`/named tool).
- `response.output_text` = text only; iterate `response.output` items for the full tool-call audit trail.

## 7. Tools & function calling — ⭐ the #1 scenario topic

### The golden distinction: WHO executes the tool?

| Tool family | Examples | Executed by |
|---|---|---|
| **Built-in / hosted** | `web_search`, `code_interpreter`, `file_search`, `AzureAISearchTool` | **The service**, server-side |
| **Custom function** | `FunctionTool`, JSON-schema functions | **YOUR code** — model proposes name + args, then STOPS |
| **OpenAPI tool** | REST API + OpenAPI spec (+ `securitySchemes` for auth) | Service calls the API over HTTP |
| **MCP tool** | MCP server connection | Service → MCP server, guarded by `require_approval` + `allowed_tools` |

### The function-calling loop (identical across every LLM API)

Request with tool schemas → model returns `function_call` (name + JSON args) and stops → **you execute** → resubmit `function_call_output` tagged with the matching **`call_id`** → final answer.

- JSON-Schema contract: `type`, `properties`, `required`, `additionalProperties: false`; zero-arg tools still need `{"type":"object","properties":{},"required":[]}`; Python param names must match schema keys (`**arguments` unpacking).
- Foundry `FunctionTool`: declared schema (server) and implementation (client) are independent — **you keep them in sync**. Frameworks (`Runner.run`, LangChain `create_agent`) automate the loop.
- **Code Interpreter** = model-written Python in an isolated, ephemeral, service-managed container → deterministic math/data-analysis/charts.
- **MCP** (Model Context Protocol) = cross-vendor standard for tool discovery/invocation: transport → session → `initialize` → `list_tools`/`call_tool`; tool results are **text content blocks** → return JSON *strings* for structured data. Azure Functions offer HTTP triggers, OpenAPI backends, and native **MCP tool triggers** as the hosting layer (auth levels: `ANONYMOUS`/`FUNCTION`/`ADMIN`; key via `x-functions-key` header, enforced by the host before your code runs).

## 8. Building agents with Foundry

- **Agent = model + instructions + tools + run loop.** `instructions` == system prompt. Define **role, goals, conversation-tracking approach, and tool schemas** up front.
- **Prompt agent** (`PromptAgentDefinition`) = stateless, per-request. **Hosted/persisted agent** + thread = server-side conversation memory. `conversations.create()` = server-side state on the OpenAI-compatible surface.
- `agent_reference` (via `extra_body`) invokes a persisted agent through a standard Responses call — its instructions, tools, `tool_choice` all apply; pin a `"version"`.
- Build/test in **portal playground or VS Code extension**; agents integrate **retrieval + function-calling + memory** in one definition.
- **Foundry IQ** ⭐ (new): a **shared knowledge platform** — register knowledge sources once, many agents ground on them with consistent, **cited** responses; improve retrieval via data optimization; RAG-as-a-platform instead of per-agent wiring.
- **Microsoft 365 integration** ⭐ (new): publish Foundry agents to **Teams and Microsoft 365 Copilot**; **Work IQ** gives access to workplace data (mail, files, chats) with the user's permissions.
- Grounding instruction *"answer only from sources; say you don't know otherwise"* = standard hallucination mitigation.

## 9. Workflows, multi-agent, Agent Framework, A2A

- **Foundry workflows** (declarative YAML / visual): trigger → `InvokeAzureAgent` nodes → `ConditionGroup` routing → `EndConversation`; capture agent output as **structured data** (`responseObject`) to route on it (e.g. intake-triage → knowledge agent vs ticket agent).
- **Microsoft Agent Framework** ⭐ = open-source SDK, successor merging **Semantic Kernel + AutoGen** — code-first complex agents and multi-agent orchestration; use Agent Service for managed hosting, Agent Framework for custom logic.
- Orchestration patterns: **sequential** (pipeline), **concurrent** (fan-out/fan-in), **group chat / debate-critic**, **handoff** (route to specialist), plus Foundry `ConnectedAgentTool`.
- **A2A protocol** ⭐ = Agent-to-Agent: standardized **discovery** (agent cards), direct communication, and coordinated task execution across **remote** agents — complements MCP (MCP connects agents→tools; **A2A connects agents→agents**).
- **Autonomous/semi-autonomous** workflows need **safeguards**: approval flow controls (human sign-off on risky actions), constraints, monitoring hooks; evaluate agent behavior and perform **error analysis** on traces.

## 10. RAG in applications

- Two styles: **bring-your-own retrieval** (your code queries the index, stuffs results into the prompt) vs **tool-based** (`AzureAISearchTool`/`FileSearchTool`/Foundry IQ retrieves server-side).
- Hybrid query mechanics: `search_text` + `VectorizableTextQuery` (integrated vectorization embeds the query for you); `k_nearest_neighbors` = vector-side candidates, `top` = final count; merged via **RRF (Reciprocal Rank Fusion)**.
- Chunking trade-off: small = precise but less context; large = more context, noisier; overlap preserves continuity.
- FAISS-in-a-notebook = learning stand-in; **the exam wants Azure AI Search / Foundry IQ** for production.

## 11. Evaluation, optimization, observability

- **Evaluate** models AND apps: fabrication/hallucination detection, relevance, quality, safety — manual (playground) + automated (evaluators, LLM-as-judge with consistent rubrics).
- ⭐ **Optimization decision ladder:** 1️⃣ **Prompt engineering** (behavior/format — try first, cheapest) → 2️⃣ **RAG** (missing/fresh/private knowledge) → 3️⃣ **Fine-tuning** (consistent style/format/domain vocabulary; training + hosting cost). *"Prompt for behavior, RAG for knowledge, fine-tune for style."* Fine-tuning ≠ adding facts. The approaches **combine**.
- **Model reflection / self-critique loops**: model critiques and revises its own output; chain-of-thought evaluations — quality vs cost/latency trade-off (each critique = another LLM call).
- **Observability:** tracing (OpenTelemetry → Application Insights → Azure Monitor alerts), **token analytics**, **safety signals**, **latency breakdowns**; collect user feedback. Instrumentation wraps *around* the client (callbacks) — agent code unchanged.
- Orchestrate **multiple models** (router/planner: cheap SLM for easy turns, LLM for hard ones) and **hybrid LLM + rules engines** (deterministic rules for compliance-critical paths, LLM for open-ended ones).
- LangChain/LangGraph still relevant as client-side orchestration: everything "agentic" there is **client-side**; LangGraph `StateGraph` = nodes + conditional edges; the 3-node `call_model ↔ tools` loop = canonical **ReAct**; no memory across `invoke()` runs unless you add a **checkpointer**.

---

# DOMAIN 3 — Computer Vision Solutions (10–15%)

> ⚠️ AI-103's vision domain = **generation + multimodal understanding + Content Understanding + visual responsible AI**. Classic AI-102 topics (Custom Vision, Face API, Video Indexer, Spatial Analysis) are **gone** — don't over-study them (see Legacy table at the end).

## 12. Image & video generation

| Operation | API | Key rules |
|---|---|---|
| Generate image | `images.generate()` | From text prompts (+ reference media); content filtering runs **before** generation |
| Prompt edit | `images.edit()` (no mask) | Less predictable — that's why masks exist |
| **Masked edit = inpainting** | `images.edit()` + `mask` | Only the masked region is regenerated |
| **Generate video** | **Sora 2** in Foundry | Text prompt (+ reference media) → video; async job (submit-then-poll); editing workflows for generated video |

⭐ **Mask rules (frequently mis-guessed):**
- **Transparent (alpha = 0) pixels get REGENERATED**; opaque pixels preserved pixel-for-pixel — opposite of intuition!
- Mask must be a **PNG with the same dimensions** as the source; source image for edits = **binary multipart file**, not URL/base64.
- Know the platform's generation/editing **controls**: size/aspect, quality, style, number of outputs, content-policy constraints.

## 13. Multimodal understanding

- Vision-enabled chat: same Responses call with `input_image` blocks — hosted URL or base64 `data:` URI (base64 ≈ +33% payload, fully private). Not every deployment supports image input.
- Build: **concise or detailed captions** (single or multiple images), **visual question-answering grounded in the image evidence**, and **alt-text / extended descriptions aligned to accessibility guidelines** (WCAG-style: convey purpose, not pixel-by-pixel).
- **Content Understanding for visuals**: extract visual characteristics, identify **objects/components/regions**, analyze **video segments**; know **single-task pipelines vs pro-mode** (pro mode = multi-step reasoning across multiple inputs/analyzers for complex extraction).
- OCR routing rule: text in **photos** → multimodal model / Vision READ; text in **documents** → Document Intelligence; **schema-driven multi-modal** → Content Understanding.

## 14. Responsible AI for multimodal content

- **Filter unsafe visual content**: image moderation via Content Safety `analyze_image` (HSSV categories) — only meaningful on multimodal deployments.
- ⭐ **Indirect prompt injection via images**: attackers embed instructions as **text inside an image**; mitigate with Prompt Shields + treating OCR'd/image text as untrusted data, never as instructions.
- **Visual policy enforcement**: watermarking generated media (provenance/C2PA-style), flagging prohibited symbols, brand-usage rules, detecting inappropriate content before publishing.

---

# DOMAIN 4 — Text Analysis Solutions (10–15%)

## 15. LLM-based vs dedicated Language analysis (know the trade-off cold)

| | LLM prompting | Azure Language (Foundry Tools) |
|---|---|---|
| Categories | Arbitrary, invented on the spot (`ticket_id`, `sla_tier`), zero training | Fixed prebuilt taxonomy (Person, Location, Organization, DateTime, Quantity, Product, URL, Email, PhoneNumber…) |
| Output | Free-form unless schema-constrained (structured outputs) | Always the same fixed schema |
| Confidence | None | Numeric `confidence_score` per entity — threshold in production |
| Consistency | Can vary run-to-run | Deterministic, auditable |

- One client, many ops: `TextAnalyticsClient` → `detect_language` (**ISO 639-1** out; detect first, then pass `language=` to other calls), `recognize_entities`, `recognize_pii_entities` (redaction), `analyze_sentiment` (sentence AND document level; `positive/neutral/negative/mixed`; opinion mining = target+assessment pairs), `extract_key_phrases`; summarization = **extractive** (picks sentences) vs **abstractive** (writes new text).
- LLM side: entity/topic/summary extraction with **structured JSON outputs** (Pydantic/JSON schema), sentiment/tone/safety/sensitive-content detection by prompt, **domain customization** (compliance summarization, domain extraction) via prompt engineering or fine-tuning.
- ⭐ **Azure Language MCP server** (new): an agent calls Language operations (detect language, NER, **PII redaction**) as MCP tools — dedicated-service accuracy inside an agentic loop.
- Translation: **Azure Translator** (100+ languages, one call → many targets via `targets` array, omit `from` = auto-detect, versioned REST + `Ocp-Apim-Subscription-Key`; **document translation** = async batch between blob containers, preserves formatting) vs **LLM translation flows** (better tone/idiom, no formal guarantees).

## 16. Speech solutions

| Mode | Method | Use for |
|---|---|---|
| Real-time / streaming | `start_continuous_recognition()` + events (`.recognizing` = interim, `.recognized` = final) | Live captioning, assistants |
| Single-shot | `recognize_once_async()` | Short commands, single turns |
| Batch | Submit file | Archives; adds per-word timestamps, **speaker diarization**, confidence |

- `SpeechConfig` from `subscription`+`region` OR `subscription`+`endpoint`; locales = `<language>-<region>` (`en-US`).
- **Neural voices** (`en-US-JennyNeural`) are the standard; plain text → `speak_text_async()`; pronunciation/pauses/pitch/multi-voice → **SSML** via `speak_ssml_async()`.
- Check `.reason` vs `ResultReason` enums (`RecognizedSpeech`, `NoMatch`, `Canceled`, `SynthesizingAudioCompleted`, `TranslatedSpeech`).
- Speech translation = its own pair: `SpeechTranslationConfig` + `TranslationRecognizer`.
- **Custom speech models** for domain vocabulary/accents (quality = WER).
- ⭐ New agentic speech (course modules): **speech as an agent modality** (STT in → agent → TTS out), **Azure Speech MCP server** (STT/TTS as MCP tools), **Voice Live API/SDK** (real-time, low-latency full-duplex voice conversations with an agent — the "live voice agent" answer), and **multimodal reasoning over audio inputs** (audio-capable models reasoning directly over sound, not just transcripts).

**🧠 Hook:** *"Once for commands, continuous for conversations, batch for backlogs, Voice Live for live voice agents."*

---

# DOMAIN 5 — Information Extraction Solutions (10–15%)

## 17. Retrieval & grounding pipelines (Azure AI Search)

- **Ingest & index multimodal content**: documents, images, audio, video — enrich on the way in.
- Retrieval modes: **keyword** (BM25) / **vector** (k-NN over embeddings) / **hybrid** (both, RRF-merged) / **semantic ranker** (re-ranks top 50 BM25 results, adds captions + answers).
- Index field attributes: `key`, `searchable`, `filterable`, `sortable`, `facetable`, `retrievable` — **filterable ≠ searchable**. 🧠 *"Key Search Filters Sort Facets Retrieved."*
- Query syntax: simple vs **full Lucene** (`queryType=full`: wildcards `azur*`, fuzzy `~`, regex, boosting `^`); `$filter`, `$orderby`, `$select`, facets; `searchMode=any|all`.
- **Enrichment skillsets** (built-in + custom skills) run during indexing: e.g. **OCR → Merge → LanguageDetection → KeyPhrase** (this chapter's lab); custom skill = `WebApiSkill` → your **Azure Function**, strict `values`/`recordId` echo contract.
- **RAG ingestion flow**: docs → **OCR/layout** → chunk → embed → index; then **connect the pipeline to agents** as tools/knowledge (`AzureAISearchTool`, Foundry IQ knowledge sources).
- Plumbing: data source → indexer (schedule ≥ 5 min, change detection, enrichment cache) → skillset → index. Admin keys (manage) vs **query keys** (read-only clients).

## 18. Document extraction — Document Intelligence & Content Understanding

### Document Intelligence prebuilt models

| Model | Returns |
|---|---|
| `prebuilt-read` | OCR text only |
| `prebuilt-layout` | Structure: text, tables, selection marks — no semantic fields |
| `prebuilt-invoice` | Layout **+** semantic fields with confidence (VendorName, InvoiceId, InvoiceDate, DueDate, SubTotal, Tax, InvoiceTotal, line items) |
| receipt / ID document / business card | Match model to document type |

- **Custom models**: **template** (fixed-layout forms) vs **neural** (varied docs), both need **5+ labeled samples** (Document Intelligence Studio); **composed model** = several customs behind one model ID with auto-classification.
- ⭐ **LRO pattern**: `begin_analyze_document` / `begin_analyze_binary` return a **poller** (submit-then-poll) — pervasive across extraction SDKs.

### Content Understanding

- **Multimodal pipelines**: OCR + layout analysis + field extraction in one; processes **documents, images, video, audio** under one API.
- Custom **analyzers** = `fieldSchema` on a `baseAnalyzerId` (e.g. `prebuilt-document`); `estimateFieldSourceAndConfidence` → value + confidence + source location per field.
- **`markdown` output** = clean, grounded, LLM-ready representation — feed it to agents/RAG; `fields` dict = programmatic access. Different analyzers → different result schemas; inspect before coding.
- ⭐ **Pipeline best practice:** extraction service extracts (reliable, schema-grounded); the LLM/agent judges (summarize, approve, flag) — keep responsibilities separate.

---

# QUICK RECAP

## 19. Legacy AI-102 topics — de-emphasized in AI-103 (skim only)

| Was on AI-102 | Status on AI-103 |
|---|---|
| CLU (intents/entities/utterances), custom QA knowledge bases | Replaced by LLM/agent-based language understanding |
| Custom Vision (classification/object detection training) | Replaced by multimodal models + Content Understanding |
| Face API (verify/identify, PersonGroups) | Not in the outline |
| Video Indexer, Spatial Analysis | Replaced by Content Understanding video analysis |
| Knowledge store projections, prompt flow DAGs | Not emphasized; workflows/Foundry IQ carry the load |

## 20. ⚡ Rapid-fire tables

### Which auth header?

| Header / mechanism | Used by |
|---|---|
| `api-key` | Azure OpenAI |
| `Ocp-Apim-Subscription-Key` | Foundry Tools REST (Language, Translator, Speech, Vision) |
| `x-functions-key` / `?code=` | Azure Functions |
| Bearer token (Entra ID, keyless) | Everything, in production |

### Sync or poll?

| Call | Pattern |
|---|---|
| Content Safety `analyze_text`/`analyze_image` | Synchronous |
| `TextAnalyticsClient` operations | Synchronous |
| DocIntel / Content Understanding `begin_*` | **LRO poller** |
| Batch transcription, document translation, video generation | Submit-then-poll |
| Real-time speech, Voice Live | Streaming + events |

### Which number was that?

| Number | Meaning |
|---|---|
| 700 | Pass mark |
| 30–35% | GenAI + agentic domain weight (biggest) |
| 0 / 2 / 4 / 6 | Content Safety severities (0–7 finer via `output_type`) |
| ≥ 4 gate | Block Medium+ before the model |
| 0–2 | `temperature` range |
| +33% | base64 image payload inflation |
| top 50 | Results the semantic ranker re-ranks |
| 5+ | Labeled samples for a custom DocIntel model |
| 5 min | Minimum indexer schedule interval |
| 100+ | Translator languages |

### Who executes it?

| Thing | Executor |
|---|---|
| `web_search`, `code_interpreter`, `file_search` | Service (server-side) |
| `FunctionTool` / function calling | **Your client code** |
| OpenAPI tool | Service → your REST API |
| MCP tool | Service → MCP server (`require_approval`, `allowed_tools`) |
| LangChain/LangGraph tools | Your client process |

### Protocol pairing (new on AI-103)

| Protocol | Connects |
|---|---|
| **MCP** | Agents → **tools** (discovery + invocation) |
| **A2A** | Agents → **agents** (discovery via agent cards, remote coordination) |

### Stateful vs stateless

| Stateless | Stateful |
|---|---|
| `responses.create()` one-off | `conversations.create()` server-side state |
| `PromptAgentDefinition` | Hosted agent + thread |
| LangGraph `invoke()` | LangGraph + checkpointer |

### SDK class ↔ purpose

| Class | Service |
|---|---|
| `AzureOpenAI` / `OpenAI` | Foundry Models (Azure OpenAI) |
| `AIProjectClient` | Foundry project |
| `SearchClient`, `VectorizableTextQuery` | AI Search |
| `TextAnalyticsClient` | Language |
| `SpeechConfig`+`SpeechRecognizer` / `SpeechTranslationConfig`+`TranslationRecognizer` | Speech |
| `ContentSafetyClient` | Content Safety |
| `DocumentIntelligenceClient` | Document Intelligence |
| `ContentUnderstandingClient` | Content Understanding |
| `AzureKeyCredential` / `DefaultAzureCredential` | Auth (key / keyless) |

## 21. 🏁 One-liner flash facts (read 10 minutes before the exam)

1. AI-103 = "Developing AI Apps and Agents on Azure" → **Azure AI Apps and Agents Developer Associate**; agents + genAI = 30–35%.
2. Entra ID / keyless > API key: short-lived tokens + RBAC; managed identity + private endpoints = "most secure" answer.
3. `model=` on Azure = **deployment name**.
4. `input_image` = Responses API; `image_url` = Chat Completions.
5. Function tools: **model proposes, YOU execute**, resubmit with matching `call_id`.
6. Built-in tools (web search, code interpreter, file search) run **server-side**.
7. `tool_choice`: auto / required / none / specific.
8. Prompt agent = stateless; hosted agent + thread/conversation = server-side memory.
9. `extra_body` smuggles Azure fields (`agent_reference`) through the OpenAI SDK; pin agent versions.
10. **MCP connects agents to tools; A2A connects agents to agents** (agent cards for discovery).
11. MCP safety levers: `require_approval` + `allowed_tools`; MCP results = JSON **strings**.
12. **Foundry IQ** = shared knowledge platform: many agents, one grounded, cited knowledge layer.
13. **Work IQ + M365**: publish agents to Teams/Copilot with workplace-data access.
14. **Microsoft Agent Framework** = Semantic Kernel + AutoGen successor, code-first multi-agent; Agent Service = managed.
15. Orchestration patterns: sequential, concurrent, group chat, handoff.
16. Autonomous agents need approval flows, constraints, oversight modes, trace-based error analysis.
17. Keyword = BM25; vector = k-NN; hybrid = RRF; semantic ranker re-ranks top 50.
18. `k_nearest_neighbors` = vector candidates; `top` = final results.
19. Decision ladder: **prompt-engineer → RAG → fine-tune** ("behavior → knowledge → style"); fine-tuning ≠ facts.
20. Evaluators: Completeness, Groundedness, Relevance (+ safety evals); run in CI/CD as quality gates.
21. Model reflection/self-critique = extra LLM calls = quality-vs-cost trade-off.
22. Observability: tracing, **token analytics**, safety signals, latency breakdowns; content recording is a separate toggle.
23. Hybrid orchestration: rules engines for deterministic paths, LLMs for open-ended ones; SLM router for cheap turns.
24. Deployment types: Standard, Global Standard, Global **Batch** (~50% off), **PTU** (reserved), Serverless/MaaS; quota = **TPM**.
25. Monitor AI-specifics: drift, safety events, grounding quality, index health, relevance.
26. F-R-P-I-T-A: Fairness, Reliability, Privacy, Inclusiveness, Transparency, Accountability; process = Identify → Measure → Mitigate → Operate.
27. Content Safety: **HSSV**, severities 0/2/4/6, sync calls, blocklists text-only; gate at ≥ 4.
28. Prompt Shields: jailbreak (direct) + **indirect injection — including text embedded in images**.
29. Groundedness detection flags claims unsupported by sources.
30. Watermark/provenance + prohibited-symbol + brand rules = visual policy enforcement.
31. Inpainting: **transparent = regenerated**, same-size PNG mask, binary multipart source.
32. **Sora 2** generates video from text/reference media in Foundry — async job.
33. Alt-text: purpose-focused, accessibility-guideline-aligned; captions can be concise or detailed, single or multi-image.
34. Content Understanding: **single-task vs pro-mode** pipelines; analyzers = `fieldSchema` + `baseAnalyzerId`; markdown output = LLM-ready.
35. OCR routing: photo → multimodal/Vision READ; document → DocIntel; multimodal schema → Content Understanding.
36. LLM = arbitrary categories, no confidence; Language = fixed schema + confidence scores — threshold them.
37. Sentiment = sentence + document level; opinion mining = target + assessment pairs.
38. Detect language first (ISO 639-1), then pass `language=` to other Language calls.
39. **Language MCP server** and **Speech MCP server** expose dedicated services as agent tools.
40. **Voice Live API** = real-time full-duplex voice agent conversations.
41. Translator `targets` array = many languages in one call; omit `from` = auto-detect; document translation = async blob-to-blob, keeps formatting.
42. SSML for pronunciation/pauses/pitch — via `speak_ssml_async()`, not a text-call parameter.
43. `TranslationRecognizer` → `ResultReason.TranslatedSpeech`; custom speech quality = WER.
44. Index attributes: key, searchable, **filterable**, sortable, facetable, retrievable.
45. Full Lucene (`queryType=full`): wildcards, fuzzy `~`, boosting `^`.
46. Admin keys manage; **query keys** for read-only clients; indexer schedule ≥ 5 min.
47. Custom skill = WebApiSkill → Azure Function; echo the `recordId`s.
48. DocIntel: read = OCR, layout = structure, invoice = fields + confidence; template vs neural customs (5+ samples); composed = one ID, auto-classify.
49. `begin_*` poller = LRO across all extraction SDKs.
50. Extraction extracts, LLM judges — keep pipeline responsibilities separate.
51. Containers: `Eula` + `Billing` + `ApiKey`; billing metadata to Azure, your data stays local.
52. Two keys per resource → zero-downtime rotation; secrets in Key Vault — or go keyless.
53. Multi-service resource = one key for many; single-service = isolation + F0 free tier.
54. CLU, custom QA, Custom Vision, Face, Video Indexer = **AI-102 legacy** — skim, don't grind.
55. Every "custom X" lifecycle: **Label → Train → Evaluate → Deploy → Consume.**

---

*Good luck — you've already built most of this in the course. The exam is recognizing these patterns dressed up as scenarios.* 🚀

➡️ **Test yourself:** [`EXAM_PRACTICE_QUESTIONS.md`](EXAM_PRACTICE_QUESTIONS.md) — 65 exam-style scenario questions with answers and explanations, weighted by the official blueprint.
