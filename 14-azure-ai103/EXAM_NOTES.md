# 🎯 AI-102/AI-103 Exam Short Notes — Complete Quick-Revision Guide

> Condensed from all 8 sections of this course (53 scripts + notebooks + labs), **plus every official-syllabus topic the course doesn't cover** (sections 16–22) — together this covers the full exam blueprint.
> **Note:** the official Microsoft certification covering this material is **AI-102: Designing and Implementing a Microsoft Azure AI Solution** — the course calls itself "AI-103", but exam tips here map to AI-102 objectives. (Microsoft retired the AI-102 exam on June 30, 2026 in favor of newer AI credentials, but the skills outline below is the definitive final syllabus and the concepts transfer directly to its successors.) Read top-to-bottom once, then use the tables for last-day recap.

### Official exam blueprint (skills measured, final Dec 2025 version)

| Domain | Weight | Covered in |
|---|---|---|
| Plan and manage an Azure AI solution | **20–25%** | §1, §2, §16 |
| Implement generative AI solutions | 15–20% | §3, §4, §7, §9, §17 |
| Implement an agentic solution | 5–10% | §4, §5, §8, §17 |
| Implement computer vision solutions | 10–15% | §9, §18 |
| Implement natural language processing solutions | 15–20% | §11, §12, §13, §19 |
| Implement knowledge mining and information extraction | 15–20% | §6, §14, §20 |

Pass mark: **700/1000**. Most questions test **GA features**; commonly-used Preview features can appear.

---

## 1. The Big Picture — Which Service / SDK for What

| Task | Service | Python SDK / Client | Auth options |
|---|---|---|---|
| Chat, reasoning, tools, agents | Azure OpenAI / AI Foundry | `openai` (`AzureOpenAI` or `OpenAI`), `azure-ai-projects` (`AIProjectClient`) | API key **or** Entra ID |
| Persisted agents + tools | Azure AI Foundry Agent Service | `azure-ai-projects` + `azure-ai-agents` | Entra ID (recommended) |
| Keyword / vector / hybrid search | Azure AI Search | `azure-search-documents` (`SearchClient`) | `AzureKeyCredential` or Entra ID |
| NER, PII, sentiment, key phrases, language detect | Azure AI Language | `azure-ai-textanalytics` (`TextAnalyticsClient`) | Key or Entra ID |
| Text translation (100+ languages) | Azure AI Translator | REST (`api-version` param, `Ocp-Apim-Subscription-Key` header) | Key |
| Speech-to-text, text-to-speech, speech translation | Azure AI Speech | `azure-cognitiveservices-speech` (`SpeechConfig`, `SpeechRecognizer`…) | Key + region/endpoint |
| OCR, layout, invoices/receipts | Azure AI Document Intelligence | `azure-ai-documentintelligence` (`begin_analyze_document`) | Key or Entra ID |
| Schema-driven multi-modal extraction | Azure AI Content Understanding | `ContentUnderstandingClient` (`begin_analyze_binary`) | Key or Entra ID |
| Harm-category moderation | Azure AI Content Safety | `azure-ai-contentsafety` (`analyze_text` / `analyze_image`) | Key or Entra ID |
| Host custom tool/skill logic | Azure Functions | `azure-functions` (v2 decorator model) | `ANONYMOUS` / `FUNCTION` / `ADMIN` |

**🧠 Memory hook:** *"OpenAI SDK for generative, `azure-ai-*` SDK for dedicated"* — every dedicated cognitive skill (Language, Speech, Search, DocIntel, Content Safety) has its own `azure-ai-*` / `azure-*` package; only generative chat rides the `openai` SDK.

---

## 2. Authentication — Asked on EVERY exam

### Two auth models (know cold)

| | API Key | Microsoft Entra ID |
|---|---|---|
| Mechanism | `api-key` / `Ocp-Apim-Subscription-Key` header, `AzureKeyCredential` | OAuth bearer token via `DefaultAzureCredential` |
| Secret lifetime | Long-lived shared secret | Short-lived, scoped tokens |
| Access control | Anyone with the key | Azure **RBAC** role assignments |
| Rotation / revocation | Manual, your responsibility | Centrally revocable |
| Exam verdict | OK for quick dev/test | **Recommended for production** |

### Credential classes

| Credential | Behavior | When |
|---|---|---|
| `DefaultAzureCredential` | Tries a **chain** (env vars → managed identity → VS → Azure CLI…) | Code that must run unmodified everywhere |
| `AzureCliCredential` | Only your `az login` session — fails fast, predictable | Local dev |
| `ClientSecretCredential` | Service principal (app id + secret) | Non-interactive services |
| Managed identity | No secret at all | Azure-hosted production workloads |

- Token scope for Foundry: `https://ai.azure.com/.default` (via `get_bearer_token_provider`).
- **RBAC roles:** *Azure AI User / Azure AI Developer / Cognitive Services User* let you **call** agents; managing (create/publish/delete) needs broader roles → always assign the **narrowest role that works** (least privilege).
- Multi-resource apps (e.g. Foundry agent + AI Search) need RBAC **per resource** — two endpoints, two role assignments.
- One script may legitimately mix auth styles (key for Content Understanding, Entra ID for Foundry) — auth choice is **per-service**, not repo-wide.
- Resilient pattern: `key if present, else DefaultAzureCredential` — same code works locally (key) and in production (managed identity).

**🧠 Memory hook:** *"Key = quick, Entra = enterprise."*

---

## 3. Azure OpenAI — Clients, Responses API, Parameters

### Client shapes (recognize both in code questions)

| Client | Signature | Notes |
|---|---|---|
| Classic | `AzureOpenAI(azure_endpoint=…, api_key=…, api_version=…)` | Versioned |
| v1 preview | `OpenAI(base_url="<resource>/openai/v1", api_key=…)` | No `api_version` |
| Via Foundry | `AIProjectClient(...).get_openai_client()` | Bridge from `azure-ai-projects` to the OpenAI-compatible surface |

- `model=` in Azure calls = your **deployment name** (chosen at deploy time), NOT necessarily the model family name. Wrong deployment name → failure.
- Azure OpenAI is **OpenAI-API-compatible** — same `openai` SDK, only `base_url`/auth/deployment change.
- `extra_body` = the SDK escape hatch for Azure-specific fields the typed SDK doesn't know (e.g. `agent_reference`).

### Responses API vs Chat Completions

| | Chat Completions | Responses API (newer, unified) |
|---|---|---|
| Input | `messages=[...]` | `input=[...]` |
| Content block types | `text`, `image_url` | `input_text`, `input_image`, `input_file` |
| State | Stateless, resend history | Native multi-turn (`conversations`) |
| Built-in tools | ✗ | ✔ web search, code interpreter, file search |
| Position | Legacy-ish | Forward path for agentic apps (supersedes Assistants API) |

**🧠 Spot-the-API trick:** `"type": "image_url"` inside `messages` → **Chat Completions**. `"type": "input_image"` inside `input` → **Responses API**.

### Key parameters

| Parameter | Range / values | Effect |
|---|---|---|
| `temperature` | 0–2 | Low = deterministic/factual, High = creative/random. For parseable output prefer **structured outputs / JSON schema** over low temperature |
| `max_output_tokens` | int | Caps length for cost/latency; truncation is **signalled**, not silent |
| Reasoning `effort` | low/medium/high | More effort = better depth but ↑latency and ↑billed reasoning tokens |
| `tool_choice` | `"auto"` / `"required"` / `"none"` / named tool object | Who decides whether/which tool is called |

- Model families differ in supported params (reasoning, structured outputs, function calling) → **probe and degrade gracefully**, don't hardcode.
- Multimodal images: hosted `https://` URL **or** inline base64 `data:` URI (base64 ≈ **+33% payload**, but fully private; no public URL needed). Not every deployment supports image input.
- `response.output_text` = concatenated message text only. For the audit trail (which tools, what code, what results) iterate `response.output` and branch on `item.type`.

---

## 4. Tools & Function Calling — The #1 Scenario Topic

### ⭐ The golden distinction: WHO executes the tool?

| Tool family | Examples | Executed by |
|---|---|---|
| **Built-in / hosted** | `web_search`, `code_interpreter`, `file_search`, `AzureAISearchTool` | **The service**, server-side, same response |
| **Custom function** | `FunctionTool`, JSON-schema function defs | **YOUR code** — model only proposes name + JSON args, then STOPS and waits |
| **OpenAPI tool** | REST API + OpenAPI spec | **The service** calls the API over HTTP directly |
| **MCP tool** | MCP server connection | Service calls MCP server (with approval controls) |

### The function-calling loop (canonical, same across all LLM APIs)

1. Send request with tool schemas → 2. Model returns `function_call` item (name + JSON args) and **stops** → 3. **You** execute the function → 4. Resubmit result as `function_call_output` tagged with the matching **`call_id`** → 5. Model produces final answer.

- `call_id` matters when a single turn triggers **multiple** tool calls — each output must match its request.
- Schema shape = standard JSON Schema: `type`, `properties`, `required`, `additionalProperties: false`.
- Zero-arg tools still need an **empty-but-valid** schema: `{"type":"object","properties":{},"required":[]}`.
- Python param names must exactly match schema property names (results are unpacked with `**arguments`).
- Foundry `FunctionTool`: declared schema (server-side) and implementation (client-side) are **independent — you keep them in sync**. The OpenAI Agents SDK's `Runner.run` automates the loop; Foundry leaves it manual for control.
- **Code Interpreter** = model-written Python runs in an **isolated, ephemeral, service-managed container** → deterministic math/data-analysis/charts (LLMs alone are unreliable at exact arithmetic).
- **MCP tools in Foundry:** `require_approval` (`never`/`always`/per-tool) = the human-in-the-loop safety lever; `allowed_tools` = defense-in-depth allow-list.
- Agent concept everywhere: **agent = model + instructions + tools + run loop**. `instructions` == system prompt (same thing, different SDK names).

**🧠 Memory hook:** *"Hosted tools run over there; function tools run right here."*

---

## 5. Azure AI Foundry — Projects, Agents, Conversations

### Vocabulary (keep straight!)

| Term | What it is |
|---|---|
| **Azure OpenAI resource** | One Cognitive Services resource with model deployments |
| **AI Foundry project** | Higher-level workspace: groups agents, connections, deployments |
| `AIProjectClient` | SDK entry point for project-level ops |
| **Connection** | Credential + endpoint indirection wiring external resources (search indexes, storage, MCP servers) into a project — reference by name, no embedded creds |

### Agent types

| | Prompt agent | Hosted / persisted agent |
|---|---|---|
| Definition | `PromptAgentDefinition` | Persisted Agent resource + Threads |
| State | **Stateless**, per-request | Server-side multi-turn memory |
| Invocation | Responses API | Threads/Runs API **or** `agent_reference` via Responses API |

- `agent_reference` (passed in `extra_body`) invokes a persisted agent by name through a standard Responses call — the agent's instructions, tools, and `tool_choice` apply automatically. Pin an explicit `"version"` for reproducibility.
- `conversations.create()` = server-side conversation state (like a Thread) — no client-side history bookkeeping; skip it for stateless single-turn calls.
- Foundry exposes an **OpenAI-SDK-compatible surface** deliberately → migrate OpenAI Assistants/Responses code with minimal changes.
- `ToolSet` bundles multiple tools for reuse across agents.
- **No-code workflows** (`workflow.yml`): trigger → `InvokeAzureAgent` nodes → `ConditionGroup` routing (e.g. Intake agent classifies → route to Knowledge agent or Ticket agent) → `EndConversation`. Conditions replace custom Python quality-gate code.

---

## 6. RAG & Azure AI Search

### Three retrieval modes (classic exam question)

| Mode | Query | Ranking |
|---|---|---|
| **Keyword / full-text** | `search_text` only | BM25 |
| **Vector** | `VectorizableTextQuery` (integrated vectorization embeds for you) | Similarity (k-NN) |
| **Hybrid** | Both together | Merged via **Reciprocal Rank Fusion (RRF)** |

- `k_nearest_neighbors` = candidates for the **vector side**; `top` = final combined result count.
- Integrated vectorization avoids managing your own embedding client and model-mismatch bugs (vs computing the query vector yourself).

### Two RAG styles

| | "Bring your own retrieval" | Tool-based / "on your data" |
|---|---|---|
| Who retrieves | Your code queries the index, stuffs results into the prompt ("prompt-stuffing RAG") | Agent's `AzureAISearchTool` / `FileSearchTool` retrieves server-side |
| Works with | Any agent/model (it's just a detailed question) | Foundry agents with tools |

- Grounding instruction *"answer only from sources; say you don't know otherwise"* = practical hallucination mitigation (responsible-AI theme).
- FAISS-in-a-notebook (LangChain) = conceptual stand-in; **the exam tests Azure AI Search** for production RAG (managed, scalable, hybrid).
- Chunking: small chunks = precise but less context; large = more context but noisier. `chunk_overlap` preserves continuity.

### AI Enrichment pipeline (indexer + skillset)

**Data source → Indexer (schedule, field mappings) → Skillset (enrichment skills) → Index (searchable fields)**

Skills seen in the lab: `OcrSkill` (image text) → `MergeSkill` (merge OCR + text into `merged_content`) → `LanguageDetectionSkill` → `KeyPhraseExtractionSkill`. Custom skills = commonly an **Azure Function** called via web API.

---

## 7. Evaluation, Monitoring & Workflows

- **LLM-as-judge** critique loops = hand-rolled version of Foundry's built-in **evaluators** (`azure-ai-evaluation` SDK / portal Evaluation): **Completeness, Groundedness, Relevance** — consistent rubrics at scale.
- Quality gates cost extra LLM calls (up to 3 per question) → **quality vs cost + latency trade-off** = governance/monitoring concern.
- **Tracing:** OpenTelemetry → Application Insights → Azure Monitor alerting/dashboards (latency, error-rate alerts on production agents).
- ⭐ `enable_content_recording` — **trace metadata and trace content are separable settings**: logging full prompts/responses has data-residency/compliance implications.
- Instrumentation wraps *around* the client (callbacks) — the agent code itself doesn't change.

---

## 8. Agent Frameworks — LangChain, LangGraph, MCP

- Everything "agentic" in LangChain is **client-side orchestration** — the Azure endpoint just sees a `tools` schema request and a maybe-`tool_calls` response.
- `@tool` docstrings are what the model reads to decide **when** to call a tool — write them precisely.
- LangChain's `create_agent` hides the manual function-calling loop; against Foundry, `model=` must be the **deployment name**.
- **LangGraph `StateGraph`** = nodes (functions over state) + edges (some conditional). The 3-node `call_model ↔ tools` loop with one conditional edge = canonical **ReAct agent** (`create_react_agent` prebuilds it). Each `invoke()` is a fresh run — **no memory unless you add a checkpointer**.
- **MCP** = cross-vendor protocol for standardized tool discovery/invocation. Client shape: transport → session → `initialize` → `list_tools`/`call_tool`. MCP tool return values are **text content blocks** → return JSON *strings* for structured data. Azure Functions now offer a native **MCP tool trigger** (Functions = hosting layer for tool logic).

---

## 9. Vision — Image Generation, Editing, Multimodal

| Operation | API | Key rules |
|---|---|---|
| Generate | `images.generate()` | From scratch, unconstrained; prompt subject to content filtering **before** generation |
| Prompt edit | `images.edit()` (no mask) | Less predictable/precise — that's why masks exist |
| Masked edit = **inpainting** | `images.edit()` + `mask` | Only masked region regenerated |

⭐ **Mask rules (frequently mis-guessed):**
- **Transparent (alpha = 0) pixels get REGENERATED**; opaque pixels are preserved pixel-for-pixel. (Opposite of intuition!)
- Mask must be a **PNG with the same dimensions** as the source image, or the request fails.
- Source image for edits = **binary multipart file**, not URL/base64.

**Azure AI Vision (Image Analysis)** = tags, captions, OCR, object detection via feature flags. **Content Understanding** = schema-driven extraction across documents/images/audio/video under one API. Know which is which.

---

## 10. Content Safety & Content Filtering

### Two different things — don't confuse!

| | Azure OpenAI **content filtering** | **Azure AI Content Safety** (separate service) |
|---|---|---|
| Where | Built into every deployment, configured **per-deployment** | Own resource type, called explicitly |
| Output | Pass / flag / block (medium+ blocked by default) | **Granular numeric severity scores** you act on programmatically |
| Extras | **Jailbreak** detection, **protected material** (verbatim copyrighted text/code) | **Custom blocklists** (exact-match, **text-only**) |

### The 4 harm categories (both systems)
**Hate, SelfHarm, Sexual, Violence** — 🧠 mnemonic: **"HSSV — Harm Stops Safe Ventures."**

- Default severity scale: **0 = Safe, 2 = Low, 4 = Medium, 6 = High** (per category, independently). Finer **0–7** scale available via `output_type`.
- `analyze_text` and `analyze_image` are **synchronous** siblings (no submit-then-poll).
- Blocklists = text-only, not applicable to images.
- Pattern the exam loves: **gate user input** (block severity ≥ 4) *before* it reaches the LLM prompt.
- Image content filtering only applies to **multimodal-capable** deployments (text-only models can't accept images at all).

---

## 11. Azure AI Language (Text Analytics)

⭐ **One client, many operations** — all on `TextAnalyticsClient`:
`detect_language`, `recognize_entities`, `recognize_pii_entities`, `analyze_sentiment`, `extract_key_phrases`.

| Feature | Key facts |
|---|---|
| **Prebuilt NER** | Zero training, **fixed Microsoft taxonomy**: Person, Location, Organization, DateTime, Quantity, Event, Product, Skill, URL, Email, PhoneNumber. `subcategory` is often `None` — code defensively |
| **Custom NER** | Domain categories (e.g. `ticket_id`, `sla_tier`) don't exist in prebuilt → label docs in **Language Studio**, train your own model |
| **PII detection** | `recognize_pii_entities` — returns entities + redacted text |
| **Sentiment** | Per **sentence AND document** in one call: `positive/neutral/negative/mixed`, each with confidence scores summing to 1.0. Mixed doc-level + true per-sentence polarity |
| **Opinion Mining** | `show_opinion_mining=True` on the same call → aspect-based pairs: *target* ("VPN client") + *assessment* ("dropping", negative), grounded in text spans |
| **Language detection** | Returns **ISO 639-1** code (`.iso6391_name`) — feed it into `language=` of other calls. Short/ambiguous text ("OK") → `unknown` + low confidence |
| **Confidence scores** | Per-entity float on nearly everything — used to **threshold/filter** in production, not just display |

**LLM vs Language service** (design-question favorite):

| | LLM prompt | Azure AI Language |
|---|---|---|
| Categories | Arbitrary, invented on the spot, zero training | Fixed prebuilt (or trained Custom) |
| Output | Free-form (unless schema-constrained) | Always same fixed schema, deterministic |
| Confidence | None | Numeric, auditable, consistent |

- Structured/JSON-schema output = an **Azure OpenAI model feature**; Language responses are inherently structured because it's a purpose-built API — similar look, different reason.
- The Responses API has **no equivalent in Language** — keep SDK surfaces straight: `openai` SDK ↔ Azure OpenAI; `azure-ai-textanalytics` ↔ Language.

---

## 12. Azure AI Translator

- Purpose-built MT: **100+ languages**, transliteration, profanity filtering, **Custom Translator** for domain terminology. LLM translation = better tone/idiom sometimes, but no formal guarantees.
- REST API: versioned URL with explicit **`api-version`** query param; key via `Ocp-Apim-Subscription-Key` header (the standard Cognitive Services REST key header).
- ⭐ **`targets` array**: one call → multiple target languages simultaneously = fewer round trips, lower latency, fewer rate-limit hits (vs LLM loop per language).
- Omit the `from` parameter → service **auto-detects** source language and returns the detected code with the translation.

---

## 13. Azure AI Speech

### Recognition modes

| Mode | Method | Use for |
|---|---|---|
| **Real-time / streaming** | `start_continuous_recognition()` + events | Live captioning, voice assistants, dictation |
| **Single-shot** | `recognize_once_async()` | Short commands, single Q&A turns |
| **Batch / file transcription** | Submit complete file | Offline archives (call centers) — throughput/accuracy over latency; richer results: per-word timestamps, **speaker diarization**, confidence |

### Must-know details

- `SpeechConfig` built from `subscription` + **`region`** (e.g. `"eastus2"`) OR `subscription` + **`endpoint`** (full URL) — interchangeable.
- Locales are **`<language>-<region>`** (`en-US`, `fr-FR`, `ja-JP`) — region affects pronunciation/vocabulary; same format for STT and TTS.
- Events: `.recognizing` = **interim** partial results (live-caption UIs); `.recognized` = **final** results.
- **Neural voices** (`en-US-JennyNeural`) = current recommended tech (standard voices deprecated); select via `speech_synthesis_voice_name`.
- ⭐ Plain text → `speak_text_async()`. Pronunciation, pauses, emphasis, rate/pitch, multiple voices → **SSML** via `speak_ssml_async()` (it is NOT a parameter of `speak_text_async`).
- Check `.reason` against **`ResultReason`** enums: `RecognizedSpeech`, `NoMatch`, `Canceled`, `SynthesizingAudioCompleted`, and for translation **`TranslatedSpeech`**.
- Speech translation = its **own object pair**: `SpeechTranslationConfig` + `TranslationRecognizer` (not a bolt-on to `SpeechRecognizer`).

**🧠 Memory hook:** *"Once for commands, continuous for conversations, batch for backlogs."*

---

## 14. Document Intelligence & Content Understanding

### Document Intelligence prebuilt models (pick-the-model questions)

| Model | Returns |
|---|---|
| `prebuilt-read` | OCR text only |
| `prebuilt-layout` | Structure: text, tables, selection marks — **no semantic fields** |
| `prebuilt-invoice` | Layout **+** semantic key-value fields with confidence: VendorName, CustomerName/Address, InvoiceId, InvoiceDate, DueDate, SubTotal, Tax, InvoiceTotal, line items |
| Others | receipt, ID document, business card — match model to document type |

- ⭐ **LRO pattern:** `begin_analyze_document(...)` returns a **poller** (submit-then-poll) — pervasive across Azure AI SDKs (DocIntel, Content Understanding `begin_analyze_binary`, Search indexer runs) because analysis outlives a sync HTTP timeout.
- **Content Understanding:** custom analyzers = **`fieldSchema` on a `baseAnalyzerId`** (e.g. `prebuilt-document`); `estimateFieldSourceAndConfidence` returns value + confidence + source location per field. `markdown` output = normalized doc text ideal for feeding an LLM. `prebuilt-imageSearch` = image representation for search scenarios (≠ field extraction).
- Different prebuilt analyzers return **different result schemas** — inspect raw output before coding against keys.
- ⭐ **Pipeline best practice:** extraction service does reliable schema-grounded extraction → serialize → LLM/agent does judgment (summarize, approve, flag anomalies). Don't ask the LLM to do raw field extraction too.

---

## 15. Azure Functions (as AI tool backends)

| Fact | Detail |
|---|---|
| Trigger types | **HTTP** (main one here), Timer, Blob, Queue, Cosmos DB, and newer **MCP tool trigger** |
| Programming models | **v2 = decorators in one file** (`func.FunctionApp()`) — current; v1 = `function.json` + `__init__.py` per function — legacy, still on exam material |
| Auth levels | `ANONYMOUS` (no key) / `FUNCTION` (per-function or host key) / `ADMIN` (master key) |
| Supplying the key | `x-functions-key` **header** (preferred — query strings leak into logs) or `?code=` query param |
| Enforcement | At the **Functions host/runtime**, *before* your code runs — invalid key → 401 from the host, your Python never executes |
| Role in AI | Hosting layer for custom logic behind agent **OpenAPI tools**, `FunctionTool` backends, AI Search **custom skills**, MCP servers |
| Routing | Path params `{order_id}` (RESTful) — must match the OpenAPI spec paths; OpenAPI `securitySchemes` metadata tells the agent tool config **how** to authenticate (anonymous/key/connection/managed identity) |

---

## 16. Plan & Manage an Azure AI Solution (biggest domain: 20–25%)

> Sections 16–22 cover official-syllabus topics **not in the course** — study these just as hard.

### Responsible AI — the six Microsoft principles (memorize!)

| Principle | Meaning |
|---|---|
| **F**airness | Treat all groups equitably; no bias amplification |
| **R**eliability & Safety | Consistent, safe operation; handle edge cases |
| **P**rivacy & Security | Protect data; secure by design |
| **I**nclusiveness | Empower everyone, including people with disabilities |
| **T**ransparency | Users understand how/why the system decides (explainability) |
| **A**ccountability | Humans are answerable for the system's behavior |

**🧠 Mnemonic: F-R-P-I-T-A — "Fair Robots Protect Individuals Through Accountability."**

### Resource planning

| | Multi-service resource | Single-service resource |
|---|---|---|
| Kind | Azure AI services (one resource) | e.g. Language, Speech, Vision, Translator |
| Endpoint/keys | **One endpoint + one key** for many services | Separate per service |
| Billing | Consolidated | Separate per service |
| Free tier | ✗ | ✔ **F0 free tier** available on many services |
| Use when | One project consuming several services | Isolate cost/access, or want free tier |

- Every resource gets **two keys (Key1/Key2)** → rotate with **zero downtime** (switch app to Key2, regenerate Key1, switch back).
- Store keys in **Azure Key Vault**; app fetches at runtime via managed identity — no secret in config.
- Endpoint format: `https://<resource-name>.cognitiveservices.azure.com/` (or regional `https://<region>.api.cognitive.microsoft.com/`).
- CI/CD: provision via ARM/Bicep/Terraform; keys injected from Key Vault in the pipeline.

### Container deployment (edge / on-prem)

- Many services (Language, Speech, Vision Read, Translator…) ship as **Docker containers** — data stays local (data-residency/compliance), low latency at the edge.
- ⭐ Containers still **must connect to Azure for billing**: run with three required args — **`Eula=accept`, `Billing=<endpoint URI>`, `ApiKey=<key>`**. Usage is metered; your data is NOT sent to Azure.
- **Disconnected containers** exist for approved scenarios (fully offline, commitment-tier licensed).
- Your app calls the **container's local endpoint** instead of the cloud endpoint — same API shape.

### Monitoring & cost

| Tool | What it gives |
|---|---|
| **Azure Monitor metrics** | Out-of-the-box: call counts, errors, latency, tokens — chartable, alertable |
| **Diagnostic settings** | Stream resource logs/metrics to **Log Analytics** (KQL queries), Event Hub, or Storage |
| **Alerts** | Fire on metric thresholds (error rate, latency) or log queries |
| **Cost management** | Costs = pricing tier + per-transaction/per-token usage; Azure OpenAI quota is in **TPM (tokens-per-minute)**, RPM scales with it |

### Harm prevention (responsible-AI toolbox)

| Feature | Detects / does |
|---|---|
| Content filters | Per-deployment severity thresholds on the 4 harm categories (configurable; lowering/disabling needs an approved request) |
| **Prompt Shields** | **Jailbreak** (direct user-prompt attacks) + **indirect attacks** (instructions hidden in documents/data you feed the model) |
| **Groundedness detection** | Flags model claims not supported by the provided source material |
| Protected material detection | Verbatim copyrighted text/code |
| Blocklists | Custom exact-match terms (text-only) |
| Governance framework | Define roles, review gates, monitoring, and incident response for AI systems — accountability in practice |

---

## 17. Generative AI Ops — Hubs, Deployments, Prompt Flow, Fine-Tuning

### Foundry structure

- **Hub** = top-level workspace holding shared config (security, networking, compute, connections). **Project** = child workspace where you actually build. Newer Foundry projects can also be resource-based without a hub — but know hub → project for the exam.
- **Model catalog** = deployable models from Microsoft/OpenAI/Meta/Mistral etc.; choose by capability, cost, context window, region availability.

### Model deployment types (frequent exam question)

| Type | Billing | Use when |
|---|---|---|
| **Standard** | Pay-per-token, regional | Default; variable workloads |
| **Global Standard** | Pay-per-token, routed globally | Higher throughput/availability, no strict data-residency need |
| **Global Batch** | ~50% cheaper, async (24 h window) | Large offline jobs, latency-insensitive |
| **Provisioned (PTU)** | Reserved capacity (provisioned throughput units) | Predictable latency + high steady volume |
| **Serverless API / MaaS** | Pay-per-token, no infra | Catalog models (e.g. Llama, Mistral) without hosting |

### Prompt flow

- Visual **DAG of tools** (LLM tool, Python tool, Prompt tool) connecting inputs → processing → outputs.
- Flow types: **Standard** (general), **Chat** (adds chat history/UX), **Evaluation** (scores other flows' outputs).
- **Variants** = A/B versions of a node (e.g. two prompts) to compare; deploy a flow to a **managed online endpoint** for real-time inference.
- **Prompt templates** parameterize reusable prompts (placeholders filled at runtime).

### Prompt engineering vs RAG vs fine-tuning (decision ladder ⭐)

| Approach | Fixes | Cost |
|---|---|---|
| 1️⃣ Prompt engineering | Behavior, format, tone — try **first** | Cheapest, instant |
| 2️⃣ RAG | Missing/fresh/private **knowledge** | Index + retrieval infra |
| 3️⃣ Fine-tuning | Consistent **style/format/domain vocabulary** prompts can't achieve; shorter prompts at scale | Training + hosting fees; retrain to update |

**🧠 Hook:** *"Prompt for behavior, RAG for knowledge, fine-tune for style."* Fine-tuning does **not** reliably add factual knowledge — that's RAG's job.

- Operationalize: model monitoring + diagnostics (performance, resource consumption), tracing + user feedback collection, **model reflection** (model critiques/improves its own output), containers for edge deployment, orchestration of multiple models (router/planner patterns).

### Agentic extras (5–10% domain)

- **Foundry Agent Service** = managed agents (the course's bread and butter). **Microsoft Agent Framework** = open-source SDK (successor merging **Semantic Kernel + AutoGen**) for complex/custom multi-agent orchestration in code.
- Multi-agent orchestration patterns: **sequential** (pipeline), **concurrent** (fan-out/fan-in), **group chat / debate**, **handoff** (route to specialist) — plus connected agents (`ConnectedAgentTool`) in Foundry.
- Lifecycle: build → **test in playground** → evaluate/trace → optimize → deploy; supports multi-user and autonomous scenarios.

---

## 18. Computer Vision (10–15% — almost none of it is in the course!)

### Azure AI Vision — Image Analysis 4.0

One call: `ImageAnalysisClient.analyze(image, visual_features=[...])`

| VisualFeature | Returns |
|---|---|
| `CAPTION` | One natural-language sentence + confidence (gender-neutral option) |
| `DENSE_CAPTIONS` | Up to 10 captions for regions, with bounding boxes |
| `TAGS` | Content tags + confidence |
| `OBJECTS` | Detected objects + **bounding boxes** |
| `PEOPLE` | People locations + confidence |
| `SMART_CROPS` | Suggested crop regions per aspect ratio |
| `READ` | **OCR — printed AND handwritten text** from images |

**🧠 OCR routing rule:** text **in photos/images** → Vision READ; text in **documents/PDFs/forms** → Document Intelligence; multi-modal + schema-driven → Content Understanding.

### Custom Vision (train your own image model)

| Fact | Detail |
|---|---|
| Project types | **Classification** (multiclass = ONE tag/image; multilabel = MANY tags/image) vs **Object detection** (tags + bounding boxes) |
| Resources | Separate **training** and **prediction** resources (or a combined one) — two endpoints/keys! |
| Data | Label images yourself; ~15+ images/tag minimum, more + varied = better |
| Metrics | **Precision** (of predicted, how many right), **Recall** (of actual, how many found), **mAP** (object detection overall) |
| Publish | Publish a trained **iteration** with a name → call it via the **prediction** endpoint |
| Edge | **Compact domains** export to ONNX / TensorFlow Lite / CoreML for offline/mobile |

### Face API

- Operations: **Detect** (box + attributes: head pose, glasses, occlusion, blur, exposure, mask), **Verify** (1:1 — same person?), **Identify** (1:N against a trained **PersonGroup**), **Find Similar**, **Group**, **Liveness detection** (anti-spoofing).
- ⭐ **Limited Access policy:** identification/verification require an approved use-case registration; emotion/age/gender attributes were **retired** — don't pick them as answers.

### Video Indexer & Spatial Analysis

- **Video Indexer**: extracts multi-channel insights from video/live streams — transcript, OCR, faces, labels, brands, topics, sentiment, scenes/shots/keyframes. Supports **custom person/brand/language models**; consume via REST API (access tokens) or embeddable **widgets**.
- **Spatial Analysis**: detects **presence and movement of people** in video (counting, zone crossing) — CCTV/retail analytics scenarios.

---

## 19. NLP — Custom Language Models & Speech Extras (not in course)

### Summarization (Language service)

- **Extractive** = picks the most important original sentences; **Abstractive** = generates new summary text. Works for documents AND conversations (call transcripts).

### Conversational Language Understanding (CLU) — successor to LUIS

| Concept | Meaning |
|---|---|
| **Intent** | What the user wants (`BookFlight`) |
| **Entity** | Data to extract (`destination = Paris`) — learned, **list**, **prebuilt**, or **regex** types |
| **Utterance** | Example phrase you label for training |
| Lifecycle | Create project → add intents/entities/utterances → **train** → evaluate (precision/recall/F1) → **deploy to a deployment slot** → predict via runtime endpoint |
| **Orchestration workflow** | A router project that sends each utterance to the right CLU / custom QA / LUIS project |
| Ops | Export/import project (backup & recovery), `None` intent catches out-of-scope input |

### Custom text classification

- **Single-label** (one class per doc) vs **multi-label** (many classes per doc). Same label → train → evaluate → deploy lifecycle as CLU/Custom NER, all in **Language Studio**.

### Custom question answering

- Project/knowledge base built from **URLs, FAQ pages, files, and chit-chat personalities** (predefined small-talk sets).
- QnA pairs support **alternate phrasings** (synonym questions) and **follow-up prompts → multi-turn conversations**.
- Lifecycle: add sources → edit pairs → **test** → **publish** → query from client (REST/SDK). **Active learning** suggests new alternate questions from real traffic. KB is **exportable** (backup/migration). Multi-language: one project per language, or translate at the edges.
- **CLU vs QA rule:** QA = static question→answer lookup; CLU = understand intent to **act**. Combine via orchestration workflow.

### Speech — beyond the course's STT/TTS

| Feature | What it is |
|---|---|
| **Custom Speech** | Train custom acoustic/language models on your audio + transcripts → better accuracy on domain vocabulary/accents; measured by **WER (word error rate)**; deploy to a custom endpoint |
| **Custom Neural Voice** | Your own branded TTS voice (Limited Access — approval required) |
| **Keyword recognition** | Offline, on-device wake-word (`KeywordRecognizer`, model from Speech Studio) |
| **Intent recognition** | `IntentRecognizer` — pattern matching or CLU-backed intent from speech |
| **Pronunciation assessment** | Scores accuracy/fluency/completeness of a speaker — language learning |
| **Speaker recognition** | Voice-based verify/identify (Limited Access) |

### Document translation (Translator's second API)

- **Text translation** = synchronous strings. **Document translation** = **async batch**: source & target **blob containers** (SAS URLs), preserves formatting (PDF, DOCX, PPTX…), submit-then-poll.
- **Custom Translator**: train on your parallel documents → category ID used in translate calls; quality measured by **BLEU score**.

---

## 20. Knowledge Mining Deep Dive — Search Internals & Custom Doc Models

### Index field attributes (memorize — constant exam fodder)

| Attribute | Enables |
|---|---|
| `key` | Unique document id (exactly one) |
| `searchable` | Full-text search over it |
| `filterable` | `$filter` expressions |
| `sortable` | `$orderby` |
| `facetable` | Facet counts/navigation |
| `retrievable` | Returned in results (off = usable internally but hidden) |

**🧠 Hook: "Key Search Filters Sort Facets Retrieved."**

### Querying

- **Simple syntax** (default) vs **Full Lucene** (`queryType=full`): wildcards (`azur*`), fuzzy (`azure~`), regex, proximity, term boosting (`azure^3`).
- `search=` (text), `$filter=`, `$orderby=`, `$select=`, `$top`, `facet=`; `searchMode=any|all`.
- **Scoring profiles** boost fields/freshness; **Semantic ranker** re-ranks the top 50 BM25 results with deep models + returns **captions and answers** — improves relevance without reindexing.

### Security & plumbing

- **Admin keys** (2, full CRUD) vs **query keys** (read-only, create many, one per client app).
- Indexer **schedules** (min 5-minute interval), change detection (high-water mark), incremental enrichment **cache** to avoid re-running skills.

### Knowledge store (enrichment output for analytics, not search)

Defined inside the **skillset** with a storage connection; three **projection** types:

| Projection | Output | For |
|---|---|---|
| **Table** | Azure Table storage rows | Power BI / analytics |
| **Object** | JSON blobs | Downstream apps |
| **File** | Extracted images/files | Image archives |

### Custom skills (in a skillset)

- `WebApiSkill` calls your endpoint (typically an **Azure Function**); strict JSON contract: request/response both use a `values` array of `{recordId, data}` — response must echo each `recordId`.

### Document Intelligence — custom & composed models

| Model | Train on | Best for |
|---|---|---|
| **Custom template** | **5+ labeled samples**, fixed layout | Structured, consistent forms |
| **Custom neural** | 5+ labeled samples, varied layout | Unstructured/varied documents |
| **Composed model** | Multiple custom models behind **one model ID** | Service auto-classifies which sub-model fits each doc |

- Label in **Document Intelligence Studio**; lifecycle: train → test → publish/copy to production resource.

---

## 21. Service-Selection Cheat Sheet (Domain 1's favorite question type)

| Scenario keyword | Pick |
|---|---|
| "chat/completion/generate text or code" | Azure OpenAI |
| "ground answers in company documents" | RAG: Azure AI Search + OpenAI (or agent File Search) |
| "agent that takes actions / calls tools" | Foundry Agent Service |
| "caption / tag / objects in photos" | Azure AI Vision — Image Analysis |
| "text in a photo / handwriting in an image" | Vision READ (OCR) |
| "train model on OUR product images" | Custom Vision |
| "recognize / verify a person's face" | Face API (Limited Access) |
| "insights from video archive" | Video Indexer |
| "people movement in camera feed" | Spatial Analysis |
| "sentiment / key phrases / PII / language detect" | Azure AI Language |
| "understand user intent to act" | CLU |
| "FAQ bot from existing docs" | Custom question answering |
| "translate text, many languages, one call" | Translator (text) |
| "translate PDFs/DOCX keeping formatting" | Translator (document translation) |
| "transcribe / synthesize / translate speech" | Azure AI Speech |
| "search over millions of docs, facets, filters" | Azure AI Search |
| "extract invoice/receipt/ID fields" | Document Intelligence prebuilt |
| "extract fields from OUR custom form" | Document Intelligence custom (template/neural) |
| "one API over docs + images + audio + video" | Content Understanding |
| "moderate user content with thresholds" | Content Safety |
| "block jailbreaks / hidden prompt attacks" | Prompt Shields |

---

## 22. Custom-Model Lifecycles — One Pattern to Rule Them All

Every "custom X" on the exam follows the same shape — **Label → Train → Evaluate → Deploy/Publish → Consume**:

| Service | Studio | Evaluate with | Deploy step |
|---|---|---|---|
| Custom Vision | Custom Vision portal | Precision / Recall / mAP | Publish iteration → prediction endpoint |
| Custom NER / text classification / CLU | Language Studio | Precision / Recall / F1 | Deployment slot |
| Custom question answering | Language Studio | Test pane | Publish KB |
| Custom Speech | Speech Studio | **WER** | Custom endpoint |
| Custom Translator | Custom Translator portal | **BLEU** | Publish → category ID |
| Doc Intelligence custom | Document Intelligence Studio | Confidence on test docs | Model ID (compose optional) |

---

## 23. ⚡ Rapid-Fire Recap Tables (Last-Day Revision)

### "Which auth header?"

| Header / mechanism | Used by |
|---|---|
| `api-key` | Azure OpenAI |
| `Ocp-Apim-Subscription-Key` | Cognitive Services REST (Language, Translator, Speech, Vision) |
| `x-functions-key` / `?code=` | Azure Functions |
| Bearer token (Entra ID) | Everything, in production |

### "Sync or poll?"

| Call | Pattern |
|---|---|
| Content Safety `analyze_text` / `analyze_image` | **Synchronous** |
| Language `TextAnalyticsClient` operations | Synchronous |
| Document Intelligence / Content Understanding `begin_*` | **LRO submit-then-poll (poller)** |
| Batch speech transcription | Submit-then-poll |
| Real-time speech | Streaming + events |

### "Which number was that again?"

| Number | Meaning |
|---|---|
| 0 / 2 / 4 / 6 | Content Safety severities: Safe / Low / Medium / High (0–7 finer scale via `output_type`) |
| ≥ 4 gate | Block Medium+ before prompting the LLM |
| 0–2 | `temperature` range |
| +33% | base64 image payload inflation |
| 100+ | Translator languages |
| 1.0 | Sum of sentiment confidence scores per sentence |

### "Prebuilt vs custom" (recurring pattern across services)

| Service | Prebuilt | Custom |
|---|---|---|
| Language NER | Fixed taxonomy (Person, Location…) | Custom NER via Language Studio training |
| Translator | General MT | Custom Translator (domain terminology) |
| Document Intelligence | read / layout / invoice / receipt / ID | Custom extraction models |
| Content Understanding | prebuilt-invoice, prebuilt-imageSearch, prebuilt-document | Custom analyzer = `fieldSchema` + `baseAnalyzerId` |
| Content Safety | 4 harm categories | Custom blocklists (text-only) |
| Vision | Image Analysis 4.0 features | Custom Vision (classification / object detection) |
| Language understanding | Prebuilt sentiment/NER/PII | CLU intents/entities, custom text classification |
| Q&A | — | Custom question answering KB |
| Speech | Standard STT/TTS, neural voices | Custom Speech (WER), Custom Neural Voice |

### "Who executes it?"

| Thing | Executor |
|---|---|
| `web_search`, `code_interpreter`, `file_search` | Service (server-side) |
| `FunctionTool` / function calling | **Your client code** |
| OpenAPI tool | Service → your REST API |
| MCP tool | Service → MCP server (`require_approval`, `allowed_tools` guard it) |
| LangChain/LangGraph tools | Your client process (framework orchestrates) |

### Stateful vs stateless

| Stateless | Stateful |
|---|---|
| `responses.create()` one-off | `conversations.create()` (server-side state) |
| `PromptAgentDefinition` | Hosted agent + Thread |
| LangGraph `invoke()` per run | LangGraph + checkpointer |
| Chat Completions (resend history) | Responses API conversations |

### SDK class ↔ purpose cheat sheet

| Class | Service |
|---|---|
| `AzureOpenAI` / `OpenAI` | Azure OpenAI |
| `AIProjectClient` | AI Foundry project |
| `SearchClient`, `VectorizableTextQuery` | AI Search |
| `TextAnalyticsClient` | Language |
| `SpeechConfig` + `SpeechRecognizer` | Speech STT |
| `SpeechTranslationConfig` + `TranslationRecognizer` | Speech translation |
| `ContentSafetyClient` | Content Safety |
| `DocumentIntelligenceClient` | Document Intelligence |
| `ContentUnderstandingClient` | Content Understanding |
| `AzureKeyCredential` / `DefaultAzureCredential` | Auth (key / Entra ID) |

---

## 24. 🏁 One-Liner Flash Facts (read 10 minutes before the exam)

1. Entra ID > API key for production: short-lived tokens + RBAC, no secret to rotate.
2. `model=` on Azure = **deployment name**.
3. `input_image` = Responses API; `image_url` = Chat Completions.
4. Function tools: **model proposes, YOU execute**, resubmit with matching `call_id`.
5. Built-in tools (web search, code interpreter, file search) run **server-side**.
6. `tool_choice`: auto / required / none / specific tool.
7. Prompt agent = stateless; hosted agent + thread = server-side memory.
8. `extra_body` smuggles Azure-specific fields (`agent_reference`) through the OpenAI SDK.
9. Keyword = BM25, vector = k-NN, hybrid = both merged by **RRF**.
10. `k_nearest_neighbors` = vector candidates; `top` = final results.
11. Skillset pipeline: OCR → Merge → LanguageDetection → KeyPhrase; custom skill = Azure Function.
12. Built-in evaluators: **Completeness, Groundedness, Relevance**.
13. Trace **metadata** and trace **content** are separate toggles (compliance).
14. Mask editing (inpainting): **transparent = regenerated**, same-size PNG required.
15. Image edit source = binary multipart, not URL/base64.
16. Content Safety categories: **Hate, SelfHarm, Sexual, Violence**; severities 0/2/4/6.
17. Content filtering (built into Azure OpenAI, per-deployment) ≠ Content Safety (own service, granular scores); filtering adds **jailbreak** + **protected material** detection.
18. Blocklists = text-only.
19. One `TextAnalyticsClient` does language detect, NER, PII, sentiment, key phrases.
20. Sentiment = sentence AND document level in one call; Opinion Mining = target + assessment pairs.
21. Language detection returns **ISO 639-1**; detect first, then pass `language=` for accuracy.
22. Custom NER needed for domain categories (ticket_id, sla_tier) — prebuilt taxonomy is fixed.
23. Translator: `targets` array = multi-language in one call; omit `from` = auto-detect.
24. `recognize_once` = short commands; continuous + events = live; batch = files (adds diarization + timestamps).
25. SSML (`speak_ssml_async`) for pronunciation/pauses/pitch — not a `speak_text_async` param.
26. Neural voices are the standard; locale = `en-US` format.
27. `TranslationRecognizer` → `ResultReason.TranslatedSpeech` (distinct from `RecognizedSpeech`).
28. `prebuilt-read` = OCR; `prebuilt-layout` = structure, no fields; `prebuilt-invoice` = fields + confidence.
29. `begin_*` poller = LRO pattern across DocIntel, Content Understanding, Search indexers.
30. Extraction service extracts; LLM judges — keep the pipeline responsibilities separate.
31. Functions auth: ANONYMOUS / FUNCTION / ADMIN; enforced by the host before your code runs; header beats query string.
32. Functions v2 = decorators (current); v1 = function.json (legacy).
33. MCP tools return JSON **strings**; Foundry MCP agents use `require_approval` + `allowed_tools`.
34. Foundry **connections** = named credential/endpoint indirection for external resources.
35. Least privilege: assign the narrowest RBAC role that can still do the job.
36. Responsible AI principles: **F-R-P-I-T-A** — Fairness, Reliability & safety, Privacy & security, Inclusiveness, Transparency, Accountability.
37. Multi-service resource = one key/endpoint for many services; single-service = separate + **F0 free tier**.
38. Two keys per resource → zero-downtime rotation; keep them in **Key Vault**.
39. Containers need `Eula=accept`, `Billing=<uri>`, `ApiKey=<key>` — billing metadata goes to Azure, **your data doesn't**.
40. Diagnostic settings stream logs to **Log Analytics** (KQL); Azure OpenAI quota = **TPM**.
41. Deployment types: Standard, Global Standard, Global **Batch** (~50% off, async), **PTU** (reserved throughput), Serverless/MaaS.
42. Decision ladder: **prompt-engineer first, RAG for knowledge, fine-tune for style** — fine-tuning ≠ adding facts.
43. Prompt flow = DAG of LLM/Python/Prompt tools; Standard / Chat / **Evaluation** flow types; variants for A/B.
44. Prompt Shields: **jailbreak** (direct) + **indirect** (hidden-in-documents) attack detection; groundedness detection flags unsupported claims.
45. Image Analysis features: CAPTION, DENSE_CAPTIONS, TAGS, OBJECTS, PEOPLE, SMART_CROPS, **READ** (OCR incl. handwriting).
46. OCR routing: image → Vision READ; document/form → Document Intelligence; multi-modal schema → Content Understanding.
47. Custom Vision: multiclass = one tag, multilabel = many; separate **training vs prediction** resources; publish an iteration; compact domains export to edge.
48. Face: Verify = 1:1, Identify = 1:N (PersonGroup); identification is **Limited Access**; emotion/age/gender attributes retired.
49. Video Indexer = transcript/OCR/faces/brands/topics from video; Spatial Analysis = people presence/movement in feeds.
50. Summarization: **extractive** picks sentences, **abstractive** writes new ones.
51. CLU = intents + entities + utterances → train → deploy slot; **orchestration workflow** routes between CLU and QA projects.
52. Custom QA: sources + alternate phrasing + follow-up prompts (multi-turn) + chit-chat; active learning suggests alternates.
53. Custom Speech quality = **WER**; Custom Translator quality = **BLEU**; both deploy to custom endpoints/categories.
54. Document translation = async batch between **blob containers** (SAS), preserves formatting.
55. Index attributes: key, searchable, **filterable**, sortable, facetable, retrievable — filterable ≠ searchable!
56. Full Lucene (`queryType=full`) unlocks wildcards, fuzzy `~`, regex, boosting `^`.
57. Semantic ranker re-ranks **top 50** BM25 results and adds captions/answers.
58. Admin keys manage; **query keys** for read-only client apps.
59. Knowledge store projections: **table** (Power BI), **object** (JSON), **file** (images) — defined in the skillset.
60. Custom skill = WebApiSkill → your Azure Function; JSON `values`/`recordId` contract must be echoed back.
61. Doc Intel custom: **template** = fixed forms, **neural** = varied docs, both need **5+ labeled samples**; **composed model** = many customs behind one ID.
62. Microsoft Agent Framework (Semantic Kernel + AutoGen successor) for code-first multi-agent; Foundry Agent Service for managed agents.
63. Every "custom X" lifecycle: **Label → Train → Evaluate → Deploy → Consume.**

---

*Good luck — you've built every one of these patterns in this course. The exam is recognizing them in scenario form.* 🚀
