# Azure AI-103 — Hands-On Labs

> Beginner-friendly guide to every `.py` file in this folder: what it teaches, what it needs to run, and how the code works.

This `Labs/` folder is a second, complementary set of materials for the "Azure AI 103" course (the main [`14-azure-ai103/`](../README.md) chapter). It's different in nature from the `NN. Section Code/` folders next door:

| | `01–08. Section Code/` | `Labs/` (this folder) |
|---|---|---|
| Style | Instructor walkthrough scripts | Hands-on lab exercises |
| Completeness | All complete, all paired with a teaching notebook | **Mostly complete**, but a few are intentionally unfinished "fill in the blank" exercises (called out below) |
| Sample data | Self-contained | **Several scripts expect a local data file/folder that isn't included in this repo** (e.g. `reviews/`, `data/events.txt`, `speech.wav`) — these came from the course's separate downloadable lab-files ZIP. You'll need to create your own sample files (formats are described per-lab below) before those scripts will run |

The folder is organized into four lab groups:

| Group folder | Labs | Topic |
|---|---|---|
| [`Generative AI (Lab-1-5)/`](<Generative AI (Lab-1-5)>) | 1–5 | Azure OpenAI chat basics: connecting via the Foundry SDK, Chat Completions vs. Responses API, streaming, async, file-search tools, and prompt engineering/evaluation |
| [`AI Agents(Lab 6-11)/`](<AI Agents(Lab 6-11)>) | 6–11 | Azure AI Foundry Agent Service: function-calling tools, MCP (Model Context Protocol) tools, and the new Microsoft Agent Framework |
| [`NLP and Speech ( Lab 12-17 )/`](<NLP and Speech ( Lab 12-17 )>) | 12–17 | Azure AI Language (text analytics) and Azure AI Speech (text-to-speech, speech-to-text, real-time voice agents) |
| [`Vision, Documents and Extended Agents (Lab 18-28)/`](<Vision, Documents and Extended Agents (Lab 18-28)>) | 18–28 | **New — not part of the original course mirror.** Multi-agent delegation, Foundry IQ/Work IQ/A2A (preview), image generation, Content Safety, Content Understanding, Document Intelligence, Azure AI Search RAG, Sora 2 video (preview), and translation |

Labs 1, 2, and 5 (originally missing) and the entire fourth group (18–28) were **added by an assistant session, not shipped by the original course** — see [AI-103 coverage](#ai-103-exam-coverage) below for why, and each new lab's own header comment for exactly what's proven-working vs. best-effort/preview.

---

## AI-103 exam coverage

The original course mirror (Labs 3–4, 6–17) only exercised two of AI-103's five exam domains — **Domain 2** (generative AI & agentic solutions, 30–35%) well, and most of **Domain 4** (text analysis, 10–15%). **Domains 1, 3, and 5** (roughly half the exam) had no hands-on lab coverage at all, even though the paired `Section Code/` notebooks elsewhere in this chapter do cover some of that ground as instructor walkthroughs. Labs 1, 2, 5, and 18–28 were added specifically to close that gap:

| Gap (per `EXAM_NOTES.md`'s course map) | Filled by | Confidence |
|---|---|---|
| Model catalog / Foundry SDK chat setup / prompt optimization & evaluation | Lab 1, 2, 5 | ✅ Proven — adapted from real working code in this repo (`11_azure_ai_foundry/00_setup/`, `03. Section Code/03_response_completeness.py`) |
| Multi-agent delegation (`ConnectedAgentTool`) | Lab 18 | ✅ Proven — ported from `11_azure_ai_foundry/06_connected_agents/` |
| Foundry IQ (shared knowledge platform) | Lab 19 | ⚠️ Best-effort/preview — no verified SDK found anywhere; illustrative only |
| Microsoft 365 / Work IQ agent publishing | Lab 20 | ⚠️ Best-effort/preview — mostly a portal/admin-center workflow, not a single SDK call |
| A2A protocol (agent-to-agent) | Lab 21 | ⚠️ Best-effort/preview — modeled on the public A2A spec, package name unconfirmed |
| Image generation & editing | Lab 22 | ✅ Proven — adapted from `05. Section Code/` |
| Content Safety (dedicated client) | Lab 23 | ✅ Proven — adapted from `05.`/`08. Section Code/` |
| Content Understanding | Lab 24 | ⚠️ Unverified SDK — the package isn't installed/resolvable in this repo; see the file's header |
| Document Intelligence | Lab 25 | ✅ Proven — adapted from `08. Section Code/01_document_intelligence.py` |
| Azure AI Search (RAG / knowledge mining) | Lab 26 | ✅ Proven — adapted from `02. Section Code/08_ai_search.py` + `10_customer_rag_client.py` |
| Sora 2 video generation | Lab 27 | ⚠️ Best-effort/preview — no video-generation code exists anywhere else in this repo to port from |
| Translation | Lab 28 | ✅ Proven — adapted from `06. Section Code/04_text_translation.py` + `09_text_translation.py` |

Every "⚠️ Best-effort/preview" lab still runs — each has a graceful fallback that explains the concept instead of crashing uninformatively if the guessed API surface turns out to be wrong on your SDK version. Read that file's top-of-file comment block before trusting any of its specific class/method names.

---

## Before you start

### 1. Environment

Same root environment as the rest of the repo:

```bash
source venv/bin/activate
pip3 install -r requirements.txt
```

### 2. Azure resources

Every lab talks to a **live Azure service** — none of these run offline. You need, at minimum, an Azure AI Foundry project (same one used in [`11_azure_ai_foundry/`](../../11_azure_ai_foundry/)) and to be logged in with `az login` (all scripts use `DefaultAzureCredential` / `AzureCliCredential` — no API keys).

Some labs also need an **agent already created in the Foundry portal** before you run the script (the script only *connects* to it — it doesn't create it). These are marked "**Needs a pre-built portal agent**" below.

### 3. Environment variables

The scripts in this folder are **not consistent with each other** about variable names (each lab was authored independently by the course). Add whichever set the lab you're running needs to your `.env`:

| Variable | Used by | Meaning |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT`, `MODEL_DEPLOYMENT` | Lab 3 (all 4 files) | Azure OpenAI endpoint + deployment name for direct Chat/Responses API calls |
| `PROJECT_ENDPOINT`, `MODEL_DEPLOYMENT_NAME` | Lab 6, 7, 8, 10 | Foundry **project** endpoint + model deployment, for creating agents in code |
| `AGENT_NAME` | Lab 6 (`agent.py`), Lab 9 | Name of an agent already created in the Foundry portal |
| `FOUNDRY_ENDPOINT`, `AGENT_NAME` | Lab 12, 13, 16 | Foundry endpoint + portal-created agent name |
| `FOUNDRY_ENDPOINT`, `FOUNDRY_KEY` | Lab 15 | Foundry endpoint + key, for the Speech SDK |
| `MODEL_ENDPOINT`, `MODEL_NAME` | Lab 14 | Azure OpenAI endpoint + deployment for the `gpt-4o-*-tts`/`whisper`-style audio models |
| `AZURE_AI_PROJECT_ENDPOINT`, `AZURE_AI_MODEL_DEPLOYMENT_NAME` | Lab 11 | Same idea as `PROJECT_ENDPOINT`/`MODEL_DEPLOYMENT_NAME`, different names — this repo's root `.env` already has `AZURE_AI_PROJECT_ENDPOINT`/`AZURE_AI_MODEL_DEPLOYMENT` (no `_NAME` suffix), so double-check before assuming it "just works" |
| `AZURE_VOICELIVE_ENDPOINT`, `AZURE_VOICELIVE_AGENT_ID`, `AZURE_VOICELIVE_PROJECT_NAME` | Lab 17 | Azure AI VoiceLive real-time voice endpoint + portal agent + project name |
| `PROJECT_ENDPOINT`, `MODEL_DEPLOYMENT_NAME` | Lab 1, 2, 5, 18, 19, 20, 26 (LLM half), 28 (LLM half) | Same Foundry project endpoint + model deployment as Labs 6/7/8/10 above |
| `KNOWLEDGE_SOURCE_NAME` | Lab 19 | Name of a knowledge source registered for Foundry IQ (⚠️ preview — see the file's header) |
| `AGENT_NAME` | Lab 20 | Name of an existing portal agent to (illustratively) publish |
| `AZURE_OPENAI_ENDPOINT`, `IMAGE_DEPLOYMENT_NAME` | Lab 22 | Azure OpenAI endpoint + an image-generation deployment (e.g. `gpt-image-2`) |
| `CONTENT_SAFETY_ENDPOINT`, `CONTENT_SAFETY_KEY` | Lab 23 | Content Safety resource, key-based auth |
| `CONTENT_UNDERSTANDING_ENDPOINT`, `CONTENT_UNDERSTANDING_KEY` | Lab 24 | Content Understanding resource (⚠️ unverified SDK — see the file's header) |
| `DOCUMENT_INTELLIGENCE_ENDPOINT`, `DOCUMENT_INTELLIGENCE_KEY` | Lab 25 | Document Intelligence resource, key-based auth |
| `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_INDEX_NAME` | Lab 26 | An Azure AI Search resource with an index already created/populated |
| `AZURE_OPENAI_ENDPOINT`, `VIDEO_DEPLOYMENT_NAME` | Lab 27 | Azure OpenAI endpoint + a video-generation deployment (⚠️ preview — see the file's header) |
| `AZURE_TRANSLATOR_ENDPOINT`, `AZURE_TRANSLATOR_KEY` | Lab 28 | Azure AI Translator resource, for the dedicated-service half |

> 💡 If a script raises `KeyError` / passes `None` as an endpoint, it's almost always a missing or misnamed `.env` variable — check the table above against the exact `os.getenv(...)` calls at the top of the file.

### 4. Packages not in the root `requirements.txt`

Install these on demand, only for the lab that needs them:

| Package | Needed by |
|---|---|
| `azure-ai-textanalytics` | Lab 12 |
| `azure-cognitiveservices-speech` | Lab 15, 16 (imported as `azure.cognitiveservices.speech`; despite the name, Lab 16's code doesn't actually call it — see below) |
| `playsound3` | Lab 14, 15 |
| `pyaudio` | Lab 17 (also needs PortAudio installed at the OS level — on macOS: `brew install portaudio` before `pip install pyaudio`) |
| `azure-ai-voicelive` | Lab 17 |
| `fastmcp` | Lab 8 (`server.py`) |
| `agent-framework`, `agent-framework-azure-ai` (import name `agent_framework`, `agent_framework.foundry`) | Lab 10, 11 — Microsoft's new (2025+) Agent Framework SDK |
| `azure-ai-agents` | Lab 18 (already in root `requirements.txt` for chapter 11) |
| `azure-ai-contentsafety` | Lab 23 |
| `azure-ai-contentunderstanding` | Lab 24 (⚠️ preview package — not installed/resolvable in this repo, see the file's header) |
| `azure-ai-documentintelligence` | Lab 25 |
| `azure-search-documents` | Lab 26 |
| `a2a` (or whatever the current official A2A SDK package is named) | Lab 21 (⚠️ preview/unverified — see the file's header) |
| `requests` | Lab 28 (Translator's REST call — already a transitive dependency, no extra install needed) |

`azure-ai-projects`, `azure-identity`, `openai`, `python-dotenv`, `pydantic`, and `mcp` are already in the root `requirements.txt`.

### 5. How to run any lab

```bash
cd "14-azure-ai103/Labs/<group folder>/Lab <N>"
python <script>.py
```

---

## Generative AI (Lab-1-5)

Azure OpenAI basics — same Azure OpenAI resource as chapter `01. Section Code/`, but focused on the difference between the older **Chat Completions API** and the newer **Responses API** (which tracks conversation state for you server-side via `previous_response_id`).

### Lab 1 — `verify_connection.py` (NEW — connection smoke test)

The AI-103 course map's first module, "Plan/prepare AI development" + "model catalog: select, deploy, evaluate" — before writing a chat app or agent, prove your Foundry project, deployment, and `az login` identity actually work. Connects via `AIProjectClient` → `get_openai_client()` → one `chat.completions.create()` call, mirroring `11_azure_ai_foundry/00_setup/verify_connection.py`. Note: there's no real SDK call anywhere (in this repo or the public SDK, as far as this lab's author could verify) that *lists* the model catalog itself — picking/deploying a model is a Foundry **portal** action; this script only verifies the deployment you already picked actually responds.

**Run:** `python verify_connection.py` from `Lab 1/`.

### Lab 2 — `chat-app-foundry-sdk.py` (NEW — chat via the Foundry SDK, not a raw endpoint)

Same interactive chat loop as Lab 3's `chat-app-responseapi.py`, but the client is built differently: instead of `OpenAI(base_url=azure_openai_endpoint, api_key=token_provider)` (Lab 3's approach — a generic client pointed at an endpoint), this goes through `AIProjectClient(...).get_openai_client()` — the same route every agent lab (6+) in this folder uses. This is the actual **"Foundry SDK"** module named in the course map, and it matters architecturally: only the Foundry-SDK route lets you later swap in agent references, project connections (Search, MCP), and managed auth without changing your chat code.

**Run:** `python chat-app-foundry-sdk.py` from `Lab 2/`.

### Lab 3 — Chat apps (4 variations of the same idea)

All four files build the exact same console chatbot ("ask a question, get an answer, loop until you type `quit`") but each demonstrates a different API shape. Reading them side-by-side is the point of this lab — diff them to see what changes.

| File | What it demonstrates | Status |
|---|---|---|
| `chat-app-chatcompletion.py` | The classic **Chat Completions API** (`client.chat.completions.create`) — you build and resend the full `messages` list yourself every turn (though this simple version doesn't even keep history — every question is a fresh conversation) | Complete |
| `chat-app-responseapi.py` | The newer **Responses API** (`client.responses.create`) — no `messages` array; you pass plain `input` text plus `previous_response_id` and Azure keeps conversation history for you server-side | Complete |
| `chat-app-responseapi-stream.py` | Same as above, but `stream=True` — tokens print as they arrive instead of waiting for the full reply. Loops over `response.response.output_text.delta` events for each chunk, and grabs the final `response.id` from the `response.completed` event | Complete |
| `chat-async.py` | Same as `chat-app-responseapi.py`, but using `AsyncOpenAI` + `asyncio` + `azure.identity.aio` — every call is `await`ed. Shows how to close the async credential in a `finally` block | Complete |

**How they work, step by step** (applies to all four):
1. `load_dotenv()` reads `AZURE_OPENAI_ENDPOINT` and `MODEL_DEPLOYMENT` from `.env`.
2. `get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")` — instead of an API key, the client authenticates with your `az login` identity (a bearer token that auto-refreshes).
3. An `OpenAI` client (not `AzureOpenAI`!) is pointed at the Azure endpoint via `base_url=`. This is the "OpenAI-compatible" way of talking to Azure OpenAI through Foundry.
4. `while True:` loop reads a prompt, sends it, prints the reply, repeats until you type `quit`.

**Run:**
```bash
cd "14-azure-ai103/Labs/Generative AI (Lab-1-5)/Lab 3"
python chat-app-chatcompletion.py      # or any of the other 3
```

### Lab 4 — File-search tool (`tools-app.py`)

⚠️ **This file is an unfinished skeleton, not a working script.** It's the lab's *starting point* — the comments (`# Initialize the OpenAI client`, `# Create vector store and upload files`, `# Get a response using tools`) are blanks a student is meant to fill in. As shipped, running it will connect but the "get a response" section does nothing (empty `try` body means every prompt is silently ignored).

**What it's meant to teach:** uploading files to Azure OpenAI, creating a vector store from them, and asking questions that get answered by searching those files (Retrieval / File Search tool) — conceptually the same pattern as chapter `07_rag/`, but using Azure OpenAI's built-in file-search tool instead of a hand-rolled RAG pipeline.

**To complete it, you'd add:**
1. An `OpenAI`/`AzureOpenAI` client (same pattern as Lab 3).
2. `openai_client.vector_stores.create(...)` + `openai_client.vector_stores.files.upload_and_poll(...)` to index some documents.
3. Inside the loop, `openai_client.responses.create(..., tools=[{"type": "file_search", "vector_store_ids": [...]}], input=input_text, previous_response_id=last_response_id)`.

### Lab 5 — `prompt_optimization_and_evaluation.py` (NEW — prompt engineering + LLM-as-judge)

The course map's "optimize" module. Two parts in one script:
1. **Few-shot prompt engineering** — `few_shot_classify()` classifies review sentiment using a prompt with worked examples baked in, instead of a bare zero-shot question, for a far more consistent one-word answer format.
2. **LLM-as-judge evaluation** — `draft_then_judge()` reproduces the exact pattern this repo's real `03. Section Code/03_response_completeness.py` uses: draft an answer, have a SECOND model call grade it ("reply with only COMPLETE or MISSING"), and only pay for a third regeneration call if the judge actually flagged a gap. Note this repo doesn't use the `azure-ai-evaluation` SDK anywhere — "evaluation" here means this hand-rolled critique pattern, not a dedicated evaluation package.

Fine-tuning (the third named "optimize" topic) is intentionally NOT covered — it's a long-running async training job that doesn't fit an interactive lab; see `06_large_language_models/03_llm_fine_tuning/` elsewhere in this repo instead.

**Run:** `python prompt_optimization_and_evaluation.py` from `Lab 5/`.

---

## AI Agents (Lab 6–11)

Azure AI Foundry **Agent Service** — persistent, named agents you create once (in code or the portal) and then converse with over a `conversation` object. Same SDKs as [`11_azure_ai_foundry/`](../../11_azure_ai_foundry/) (`azure-ai-projects`, `azure-identity`), but using the newer `PromptAgentDefinition` / `create_version` / `conversations` API shape rather than the older `threads`/`runs` shape.

### Lab 6 — `agent_with_functions.py` (IT support agent, portal-created)

**Needs a pre-built portal agent** (set `AGENT_NAME`, default `it-support-agent`) that already has tools like **code interpreter** enabled in the Foundry portal — this script doesn't define any tools itself.

**What it teaches:** how to *consume* an agent that can return more than text — code-interpreter-generated charts, downloadable files, or file citations — and save that output locally.

**Key mechanics:**
- `project_client.agents.get(agent_name=...)` loads the existing agent (no creation step).
- A single shared `conversation` object is created once, then every turn adds a message via `conversations.items.create(...)` and requests a reply via `responses.create(conversation=conversation.id, extra_body={"agent_reference": {...}})`.
- The interesting part is `format_output_text()` / `download_container_file()`: when the agent's code interpreter produces a file (e.g. a chart image), the response only contains a *citation* pointing at a sandboxed container file — this code downloads that file via `openai_client.containers.files.content.retrieve(...)` and saves it to `agent_outputs/` so you can actually open it.
- Images the agent generates directly (`item.type == "image"`) are base64-decoded and saved as `chart_N.png`.

**Run:** `python agent_with_functions.py` from inside `Lab 6/`.

### Lab 7 — `agent.py` + `functions.py` (astronomy agent, function-calling from scratch)

**Complete, self-contained example of function calling** — the clearest one to study if you're new to agent tools.

- **`functions.py`** defines three plain Python functions (`next_visible_event`, `calculate_observation_cost`, `generate_observation_report`) that read from `data/events.txt`, `data/telescope_rates.txt`, and `data/priority_multipliers.txt` — **these three files aren't included in this repo**; you must create them yourself. Formats, inferred from the loader code:
  - `data/events.txt` — pipe-delimited, one event per line: `EventName|EventType|MM-DD|location1;location2;location3`
  - `data/telescope_rates.txt` and `data/priority_multipliers.txt` — pipe-delimited `key|number` per line, e.g. `standard|50.0`
- **`agent.py`** builds a `FunctionTool` for each function (JSON-schema description of its parameters), creates a new agent (`project_client.agents.create_version(...)`) with all three tools, then runs a chat loop:
  1. Send the user's message.
  2. If the reply contains `function_call` items, look up the matching Python function by name, call it with `json.loads(item.arguments)`, and wrap the result as a `FunctionCallOutput`.
  3. Send those outputs back with `previous_response_id=response.id` to get the agent's final natural-language answer.
  4. At the very end, the agent version is deleted (`agents.delete_version(...)`) — this script cleans up after itself, unlike Lab 6.

**Run:** `python agent.py` from `Lab 7/` (needs the `data/` files above created first).

### Lab 8 — Two independent MCP demos

This folder actually contains **two unrelated exercises** that happen to share a directory:

**A) `agent.py` — hosted/remote MCP tool**
Creates an agent whose only tool is a *remote* MCP server (`MCPTool(server_url="https://learn.microsoft.com/api/mcp", require_approval="always")`) — Microsoft's public Learn docs MCP server. Because `require_approval="always"`, every tool call the agent wants to make comes back as an `mcp_approval_request` item that your code must explicitly approve (`McpApprovalResponse(approve=True, ...)`) before the agent can proceed — this loop repeats until there are no more pending approvals. This is the safety pattern real MCP integrations use to avoid an agent silently calling untrusted external tools.

**B) `server.py` + `client.py` — your own local MCP server**
- `server.py` is a **tiny MCP server** built with `fastmcp`. It exposes two tools, `get_inventory_levels()` and `get_weekly_sales()`, that just return hardcoded dicts — this is the simplest possible MCP server you can write.
- `client.py` **launches `server.py` as a subprocess** over stdio (`StdioServerParameters(command="python", args=["server.py"])`), lists its tools via MCP, then **wraps every MCP tool as a Foundry `FunctionTool`** so a Foundry agent can call it. When the agent calls a "function", the code actually forwards that call over MCP to the local server and returns its result. This is the pattern for bridging *any* MCP server into Foundry's agent tool-calling.
- Run **`client.py`** (it starts the server for you — don't run `server.py` directly): `python client.py` from `Lab 8/`.

### Lab 9 — `agent_client.py` (minimal chat client, portal-created agent)

**Needs a pre-built portal agent** (`PROJECT_ENDPOINT` + `AGENT_NAME`). This is the simplest possible "talk to an existing Foundry agent" script — no tools, no function calling, just: connect → get agent → create conversation → loop sending messages and printing `response.output_text`. Good starting point if Lab 6/7's extra machinery is overwhelming — read this one first.

### Lab 10 — `agent-framework.py` (expense-claim agent, Microsoft Agent Framework)

Switches SDKs entirely: instead of `azure-ai-projects`, this uses **Microsoft's Agent Framework** (`agent_framework`, `agent_framework.foundry`) — a higher-level, framework-style API (decorator-based tools, `async with Agent(...) as agent:`) that sits on top of Foundry.

- **Needs a `data.txt` file in the same folder** (not included) containing sample expense data as plain text — the script reads it and hands it to the agent along with your prompt.
- `@tool(approval_mode="never_require")` turns a plain Python function (`submit_claim`) into an agent tool with **zero approval friction** — contrast this with Lab 8's `require_approval="always"` MCP tool, showing the two ends of the trust spectrum.
- The agent's instructions tell it to draft an expense email and call `submit_claim(to, subject, body)` — which just prints the "email" to the console rather than actually sending one.

**Run:** `python agent-framework.py` from `Lab 10/` (create `data.txt` first, e.g. a few lines like `Taxi - $45.00`, `Hotel - $210.00`).

### Lab 11 — `agents.py` (sequential multi-agent pipeline)

Also uses the Agent Framework, but demonstrates **orchestrating multiple agents in sequence** via `agent_framework.orchestrations.SequentialBuilder` — the output of one agent becomes the input to the next.

- Three agents, each with narrow instructions: `summarizer` (condense feedback to one sentence) → `classifier` (label it Positive/Negative/Feature request) → `action` (suggest next step).
- `SequentialBuilder(participants=[...], output_from="all").build()` wires them into a pipeline; `workflow.run(...)` runs all three in order and `result.get_outputs()` returns every agent's message, which the script prints numbered and labeled by author.
- This is a hardcoded example (fixed `feedback` string) meant purely to show the orchestration pattern, not to take live input.
- Uses `AZURE_AI_PROJECT_ENDPOINT` / `AZURE_AI_MODEL_DEPLOYMENT_NAME` (see the env-var table above for the naming mismatch vs. this repo's root `.env`).

**Run:** `python agents.py` from `Lab 11/`.

---

## NLP and Speech (Lab 12–17)

Azure AI **Language** (text analytics) and **Speech** services — some labs call the dedicated Language/Speech SDKs directly, others go through a Foundry agent that has those capabilities configured in the portal.

### Lab 12 — `text-analysis.py`

Calls **Azure AI Language** directly via `azure.ai.textanalytics.TextAnalyticsClient` (same service as `06. Section Code/` in the main chapter, but the dedicated-SDK approach rather than an LLM prompt).

- **Needs a `reviews/` folder in the same directory** containing `.txt` files (not included) — the script iterates every file in it.
- For each review file, it prints: detected language (`detect_language`), named entities (`recognize_entities`), and PII entities plus a redacted version of the text (`recognize_pii_entities` → `.redacted_text`).
- Good first example of the "batch documents in, structured predictions out" shape common to all Text Analytics operations.

**Run:** `python text-analysis.py` from `Lab 12/` (create a `reviews/` folder with a few `.txt` files first, e.g. product review text containing a name/email to see PII redaction in action).

### Lab 13 — `text-agent.py`

**Needs a pre-built portal agent** (`FOUNDRY_ENDPOINT` + `AGENT_NAME`). The simplest possible one-shot pattern: read one prompt, send it to the agent, print the reply, exit — no loop. Useful as a smoke test that your agent/credentials are wired up correctly before running the more elaborate labs.

### Lab 14 — Text-to-speech and speech-to-text via Azure OpenAI audio models

Two independent scripts, both using `AzureOpenAI` (not the Speech SDK) — i.e. **LLM-hosted audio models** rather than the dedicated Speech service. Contrast with Lab 15/16, which use the dedicated Speech SDK for the same kind of task.

| File | What it does |
|---|---|
| `generate-speech.py` | `client.audio.speech.with_streaming_response.create(model=..., voice="alloy", input="My voice is my passport!", instructions="Speak in a serious tone.")` — streams synthesized speech to `speech.mp3` in the same folder, then plays it with `playsound3` |
| `transcribe-speech.py` | Opens **`speech.wav`** (not included — you'll need your own sample audio file, or point it at the `speech.mp3` the first script generates by adjusting the filename/extension) and calls `client.audio.transcriptions.create(model=..., file=..., response_format="text")` to transcribe it |

Both need `MODEL_ENDPOINT` / `MODEL_NAME` pointing at an Azure OpenAI audio-capable deployment (a TTS model like `gpt-4o-mini-tts` for the first script, a transcription model like `gpt-4o-transcribe`/`whisper` for the second).

**Run:** `python generate-speech.py` then `python transcribe-speech.py`, both from `Lab 14/`.

### Lab 15 — `voice-mail.py` (dedicated Speech SDK, record + transcribe)

Uses `azure.cognitiveservices.speech` (the classic Azure AI Speech SDK) directly instead of an LLM audio model — a menu-driven console app:

- **Option 1 — `record_greeting()`**: takes typed text, synthesizes it to `greeting.wav` using `SpeechSynthesizer` with the neural voice `en-US-Serena:DragonHDLatestNeural`.
- **Option 2 — `transcribe_messages()`**: iterates every `.wav` file in a **`messages/` folder** (not included — add your own short `.wav` recordings), plays each one, then transcribes it with `SpeechRecognizer.recognize_once_async()`.
- Authenticates via `SpeechConfig(token_credential=DefaultAzureCredential(), endpoint=foundry_endpoint)` — Entra ID auth, no key needed despite `FOUNDRY_KEY` being loaded from `.env` (it's read but unused in this version, a leftover from an older key-based pattern).

**Run:** `python voice-mail.py` from `Lab 15/` (create a `messages/` folder with some `.wav` files to use option 2).

### Lab 16 — `speech-client.py`

**Needs a pre-built portal agent** (`FOUNDRY_ENDPOINT` + `AGENT_NAME`). Despite the filename, **this code is a plain text chat loop** — structurally identical to Lab 13/Lab 9, just with a `while True` loop added. There's no audio capture or synthesis code here at all. The "speech" part of this lab is almost certainly configured on the *agent itself* in the Foundry portal (e.g. an agent with voice/telephony channel enabled) rather than in this script — treat this file as "how to talk to that agent from Python once it exists," not as a speech-processing example.

### Lab 17 — `chat-client.py` (real-time voice agent, Azure AI VoiceLive)

The most advanced file in this folder — a **full-duplex, real-time voice conversation** with a Foundry agent using the `azure-ai-voicelive` SDK and your computer's microphone/speakers via `pyaudio`. Read this one last, after the others.

**Two classes:**
- **`VoiceAssistant`** — connects to the VoiceLive WebSocket endpoint (`connect(endpoint=..., credential=..., agent_config=...)`), configures the session (`RequestSession`: both text+audio modalities, 16-bit PCM audio, **`AzureSemanticVadMultilingual`** for automatic turn-detection — it decides when you've stopped talking — plus echo cancellation and noise reduction), then loops forever handling server events: session ready → start listening; speech transcribed → print it; agent audio arriving → queue it for playback; user starts talking → clear the playback queue (barge-in / interruption support).
- **`AudioProcessor`** — the low-level plumbing: opens a microphone input stream that base64-encodes each captured chunk and sends it to VoiceLive (`connection.input_audio_buffer.append(...)`), and a speaker output stream that pulls decoded audio off a `queue.Queue` and feeds it to PyAudio's callback, padding with silence if the queue runs dry to avoid audio glitches.

This is the only lab in the whole folder that captures/plays real audio hardware — it needs a working microphone and speakers, and (on macOS) `brew install portaudio` before `pip install pyaudio` will succeed.

**Run:** `python chat-client.py` from `Lab 17/`, then just talk — press `Ctrl+C` to exit.

---

## Vision, Documents and Extended Agents (Lab 18–28)

**Everything in this group is new — added to close the AI-103 coverage gap described above, not part of the original course mirror.** Covers Domains 1, 3, and 5 of the exam blueprint, which had zero hands-on lab coverage before. See the coverage table above for which labs are proven-working vs. best-effort/preview.

### Lab 18 — `connected_agents.py` (multi-agent delegation, proven)

Foundry's native multi-agent DELEGATION pattern — different from Lab 11's `SequentialBuilder` (which pipes agents one after another). Here, one orchestrator agent ("concierge") has other, already-created agents attached to it as tools (`ConnectedAgentTool`), and decides per-request which specialist(s) to call and how to combine their answers. Uses `AgentsClient` (not `AIProjectClient`) and the older thread/run model (`threads.create()`, `runs.create_and_process()`). Ported from this repo's own real, working `11_azure_ai_foundry/06_connected_agents/main.py`. Cleans up all three agents it creates at the end.

**Run:** `python connected_agents.py` from `Lab 18/`.

### Lab 19 — `foundry_iq_knowledge.py` ⚠️ (preview/best-effort)

Illustrates the CONCEPT of Foundry IQ — a shared knowledge platform where a knowledge source is registered once and many agents ground on it with consistent citations. No verified SDK class exists for this anywhere in this repo; the `KnowledgeTool` import is a guess modeled on the real `MCPTool`/`FunctionTool` shape. The script tries the import and, if it fails (expected), gracefully prints a plain-English explanation of the concept instead of crashing.

### Lab 20 — `work_iq_m365_publishing.py` ⚠️ (preview/best-effort)

Illustrates publishing a Foundry agent into Microsoft Teams / M365 Copilot with Work IQ data access. The agent-lookup half is real (`project_client.agents.get(...)`); the actual "publish" step is illustrative pseudocode — this is mostly a portal/admin-center workflow (Teams Admin Center app registration), not a single Python SDK call.

### Lab 21 — `a2a_protocol.py` ⚠️ (preview/best-effort)

Illustrates A2A (Agent2Agent) — an open, cross-vendor protocol for agents to discover and talk to OTHER agents (as opposed to MCP, which connects agents to tools — memorize that pairing, it's a recurring exam theme). Shows a realistic Agent Card JSON shape (the discovery document) and a best-effort `A2AClient` call that's expected to fail with an `ImportError` unless you `pip install` the real current A2A package yourself.

### Lab 22 — `image_generation_and_editing.py` (proven)

Three steps in one script: generate an image from a text prompt, edit an existing image with a text instruction, and masked edit/inpainting (only a masked region changes). Adapted from `05. Section Code/02-04`. Needs your own `input_image.png` (and optionally `mask.png`) in this folder for steps 2-3 — step 1 runs standalone.

**Run:** `python image_generation_and_editing.py` from `Lab 22/`.

### Lab 23 — `content_safety.py` (proven)

The dedicated `ContentSafetyClient` (key-based auth) — moderates text and image content, printing a 0-7 severity score per harm category (Hate/SelfHarm/Sexual/Violence) rather than a single pass/fail flag. Contrast with the "read `content_filters` off an agent's response" pattern shown in `05. Section Code/05-06` (passive moderation vs. this lab's active, dedicated-client moderation).

**Run:** `python content_safety.py` from `Lab 23/`.

### Lab 24 — `content_understanding.py` ⚠️ (unverified SDK)

Adapted from `08. Section Code/03_cloudxeus_invoice_agent.py`'s `begin_analyze` + `AnalysisInput` shape. Flagged unverified because the `azure-ai-contentunderstanding` package isn't installed/resolvable in this repo's environment, and the two Section Code files that use it call **different method names** for the same operation (`begin_analyze_binary` vs `begin_analyze`) — a sign this SDK was still moving when they were written. The script's import is wrapped in try/except so it explains the concept instead of crashing if the package isn't installed. Needs a `sample_invoice.pdf` in this folder.

### Lab 25 — `document_intelligence.py` (proven)

The dedicated `DocumentIntelligenceClient`, adapted directly from `08. Section Code/01_document_intelligence.py` — unlike Content Understanding, this is a long-established, stable SDK. Runs a `prebuilt-layout` analysis (extracts text + tables as Markdown) against a public sample PDF URL by default — no setup needed to try it, though you can point `SAMPLE_DOCUMENT_URL` at your own document.

**Run:** `python document_intelligence.py` from `Lab 25/`.

### Lab 26 — `azure_ai_search_rag.py` (proven)

"Bring your own retrieval" RAG — queries `SearchClient` directly (hybrid keyword + vector search via `VectorizableTextQuery`, merged server-side via RRF) and manually stuffs the results into a grounding prompt, adapted from `02. Section Code/08_ai_search.py` + `10_customer_rag_client.py`. Note: `AzureAISearchTool` (a Foundry-native "search as an agent tool" class some docs mention) is never actually called anywhere in this repo — this client-side pattern is the only proven-working Search-as-RAG example here (Lab 8's MCP-based search-knowledge-base tool is the only proven example of search wired in as a native agent *tool*, via a different mechanism). Needs an Azure AI Search index that's already created and populated.

**Run:** `python azure_ai_search_rag.py` from `Lab 26/`.

### Lab 27 — `sora2_video_generation.py` ⚠️ (preview/best-effort)

Illustrates video generation as an async submit-then-poll job (`client.videos.create()` → poll `client.videos.retrieve()` → `client.videos.download_content()`), modeled on OpenAI's public video-generation API pattern extended to Azure the same way Lab 22's image generation is. No video-generation code exists anywhere else in this repo to port from, so this is entirely a best-effort guess — wrapped in a broad try/except that explains what went wrong if your resource doesn't support it or the real API shape differs.

### Lab 28 — `translation.py` (proven)

Two approaches side by side, exactly the trade-off AI-103 tests: **LLM-prompted translation** (one prompt per target language, flexible instructions) vs. **Azure AI Translator** (a single REST call — no dedicated Python SDK package, just `requests` + an `Ocp-Apim-Subscription-Key` header — that can translate into MANY target languages at once via a `targets` array). Adapted from `06. Section Code/04_text_translation.py` + `09_text_translation.py`. Each half runs independently depending on which `.env` variables you've set.

**Run:** `python translation.py` from `Lab 28/`.

---

## Previous / Related

← [Chapter 14 — Azure AI-103 overview](../README.md) (the paired-notebook `Section Code` walkthroughs)
← [Chapter 11 — Azure AI Foundry labs](../../11_azure_ai_foundry/) (the `azure-ai-projects` / `DefaultAzureCredential` pattern these labs build on)
