# Azure AI-103 — Hands-On Labs

> Beginner-friendly guide to every `.py` file in this folder: what it teaches, what it needs to run, and how the code works.

This `Labs/` folder is a second, complementary set of materials for the "Azure AI 103" course (the main [`14-azure-ai103/`](../README.md) chapter). It's different in nature from the `NN. Section Code/` folders next door:

| | `01–08. Section Code/` | `Labs/` (this folder) |
|---|---|---|
| Style | Instructor walkthrough scripts | Hands-on lab exercises |
| Completeness | All complete, all paired with a teaching notebook | **Mostly complete**, but a few are intentionally unfinished "fill in the blank" exercises (called out below) |
| Sample data | Self-contained | **Several scripts expect a local data file/folder that isn't included in this repo** (e.g. `reviews/`, `data/events.txt`, `speech.wav`) — these came from the course's separate downloadable lab-files ZIP. You'll need to create your own sample files (formats are described per-lab below) before those scripts will run |

The folder is organized into three lab groups, numbered to match the course:

| Group folder | Labs | Topic |
|---|---|---|
| [`Generative AI (Lab-1-5)/`](<Generative AI (Lab-1-5)>) | 3, 4 | Azure OpenAI chat basics: Chat Completions vs. Responses API, streaming, async, and file-search tools |
| [`AI Agents(Lab 6-11)/`](<AI Agents(Lab 6-11)>) | 6–11 | Azure AI Foundry Agent Service: function-calling tools, MCP (Model Context Protocol) tools, and the new Microsoft Agent Framework |
| [`NLP and Speech ( Lab 12-17 )/`](<NLP and Speech ( Lab 12-17 )>) | 12–17 | Azure AI Language (text analytics) and Azure AI Speech (text-to-speech, speech-to-text, real-time voice agents) |

(Labs 1, 2, and 5 don't have a folder here — this mirror only includes the labs that ship a `.py` file.)

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

`azure-ai-projects`, `azure-identity`, `openai`, `python-dotenv`, `pydantic`, and `mcp` are already in the root `requirements.txt`.

### 5. How to run any lab

```bash
cd "14-azure-ai103/Labs/<group folder>/Lab <N>"
python <script>.py
```

---

## Generative AI (Lab-1-5)

Azure OpenAI basics — same Azure OpenAI resource as chapter `01. Section Code/`, but focused on the difference between the older **Chat Completions API** and the newer **Responses API** (which tracks conversation state for you server-side via `previous_response_id`).

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

## Previous / Related

← [Chapter 14 — Azure AI-103 overview](../README.md) (the paired-notebook `Section Code` walkthroughs)
← [Chapter 11 — Azure AI Foundry labs](../../11_azure_ai_foundry/) (the `azure-ai-projects` / `DefaultAzureCredential` pattern these labs build on)
