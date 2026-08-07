# 📝 AI-103 Practice Questions — Foundry Agent Service Deep Dive (Q66–Q97)

> **32 original practice questions** extending [`EXAM_PRACTICE_QUESTIONS.md`](EXAM_PRACTICE_QUESTIONS.md) with a deeper pass through Domain 2 (agentic solutions) topics: agent identity & deployment governance, tool orchestration, workflows, safety/compliance controls, and observability. Numbering continues from the 65 questions in the main set, so a full run-through is Q1–Q97.
>
> ⚠️ **Not exam dumps.** These are freshly written, scenario-based questions grounded in [`EXAM_NOTES.md`](EXAM_NOTES.md) and official Microsoft Foundry documentation — not reproductions of real (NDA-protected) exam items. If a question here and Microsoft documentation ever disagree, trust the documentation. See Microsoft's free Practice Assessment for officially sanctioned practice questions.
>
> **How to use:** cover the answer, commit to a choice, then expand.

---

## Section A — Identity, Deployment & Access Governance

**Q66.** A Python service must call a Foundry-deployed model's Responses API while authenticating purely via Microsoft Entra ID — no API key anywhere in the code or config. Which two lines complete the client setup?

- A. `credential = AzureKeyCredential(os.environ["KEY"])` then `openai_client.responses.retrieve(...)`
- B. `credential = DefaultAzureCredential()` then `openai_client.responses.create(...)`
- C. `credential = ClientSecretCredential(tenant, client_id, secret)` then `openai_client.responses.compact(...)`
- D. `credential = DefaultAzureCredential()` then `openai_client.responses.retrieve(...)`

<details><summary><b>Answer</b></summary>

**B.** `DefaultAzureCredential` authenticates via Entra ID (managed identity in Azure, a chain of local sources for dev) with no stored key. `.responses.create()` is the call that actually sends a new prompt to the model; `.retrieve()` only fetches a previously created response by ID, and `.compact()` isn't a real Responses API method. A uses a key (violates the requirement); C hardcodes a client secret — a long-lived credential, not the keyless pattern being asked for.
</details>

**Q67.** Developers get HTTP 403 when calling a Foundry model deployment's inference endpoint with `DefaultAzureCredential`, even after `az login`. They must never create, delete, or reconfigure deployments — only run inference. Which RBAC role satisfies least privilege?

- A. Cognitive Services User
- B. Cognitive Services OpenAI User
- C. Contributor
- D. Cognitive Services Data Reader

<details><summary><b>Answer</b></summary>

**B.** Cognitive Services OpenAI User grants exactly "call the model for inference," nothing else. Cognitive Services User (A) is the non-OpenAI-specific cousin and doesn't cover Azure OpenAI/Foundry model calls the same way; Contributor (C) can manage/delete the resource — far beyond least privilege; Data Reader (D) is about reading resource metadata, not invoking inference.
</details>

**Q68.** A support agent's model deployment must (1) dynamically scale to bursty traffic without provisioning reserved capacity, and (2) never silently change model version underneath the running agent. Which TWO settings should you choose? (Choose two.)

- A. Deployment type: Global Standard
- B. Deployment type: Provisioned (PTU)
- C. Version update policy: opt out of automatic upgrades (pin the version)
- D. Version update policy: upgrade automatically once a new default version is released

<details><summary><b>Answer</b></summary>

**A and C.** Global Standard is pay-per-token with elastic global capacity — no PTU reservation needed. Pinning the version (opting out of auto-upgrade) is the only policy that guarantees the exact model version never changes without a deliberate action. B is the opposite of "no reserved capacity"; D is the opposite of "never silently changes."
</details>

**Q69.** After fine-tuning and publishing a custom Speech-to-text model, your agent's calls to the Speech-to-text REST API fail with "invalid project ID." Which value goes in the `project` property?

- A. The Speech resource's regional endpoint URL
- B. The custom-model deployment's endpoint URL
- C. The ID of the Speech Studio/Foundry project the custom model belongs to
- D. The Speech resource's subscription key

<details><summary><b>Answer</b></summary>

**C.** Custom Speech models are scoped to a project; the REST API's `project` property expects that project's **ID**, not a URL or key. A and B are endpoints (used elsewhere in the request, not as the `project` value); D is a credential, not an identifier.
</details>

**Q70.** A custom Speech-to-text model deployed to a dedicated endpoint is about to hit its expiration date, and no action is taken. What happens to recognition requests against that endpoint afterward?

- A. They start failing with 4xx errors until a replacement model is deployed
- B. They keep using the expired model indefinitely until someone deletes it
- C. They automatically fall back to the latest base model for the same locale
- D. The endpoint and custom model are deleted automatically

<details><summary><b>Answer</b></summary>

**C.** Expired custom models gracefully fall back to the current base model for their locale rather than hard-failing or silently running stale code — a "don't break production on model expiry" default. A, B, and D all describe outcomes Azure Speech specifically avoids.
</details>

---

## Section B — Agent Tools, Workflows & Orchestration

**Q71.** Three specialized agents (intake classifier, knowledge lookup, ticket writer) must run in a deterministic sequence with conditional branching, shared state between steps, and an optional ticket-creation step gated on the classification result — built with minimal custom code. What should you use?

- A. A Foundry workflow (declarative, node-based orchestration)
- B. Threads and runs coordinated manually, with no workflow definition
- C. A free-form multi-agent group chat
- D. Separate agent invocations glued together in application code

<details><summary><b>Answer</b></summary>

**A.** Foundry workflows are built for exactly this: deterministic step order, condition-based branching, shared state across nodes, and conditional actions — all declared rather than hand-coded. B and D push all that orchestration logic into your own code; C (group chat) is for open-ended multi-agent conversation, not a controlled pipeline.
</details>

**Q72.** A finance-analyst agent needs three new capabilities: run arithmetic/statistical calculations mid-conversation, pull current information from the public web, and answer questions from spreadsheets a user uploads directly into the chat. Match each capability to the right built-in tool.

- A. Calculations → Code interpreter · Public web → Grounding with Bing Search · Uploaded files → File search
- B. Calculations → File search · Public web → Code interpreter · Uploaded files → Grounding with Bing Search
- C. Calculations → Grounding with Bing Search · Public web → File search · Uploaded files → Code interpreter
- D. All three → Code interpreter

<details><summary><b>Answer</b></summary>

**A.** Code interpreter executes real computation; Grounding with Bing Search retrieves live public-web results; File search indexes and retrieves from files attached directly to the agent/thread. The other pairings assign tools that don't fit the job (File search can't run arithmetic; Bing grounding can't read a user's uploaded spreadsheet).
</details>

**Q73.** Five agents in one Foundry project all need to query the same Azure AI Search index. You want the search credentials configured and rotated in exactly one place, automatically applied to every agent that uses that tool. What should you do?

- A. Enable RBAC on the Azure AI Search resource and stop there
- B. Add a project-level connection to the Azure AI Search resource and point each agent's search tool at it
- C. Disable key-based auth on Azure AI Search
- D. Create a managed private endpoint from the project to Azure AI Search

<details><summary><b>Answer</b></summary>

**B.** A project connection is the single, centrally managed credential object that any agent's tool configuration references — update it once, every agent picks up the change. A and C are security hardening steps (good practice, but don't solve "one place to manage credentials for many agents"); D solves network reachability, not credential management.
</details>

**Q74.** An agent calls an external API through an OpenAPI 3.0 tool. The API requires a key in a custom HTTP header, already stored as a project connection. You need the key attached to every call automatically, with no code in the tool itself. What do you add to the OpenAPI spec?

- A. A literal header parameter with the key value hardcoded per operation
- B. An Azure Key Vault reference inside the spec
- C. A security scheme of type `apiKey` (in `header`), then link the connection to satisfy it
- D. A security scheme of type `http` with `bearer` scheme

<details><summary><b>Answer</b></summary>

**C.** Declaring an `apiKey`-type security scheme tells the OpenAPI tool "this call needs a header credential," and Foundry fills it in from the linked connection automatically. A hardcodes a secret into the spec; B references a vault the OpenAPI tool doesn't natively resolve; D is for bearer/OAuth-style tokens, not a static header key.
</details>

**Q75.** An OpenAPI tool calls an external API that needs an API key stored in project connection `Conn1`. Trace logs show the API returning 401 because the key header is simply never sent. What's the fix?

- A. Turn on identity passthrough so the tool forwards the caller's Entra ID token
- B. Manually add the key as a literal header value in the spec
- C. Point the tool at the project's default connection
- D. Explicitly connect the OpenAPI tool to `Conn1`

<details><summary><b>Answer</b></summary>

**D.** A 401 with "no header sent at all" almost always means the tool was never wired to the connection holding the key — connect it and security-scheme resolution starts working. A swaps in a different (Entra) auth mechanism the external API doesn't necessarily accept; B hardcodes a secret; C only helps if the default connection happens to be `Conn1`, which isn't stated.
</details>

**Q76.** An agent has an MCP tool over a knowledge base, but some runs answer straight from the base model — skipping retrieval and producing unsupported claims. You need every run to invoke the MCP tool, no exceptions. What do you set on the run?

- A. `tool_choice="auto"`
- B. `tool_choice="required"`
- C. `tools=[]`
- D. `response_format={"type": "json_object"}`

<details><summary><b>Answer</b></summary>

**B.** `"required"` forces the model to make at least one tool call before it can answer — the only setting that removes the model's discretion. `"auto"` (A) is exactly the current, misbehaving default; C removes the tool entirely; D shapes output formatting and has nothing to do with tool invocation.
</details>

**Q77.** You're building the `run_payload` dict passed to a triage agent's run and need to guarantee it always calls a tool rather than answering directly. Which key:value pair belongs in that dict?

- A. `"tool_choice": "auto"`
- B. `"tools": "required"`
- C. `"tool_choice": "required"`
- D. `"type": "required"`

<details><summary><b>Answer</b></summary>

**C.** The run payload's `tool_choice` key controls invocation behavior, and the literal value `"required"` forces a call. A leaves it optional (the bug you're fixing); B and D put `"required"` under the wrong key, which the API won't interpret as a tool-forcing instruction.
</details>

**Q78.** You're publishing an agent for a compliance workflow with three hard constraints: every run must retrieve before answering, tool calls must run under the published agent's *own* identity (not the caller's, not a shared project identity), and that identity's activity must be independently audit-traceable. What should you configure?

- A. `tool_choice="auto"`, tools authenticated via the shared project agent identity
- B. `tool_choice="required"`, tools authenticated via a distinct agent identity bound to the published agent
- C. `tool_choice="none"`, tools authenticated via a user-assigned identity shared across all agents
- D. `tool_choice="required"`, tools authenticated via caller-passthrough Entra tokens

<details><summary><b>Answer</b></summary>

**B.** `"required"` guarantees the mandatory retrieval step; a distinct identity bound to *this* published agent (instead of a shared project identity) gives both the isolation and the clean audit trail the requirements call for. A doesn't force retrieval and uses a shared identity; C disables tool calls outright; D authenticates as whoever called the agent, not as the agent itself.
</details>

**Q79.** A high-volume chat app is mostly simple FAQs with an occasional question needing deep reasoning. You must cut cost and latency on the common case without degrading answers on the hard case. What's the right move?

- A. Route every request to the smallest available model
- B. Route every request to the most capable (and most expensive) model
- C. Raise `max_tokens` globally so responses are never truncated
- D. Use a model cascade: a small/cheap model attempts first, escalating to a larger model when needed

<details><summary><b>Answer</b></summary>

**D.** A cascade gets the cost/latency win on the easy majority of traffic while still reaching the capable model for the hard minority — the only option that improves *both* metrics without sacrificing quality anywhere. A saves cost everywhere but wrecks quality on hard questions; B protects quality but keeps cost/latency high everywhere; C doesn't address cost or latency at all.
</details>

**Q80.** In a Foundry workflow, an "Ask a question" node stores the reply in `Local.Var01`. You need an if/else condition that only proceeds when the user actually answered, and a Send-message expression that echoes the answer in uppercase. Which pair of Power Fx expressions is correct?

- A. `IsBlank(Local.Var01)` and `{Local.Var01}`
- B. `Not(IsBlank(Local.Var01))` and `{Upper(Local.Var01)}`
- C. `IsEmpty(Local.Var01)` and `{Upper(Var01)}`
- D. `Not(IsBlank(Local.Var01))` and `{Local.Var01}`

<details><summary><b>Answer</b></summary>

**B.** `Not(IsBlank(...))` is true exactly when the variable *has* a value, matching "proceeds only when answered"; `{Upper(Local.Var01)}` transforms and echoes it in uppercase. A's condition is inverted (true when blank); C's `IsEmpty` targets collections/tables, not a scalar text variable, and drops the `Local.` scope; D never uppercases the echoed text.
</details>

**Q81.** A chat app queries an Azure AI Search vectorized index. Requirements: complex questions must pull from several chunks at once, multi-turn context must actively shape what gets retrieved next, and retrievals should run in parallel to keep latency down. Which retrieval approach fits?

- A. Classic Retrieval Augmented Generation (single-shot vector search)
- B. Chain-of-thought prompting on the base model
- C. Agentic Retrieval Augmented Generation (a planning step decomposes the query and issues parallel sub-retrievals)
- D. Sequential iterative retrieval, one query refinement at a time

<details><summary><b>Answer</b></summary>

**C.** Agentic RAG adds a planning/decomposition layer that breaks a complex, context-aware question into multiple sub-queries and fires them in parallel — covering all three requirements at once. Classic RAG (A) does one retrieval pass and ignores conversation history; chain-of-thought (B) is a reasoning technique, not a retrieval mechanism; sequential iterative retrieval (D) explicitly does not run in parallel.
</details>

**Q82.** An agent is grounded on a retailer's own product catalog but users keep steering it into unrelated general-knowledge chats. Business rules require it to only ever discuss the retailer's products. What's the most direct fix?

- A. Increase the `temperature` parameter
- B. Add more few-shot examples to the retrieval index
- C. Tighten the system message / instructions to explicitly scope the agent's allowed topics
- D. Switch to top-p sampling

<details><summary><b>Answer</b></summary>

**C.** Scope restriction is an instruction-following problem, and the system message is where you declare hard boundaries like "only answer questions about our products; decline everything else." A and D are sampling-randomness knobs and don't teach the model what topics to refuse. B affects retrieval quality, not topical boundaries.
</details>

**Q83.** Customers reopen the same support case days later and expect the agent to pick up with full context — every prior user message, agent message, tool call, and tool output — automatically, with no manual replay by the client app. What should you implement?

- A. Store only the model's last reply client-side and prepend it to future prompts
- B. Turn on an agent-level "memory summarization" toggle and stop tracking anything else
- C. Create a conversation once, persist its ID, and pass that same ID on every subsequent request (same session or a new one days later)
- D. Re-send the entire chat transcript as plain text in the system message each time

<details><summary><b>Answer</b></summary>

**C.** A persisted conversation ID is the server-side handle that gives you full-fidelity history — messages, tool calls, and tool outputs — reloaded automatically on each new turn, across sessions. A throws away everything except the final reply; B isn't a real mechanism for full-fidelity replay; D is possible but manual, unbounded in size, and exactly the client-side burden the requirement is trying to avoid.
</details>

---

## Section C — Safety, Compliance & Document Pipelines

**Q84.** A multimodal agent extracts and acts on text found inside uploaded images. Attackers discover they can embed adversarial instructions as near-invisible text inside an otherwise unsafe-looking image to hijack the agent's behavior. **Proposed fix: turn on image content moderation to block explicit/unsafe visual content before processing.** Does this fully address the risk?

- A. Yes
- B. No

<details><summary><b>Answer</b></summary>

**B.** Content moderation screens for unsafe *imagery* (violence, explicit content, etc.) — it has no concept of adversarial *text* hidden inside a technically "safe-looking" image, so the injection attack still gets through untouched.
</details>

**Q85.** Same scenario as Q84. **Proposed fix: apply Prompt Shields' indirect-attack detection to the document/image content itself, before it's treated as trusted input.** Does this fully address the risk?

- A. Yes
- B. No

<details><summary><b>Answer</b></summary>

**A.** Prompt Shields' indirect-injection detection is purpose-built for this case — instructions smuggled inside documents or images rather than typed by the user — and flags/blocks that hidden content before the agent can act on it as if it were a legitimate instruction.
</details>

**Q86.** Same scenario as Q84. **Proposed fix: apply Prompt Shields to the user's typed prompt text only.** Does this fully address the risk?

- A. Yes
- B. No

<details><summary><b>Answer</b></summary>

**B.** The malicious instructions live inside the image's extracted text, not in what the user typed — screening only the literal user prompt leaves the actual attack surface (the image content) completely unchecked.
</details>

**Q87.** Same scenario as Q84. **Proposed fix: enable protected-material detection.** Does this fully address the risk?

- A. Yes
- B. No

<details><summary><b>Answer</b></summary>

**B.** Protected-material detection flags reproduced copyrighted text/code — it isn't a jailbreak or prompt-injection control, so it does nothing against instructions hidden inside an image.
</details>

**Q88.** An agent must never disclose customer-identifying details in its responses, even in the worst case where a document containing customer data is accidentally dropped into its otherwise-generic knowledge repository. Which Foundry Tools capability should you add to close that gap?

- A. Self-harm content filtering
- B. Prompt Shields
- C. Personally Identifiable Information (PII) Detection
- D. Violence content filtering

<details><summary><b>Answer</b></summary>

**C.** PII Detection is aimed squarely at recognizing and redacting/blocking personal data in generated output — exactly the failure mode described. The other three filters target harmful-content categories (self-harm, jailbreaks, violence), none of which is about protecting personal data.
</details>

**Q89.** An agent receives a blob URL (from an internal ticketing tool) pointing at a user-uploaded screenshot and must run image moderation on it before returning any response that references the image, while granting Azure AI Content Safety the narrowest possible access to that storage account. Which TWO should you configure? (Choose two.)

- A. A guardrail that inspects tool responses/output and blocks (not just annotates) unsafe results
- B. Storage account access keys shared with the Content Safety resource
- C. A system-assigned managed identity for Content Safety, granted the Storage Blob Data Reader role
- D. A user-assigned identity granted the Storage Queue Data Contributor role

<details><summary><b>Answer</b></summary>

**A and C.** Blocking (not merely annotating) is required to actually "prevent harmful content from being returned," and a system-assigned managed identity scoped to read-only Blob access is the least-privilege way to let Content Safety fetch the image by URL. B reintroduces a long-lived shared secret; D grants Queue permissions, which have nothing to do with reading a blob.
</details>

**Q90.** You're standing up two Azure Content Understanding pipelines for supplier documents: Pipeline1 must cheaply process large volumes of standalone, single PDF invoices; Pipeline2 must cross-validate a submitted document against separate reference data using multi-step reasoning. How should you configure each?

- A. Pipeline1: single-file task, standard mode · Pipeline2: multi-file task, pro mode
- B. Pipeline1: multi-file task, pro mode · Pipeline2: single-file task, standard mode
- C. Both pipelines: multi-file task, pro mode
- D. Both pipelines: single-file task, standard mode

<details><summary><b>Answer</b></summary>

**A.** Standard-mode single-file tasks are the cost-effective, high-throughput path for one-document-at-a-time extraction (the invoice case); pro mode's multi-file, multi-step reasoning is what's needed to reconcile a document against separate reference data. Using pro mode for high-volume standalone invoices (B, C) wastes the cost/latency budget the requirement rules out; standard mode alone (D) can't do the cross-document reasoning Pipeline2 needs.
</details>

---

## Section D — Observability & Evaluation

**Q91.** A customer-support agent calls an internal knowledge API before answering. Some runs take 15+ seconds and some answers are wrong even though the API returned correct data. You need to see the exact ordered sequence of LLM calls, tool invocations, and their individual timings for a specific run. What do you use?

- A. Token usage analytics
- B. General monitoring dashboards
- C. Safety/risk metrics
- D. Tracing (per-run execution trace)

<details><summary><b>Answer</b></summary>

**D.** A per-run trace reconstructs the ordered timeline of LLM calls and tool invocations with timing for one specific run — exactly what's needed to see whether the slowdown or wrong answer originated in the tool call, a particular LLM step, or elsewhere. Token usage and monitoring dashboards are aggregate views, not single-run timelines; safety metrics are unrelated to latency or correctness debugging.
</details>

**Q92.** A high-traffic agent's operational cost jumps sharply after a release, but request *volume* is unchanged per monitoring. You suspect the request/response shape itself changed — bigger prompts, longer answers, or more tool calls. Which capability isolates the actual cost driver?

- A. Latency metrics
- B. Run success rate
- C. Token usage analytics (broken out by input/output/tool tokens)
- D. Evaluation metrics (groundedness, relevance, etc.)

<details><summary><b>Answer</b></summary>

**C.** Cost scales with tokens, so a token-usage breakdown by input, output, and tool-call tokens is the only signal that directly attributes a cost increase to one of those three causes. Latency (A) and success rate (B) can move for unrelated reasons and don't explain cost; evaluation metrics (D) measure answer quality, not spend.
</details>

**Q93.** An internal Q&A agent shows two symptoms: a rising rate of "no relevant information found" responses, and periodic HTTP 429s during peak hours. You need telemetry that lets you tell apart model unavailability, throughput/resource limits, and inference-level failures. What should you enable?

- A. Only a "requests with status 200" counter, plus an `audit` diagnostic log
- B. Model availability rate + provisioned utilization metrics, plus a `trace` diagnostic log
- C. Only a token-cache match-rate metric, plus a `RequestResponse` diagnostic log
- D. No additional metrics — rely on the existing application logs

<details><summary><b>Answer</b></summary>

**B.** Availability-rate and utilization metrics separate "the model was down" from "you're saturating your provisioned/rate-limit capacity," while a detailed execution trace lets you drill into individual failed inference calls to see what went wrong. A's status-200-only counter and audit log don't surface *why* the other requests failed; C's cache-hit metric is irrelevant to either symptom; D gives up on root-causing the issue at all.
</details>

**Q94.** A prompt agent is invoked entirely from a custom Python backend service that never touches the Foundry portal UI. You still need end-to-end tracing — latency breakdowns and exceptions — across every agent run. Which TWO components combine to deliver that? (Choose two.)

- A. OpenTelemetry instrumentation in the service
- B. Microsoft Sentinel
- C. Application Insights as the trace destination
- D. The Azure Monitor Agent (VM/host telemetry agent)

<details><summary><b>Answer</b></summary>

**A and C.** OpenTelemetry is how the Python service emits standardized traces/spans around each agent call, and Application Insights is where those traces land for latency breakdowns and exception analysis — a portal-independent pipeline. Sentinel (B) is a SIEM for security signals, not application tracing; the Azure Monitor Agent (D) collects OS/VM-level telemetry, not application-level distributed traces.
</details>

**Q95.** You need to continuously verify that an agent's answers — generated using retrieved product-sheet content — stay relevant, complete, and factually supported by that retrieved content. What should you add to the evaluation pipeline?

- A. A generic Retrieval Augmented Generation (RAG) evaluator with no groundedness component
- B. A custom guardrail with no scoring or reporting
- C. Model fine-tuning on the product sheets
- D. A groundedness evaluator (checking claims against the retrieved source content)

<details><summary><b>Answer</b></summary>

**D.** Groundedness evaluation is specifically designed to score whether generated claims are actually supported by the retrieved context — exactly "relevant, complete, and accurate relative to the source material." A generic RAG evaluator without a groundedness dimension (A) misses the "supported by the source" check; a guardrail (B) blocks/annotates at runtime but isn't a quality-scoring evaluation; fine-tuning (C) changes model behavior but gives no ongoing measurement of accuracy.
</details>

**Q96.** You're choosing a model to power an agent that must ground answers in a large internal knowledge base, perform deep multi-step reasoning across long retrieved context, and produce detailed natural-language explanations. Which model class fits?

- A. A multimodal model
- B. A small language model (SLM)
- C. A key-phrase extraction model
- D. A large language model (LLM)

<details><summary><b>Answer</b></summary>

**D.** Deep multi-step reasoning over long context plus detailed natural-language generation is squarely an LLM's strength. A multimodal model (A) adds image/audio handling that isn't needed here; an SLM (B) trades away exactly the reasoning depth and context length the requirement calls for; a key-phrase extraction model (C) pulls out terms — it doesn't reason or generate prose at all.
</details>

**Q97.** An automated validation harness flags mismatches because a deployed chat model's wording varies slightly between otherwise-identical requests. You must tighten output stability while still maximizing the model's reasoning quality. Which two request settings should you set?

- A. `temperature=2`, reasoning `effort="low"`
- B. `temperature=0`, reasoning `effort="high"`
- C. `temperature=1`, reasoning `effort="medium"`
- D. Leave `temperature` unset, reasoning `effort="high"`

<details><summary><b>Answer</b></summary>

**B.** `temperature=0` makes token selection as deterministic as the model allows (minimizing wording drift between runs), and `effort="high"` spends the most reasoning budget on quality — the only combination that pushes both stability and reasoning quality to their maximum simultaneously. A and C leave meaningful randomness in place (and A also caps reasoning effort low); D never pins down the randomness actually causing the validation mismatches.
</details>

---

## Score yourself

| Score (out of 32) | Reading |
|---|---|
| 29–32 | Exam-ready on agent/observability depth — do the official Practice Assessment to confirm |
| 23–28 | Solid — re-read the sections where you missed, then retry |
| 16–22 | Re-study `EXAM_NOTES.md` §§ on Foundry Agent Service, tracing, and content safety |
| < 16 | Work back through the `02_foundry_agent_service` and `03_conversations_and_evaluation` notebooks first |
