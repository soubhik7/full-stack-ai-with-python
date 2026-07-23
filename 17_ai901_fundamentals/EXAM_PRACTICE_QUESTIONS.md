# 📝 AI-901 Practice Questions — Exam-Style Scenarios with Answers

> **60 original practice questions** written in the style of the real AI-901 exam (scenario-based, single/multi-select), distributed across the two domains in proportion to the official blueprint weights (40–45% / 55–60%). Every question is grounded in the syllabus facts from [`EXAM_NOTES.md`](EXAM_NOTES.md), which is itself built from the public, official [AI-901 skills-measured outline](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/ai-901) on Microsoft Learn.
>
> ⚠️ These are **not leaked exam items** — real questions are protected by the Microsoft Certification Agreement's non-disclosure terms, and reproducing them would also be copyright infringement. Sites that claim to sell or crowd-source "real" AI-901 questions (exam dumps/braindumps) are violating that agreement — using them can get your certification revoked and doesn't test the understanding the exam is actually designed to measure. For officially sanctioned questions, take Microsoft's **free Practice Assessment** via AI Skills Navigator (linked from the AI-901 exam page). If a practice question here and Microsoft documentation ever disagree, trust the documentation.
>
> **How to use:** cover the answer, commit to a choice, then expand. Wrong answers here are designed around the same misconceptions the real exam exploits.

---

## Domain 1 — Identify AI Concepts and Capabilities (Q1–Q26)

### Responsible AI (§1)

**Q1.** A recruiting model trained mostly on historical resumes from one demographic performs noticeably worse when screening candidates from underrepresented groups. Which responsible AI principle is most directly at risk?

- A. Transparency
- B. Fairness
- C. Accountability
- D. Inclusiveness

<details><summary><b>Answer</b></summary>

**B.** Fairness asks whether the system treats similar people/groups similarly — a model that performs worse for one demographic due to skewed training data is the textbook fairness failure. Inclusiveness (D) is about whether the product works for people of varying abilities/backgrounds by design, a related but distinct concern.
</details>

**Q2.** A customer support chatbot never discloses that the user is talking to an AI system rather than a human agent. Which principle does this violate?

- A. Privacy & security
- B. Reliability & safety
- C. Transparency
- D. Fairness

<details><summary><b>Answer</b></summary>

**C.** Transparency requires that people understand how a system works and, at minimum, know when they're interacting with AI rather than a human.
</details>

**Q3.** An image-tagging model crashes or returns nonsensical labels whenever it receives a corrupted or unusually formatted image, instead of failing gracefully. Which principle is most relevant?

- A. Reliability & safety
- B. Accountability
- C. Inclusiveness
- D. Privacy & security

<details><summary><b>Answer</b></summary>

**A.** Reliability & safety is about consistent performance and graceful failure, including on edge-case or malformed input — a crash on unexpected input is a reliability/safety gap, not a fairness or privacy issue.
</details>

**Q4.** A healthcare AI system logs and retains far more patient data than it needs to make a diagnosis, and that data isn't encrypted at rest. Which principle addresses this?

- A. Privacy & security
- B. Fairness
- C. Transparency
- D. Inclusiveness

<details><summary><b>Answer</b></summary>

**A.** Privacy & security is about protecting and minimizing personal data across its lifecycle — over-collection and lack of encryption are squarely privacy/security failures.
</details>

**Q5.** A voice assistant only understands one regional accent well and effectively excludes speakers with other accents or speech patterns from using the product. Which principle applies?

- A. Fairness
- B. Inclusiveness
- C. Accountability
- D. Reliability & safety

<details><summary><b>Answer</b></summary>

**B.** Inclusiveness is specifically about the product working for people of varying abilities, languages, accents, and backgrounds — this is a design/coverage gap, distinct from a bias-in-outcomes fairness problem.
</details>

**Q6.** After an AI-driven loan-denial system causes public harm, no team, role, or process exists to review the decision or take responsibility. Which principle was missing from the start?

- A. Accountability
- B. Transparency
- C. Privacy & security
- D. Fairness

<details><summary><b>Answer</b></summary>

**A.** Accountability means a person or organization is answerable for the system's behavior, with governance in place — the absence of anyone responsible for reviewing harmful decisions is an accountability gap.
</details>

### How generative AI models work (§2)

**Q7.** What is the fundamental task a large language model is trained to perform, from which chat, summarization, and code generation all derive?

- A. Classifying images into fixed categories
- B. Predicting the next token in a sequence
- C. Clustering similar documents together
- D. Detecting anomalies in structured data

<details><summary><b>Answer</b></summary>

**B.** Every downstream LLM capability is built on the core training objective of predicting the next token given the preceding context.
</details>

**Q8.** A developer notices that a model correctly resolves a pronoun referring back to a noun mentioned several paragraphs earlier. Which architectural feature is most responsible for this capability?

- A. Tokenization
- B. The attention mechanism in the transformer architecture
- C. Top P sampling
- D. Max token limit

<details><summary><b>Answer</b></summary>

**B.** Attention lets the model weigh the relevance of every other token when predicting the next one, which is what enables resolving long-range references across a document.
</details>

**Q9.** A team needs to build semantic search that finds documents with *similar meaning* to a query, not just matching keywords. What underlying representation makes this possible?

- A. Embeddings
- B. Stop sequences
- C. System prompts
- D. Speech synthesis

<details><summary><b>Answer</b></summary>

**A.** Embeddings map text into vectors positioned so that semantically similar content sits close together — the foundation of semantic search and retrieval-augmented generation.
</details>

### Choosing an AI model (§3)

**Q10.** A team needs a model to run a simple, narrowly scoped classification task on a resource-constrained edge device with tight cost limits. What is the best-fit choice?

- A. A flagship multimodal reasoning model
- B. A small language model (SLM)
- C. An image-generation model
- D. A Global Batch deployment of the largest available model

<details><summary><b>Answer</b></summary>

**B.** SLMs trade some general capability for a smaller footprint, making them well suited to narrow tasks on constrained hardware where cost and latency matter more than broad general-purpose ability.
</details>

**Q11.** A task requires solving a multi-step logical puzzle that depends on chaining several inferences correctly. Which class of model is best suited, and what's the trade-off?

- A. A reasoning model; trade-off is higher latency and token cost
- B. A small language model; trade-off is reduced accuracy
- C. A speech-to-text model; trade-off is no text understanding
- D. An image-generation model; trade-off is no text output

<details><summary><b>Answer</b></summary>

**A.** Reasoning models spend extra inference-time computation working through steps before answering, which improves multi-step logic at the cost of latency and spend — not appropriate for simple lookups, but the right fit here.
</details>

### Deployment options and configuration parameters (§4)

**Q12.** A nightly job summarizes two million support tickets. Latency is irrelevant but cost per token matters a great deal. Which deployment type fits best?

- A. Provisioned (PTU)
- B. Standard
- C. Global Batch
- D. Global Standard

<details><summary><b>Answer</b></summary>

**C.** Global Batch is roughly half the cost of Global Standard with an asynchronous, up-to-24-hour turnaround — exactly the "large, latency-insensitive, cost-sensitive" profile.
</details>

**Q13.** A production customer-facing app needs predictable, consistently low latency at high, steady request volume. Which deployment type should be recommended?

- A. Global Batch
- B. Provisioned (PTU)
- C. Standard
- D. None — deployment type doesn't affect latency

<details><summary><b>Answer</b></summary>

**B.** Provisioned throughput reserves capacity via purchased PTUs, giving predictable, low, consistent latency for steady high-volume production traffic — the opposite trade-off from Batch.
</details>

**Q14.** A customer reports that identical prompts sent to the same deployment produce wildly different, sometimes off-topic answers, and asks for more consistent output. What should be adjusted first?

- A. Switch to a Global Batch deployment
- B. Lower the temperature (and/or top P)
- C. Increase max tokens
- D. Add more stop sequences

<details><summary><b>Answer</b></summary>

**B.** Temperature (and top P) control the randomness of generation — lowering them produces more focused, consistent output for the same prompt. Deployment type (A) affects cost/latency, not output variability; max tokens (C) only caps length.
</details>

**Q15.** Which configuration element is set once and shapes the model's role, tone, and constraints across an entire conversation, rather than being specified on every turn?

- A. The user prompt
- B. The system message
- C. The stop sequence
- D. The max token limit

<details><summary><b>Answer</b></summary>

**B.** The system message persists across the conversation and defines role, tone, and constraints; the user prompt (A) is the per-turn request.
</details>

### AI workload scenarios (§5)

**Q16.** A system autonomously plans a multi-step itinerary, calling a flights API, then a hotels API, then a calendar API, without a human directing each individual step. Which workload category best describes this?

- A. Text analysis
- B. Agentic AI
- C. Computer vision
- D. Information extraction

<details><summary><b>Answer</b></summary>

**B.** Agentic AI describes a system that plans multi-step actions and calls tools/APIs autonomously toward a goal, rather than just responding to one prompt at a time.
</details>

**Q17.** A company wants to pull structured fields — vendor name, invoice number, total due — out of scanned PDF invoices. Which workload is this?

- A. Information extraction
- B. Speech
- C. Generative AI
- D. Sentiment analysis (a subset of text analysis)

<details><summary><b>Answer</b></summary>

**A.** Extracting structured fields from documents is information extraction — distinct from generative AI (open-ended content creation) or general text analysis (unstructured signals like sentiment or entities without a fixed schema).
</details>

**Q18.** A marketing team wants a tool that drafts multiple variations of ad copy from a short product description. Which workload fits?

- A. Agentic AI
- B. Generative AI
- C. Information extraction
- D. Computer vision

<details><summary><b>Answer</b></summary>

**B.** Drafting/rewriting natural-language content from a prompt is the core generative AI use case.
</details>

### Text analysis techniques (§6)

**Q19.** A support team wants to automatically flag which incoming emails mention specific product names, dates, and company names. Which technique is this?

- A. Sentiment analysis
- B. Entity detection (Named Entity Recognition)
- C. Summarization
- D. Key phrase extraction

<details><summary><b>Answer</b></summary>

**B.** Entity detection identifies and categorizes named things — people, organizations, dates, products — which is exactly what's needed here.
</details>

**Q20.** A product wants to score customer reviews as positive, negative, neutral, or mixed, with a confidence value attached. Which technique applies?

- A. Key phrase extraction
- B. Summarization
- C. Sentiment analysis
- D. Entity detection

<details><summary><b>Answer</b></summary>

**C.** Sentiment analysis scores text along a positive/negative/neutral/mixed scale, typically with a confidence score.
</details>

**Q21.** A legal team needs a shorter version of a 40-page contract that preserves the key points without needing every clause. Which technique fits?

- A. Summarization
- B. Key phrase extraction
- C. Entity detection
- D. Sentiment analysis

<details><summary><b>Answer</b></summary>

**A.** Summarization condenses a long document into a shorter version while preserving the key content — key phrase extraction (B) surfaces topics/phrases but doesn't produce a coherent condensed narrative.
</details>

### Speech (§7)

**Q22.** A call-center app needs to produce a written transcript of a live customer phone call as it happens. Which capability is required?

- A. Speech synthesis (text-to-speech)
- B. Speech recognition (speech-to-text)
- C. Sentiment analysis
- D. Image generation

<details><summary><b>Answer</b></summary>

**B.** Speech recognition converts spoken audio into text, which is exactly the live-transcription requirement.
</details>

**Q23.** A navigation app needs to read turn-by-turn directions aloud in a natural-sounding voice. Which capability is required?

- A. Speech recognition
- B. Speech translation
- C. Speech synthesis (text-to-speech)
- D. Entity detection

<details><summary><b>Answer</b></summary>

**C.** Speech synthesis converts text into natural-sounding spoken audio — the correct fit for reading directions aloud.
</details>

### Computer vision and image generation (§8)

**Q24.** A retail app needs to read the printed serial number off a product label photo. Which feature is required?

- A. Object detection
- B. Optical character recognition (OCR / Read)
- C. Face detection
- D. Image generation

<details><summary><b>Answer</b></summary>

**B.** OCR/Read extracts printed or handwritten text from images — exactly what's needed to read a printed serial number.
</details>

**Q25.** A marketing team wants to produce a brand-new promotional image from a written text description, not analyze an existing photo. Which capability fits?

- A. Object detection
- B. Image generation
- C. Face detection
- D. OCR

<details><summary><b>Answer</b></summary>

**B.** Image generation produces new visual content from a text prompt — the other options all analyze existing images rather than creating new ones.
</details>

### Extraction across content types (§9)

**Q26.** A company wants one consistent approach to pull a custom set of structured fields (e.g., "topic," "sentiment," "action items") out of documents, images, call recordings, and video, using the same underlying mental model across all four. What is that unifying concept?

- A. A user-defined schema that a generative model reasons over to fill in
- B. A single fixed set of fields identical across all content types
- C. Manual data entry after human review
- D. Optical character recognition applied uniformly to every content type

<details><summary><b>Answer</b></summary>

**A.** The unifying idea across text, image, audio, and video extraction is: define a schema of the fields you want, and a generative model reasons over the raw content to fill it in — this is the core mental model behind Azure Content Understanding, covered in Domain 2.
</details>

---

## Domain 2 — Implement AI Solutions by Using Microsoft Foundry (Q27–Q60)

### The Foundry platform shape (§10)

**Q27.** Before writing any code, a developer wants to interactively try out a catalog model with different prompts and parameters. Which Foundry portal area should they use?

- A. The Playground
- B. The Tools tab
- C. Deployments
- D. The model catalog listing page alone

<details><summary><b>Answer</b></summary>

**A.** The Playground is built for interactive testing of a deployed model or prompt agent, adjusting prompts and parameters before writing client code.
</details>

**Q28.** Which Foundry portal area is used specifically to discover and connect tool integrations (function calling, code interpreter, and external business-system connections) that an agent can call?

- A. Model catalog
- B. Tools tab
- C. Playground
- D. Deployments

<details><summary><b>Answer</b></summary>

**B.** The Tools tab is the entry point for discovering, connecting, and managing agentic tool integrations.
</details>

**Q29.** What is the current umbrella name for the family of pre-built AI APIs formerly known as Azure AI Services (Language, Speech, Vision, Translator, Document Intelligence, Content Understanding)?

- A. Foundry Tools
- B. Azure Cognitive Suite
- C. Model Catalog Services
- D. Foundry Agent Service

<details><summary><b>Answer</b></summary>

**A.** Foundry Tools is the current name for that family of pre-built AI APIs, now callable from the same Foundry project as everything else. Foundry Agent Service (D) is specifically the agent-hosting platform, a different piece.
</details>

### Prompt engineering (§11)

**Q30.** An app's responses are inconsistent in tone and format across different users asking similar questions. What is the most direct fix?

- A. Increase the max token limit
- B. Tighten and clarify the system prompt
- C. Switch to a Global Batch deployment
- D. Add more stop sequences

<details><summary><b>Answer</b></summary>

**B.** Tone and format consistency across turns and users is governed by the system prompt, which persists across the conversation — not by deployment type or stop sequences.
</details>

**Q31.** A developer wants a model to reliably follow a specific output format (e.g., always return a bulleted list of exactly three items) without fine-tuning. What prompting technique best achieves this?

- A. Raising the temperature to encourage variety
- B. Few-shot prompting with one or two examples of the desired format
- C. Increasing top P to broaden token choices
- D. Switching to a smaller language model

<details><summary><b>Answer</b></summary>

**B.** Few-shot prompting — including a small number of examples of the desired output format in the prompt — is the standard technique for making format-following more reliable without fine-tuning.
</details>

**Q32.** What is the key difference between a system prompt and a user prompt?

- A. There is no difference; they're interchangeable terms for the same thing
- B. The system prompt is set once and persists across the conversation; the user prompt is the individual request in a given turn
- C. The user prompt controls model latency; the system prompt controls cost
- D. Only the system prompt can contain instructions

<details><summary><b>Answer</b></summary>

**B.** The system prompt sets persistent role/tone/constraints for the whole conversation, while the user prompt is the specific question or request for that turn.
</details>

### Deploying and testing a model in the portal (§12)

**Q33.** What is the correct high-level order of steps to get a catalog model callable from application code via the Foundry portal?

- A. Deploy → browse the model catalog → test in Playground → copy endpoint info
- B. Browse the model catalog → deploy, choosing a deployment type and region → test in Playground → copy endpoint info
- C. Copy endpoint info → browse the model catalog → deploy → test in Playground
- D. Test in Playground → browse the model catalog → deploy → copy endpoint info

<details><summary><b>Answer</b></summary>

**B.** The typical flow is: pick a model from the catalog, deploy it (choosing deployment type and region), test it interactively in the Playground, then take the endpoint and credential info into code.
</details>

**Q34.** During deployment, what must be selected in addition to the model itself?

- A. Only the model's version number
- B. A deployment type (e.g., Standard, Global Batch, Provisioned) and region
- C. The end user's preferred language only
- D. Nothing further is required beyond picking the model

<details><summary><b>Answer</b></summary>

**B.** Deployment requires choosing a deployment type (which governs pricing/latency trade-offs) and a region/data-zone option (which affects residency and availability).
</details>

**Q35.** After deployment completes in the Foundry portal, where does a developer typically land to interactively try the model before writing code?

- A. The Foundry Playground
- B. A locally hosted Jupyter notebook
- C. The Azure billing dashboard
- D. The model catalog's benchmark comparison page

<details><summary><b>Answer</b></summary>

**A.** Completing a deployment lands you on the Playground, where the newly deployed model can be tested interactively.
</details>

### Building a chat client with the Foundry SDK (§13)

**Q36.** Which credential class should a script use to authenticate to a Foundry project using the developer's cached `az login` session, so that it fails fast and predictably if they're not logged in?

- A. `ClientSecretCredential`
- B. `AzureKeyCredential`
- C. `AzureCliCredential`
- D. A hardcoded API key

<details><summary><b>Answer</b></summary>

**C.** `AzureCliCredential` uses only the CLI's cached login and fails predictably and quickly if that session doesn't exist — the right fit for a developer-laptop script. `ClientSecretCredential` (A) is for service principals; `AzureKeyCredential` (B) and a hardcoded key (D) are the long-lived-secret patterns the keyless approach is meant to avoid.
</details>

**Q37.** Which client class from `azure-ai-projects` is the primary entry point for accessing both inference (chat completions) and agent operations in a Foundry project?

- A. `AzureOpenAI`
- B. `AIProjectClient`
- C. `DefaultAzureCredential`
- D. `ChatCompletionsClient` (standalone, with no project context)

<details><summary><b>Answer</b></summary>

**B.** `AIProjectClient` is the primary entry point into a Foundry project, from which you retrieve an inference client for chat completions and an agents client for agent operations.
</details>

**Q38.** What is the primary security advantage of the Foundry SDK's typical authentication pattern (`AIProjectClient` + `DefaultAzureCredential`/`AzureCliCredential`) over an API-key-based client?

- A. It's faster over the network
- B. It's keyless — no long-lived secret exists anywhere, and access is governed by short-lived tokens and RBAC
- C. It removes the need for a deployed model
- D. It automatically encrypts all data at rest, which API-key auth does not

<details><summary><b>Answer</b></summary>

**B.** The Entra ID-based pattern is keyless: no long-lived secret to leak or rotate, with access governed by short-lived tokens and RBAC role assignments — the same "keyless is preferred" pattern tested across Azure AI exams.
</details>

**Q39.** Once an `AIProjectClient` is created, what is the typical next step to send a chat message to a deployed model?

- A. Retrieve an OpenAI-compatible chat-completions client from the project client, then send a system/user message list
- B. Directly query the model catalog's REST API with no client library
- C. Create a new Azure subscription
- D. Bypass the project client and use only the Azure CLI

<details><summary><b>Answer</b></summary>

**A.** From the project client you retrieve an inference (chat-completions) client, then send the familiar system-message-plus-user-messages request shape to get a response.
</details>

### Single-agent solutions (§14)

**Q40.** A team wants to build a simple tool-using assistant quickly, with no application code, containers, or compute to manage themselves. Which Foundry Agent Service option fits?

- A. A hosted agent
- B. A prompt agent
- C. A Global Batch deployment
- D. A standalone Docker container they manage themselves

<details><summary><b>Answer</b></summary>

**B.** Prompt agents are authored in the portal (or via SDK/REST) and run entirely by Foundry — no application code or infrastructure for the developer to manage.
</details>

**Q41.** A team already has custom agent orchestration logic built with an existing agent framework and wants Foundry to run it with a managed endpoint. Which agent type fits?

- A. A prompt agent
- B. A hosted agent
- C. Neither — Foundry Agent Service only supports prompt agents
- D. A Global Standard chat deployment

<details><summary><b>Answer</b></summary>

**B.** Hosted agents let you bring your own code (with an agent framework or your own implementation), package it as a container, and have Foundry run it with a managed endpoint, scaling, and identity — the right fit for existing custom orchestration logic.
</details>

**Q42.** Where can a newly created single-agent solution be interactively tested before it's called from a client application?

- A. The Foundry Playground
- B. Only via a production client app — there's no interactive testing option
- C. The Azure billing dashboard
- D. A local text editor

<details><summary><b>Answer</b></summary>

**A.** All prompt agents in a project surface directly in the Playground for interactive testing, the same as deployed chat models.
</details>

**Q43.** What is the main trade-off a team accepts by choosing a hosted agent instead of a prompt agent?

- A. Hosted agents cannot use tools at all
- B. The team takes on responsibility for the agent's code and container packaging, in exchange for more orchestration control
- C. Hosted agents cannot be tested in the Playground under any circumstance
- D. There is no trade-off — hosted agents are strictly better in every scenario

<details><summary><b>Answer</b></summary>

**B.** Hosted agents trade "no code to write or maintain" for the ability to bring custom orchestration logic — the team now owns the agent's code and container, in exchange for that flexibility.
</details>

### Lightweight agent client applications (§15)

**Q44.** In a lightweight agent client, what gives the agent conversational memory across multiple turns, as opposed to a one-shot chat-completions call?

- A. A thread
- B. The system prompt alone
- C. The deployment type
- D. Increasing max tokens

<details><summary><b>Answer</b></summary>

**A.** A thread holds the conversation's messages across turns, giving the agent memory of prior exchanges — distinct from a single stateless chat-completions request.
</details>

**Q45.** What is the correct order of operations for a minimal agent client interaction?

- A. Run the agent → create a thread → post a message → authenticate
- B. Authenticate → get the agent client → create/reuse a thread → post a user message → run the agent → read the response
- C. Post a user message → authenticate → run the agent → create a thread
- D. Read the response → authenticate → run the agent → post a message

<details><summary><b>Answer</b></summary>

**B.** The natural order is: authenticate, get the project's agent client, create or reuse a thread, post the user's message to it, run the agent, then read its response from the thread.
</details>

**Q46.** A developer wants to continue an existing conversation with an agent rather than start fresh. What should the client do?

- A. Create a brand-new thread for every single message
- B. Reuse the existing thread ID when posting the next message
- C. Ignore threads entirely and resend the entire conversation history as a system prompt every time
- D. Switch the agent to a hosted agent

<details><summary><b>Answer</b></summary>

**B.** Reusing the existing thread preserves the conversation's context; creating a new thread each time would lose prior turns.
</details>

### Text analysis apps (§16)

**Q47.** A lightweight app needs to extract entities, run sentiment analysis, and summarize customer feedback text. Which Foundry Tools service is purpose-built for this?

- A. Azure Speech in Foundry Tools
- B. Azure Language in Foundry Tools
- C. Azure Vision in Foundry Tools
- D. Content Understanding's video analyzer

<details><summary><b>Answer</b></summary>

**B.** Azure Language in Foundry Tools is the dedicated text-analysis service (entities, sentiment, key phrases, summarization).
</details>

**Q48.** Besides calling a dedicated Language service, what is an alternative way to perform a text-analysis task like summarization in a Foundry-based app?

- A. There is no alternative — a dedicated service is strictly required
- B. Prompt a deployed generative model directly to perform the same task
- C. Use the image analyzer from Content Understanding
- D. Use Azure Speech synthesis

<details><summary><b>Answer</b></summary>

**B.** A deployed LLM can be prompted to perform tasks like summarization or entity extraction directly, as an alternative (with different accuracy/consistency/cost trade-offs) to calling a dedicated Language endpoint.
</details>

**Q49.** A text-analysis app returns structured JSON with entities, confidence scores, and a summary string. What should the app do with this output?

- A. Discard it — text analysis output is not meant to be consumed programmatically
- B. Surface the structured result to the user or pass it to a downstream system
- C. Convert it into an image before use
- D. Re-run OCR on it

<details><summary><b>Answer</b></summary>

**B.** The structured result (entities, scores, summary) is meant to be surfaced directly to the end user or consumed by a downstream system/workflow.
</details>

### Spoken prompts and Azure Speech in Foundry Tools (§17)

**Q50.** A simple "talk to the assistant" prototype needs to accept a spoken audio clip and respond, with minimal setup and no separate Speech resource. What is the simplest approach?

- A. Provision a dedicated Speech resource, custom vocabulary, and a TTS voice before writing any code
- B. Send the audio clip directly to a deployed multimodal model and let it return a response
- C. Manually transcribe the audio by hand before sending it to the model
- D. This scenario is not possible without Content Understanding's audio analyzer

<details><summary><b>Answer</b></summary>

**B.** A deployed multimodal model can accept audio directly as part of the request, collapsing the traditional STT → LLM → TTS pipeline into one call — the simplest option for a minimal prototype.
</details>

**Q51.** An enterprise voice application requires custom vocabulary support, precise word-level timestamps, and a specific branded synthesized voice. Which approach is most appropriate?

- A. A generic multimodal model handling audio directly, with no further configuration
- B. Azure Speech in Foundry Tools, configured with custom vocabulary and a specific voice
- C. Content Understanding's document analyzer
- D. Azure Vision's OCR feature

<details><summary><b>Answer</b></summary>

**B.** When fine-grained control is needed — custom vocabulary, precise timestamps, specific synthesized voices — the dedicated Azure Speech service is the right call rather than relying on a general multimodal model's audio handling.
</details>

**Q52.** What does speech translation combine to convert spoken input in one language into text or speech in another?

- A. Speech recognition and translation
- B. Image generation and OCR
- C. Sentiment analysis and summarization
- D. Entity detection and key phrase extraction

<details><summary><b>Answer</b></summary>

**A.** Speech translation combines recognition (converting speech to text) with translation (converting that text into another language), optionally followed by synthesis to produce spoken output.
</details>

**Q53.** In a lightweight speech app that needs full control over transcription accuracy and reply voice, what is the typical processing order?

- A. Synthesize speech → transcribe → route through a model
- B. Transcribe speech to text (Speech) → optionally route text through a model → synthesize the reply (Speech)
- C. Route directly to Content Understanding's video analyzer
- D. Skip transcription and synthesize a reply from silence

<details><summary><b>Answer</b></summary>

**B.** The dedicated-service pipeline is: transcribe the spoken input, optionally process the resulting text with a model, then synthesize a spoken reply — giving control at each stage that a single multimodal call wouldn't provide.
</details>

### Vision and image generation in Foundry (§18)

**Q54.** An app needs to answer a user's question about the contents of a photo they just uploaded (e.g., "what's wrong with this device in the picture?"). What is the simplest implementation approach?

- A. Send the image and question together to a deployed multimodal model
- B. Use Content Understanding's audio analyzer
- C. Use an image-generation model
- D. Use speech synthesis on the image

<details><summary><b>Answer</b></summary>

**A.** A deployed multimodal model can accept an image alongside a text prompt and answer questions about it directly — no separate vision resource required for this kind of Q&A.
</details>

**Q55.** A marketing tool needs to produce a brand-new image from a text description entered by the user. Which type of model is required, and why can't an image-understanding multimodal model do this instead?

- A. An image-generation model is required; an understanding model interprets existing images but doesn't create new ones
- B. Any multimodal model works identically for both understanding and generating images
- C. Azure Speech in Foundry Tools is required
- D. Content Understanding's document analyzer is required

<details><summary><b>Answer</b></summary>

**A.** Understanding ("what's in this image?") and generation ("create an image from this description") are different capabilities — a model built for understanding visual input doesn't necessarily generate new images, and vice versa.
</details>

**Q56.** A team wants a lightweight vision app that both answers questions about uploaded images and produces new promotional images on request. What does this most likely require?

- A. A single capability handles both — no distinction needed
- B. Both a multimodal understanding-capable model and an image-generation model, wired into the same app
- C. Only Azure Speech in Foundry Tools
- D. Only Content Understanding's video analyzer

<details><summary><b>Answer</b></summary>

**B.** Since understanding and generation are distinct capabilities, an app that does both needs both a multimodal model for interpreting uploaded images and a separate image-generation model for creating new ones.
</details>

**Q57.** A retail app needs to detect and label multiple distinct objects in a shelf photo with bounding boxes, not just produce a one-sentence caption. Which capability is this?

- A. Image captioning
- B. Object detection
- C. Face detection
- D. Image generation

<details><summary><b>Answer</b></summary>

**B.** Object detection locates and labels multiple objects with bounding boxes, distinct from a single overall caption.
</details>

### Content Understanding for information extraction (§19)

**Q58.** A team wants to extract a custom set of fields (invoice number, vendor, total) from scanned invoices that vary widely in layout and are sometimes handwritten. Which Foundry Tools capability fits best?

- A. Azure Speech in Foundry Tools
- B. Content Understanding's document analyzer
- C. Image generation
- D. Entity detection alone, with no schema

<details><summary><b>Answer</b></summary>

**B.** Content Understanding's document analyzer extracts schema-defined fields from forms/documents, including messier, multimodal, or handwritten content — complementing Document Intelligence's deterministic extraction from well-structured documents.
</details>

**Q59.** A company wants structured metadata — topics discussed, overall sentiment, and action items — pulled automatically from recorded sales calls. Which Content Understanding analyzer applies?

- A. Document analyzer
- B. Image analyzer
- C. Audio analyzer
- D. Video analyzer

<details><summary><b>Answer</b></summary>

**C.** The audio analyzer builds structured fields (topics, sentiment, action items) on top of transcription for audio content like call recordings.
</details>

**Q60.** A schema requires deep reasoning over complex, ambiguous source material to fill in correctly, and the team is willing to accept higher cost/latency for better extraction quality. Which Content Understanding mode should they choose?

- A. Standard mode
- B. Pro mode
- C. Neither mode — Content Understanding has no mode selection
- D. Global Batch mode

<details><summary><b>Answer</b></summary>

**B.** Pro mode trades cost/speed for deeper reasoning over complex content, appropriate when a schema needs more than straightforward, well-defined field extraction. Standard mode (A) is the faster/cheaper option for simpler schemas. "Global Batch" (D) is a model-deployment concept, not a Content Understanding mode.
</details>

---

## 🔗 See also

- [`EXAM_NOTES.md`](EXAM_NOTES.md) — the full study notes these questions are drawn from
- [`README.md`](README.md) — chapter overview and how this material was built
