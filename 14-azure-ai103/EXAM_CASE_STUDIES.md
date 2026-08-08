# 📚 AI-103 Practice Case Studies — Exam-Style Scenarios with Answers

> **3 original case studies, 20 questions total (CS1: Q1–Q7, CS2: Q1–Q7, CS3: Q1–Q6)**, written in the format the real AI-103 (and its predecessor AI-102) uses for its case-study section: a **Company Overview**, an **Existing Environment**, and **Requirements** (business + technical), followed by a battery of questions that all draw on the same scenario — multiple-choice, multi-select, "Yes/No per statement," ordered-steps, and fill-in-the-blank (hotspot-style) items.
>
> ⚠️ **Not exam dumps.** These are freshly written, scenario-based case studies grounded in [`EXAM_NOTES.md`](EXAM_NOTES.md) and official Microsoft documentation — not reproductions of real (NDA-protected) exam items. If a question here and Microsoft documentation ever disagree, trust the documentation. For officially sanctioned questions, take Microsoft's free Practice Assessment on the AI-103 exam page.
>
> **How the real exam's case studies work — and how to practice that way here:**
> - On the real exam you get a **scrollable case study pane** (Overview / Existing Environment / Requirements tabs) and a **separate list of questions** about it. You can revisit the case study tabs at any time while answering its questions, but **once you leave the case-study question set, you cannot return to it.** Read the whole case once, fully, before answering anything.
> - Case-study questions are **deliberately cross-domain** — a single scenario tests planning/security choices (Domain 1) alongside generative AI/agent design (Domain 2) and one of vision/text/extraction (Domains 3–5), because that's how real architecture decisions actually interact.
> - Cover the answer, commit to a choice using only facts stated in the case, then expand.

---

# Case Study 1 — CloudXeus IT Helpdesk Modernization

*(Domains exercised: 1 — Plan & Manage, 2 — Generative AI & Agents, 4 — Text Analysis)*

## Company Overview

CloudXeus is a managed IT services provider with 400 employees across three regions (US, EU, APAC). It runs an internal helpdesk that currently routes every ticket to a human agent, regardless of complexity. Average first-response time is 6 hours, and 70% of tickets are repetitive password-reset, VPN-access, and software-license requests.

## Existing Environment

- CloudXeus has a Microsoft Foundry project (`cloudxeus-project`) with a GPT-4o-class model deployed as `cloudxeus-gpt4o`, currently on a **Standard** deployment.
- Ticket text is stored in a SQL database. No customer PII beyond employee name and email is ever entered in ticket free text, but EU works-council rules require that free-text ticket bodies **never leave EU boundaries** for EU-origin tickets.
- The current helpdesk web app authenticates to Foundry using an API key stored in the app's `appsettings.json` file, checked into a private Git repo.
- A security audit flagged the API key as a finding: keys in source control are considered a high-severity risk regardless of repo visibility.
- CloudXeus has an existing Azure Function, `helpdesk_functions.py`, exposing `reset_password()` and `check_license_seats()` as callable operations, currently invoked only by a legacy scheduled job — not by any agent.

## Business Requirements

- Reduce average first-response time for the 70% of tickets that are repetitive, routine requests.
- Satisfy the EU works-council data-residency rule for EU-origin ticket text.
- Remediate the audit finding about the stored API key before the next release.
- Give IT managers a way to see, after the fact, which tool calls an automated response made and why — for audit purposes.

## Technical Requirements

- The solution must use the existing `cloudxeus-project` Foundry deployment and the existing `helpdesk_functions.py` operations wherever possible — do not rebuild them.
- Whatever authentication mechanism is chosen must not require CloudXeus to manage or rotate any secret.
- The support team wants tickets that are ambiguous or emotionally charged (e.g., an angry customer) to still reach a human, not the automated flow.

---

### CS1 Q1 — Authentication

You must remediate the audit finding about the API key in `appsettings.json` without adding any secret-rotation burden. What should you implement?

- A. Move the key into an environment variable on the app host
- B. Move the key into Azure Key Vault and reference it via Key Vault references
- C. Replace the key with a managed identity on the app host and grant it an RBAC role on the Foundry resource
- D. Rotate the key manually every 90 days and store the new value in the repo's CI/CD secrets

<details><summary><b>Answer</b></summary>

**C.** The technical requirement explicitly says "must not require CloudXeus to manage or rotate any secret" — that phrase is the exam's signal for **managed identity**, the only keyless option among the four. B (Key Vault) still stores a long-lived secret that must eventually rotate; A is strictly worse than B; D directly violates the "no rotation burden" requirement.
</details>

---

### CS1 Q2 — Deployment type

The routine 70%-of-tickets flow will run continuously during business hours at a fairly steady request rate, and the business requirement is *faster* first response. Which deployment type best fits, and why?

- A. Global Batch — cheapest per token
- B. Standard — no change needed
- C. Provisioned (PTU) — reserved capacity gives predictable low latency at steady volume
- D. Global Standard — highest burst capacity, spiky traffic pattern

<details><summary><b>Answer</b></summary>

**C.** The business requirement is to *reduce first-response time* for a *steady, high-volume, business-hours* workload — that is the textbook PTU profile (predictable latency, reserved capacity). Global Batch (A) is asynchronous and can take up to 24 hours, the opposite of "reduce first-response time." Global Standard (D) is for bursty/unpredictable traffic, which this case does not describe — the ticket load is steady.
</details>

---

### CS1 Q3 — Data residency

For each statement about the EU data-residency requirement, select **Yes** if the approach satisfies it, or **No** if it does not.

| Statement | Yes / No |
|---|---|
| Deploy a second Foundry project with a model in an EU Azure region, and route EU-origin tickets to it. | ? |
| Keep a single global Foundry deployment, but encrypt EU ticket text with a customer-managed key before sending it. | ? |
| Use Global Batch deployment for EU tickets, since batch jobs are always processed in-region. | ? |

<details><summary><b>Answer</b></summary>

**Yes / No / No.**
- Statement 1 — **Yes.** Deploying to an EU-region Foundry project and routing EU traffic to it is the standard way to satisfy data-residency requirements; the model processes the data inside the required boundary.
- Statement 2 — **No.** Encryption in transit/at rest doesn't change *where* the data is processed — the plaintext still has to be decrypted for the model to read it, and that processing still happens wherever the deployment lives. Encryption addresses confidentiality, not residency.
- Statement 3 — **No.** Deployment type (Batch vs. Standard vs. PTU) controls latency/cost/throughput trade-offs, not the geographic region of processing. Region is chosen when you create the resource/deployment, independent of deployment type.
</details>

---

### CS1 Q4 — Reusing the existing Function

The routine-ticket flow should call the existing `reset_password()` and `check_license_seats()` operations from `helpdesk_functions.py` without rewriting them. What is the correct way to make a Foundry agent use them?

- A. Rewrite them as a new Foundry-native connector
- B. Register them as a custom **function tool** on the agent, with the agent's model deciding when to call each function based on the ticket text
- C. Have the agent generate Python code at runtime that re-implements password reset
- D. Add them as a **code interpreter** tool so the model can execute them in a sandbox

<details><summary><b>Answer</b></summary>

**B.** This is exactly what function tools are for: expose existing application logic (already-written functions) as callable tools, described to the model with a name/description/parameter schema, and let the model decide when a ticket's content calls for `reset_password()` vs. `check_license_seats()` vs. neither. A violates "do not rebuild"; C is unsafe and unnecessary — the function already exists; D (code interpreter) is for the model to run *arbitrary generated code* in a sandbox, not for calling your own pre-existing, trusted functions.
</details>

---

### CS1 Q5 — Audit trail

IT managers want to see which tool calls an automated response made and why, after the fact. Which Foundry/Foundry-Agent-Service capability should you enable?

- A. Content filtering
- B. Tracing with content recording enabled
- C. Prompt Shields
- D. A higher TPM quota

<details><summary><b>Answer</b></summary>

**B.** Tracing captures the reasoning trail — which tools were invoked, with what arguments, and the model's intermediate steps — which is exactly "which tool calls were made and why." Content filtering (A) and Prompt Shields (C) are safety controls, not observability; a TPM quota (D) is a capacity setting, unrelated to auditability.
</details>

---

### CS1 Q6 — Escalation to a human

Ambiguous or emotionally charged tickets must still reach a human. Which design best satisfies this without CloudXeus building a separate sentiment-detection pipeline from scratch?

- A. Instruct the agent's system prompt to always answer every ticket itself
- B. Add a routing step that runs sentiment/intent analysis (e.g., via Azure AI Language or a classification prompt) and forces a human handoff below a confidence threshold or on negative sentiment
- C. Disable the agent for all tickets containing an exclamation point
- D. Increase the model's temperature so answers sound more empathetic

<details><summary><b>Answer</b></summary>

**B.** A routing/classification step that scores intent clarity and sentiment, with an explicit threshold that forces human handoff, is the standard "know when *not* to automate" pattern — and it reuses an existing service (Azure AI Language sentiment analysis) rather than building new ML. A ignores the requirement entirely; C is a brittle keyword hack, not a reliable signal; D changes *tone*, not whether a human sees the ticket, and doesn't address ambiguity at all.
</details>

---

### CS1 Q7 — Ordering the fix

Arrange the remediation steps for the API-key audit finding in the correct order.

- [ ] Grant the app's managed identity an RBAC role scoped to the Foundry resource (e.g., Cognitive Services OpenAI User)
- [ ] Remove the API key from `appsettings.json` and from source control history
- [ ] Enable a system-assigned (or user-assigned) managed identity on the app host
- [ ] Update the app code to authenticate via `DefaultAzureCredential`/`ManagedIdentityCredential` instead of an `AzureKeyCredential`

<details><summary><b>Answer</b></summary>

**Correct order:**
1. Enable a system-assigned (or user-assigned) managed identity on the app host
2. Grant the app's managed identity an RBAC role scoped to the Foundry resource (e.g., Cognitive Services OpenAI User)
3. Update the app code to authenticate via `DefaultAzureCredential`/`ManagedIdentityCredential` instead of an `AzureKeyCredential`
4. Remove the API key from `appsettings.json` and from source control history

You must have an identity to grant permissions *to* before granting them, then switch the code to use it, and only remove the old key **last** — once the new path is proven to work, so there's never a gap in which the app can't authenticate at all. Removing history (old commits) matters too: a key merely deleted from the latest commit is still readable in Git history.
</details>

---

# Case Study 2 — Fabrikam Retail Visual Catalog & Invoicing

*(Domains exercised: 1 — Plan & Manage, 3 — Computer Vision, 5 — Information Extraction)*

## Company Overview

Fabrikam Retail sells furniture online. Product photography is expensive and slow — a new SKU can take two weeks to get studio-quality images. Separately, Fabrikam's accounts-payable team manually keys in ~3,000 supplier invoices per month from PDF and scanned-paper sources, in several layouts that change per supplier.

## Existing Environment

- Fabrikam has an Azure OpenAI resource with an image-generation-capable model deployed.
- Product photos, once generated, are reviewed by a human merchandiser before publishing — this human review step is contractually required by Fabrikam's brand-safety policy and **cannot be removed**.
- Accounts-payable currently emails PDF invoices to a shared mailbox; a junior analyst opens each one and re-types line items into an ERP system.
- Some suppliers send **scanned, handwritten** delivery notes alongside typed invoices.
- Fabrikam has no existing Document Intelligence or Content Understanding resource today.
- Legal requires that any AI-generated marketing image be **detectably labeled** as AI-generated when published externally.

## Business Requirements

- Cut new-SKU photography turnaround from two weeks to under two days.
- Reduce manual invoice keying effort by at least 80%.
- Meet the AI-generated-image labeling requirement without manual tagging by the merchandising team.
- Handle both the typed invoices and the scanned handwritten delivery notes with a single extraction pipeline where possible.

## Technical Requirements

- The image pipeline must still route every generated image through the existing human merchandiser review step.
- The invoice/document solution should minimize custom model training if a prebuilt or lightly-customized option can meet the requirement.

---

### CS2 Q1 — Image generation pipeline

Which pipeline satisfies both the turnaround requirement and the mandatory human-review requirement?

- A. Generate images and auto-publish directly to the storefront, skipping review to save time
- B. Generate candidate images with the image-generation model, route them to the existing merchandiser review queue, and publish only approved images
- C. Use a code interpreter tool to programmatically approve images that pass a resolution check
- D. Batch-generate all images once per quarter to reduce API cost

<details><summary><b>Answer</b></summary>

**B.** This is the only option that both speeds up generation *and* keeps the contractually-required human review step intact. A violates the requirement outright ("cannot be removed"). C replaces human judgment about brand fit with an automated resolution check — not the same thing, and still effectively removes human review. D doesn't address the two-day turnaround goal at all.
</details>

---

### CS2 Q2 — AI-generated content labeling

How should Fabrikam meet the requirement that AI-generated marketing images be detectably labeled, without relying on the merchandising team to manually tag each one?

- A. Ask merchandisers to add a text watermark manually before publishing
- B. Rely on the image generation service's built-in content credentials / provenance metadata (e.g., C2PA-style signing) that Azure OpenAI image generation attaches automatically
- C. Rename the image file to include "AI" in the filename
- D. Add a disclaimer to the website's general terms and conditions page

<details><summary><b>Answer</b></summary>

**B.** Azure OpenAI's image generation service attaches provenance/content-credential metadata to generated images automatically — satisfying "detectably labeled" without a manual step. A and C are manual, error-prone, and easily dropped; D doesn't label the *image itself* and wouldn't satisfy a per-image labeling requirement.
</details>

---

### CS2 Q3 — Choosing the extraction service

Which service should Fabrikam adopt for the invoice/delivery-note extraction pipeline, given the technical requirement to minimize custom training?

- A. A general LLM prompt asking the model to "read this PDF and list the line items," with no structured schema
- B. Azure AI Document Intelligence prebuilt invoice model, supplemented by Content Understanding for the handwritten delivery notes
- C. Train a custom object-detection model on labeled invoice images from scratch
- D. Azure AI Vision Image Analysis's OCR "Read" feature alone, with all downstream parsing done in application code

<details><summary><b>Answer</b></summary>

**B.** Document Intelligence ships a **prebuilt invoice model** that already understands common invoice fields (vendor, line items, totals) across varying layouts — no training required for the typed invoices. Content Understanding's document/analysis capabilities handle the harder, more heterogeneous case (handwritten notes) with schema-driven extraction, again without training a model from zero. A produces unstructured, unreliable output with no guaranteed schema — poor fit for feeding an ERP system. C directly violates "minimize custom training." D (raw OCR text only) pushes all the structuring work into custom app code that Document Intelligence already does for you — more effort, not less.
</details>

---

### CS2 Q4 — Handling mixed typed + handwritten input

For each statement about handling the handwritten delivery notes, select **Yes** if it is an appropriate approach or **No** if it is not.

| Statement | Yes / No |
|---|---|
| Content Understanding / Document Intelligence's handwriting-capable OCR can extract handwritten text alongside typed text in the same pipeline. | ? |
| Handwritten content should be silently dropped, since no Azure service can process handwriting reliably. | ? |
| A single extraction pipeline can output a common schema (e.g., line items, quantities, totals) regardless of whether the source page was typed or handwritten. | ? |

<details><summary><b>Answer</b></summary>

**Yes / No / Yes.**
- Statement 1 — **Yes.** Both Document Intelligence and Content Understanding support handwriting recognition as part of their OCR layer, not just typed/printed text.
- Statement 2 — **No.** This is false and would fail the business requirement to handle both document types "with a single extraction pipeline where possible" — handwriting support exists and should be used.
- Statement 3 — **Yes.** Schema-driven extraction (defining the fields you want, like `line_items[].description`, `quantity`, `total`) is exactly how these services normalize output regardless of the input's visual format — that's the point of a defined output schema versus raw OCR text.
</details>

---

### CS2 Q5 — Cost/latency for the invoice batch

Accounts-payable processes invoices once per night in a batch of a few hundred at a time; nobody is waiting on the result in real time. Which choice minimizes cost for the *model calls in this pipeline* (e.g., any LLM step used for line-item normalization after extraction) without harming the business requirement?

- A. Provisioned (PTU) deployment, since batches need guaranteed throughput
- B. Global Batch deployment, since the workload is asynchronous and latency-insensitive
- C. Global Standard deployment, to handle unpredictable spikes
- D. Multiple Standard deployments across regions for redundancy

<details><summary><b>Answer</b></summary>

**B.** "Once per night, nobody waiting in real time" is the canonical Global Batch scenario — lowest per-token cost, in exchange for asynchronous (up to 24-hour) turnaround the business doesn't need faster than nightly anyway. A, C, and D are all built for latency-sensitive or unpredictable-traffic scenarios that don't describe this workload, and all cost more than Batch for no benefit here.
</details>

---

### CS2 Q6 — Data sensitivity in invoices

A supplier's invoice PDF happens to contain an individual's home address in a "ship-to" field. Fabrikam's legal team wants this flagged before the record reaches long-term storage. Which capability should be added to the pipeline?

- A. Azure AI Language PII detection over the extracted text fields
- B. A stronger image-generation content filter
- C. Prompt Shields on the invoice text
- D. Increasing the Document Intelligence confidence threshold

<details><summary><b>Answer</b></summary>

**A.** PII detection is a text-analysis capability purpose-built to flag categories like addresses, names, and phone numbers in extracted text. B is unrelated (that's for generated images, not invoices). C targets prompt-injection attacks, not PII, though it's a reasonable *additional* control if invoice text is later fed to an LLM — it doesn't by itself flag PII. D changes extraction accuracy, not sensitive-data detection.
</details>

---

### CS2 Q7 — Architecture ordering

Arrange the invoice-processing pipeline steps in the correct order, from a supplier's email landing in the shared mailbox to a record appearing in the ERP system.

- [ ] Run PII detection over extracted fields and flag any matches for legal review
- [ ] Extract structured fields (vendor, line items, totals) using Document Intelligence's prebuilt invoice model or Content Understanding
- [ ] Automatically route the incoming PDF/scanned attachment out of the mailbox into blob storage
- [ ] Write the validated, structured record into the ERP system

<details><summary><b>Answer</b></summary>

**Correct order:**
1. Automatically route the incoming PDF/scanned attachment out of the mailbox into blob storage
2. Extract structured fields (vendor, line items, totals) using Document Intelligence's prebuilt invoice model or Content Understanding
3. Run PII detection over extracted fields and flag any matches for legal review
4. Write the validated, structured record into the ERP system

Ingest first (get the raw file somewhere a pipeline can reach it), then extract structure, then apply data-governance checks (PII) on that now-structured text — checking unstructured raw bytes for PII is far less reliable than checking labeled fields — and only then commit to the system of record.
</details>

---

# Case Study 3 — Northwind Traders Multilingual Contact Center

*(Domains exercised: 1 — Plan & Manage, 2 — Generative AI & Agents, 4 — Text Analysis / Speech)*

## Company Overview

Northwind Traders runs a phone-and-chat contact center supporting customers in English, French, and Japanese. Call volume is highly seasonal — quiet most of the year, then 5x normal volume during a two-week holiday sales event.

## Existing Environment

- Chat is text-based today, handled by human agents reading a shared inbox.
- Phone support uses a legacy IVR with no AI capability; recordings are archived but never analyzed.
- Northwind's Foundry project currently has one model deployment, sized for average (non-peak) chat volume.
- Compliance requires that call recordings **not be permanently stored with associated customer identity** beyond 30 days, and any transcript must have direct identifiers (names, phone numbers) removed before being used for quality-analysis reporting.
- Northwind wants agents' "system prompt" / instructions to remain identical across languages — the same agent, just capable of responding fluently in the customer's language.

## Business Requirements

- Extend AI assistance to phone calls, not just chat, including real-time transcription during the call.
- Handle the 5x holiday traffic spike without paying for 5x capacity year-round.
- Produce a de-identified transcript archive usable for quality reporting, compliant with the 30-day / no-identity-retention rule.
- Avoid maintaining three separate per-language versions of the agent's instructions.

## Technical Requirements

- Whatever handles multilingual behavior should not require the support team to translate and maintain the system prompt in three languages by hand.
- The chat and phone experiences should be able to call the same underlying agent/tool logic where possible.

---

### CS3 Q1 — Handling the seasonal spike

Which deployment strategy best satisfies "handle 5x holiday traffic without paying for 5x capacity year-round"?

- A. Provision PTU sized for peak (5x) volume permanently
- B. Use a Standard or Global Standard deployment that scales with pay-per-token pricing, reserving PTU (if at all) only for the peak window
- C. Keep a single fixed-size deployment and let requests queue during the spike
- D. Turn off the model entirely outside the peak window

<details><summary><b>Answer</b></summary>

**B.** Standard/Global Standard deployments are pay-per-token and elastic with Azure-managed capacity — you're not paying for idle reserved capacity the rest of the year. If truly predictable low latency during the peak window specifically is required, a temporary PTU reservation just for those two weeks is a valid refinement, but sizing PTU for peak *permanently* (A) is exactly the "pay for 5x capacity year-round" the requirement rules out. C harms customer experience during the exact window that matters most; D isn't viable — the service is needed for chat/phone year-round, just at lower volume.
</details>

---

### CS3 Q2 — Real-time phone transcription

Which Azure AI Speech capability directly satisfies "real-time transcription during the call"?

- A. Speech-to-text batch transcription API
- B. Real-time speech-to-text (streaming recognition)
- C. Text-to-speech neural voices
- D. Speech translation configured for offline files only

<details><summary><b>Answer</b></summary>

**B.** "Real-time... during the call" means streaming recognition, which returns partial and final transcription results as audio arrives, not after the fact. A (batch) processes a completed recording after the call ends — useful for archival analysis, wrong for in-call assistance. C converts text to speech, the opposite direction. D is explicitly offline-file-only, which rules out live calls.
</details>

---

### CS3 Q3 — One agent, three languages

How should Northwind avoid maintaining three separate per-language versions of the agent's system prompt/instructions?

- A. Write the instructions once in English; rely on the underlying LLM's native multilingual understanding and instruct it (once, in the single prompt) to respond in the customer's detected language
- B. Maintain three Foundry agents, one per language, each with translated instructions
- C. Pre-translate every customer message to English before sending it to the agent, and translate every response back
- D. Restrict the contact center to English only, and require customers to type in English

<details><summary><b>Answer</b></summary>

**A.** Modern GPT-4o-class models are natively multilingual — a single English-language instruction set that says "detect the customer's language and respond fluently in it" is enough; there's no need to translate the instructions themselves into French/Japanese. B directly creates the three-copies maintenance burden the requirement asks to avoid. C (translate-then-respond-then-translate-back) adds latency, translation-error risk, and an extra pipeline stage for no benefit when the model can just respond natively. D fails the business requirement outright — Northwind explicitly supports three languages.
</details>

---

### CS3 Q4 — De-identified transcript archive

For each statement about the compliant transcript archive, select **Yes** if it satisfies the requirement, or **No** if it does not.

| Statement | Yes / No |
|---|---|
| Run PII/PHI-style detection (e.g., Azure AI Language PII detection) over transcripts and redact detected names/phone numbers before archiving for quality reporting. | ? |
| Store the full transcript with customer name and phone number indefinitely, since it's needed to resolve future disputes. | ? |
| Set an automated retention/deletion policy so identity-linked recordings are removed after 30 days, while the redacted transcript persists longer for reporting. | ? |

<details><summary><b>Answer</b></summary>

**Yes / No / Yes.**
- Statement 1 — **Yes.** This is exactly the de-identification step the requirement asks for — PII detection and redaction before the transcript is used for quality-analysis reporting.
- Statement 2 — **No.** This directly violates the stated compliance rule ("recordings not permanently stored with associated customer identity beyond 30 days").
- Statement 3 — **Yes.** Separating the identity-linked recording (short retention) from the de-identified transcript (which no longer carries direct identifiers, so it can be retained longer for reporting) satisfies both the 30-day rule and the reporting business need simultaneously.
</details>

---

### CS3 Q5 — Shared logic between chat and phone

The technical requirement says chat and phone should call the same underlying agent/tool logic where possible. Which design achieves this?

- A. Build the phone assistant as a completely separate application with its own copy of every tool implementation
- B. Have both the chat front-end and the phone front-end (after speech-to-text converts audio to text) call the same Foundry agent and its registered function tools; only the input/output modality differs
- C. Only add AI to chat; keep phone as a plain human-staffed IVR forever
- D. Give the phone channel a cheaper, less-capable model than chat to reduce cost

<details><summary><b>Answer</b></summary>

**B.** Once speech-to-text turns the caller's audio into text, that text can be handed to the *same* agent/tool stack chat already uses — the agent doesn't need to know or care whether the text originated from typing or from a transcribed call. This is the core reason to route both channels through one Foundry agent rather than duplicating logic. A duplicates maintenance burden, which is exactly what the requirement rules out; C fails the business requirement to extend AI to phone; D isn't asked for anywhere in the case and would create inconsistent answer quality across channels for no stated reason.
</details>

---

### CS3 Q6 — Monitoring the peak window

During the two-week holiday spike, Northwind's operations team wants an early warning if the deployment is about to be throttled, before customers start seeing failed responses. What should they configure?

- A. Wait for customer complaints, then investigate
- B. Set up monitoring/alerting on token-usage and request-rate metrics against the deployment's TPM/RPM quota, with an alert threshold below 100% utilization
- C. Increase the content filter strictness
- D. Disable retries so failures surface immediately in application logs only

<details><summary><b>Answer</b></summary>

**B.** Proactive quota monitoring — alerting *before* you hit the ceiling (e.g., at 80% of TPM/RPM) — is what turns "early warning" into something operations can act on ahead of throttling, by requesting more quota or shifting load. A is reactive, not an early warning by definition. C is unrelated to throttling. D removes a resilience mechanism (retry/backoff) and doesn't provide any advance warning — it just changes where a failure is logged after it already happened.
</details>

---

## Previous File

← [`EXAM_PRACTICE_QUESTIONS_AGENTS_DEEPDIVE.md`](EXAM_PRACTICE_QUESTIONS_AGENTS_DEEPDIVE.md) (Q66–Q97)
