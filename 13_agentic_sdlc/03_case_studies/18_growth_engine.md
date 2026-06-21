# Chapter 18 — Building a Growth Engine

> Source: *The Agentic SDLC Handbook* (Daniel Meppiel, v0.9.2), pp. 220–223.
> The methodology's limit-testing chapter: applied to non-engineering work (marketing/growth infrastructure), this case study is less about what got built and more about where agent automation hit genuine platform and policy walls that no amount of retrying could cross.

---

## Very Important

### What Was Built and Why This Case Study Matters
**Scope:** Checkpoints 014–019 — 6 phases — 18 implementation tasks.
**Duration:** ~8 hours wall-clock across multiple sessions.
**Theme:** what happens when PROSE methodology meets non-engineering domains — and the pivots, platform limitations, and policy constraints that emerge.

What was built: a growth engine for a free technical book — email capture, landing page, DNS and email infrastructure, a PII audit, and launch preparation. The book is explicit that **the interesting story is not what got built, but how many times the plan changed and where the methodology hit its limits.**

### Panel Strategy and the Scope-Correction Override
Expert panels drove the strategic decisions. Across six phases, the orchestrator convened panels with **fifteen domain-specific personas** (publishing strategy, growth, branding, LinkedIn, security) that produced synthesized recommendations through moderator agents. Early panels over-indexed on career coaching, prompting a direct user override: *"Focus on the growth engine — brand, followers, industry gravitas, tactical distribution."* The orchestrator reframed the brief and produced an 18-task implementation plan. The book identifies this explicitly as **the Architect role from Chapter 13: scope correction when agent output drifts from the actual objective.**

### The Kit Automation Escalation
The most instructive failure sequence in the chapter: automating the Kit email platform, where **three distinct approaches each hit a different wall**.

**What happened:** Playwright scripts handled login, tag creation, and form creation — even form *publishing* succeeded. But Kit's form builder uses React-controlled components: the textarea and combobox dropdowns are not native HTML elements. Standard DOM manipulation (`element.fill()`, keyboard shortcuts, JavaScript value setters) all failed because React reconciles its virtual DOM and discards external mutations.

| Attempt | Tool | Result |
|---|---|---|
| 1 | Playwright scripts | Login, tags, form creation succeeded; **FAIL** on React textarea (`.fill()`, Meta+a, JS setter all fail) and React combobox (not a native select) |
| 2 | Playwright MCP browser control | `navigate`, `click`, `snapshot` work; **FAIL** — `browser_run_code` throws SyntaxError on all inputs; still cannot solve the React combobox |
| 3 | Kit V3 REST API directly | List forms and tags succeeded; **FAIL** — `/v3/automations` returns 404 (not exposed) |
| — | Human escalation | 3-step manual checklist completed in 2 minutes |

The book's framing of the value here is sharp: **the methodology's value was not in automating this task — it was in recognizing when to stop trying.** Each approach was trusted to work based on partial success before validating the fundamental constraint: React's virtual DOM rejects external mutations. The discipline was stopping after three genuine platform limitations instead of attempting a fourth DOM hack. The book's reusable artifact is the escalation checklist itself:

**Kit Form Automation — Human Escalation Checklist:**
1. Open Kit form editor > select the confirmation message textarea
2. Paste: "Check your email to confirm your subscription"
3. In the automation dropdown, select "Add subscriber to sequence: Welcome"

Three automated approaches failed on React internals; the human completed this in 2 minutes. The stated discipline: **recognize when the last 10% is a platform limitation, not an application-logic problem.**

### The Persona Drift Correction
During landing page work, the orchestrator's authority profile described the author as a "software engineer." The user corrected this immediately: *"I am NOT a Software Engineer. I'm a Global Black Belt with 14+ years enterprise strategy."*

This is identified explicitly as **Anti-pattern #17 — Persona Drift**: the agents had latched onto the most common tech-industry persona in their training data rather than the actual role described in the source documents. It also surfaced **Anti-pattern #7 — The Trust Fall**: the authority profile had been generated in an earlier wave and accepted without human verification. A single factual error — job title — cascaded into every downstream deliverable that referenced the author's credentials (landing page, preface, chapter bios, panel briefs). The fix was cheap (find-and-replace across files), but only because it was caught early; the book notes that had the book shipped with "software engineer" in the bio, the credibility damage would have been real.

### The PII Audit Pipeline
Four parallel agents scanned the repository for sensitive data; three completed normally, finding issues across 25+ files. **One agent refused to scan** — the career directory contained personal documents and the agent's safety guardrails triggered, declining to process content it classified as sensitive personal information.

The book frames this as a genuine edge case: the guardrail was correct *in principle* (the files genuinely were sensitive), but the task was finding sensitive content in order to *remove* it — the agent couldn't distinguish audit-to-remove from audit-to-exploit, an edge case where safety boundaries and task objectives were aligned but the agent still couldn't proceed. The orchestrator adapted with manual `grep` analysis instead. The pipeline: four parallel agents (reviewer, audits+agents, config files, career-folder agent) → findings (three) + refusal (one, citing safety guardrails) → manual grep analysis folded in alongside the three automated findings → a single remediation plan → `git-filter-repo` history scrub → clean history, originals preserved locally.

### Constraint Discovery — Two Walls No Panel Anticipated
**Email infrastructure.** The domain registrar had silently discontinued free email forwarding. The orchestrator discovered this during DNS setup, pivoted to ImprovMX (free tier), configured MX records, added SPF and DKIM entries, and verified delivery — a pure infrastructure pivot, absorbed without escalation.

**CELA risk — the chapter's novel finding.** A PR adding a handbook link to the APM README (a Microsoft org repository) was identified by the user as a potential compliance risk — personal lead generation via a corporate open-source asset. The PR was closed and a new constraint was added: no promotional links from `microsoft/` org repos to personal content. The book is explicit about why this matters: this is **the Architect role exercising judgment that no agent could have made** — organizational policy awareness that exists nowhere in the training data and cannot be inferred from public documentation. No panel suggested this constraint; no agent flagged it. The entire deliberation architecture — fifteen personas across seven panels — missed it, because organizational policy is not in the training data and cannot be inferred from public documentation.

A secondary lesson on agent inputs surfaced in the same phase: agents needed specific artifacts in context to be useful, not just instructions. The first two landing-page agents failed — one stalled for eight minutes, the other produced generic copy — because they lacked the actual HTML. The third attempt embedded the full page source and immediately produced field-by-field rewrites. The book's stated lesson: **progressive disclosure means providing the *right* context, not the *least* context.**

### What Held True — and the Permanent Boundary Found
The structural properties held where applicable, but the chapter's **novel and most important finding** is stated as a hard limit on the methodology itself: **organizational policy awareness lives nowhere in training data.** This is presented as a **permanent boundary of agentic methodology that no model improvement will address** — not a current limitation expected to close with better models, but a structural one.

The growth engine shipped. The book is careful to state *why* it shipped: not because every agent succeeded — the Kit automation plainly failed three times — but because **the orchestrator knew when to pivot and when to stop.**

| Metric | Value |
|---|---|
| Implementation phases | 6 |
| Expert panels | 7+ |
| Agent personas | 15 |
| Kit automation attempts | 3 (all blocked by platform limitations) |
| Manual workarounds | 1 |
| CELA risk discoveries | 1 |
| Anti-patterns observed | 3 (#17 Persona Drift, #7 The Trust Fall, #5 Scope Creep) |
| Methodology limitation discovered | Organizational policy awareness |

---

## Important

### Key Takeaways (Chapter's Own TL;DR)
Stated up front in the chapter as the condensed summary:
- When three automated approaches each hit a genuine platform limitation, escalating to a human with a precise checklist is the correct move — not a fourth attempt.
- A single factual error (wrong job title) cascades into every downstream deliverable — catch persona drift early.
- Organizational policy awareness lives nowhere in training data; human judgment remains the irreplaceable input.
- Pivots are normal: the plan changed repeatedly, and the orchestration protocol absorbed each one.

### Anti-Pattern Catalogue Observed in This Chapter
Three named anti-patterns surfaced concretely in this case study, each with a distinct trigger:

| Anti-pattern | Trigger in This Chapter |
|---|---|
| #17 Persona Drift | Orchestrator described the author as a "software engineer," defaulting to training-data stereotype over source-document fact |
| #7 The Trust Fall | The incorrect authority profile was generated in an earlier wave and propagated without human verification |
| #5 Scope Creep | Early panels drifted into career-coaching territory before a user override reframed the brief to the actual growth-engine objective |

### Infrastructure Pivots Absorbed Without Escalation
Not every constraint triggered an escalation event — some were absorbed as ordinary pivots within the existing plan: the email-forwarding registrar change (pivot to ImprovMX, MX/SPF/DKIM reconfiguration) was discovered and resolved within the DNS-setup phase itself, illustrating that the orchestration protocol's job is partly to absorb this kind of plan change routinely, reserving escalation for genuine platform or policy walls.

---

## Key Terms

- **CELA risk** — a compliance/legal risk (Corporate, External & Legal Affairs-style review) flagged when a Microsoft org repository linked to personal lead-generation content; the chapter's novel finding that organizational policy constraints cannot be inferred by agents from public information.
- **Persona Drift (Anti-pattern #17)** — an agent defaulting to the most statistically common identity/role in its training data instead of the actual role stated in source documents.
- **Kit** — the email marketing platform whose React-controlled form builder defeated three separate automation approaches (Playwright scripts, Playwright MCP, REST API).
- **Escalation checklist** — the chapter's reusable artifact: a precise, numbered manual procedure handed to a human once automated approaches have exhausted genuine platform limitations.
- **ImprovMX** — the free-tier email-forwarding service adopted after discovering the domain registrar had discontinued its own free forwarding.
