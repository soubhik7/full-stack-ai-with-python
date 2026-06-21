# Chapter 16 — Writing a ~68,000-Word Book with Agent Fleets

> Source: *The Agentic SDLC Handbook* (Daniel Meppiel, v0.9.2), pp. 206–213.
> A self-referential case study: this book — all 15 chapters, ~68,000 words — was drafted and composed using the same agentic orchestration patterns the book itself teaches, testing whether code-oriented patterns transfer to editorial composition.

---

## Very Important

### The Meta-Narrative
**Scope:** 15 chapters, ~68,000 words — a full technical book on AI-native software development.
**Duration:** single extended Copilot CLI session.
**Theme:** whether code orchestration patterns transfer to editorial composition.

*The Agentic SDLC Handbook* was drafted and composed through agentic orchestration in a single Copilot CLI session. The human author — Daniel Meppiel, Global Black Belt at Microsoft and creator of APM — provided domain expertise, intellectual property context, and editorial direction; Copilot CLI managed the agent team. The agent team itself was packaged and distributed using APM: 8 persona definitions as `.agent.md` files, an orchestration workflow as `SKILL.md`, declared as a dependency in `apm.yml`, and deployed via `apm install`. As the book puts it: **APM was used to build the book that teaches about APM.** This case study maps execution to the handbook's own vocabulary (Ch. 14) and tests its structural properties (Ch. 15) under editorial rather than engineering conditions.

### Team Topology: 11 Personas, Four Pods
The orchestrator designed an 8-persona expert panel at project inception, then **dynamically created 3 more** as needs emerged during execution — for a final total of 11 personas organized into four functional pods:

- **Domain Expert Pod** — C-Suite Strategist (executive lens), Practitioner Authority (org credibility), Market Analyst (landscape), Platform Strategist (architecture)
- **Editorial Pod** — Chief Editor (coherence), Thought Leadership (positioning)
- **Audit Pod (dynamic)** — Illustrator (visual strategy), Fact-Ref-Checker (claims audit), Publishing Advisor (distribution)
- **Review Pod** — CTO Proxy ("buy for my team?"), Dev Lead Proxy ("engineers use this?")

Each persona was a **primitive** — a composable instruction unit defined in a single `.agent.md` file. The orchestrator never prompted agents as generic LLMs; every dispatch carried persona context, scope boundaries, and output format requirements.

### Execution Timeline: Four Waves Plus Integration
The manuscript was produced in a phased pipeline: **corpus audit → architecture → four writing waves → integration review → polish**. Each wave followed checkpoint discipline: draft, review, revise, commit.

Wave ordering was deliberate, not arbitrary:
- **Wave 0** tested the pipeline with a single chapter.
- **Wave 1** targeted chapters with the most source material (lowest risk).
- **Wave 2** took on the hardest chapters requiring fresh writing.
- **Wave 3** handled integration chapters needing cross-references to earlier work.

This sequencing is **context budgeting** applied at the editorial level — right-sizing each wave's scope to what the agents could handle with available context, same principle as Chapter 15's file-splitting, applied to prose instead of code.

### The Draft→Review→Revise Cycle
Every chapter passed through an identical three-stage pipeline — the same quality loop the book describes in Chapter 13 for code:

1. Orchestrator dispatches a domain-specialist draft agent with the architecture spec + source material → chapter draft (3,000–5,000 words).
2. Parallel review: CTO Proxy reviews for executive audience, Dev Lead Proxy for practitioner audience, Chief Editor for coherence and voice — each returns a REVISE verdict plus fixes.
3. Orchestrator synthesizes consensus fixes, dispatches a revision agent to apply them, then checkpoints (commit draft + revision).

In later waves, the orchestrator **batched reviews by persona rather than by chapter** — sending one reviewer all 5 chapters at once rather than dispatching 15 separate reviews. This reduced dispatch overhead without sacrificing coverage: a practical application of *reduced scope* at the orchestration level itself, not just at the task level.

### Dynamic Persona Creation
Three of the eleven personas did not exist at project inception — they were created in response to gaps the process itself surfaced:

| Trigger | New Persona | Outcome |
|---|---|---|
| Integration review found visual anchors missing | Illustrator (Visual Strategist) | Fleet of 2 (Block 1 + Block 2): 40 visual opportunities, 25 Mermaid diagrams embedded |
| Integration review found inconsistent statistics across chapters | Fact-Ref-Checker (Claims Auditor) | Fleet of 2: 75 flags found, 5 critical incl. a PR-394 statistic contradiction |
| "How to publish?" required publishing expertise | Publishing Advisor | 5 paths evaluated → recommended Open Core + Premium model |

This validates the handbook's claim that **primitives should be created and iterated throughout a project, not frozen at inception**. Anti-pattern #10 (*Not Fixing the Primitives*) warns against correcting symptoms manually instead of updating the instruction set — each dynamic persona was the fix: a new primitive addressing a structural gap, not a one-off patch. The book's worked example for the pattern is the Fact-Ref-Checker, defined with an explicit scope and output format:

```markdown
# .github/agents/fact-ref-checker.agent.md
---
name: Fact-Ref-Checker
role: Claims Auditor
---
You audit manuscripts for factual accuracy. For every claim:
1. Is it sourced? Flag unsourced statistics.
2. Is it consistent? Cross-reference numbers across chapters.
3. Is it falsifiable? Flag unfalsifiable superlatives.

Output: severity-ranked findings (CRITICAL / HIGH / MEDIUM).
```
The key, per the book: define the persona's scope and output format explicitly — generic "review this" dispatches produce generic output.

### Escalation Events and Anti-Pattern Mapping
Four escalation events interrupted normal flow, each mapped to a specific anti-pattern from Chapter 14:

1. **Architecture Agent Timeout.** A single agent was dispatched to produce the full 15-chapter architecture; it failed after 26 minutes — a connection error caused by context window exhaustion. **Anti-pattern #11 (Context Window Exhaustion) compounded by #6 (The Solo Hero)** — the orchestrator assigned monolithic scope to one agent, violating the one-file/one-agent-per-wave isolation principle. **Resolution:** split into two parallel agents (Part 1: Chapters 1–8; Part 2: Chapters 9–15 plus cross-cutting concerns). Both completed successfully, producing 1,020 lines of chapter specifications. **What held true:** context will remain finite and fragile — regardless of model capability, there is always a limit to how much an agent can consider effectively.
2. **Chief Editor Synthesis Scope.** After four corpus-audit specialists completed parallel scans, their outputs needed cross-cutting synthesis no individual specialist could produce. **Anti-pattern #2 (Context Dumping)** was the temptation — dumping all four audit outputs into one agent's context window. Instead, the orchestrator escalated to the Chief Editor persona, whose explicit role was cross-chapter coherence. **Resolution:** Chief Editor ran as synthesizer, producing 7 consensus themes, 6 resolved tensions, 10 cuts, and 9 identified gaps. **What held true:** composition will remain necessary — complex analysis required coordination across specialists and structured integration of results.
3. **User Vetoed `apm compile`.** During the APM strategic-insertion phase, the orchestrator prepared six surgical insertions. The human author intervened: "Do not mention `apm compile` — niche feature." All six insertions were adapted to respect the constraint. This was not persona drift or agent failure — every agent was following its persona correctly; the user exercised editorial authority, overriding correct-but-misaligned agent output. This is the Architect role in practice: human judgment as final arbiter of what the book should and should not say. **What held true:** human judgment remains the bottleneck and the differentiator — the scarce resource was not token generation but the ability to decide what the book should and should not say.
4. **Panel Disagreement on APM Prominence.** After discovering "zero APM mentions in ~68,000 words," the Market Analyst wanted more visibility; the Thought Leadership Advisor wanted less — a genuine panel disagreement. The Chief Editor resolved it with a single governing principle: *"The book is 100% useful without APM. APM appears as proof, not prerequisite."* Once articulated, the principle made every subsequent decision mechanical. **Resolution:** 6 surgical insertions, 5 name-mentions, ~475 words across 5 chapters — the lightest possible touch. **What held true:** explicit knowledge is more valuable than implicit knowledge — the governing principle, once stated, made every subsequent decision mechanical.

A notable finding the book highlights on its own: a book about AI-native software development, written by APM's creator, using APM's orchestration infrastructure, initially contained **zero mentions of APM across ~68,000 words**. The book frames this as the methodology working correctly, not a bug — each agent was dispatched with chapter-specific scope, and no agent had "promote the author's project" in its persona. The absence proved the review process was honest, and it is precisely what prompted the four-wave strategic assessment that produced the surgical insertions above.

### What Held True
The five structural properties (introduced in full in the APM Overhaul case study, Ch. 15) held under editorial conditions too. The chapter's **novel finding**: **composition-level orchestration** — coordinating agents that write prose, not code — scaled without modification to the wave model.

| Metric | Value |
|---|---|
| Words | ~68,000 |
| Chapters | 15 |
| Personas | 11 (8 + 3 dynamic) |
| Dispatches | ~50+ |
| Waves | 4 writing waves + integration + polish |
| Mermaid diagrams | 25 |
| Fact-check flags | 75 (5 critical) |
| Visual opportunities found | 40 |
| APM mentions | 5 (~475 words) |
| Escalation events | 4 |
| Structural properties tested | 5/5 |

---

## Important

### The Authenticity Question
A separate three-expert panel (HN Skeptic, Thought Leadership Strategist, Meta-Authenticity Analyst) evaluated the risk of openly documenting the AI-assisted writing pipeline. The consensus: **net positive**. The panel's key test, quoted directly: *"Could this person have written a credible book without AI? If yes, AI reads as methodology demonstration."* The framing rule adopted: **"built using the same methodology it teaches"** — never "AI-written." Expected risk: some initial skepticism from readers who dismiss anything AI-assisted, but the panel's judgment was that engaged readers would find the transparency more impressive than the alternative. The README was rewritten to showcase the pipeline openly: an 11-agent team table, a 5-stage pipeline diagram, and links to review artifacts — explicit knowledge over implicit knowledge, a property that held true here as well as in the writing process itself. The book notes that hiding the process would have directly contradicted the book's own thesis.

### Key Takeaways (Chapter's Own TL;DR)
Stated up front in the chapter as the condensed summary:
- The same primitives that orchestrate code changes — personas, skills, file-scoped rules — orchestrate prose at book scale.
- Wave ordering, checkpoint discipline, and context budgeting matter more than model capability.
- Dynamic persona creation mid-project beats freezing the team at inception — three of eleven personas were created in response to gaps the process surfaced.
- When agents unanimously agree, a single human override can still be the correct call (the `apm compile` veto).
- Composition is load-bearing: 15 chapters required 50+ dispatches, 4 waves, and 2 integration passes. No single agent could hold the full manuscript.

---

## Key Terms

- **`.agent.md`** — the file format used to define a single persona as a composable primitive (name, role, scope, output format).
- **Pod** — a functional grouping of personas serving a shared purpose (Domain Expert, Editorial, Audit, Review).
- **Dynamic persona creation** — creating a new persona mid-project in response to a gap the process surfaces, rather than freezing the team roster at inception; contrasted with Anti-pattern #10 (Not Fixing the Primitives).
- **Composition-level orchestration** — this chapter's novel finding: the same wave/checkpoint orchestration model used for code generalizes to coordinating agents that produce prose.
- **The `apm compile` veto** — the escalation event where the human author overrode a unanimous agent plan to exclude mention of a niche feature, illustrating human judgment as final editorial authority.
