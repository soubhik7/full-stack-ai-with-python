# Chapter 13 — The Agentic SDLC Handbook (Study Notes)

> **Study notes for "The Agentic SDLC Handbook" by Daniel Meppiel (v0.9.2, March 2026)** —
> a methodology book for AI-native software development, built around the **PROSE**
> framework: five architectural constraints for reliable human-AI collaboration.

The source PDF lives at [`The-Agentic-SDLC-Handbook.pdf`](The-Agentic-SDLC-Handbook.pdf)
in this folder. Everything else here is extracted notes, organized to mirror the
book's own five-part structure, with every concept tagged **Very Important** or
**Important** inside its chapter file.

---

## Why This Chapter Exists

Chapters 10–12 of this curriculum teach you to *use* AI tooling (RAG, agents, MCP,
Claude Code's extensibility model). This book is the missing layer above that:
a methodology for doing it **reliably on a real, messy, pre-existing codebase** —
not a greenfield demo. Its central diagnosis, the "Vibe Coding Cliff," is the
formal name for exactly the failure mode anyone who's pointed an agent at a large
repo has already felt.

---

## Learning Path

```
00_foundation/          ← Part I (Ch 1): the thesis — why tools alone don't fix it, and the PROSE framework
01_for_leaders/          ← Part II (Ch 2-7): market landscape, business case, architecture, governance, teams, transition
02_for_practitioners/    ← Part III (Ch 8-14): mindset, instrumented codebase, PROSE spec, context engineering,
                            multi-agent orchestration, execution meta-process, anti-patterns
03_case_studies/         ← Part IV (Ch 15-18): four real, worked executions of the methodology
04_closing/              ← Part V (Ch 19): what's durable vs. what will age out within 18 months
```

---

## Chapter Index

### `00_foundation/` — Part I: The Foundation
| File | Chapter |
|------|---------|
| `01_agentic_sdlc_thesis.md` | Ch 1 — The Vibe Coding Cliff, why better models don't fix it, the five PROSE constraints |

### `01_for_leaders/` — Part II: For Leaders
| File | Chapter |
|------|---------|
| `02_ai_native_landscape.md` | Ch 2 — Market velocity, coding tools vs. delivery platforms, the 8-phase evaluation framework |
| `03_business_case.md` | Ch 3 — The productivity paradox, true cost, building an honest ROI case |
| `04_reference_architecture.md` | Ch 4 — The three-layer reference architecture, the Context Moat, build/buy/compose |
| `05_governance.md` | Ch 5 — Governance readiness checklist, risk taxonomy, board reporting |
| `06_team_structures.md` | Ch 6 — How roles evolve, new roles, team topologies, the junior pipeline |
| `07_planning_the_transition.md` | Ch 7 — Readiness assessment, phased adoption, transition pitfalls |

### `02_for_practitioners/` — Part III: For Practitioners
| File | Chapter |
|------|---------|
| `08_practitioners_mindset.md` | Ch 8 — The Autocomplete Trap, your three roles, when to use agents vs. code manually |
| `09_instrumented_codebase.md` | Ch 9 — The seven primitive types, directory structure, cross-tool portability |
| `10_prose_specification.md` | Ch 10 — The full PROSE spec: P/R/O/S/E in detail, compliance checklist |
| `11_context_engineering.md` | Ch 11 — Context budgets, instruction hierarchy, agent/skill design, the context audit |
| `12_multi_agent_orchestration.md` | Ch 12 — Specialization patterns, parallelization, conflict resolution, the coordination tax |
| `13_execution_meta_process.md` | Ch 13 — The five-phase meta-process, wave decomposition, checkpoint discipline |
| `14_anti_patterns_failure_modes.md` | Ch 14 — The full anti-pattern taxonomy, recovery playbook, security practices |

### `03_case_studies/` — Part IV: Case Studies
| File | Chapter |
|------|---------|
| `15_apm_auth_logging_overhaul.md` | Ch 15 — A 75-file architecture change across 25 agents (the book's primary evidence, "PR #394") |
| `16_writing_a_book_with_agent_fleets.md` | Ch 16 — This book itself, written by an 11-persona, four-pod agent fleet |
| `17_publishing_pipeline.md` | Ch 17 — Shipping the book: conversion, CI/CD, licensing |
| `18_growth_engine.md` | Ch 18 — Building a growth engine; where automation hit a wall and had to escalate to a human |

### `04_closing/` — Part V: Closing
| File | Chapter |
|------|---------|
| `19_what_comes_next.md` | Ch 19 — Three-horizon predictions, what won't change, when *not* to use agentic workflows, what the author probably got wrong |

---

## The Core Framework at a Glance

**PROSE** — five architectural constraints, positioned the way Roy Fielding's REST
positioned constraints for distributed systems: not a tool, a style.

| Constraint | Principle |
|---|---|
| **P**rogressive Disclosure | Context arrives just-in-time, not just-in-case |
| **R**educed Scope | Match task size to context capacity |
| **O**rchestrated Composition | Simple things compose; complex things collapse |
| **S**afety Boundaries | Autonomy within guardrails |
| **E**xplicit Hierarchy | Specificity increases as scope narrows |

Full detail: [`02_for_practitioners/10_prose_specification.md`](02_for_practitioners/10_prose_specification.md).
First introduced: [`00_foundation/01_agentic_sdlc_thesis.md`](00_foundation/01_agentic_sdlc_thesis.md).

---

## How to Read This

Same dual-path the book itself recommends:
- **If you lead a team:** start with `01_for_leaders/`, then skim `00_foundation/` for the thesis and `04_closing/` for what to do Monday.
- **If you write the code:** start with `00_foundation/`, then go straight to `02_for_practitioners/` — especially `09_instrumented_codebase.md` and `10_prose_specification.md`.
- **If you want proof it's not just theory:** read `03_case_studies/` — every claim in the practitioner chapters traces back to one of these four executions.

---

## Next Step

```
... → Azure AI Foundry (11) → Claude Code (12) → Agentic SDLC (13)  ← You are here
```

This closes the loop on the curriculum: chapters 10–12 taught you the *mechanics*
of agentic tooling (MCP, Claude Code extensibility); this chapter teaches the
*methodology* for applying that tooling reliably at production scale.
