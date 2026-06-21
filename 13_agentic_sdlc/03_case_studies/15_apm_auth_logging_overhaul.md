# Chapter 15 — The APM Auth + Logging Overhaul

> Source: *The Agentic SDLC Handbook* (Daniel Meppiel, v0.9.2), pp. 199–205.
> The book's primary evidence chapter: one real, public, inspectable pull request (PR #394 against `microsoft/apm`) used to show what multi-agent orchestration looks like — failures included — under actual production conditions.

---

## Very Important

### The Case Study's Scope and Stakes
**Scope:** PR #394 against `microsoft/apm` — auth consolidation, logging abstraction, diagnostic collection.
**Duration:** ~16 hours wall-clock across 2 sessions.
**Theme:** Multi-agent orchestration against a production codebase — the methodology's home turf.

The session operated at three scales simultaneously: a **6-expert audit panel** for diagnosis, a **fleet of ~25 agents** for implementation, and **individual agents** for targeted fixes. Every metric in the chapter comes from session checkpoint logs, not from author recollection.

### The Problem That Forced a 75-File Change
A user reported confusing UX when `apm install` failed for a GitHub Enterprise Managed Users (EMU) org package. Investigation traced this to a single root cause branching into three systemic failures:

1. **Auth bypass** — `_validate_package_exists()` ran bare `git ls-remote` with no credentials; `GITHUB_APM_PAT` was ignored for github.com hosts.
2. **Auth fragmentation** — four inconsistent auth implementations scattered across the install, download, copilot, and operations modules.
3. **Observability gap** — 766 ad-hoc `_rich_*` logging calls across 27 files with no shared abstraction. When verbose logging silently failed (a `NameError` caught by an outer `try/except`), there was no way to know.

A point fix would have patched `_validate_package_exists()` alone. The structural fix instead required centralized auth resolution, a command-logger abstraction, and diagnostic collection — touching 75 files in total.

### Eight Plan Iterations — The Plan Gate Working as Designed
The plan went through eight versions before approval. The book frames this explicitly as **the meta-process working correctly, not a failure**:

| Version | Scope | Trigger |
|---|---|---|
| v1 | 10 UX message fixes | Initial triage |
| v2 | Removed v0.8.2-only bug | User correction |
| v3 | Added auth gap as root cause | Expert panel findings |
| v4 | Unauth-first → auth-fallback | Architecture expert |
| v5 | Architecture-first, 4 phases | Orchestrator restructure |
| v6 | Added agent/skill primitives | Instrumented codebase needs |
| v7 | ALL commands covered | **User escalation (L4)** |
| v8 | 25 todos, 47 files, 5 phases | Approved |

**Version 7 was the critical turn.** The user rejected a plan scoped to `install` only and demanded every command route through `AuthResolver` and `CommandLogger`. The orchestrator dispatched three more explore agents to audit all logging calls (766+), all auth touchpoints (95+), and per-dependency auth paths. This is **Scope Creep (Anti-pattern #5) handled correctly** — the scope expanded *before* execution began, through the plan gate, not mid-wave. The plan targeted 47 primary source files; the final PR touched 75 — the additional 28 were test files, configuration updates, and documentation changes discovered during execution, which the book calls typical of dependency-following refactors.

### Wave Execution and Checkpoint Discipline
The approved plan decomposed into five phases. Each wave followed **checkpoint discipline**: full test suite after every wave, commit before the next. Tests climbed from 2,829 to 2,897 across five waves.

| Wave | Agents | Scope | Checkpoint |
|---|---|---|---|
| Foundation | 3 parallel | AuthResolver, CommandLogger, DiagnosticCollector | 2,839 tests |
| Auth wiring | 8 parallel | One file per agent (downloader, install, copilot, operations, errors) | 2,846 tests |
| Logger wiring | 7 parallel | All commands through CommandLogger. `install.py` (58 calls) **got stuck → escalated** | 2,874 tests |
| Tests | parallel | 78 unit + 26 integration + 11 diagnostics | 2,897 tests |
| Ship | sequential | Docs, skills, CHANGELOG, PR review fixes | Released as v0.8.4 |

115 new tests were written (78 unit, 26 integration, 11 diagnostics); 47 legacy tests were consolidated or replaced during the refactor, for a net gain of 68. The **one-file-one-agent-per-wave rule** (Ch. 12) prevented merge conflicts across all parallel dispatches — the one exception, `install.py` in the logger wave, is Escalation #1 below.

### Five Escalation Events Mapped to Anti-Patterns
Escalation severity follows a three-tier model:

| Level | Meaning | Action |
|---|---|---|
| **L2** | Agent needs guidance | Orchestrator adjusts prompt and re-dispatches |
| **L3** | Agent cannot complete | Orchestrator takes over the task manually |
| **L4** | Plan scope changes | New todos added, potentially new wave |

Five escalations occurred, each mapping to a specific anti-pattern and PROSE constraint:

| Escalation | Anti-pattern | PROSE Constraint | Resolution |
|---|---|---|---|
| `install.py` stuck | #11 Context Window Exhaustion | Progressive Disclosure | Split file across waves; manual completion |
| Unicode persistence | #12 Hallucinated Edits | Safety Boundaries | File-state verification after every dispatch |
| Unicode persistence | #7 The Trust Fall | Safety Boundaries | Never accept self-report without diff check |
| Token type error | #13 Stale Context Between Waves | Progressive Disclosure | Re-validate expert findings before wiring |
| PAT 403 | #5 Scope Creep | Reduced Scope | L4 escalation through plan gate |
| Silent NameError | #9 Skipping Checkpoints | Safety Boundaries | Assert on observable behaviour, not just pass/fail |

The five escalations in detail:

1. **The `install.py` agent (Context Window Exhaustion).** The agent migrating `install.py` (58 `_rich_*` calls, the largest single file) ran for 45+ minutes and stopped producing coherent edits — its context window filled with its own prior output. Recovery: the orchestrator escalated to L3, wrote a Python script to strip 30 dead `else`-branch fallbacks, manually fixed 3 duplicate calls, and committed. Lesson: 58 call sites in one dispatch is too many; the file should have been split across two waves (structural calls in Wave 3a, verbose calls in Wave 3b).
2. **Unicode agent persistence (Hallucinated Edits).** The Wave 3 unicode cleanup agent reported all replacements complete; file inspection showed zero changes — the agent had written to a temporary copy. Recovery: the orchestrator performed all replacements manually across 4 files. The Trust Fall (Anti-pattern #7) was also in play — the orchestrator initially accepted the agent's success report without verification.
3. **Token type correction (Stale Context Between Waves).** The expert panel classified `ghu_` tokens as EMU-specific. The user corrected this: `ghu_` is OAuth; EMU users receive standard `ghp_`/`github_pat_` tokens. The security constraint "host-gating global env vars" had been built on stale expert output. Recovery: orchestrator updated `AuthResolver` to remove the incorrect host-gating logic and re-ran auth tests.
4. **Fine-grained PAT 403 failure (L4 — plan scope change).** Auth still failed with a valid fine-grained PAT. Root cause: the `x-access-token:{token}@host` URL format sends Basic auth, which GitHub rejects for fine-grained PATs. Recovery: pivoted validation entirely from `git ls-remote` to the GitHub REST API — one code path for all token types. This was an L4 escalation; the orchestrator chose one validation strategy over a branching tree of token-type logic, avoiding architecture-level complexity.
5. **Verbose logging silent `NameError`.** `_rich_echo` was never imported in `install.py`. The `verbose_log` lambda triggered a `NameError` caught by an outer `try/except`; verbose mode silently did nothing. No test caught it because the test suite never asserted on verbose output — 2,829 passing tests did not catch a silent `NameError`. This is why checkpoint discipline (Ch. 12) must include *behavioural* assertions, not just "tests pass."

### What Held True Regardless of the Model
Chapter 15 tests three of the book's five cross-chapter structural properties under production conditions:

- **"Context will remain finite and fragile."** The `install.py` agent proved it: 58 call sites exhausted the context window. No amount of model improvement eliminates the need for **context budgeting** — partitioning work to fit the window with room for reasoning.
- **"Output will remain probabilistic."** The unicode agent reported success on changes it never persisted. The same prompt, re-dispatched, might have worked. Reliability was architected through **checkpoint discipline** and file-state verification — not by trusting any single execution.
- **"Human judgment will remain the bottleneck and the differentiator."** The user's v7 escalation — demanding all commands be covered, not just `install` — was the highest-leverage decision in the session. No agent suggested it. The 8 plan iterations were not wasted work; they were the mechanism through which human judgment shaped the architecture.

The evidence chain is inspectable at PR #394. The five escalations above are the full log — nothing is summarized away.

---

## Important

### Expert Panel Composition and Findings
Six experts ran in parallel during the audit phase, each covering a distinct domain: GitHub auth patterns, EMU constraints, Azure DevOps auth, architecture design, CLI UX, and documentation. Each produced severity-ranked findings, synthesized into a single source-of-truth document. Two notable findings: the EMU Expert identified that host-gating `ghu_` tokens was incorrect (later independently validated as Escalation #3); the Architecture Expert proposed **Strategy + Chain of Responsibility** as the auth consolidation pattern; the CLI UX Expert found the 766 ad-hoc logging calls with no shared abstraction. The book's framing: **expert panels are audits, not oracles** — panel findings must be validated before they become wiring instructions (this is exactly what Escalation #3 demonstrates when a panel finding turned out to be stale).

### Filesystem Verification After Agent Dispatch
The chapter's concrete checklist for trusting agent output:

```bash
# After agent claims "all unicode replacements complete":
git diff --stat            # Did any files actually change?
grep -rn "\|\\|\|" src/    # Are the old characters still there?
```
Agent success messages are probabilistic output; `diff` is deterministic. The book's instruction: build this check into every checkpoint, never trust self-reports alone.

### Canonical Metrics — PR #394
This table is referenced by other chapters throughout the book as the canonical numeric baseline:

| Metric | Value | Notes |
|---|---|---|
| Files changed | 75 | 47 planned + 28 dependency/test/config |
| Lines changed | +7,832 / −1,074 | |
| Tests before | 2,829 | Baseline at start |
| Tests after | 2,897 | +115 written, −47 consolidated = +68 net |
| Agent dispatches | ~25 | Across 5 waves in 5 phases |
| Audit agents | 6 | Expert panel phase |
| Plan iterations | 8 | v1–v8, approved at v8 |
| Escalations | 5 | 1×L2, 2×L3, 2×L4 |
| Execution interventions | 3 | During wave execution (L3/L4 escalations) |
| Verification interventions | 2 | During review/validation |
| Wave execution time | ~90 minutes | Active agent + human review time |
| Total wall-clock time | ~16 hours | 2 sessions including planning, monitoring, breaks |
| Human time breakdown | ~30% planning, ~20% monitoring, ~25% interventions, ~25% review | Approximate, single execution |

The book is explicit that other chapters reference these same numbers — "~90 minutes" means wave execution time, "~16 hours" means total elapsed time including planning and breaks. It also offers an honest comparison point: in the author's experience, similar-scope *manual* refactors (consolidating auth patterns across 40+ files) have taken 3–5 days of focused engineering time — but this estimate is explicitly flagged as not formally benchmarked.

### The Five Lessons (Chapter's Own TL;DR)
Stated up front in the chapter as the condensed takeaway:

1. **Budget context before you dispatch.** Count call sites; split at ~25 per agent. 58 was too many.
2. **Verify filesystem, not self-reports.** `diff` target files after every dispatch. Agent success messages are probabilistic output.
3. **Expert panels are audits, not oracles.** Panel findings must be validated before they become wiring instructions.
4. **Expand scope through the plan gate.** Mid-planning expansion is healthy. Mid-wave expansion is dangerous.
5. **Checkpoints must assert behaviour.** 2,829 passing tests did not catch a silent `NameError`. Assert on observable output.

---

## Key Terms

- **PR #394** — the real, public pull request against `microsoft/apm` that is this chapter's (and much of the book's) primary evidence source.
- **EMU (Enterprise Managed Users)** — a GitHub Enterprise org-package configuration whose auth requirements exposed the original bug.
- **AuthResolver / CommandLogger / DiagnosticCollector** — the three foundational abstractions built in Wave 1 (Foundation) to centralize auth, logging, and diagnostics respectively.
- **Escalation levels (L2/L3/L4)** — L2: agent needs guidance and is re-dispatched; L3: agent cannot complete and the orchestrator takes over; L4: the plan's scope itself changes.
- **Checkpoint discipline** — running the full test suite and committing after every wave; the chapter argues this must include behavioural assertions, not just pass/fail counts.
- **The Trust Fall (Anti-pattern #7)** — accepting an agent's self-reported success without independent verification against filesystem state.
