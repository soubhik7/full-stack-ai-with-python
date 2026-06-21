# Chapter 13 — The Execution Meta-Process

> Source: *The Agentic SDLC Handbook* (Daniel Meppiel, v0.9.2), pp. 173–182.
> The agent is the easy part of using AI on a real codebase — the hard part is knowing what to ask, in what order, and when to stop asking and start verifying; this chapter is the five-phase process that answers that.

---

## Very Important

### The Five Phases (Audit, Plan, Wave, Validate, Ship)
The methodology is a **five-phase meta-process** that works regardless of which AI coding tool is used, because it operates at a level above any specific tool's mechanics — the tool dispatches agents, runs tests, and manages files; the human manages the process.

| Phase | Purpose | Human decision | Key rule | Output |
|---|---|---|---|---|
| **Audit** | Build a multi-perspective understanding of the code about to change | Review audit reports, decide what matters — not every finding warrants action now | Audits are **read-only**; agents explore, never modify | Prioritized findings with citations |
| **Plan** | Transform audit findings into an executable specification: what changes, in what order, by which agents, with what constraints | Approve the plan — "the highest-leverage moment in the entire process" | **No implementation starts until the plan is approved** | An approved plan: scope, teams, waves, principles, constraints |
| **Wave** | Execute the plan in dependency-ordered batches, with validation between each | Approve each wave launch (or authorize the full sequence if trusting the plan); triage test failures; intervene on escalation | **Every wave ends with green tests and a commit. No exceptions.** | A commit per wave, all tests passing |
| **Validate** | Confirm the final state before shipping | Decide whether changes are ready to ship | Full test suite + spot-checks on complex/boundary files | A validated changeset |
| **Ship** | Commit, push, merge | — | Commit history is one commit per wave, each with passing tests — reviewable and bisectable | A merged PR |

The **ADAPT loop** connects wave execution back to planning: when a wave fails (a test breaks, an agent gets stuck, a dependency was missed), you diagnose, adjust the plan, and re-execute. **The loop is not a sign of failure — it is the mechanism that makes the process resilient.**

**Phase 1 (Audit) in detail.** Dispatch expert agents — typically 2 to 4, running in parallel — each with a distinct audit lens (an architecture expert examines patterns/coupling/separation of concerns; a domain expert examines the specific subsystem; a security expert checks vulnerabilities). Each produces ranked findings with severity levels, exact file-and-line citations, and remediation guidance. Audits are read-only, which means audit agents can be dispatched freely without worrying about partial changes or file conflicts. Enabling capability: background agents with parallel dispatch and session isolation, each in its own context window with read-only codebase access.

**Phase 2 (Plan) in detail.** Define scope (what's in, out, deferred), agent teams (which personas own which concerns — this decomposition pattern echoes Conway's Law: agent team boundaries are made to mirror module boundaries, by design, not accident), and wave structure (dependency-ordered batches). The plan includes **principles** — priority-ordered values that anchor every decision when trade-offs arise — and **constraints** — what must not change. A well-structured plan example includes a priority-ordered principles list: 1. SECURITY (no token leaks, no path traversal), 2. CORRECTNESS (tests pass, behavior preserved), 3. UX (world-class developer experience in every message), 4. KISS (simplest correct solution), 5. SHIP SPEED (favor shipping over perfection) — plus explicit constraints like "Do NOT modify test infrastructure," "Do NOT change CLI command signatures," "Do NOT alter public API return types." Footnote: Brooks makes the parallel argument in *The Mythical Man-Month* (1975) — "Plan to throw one away; you will, anyhow." The meta-process inverts this: invest disproportionately in the plan so you don't have to throw the execution away.

**Phase 3 (Wave Execution) in detail.** Each wave is a set of tasks with no unmet dependencies; the orchestrating tool dispatches parallel agents for each task, grouped so no two agents edit the same file simultaneously. Each agent receives precise instructions (which files to change, what patterns to follow, what constraints to respect, what verification to run before reporting completion). When all agents in a wave finish, the full test suite runs — pass and the wave is committed, fail and you triage. This produces a clean commit history (one commit per wave) that guarantees you can bisect regressions to a specific batch of changes.

**Phase 4 (Validate) in detail.** Full test suite (unit, acceptance, optionally integration/end-to-end) plus human spot-checks on the files with the most complex modifications, the boundary conditions specified in the plan, and the areas most likely to harbor subtle errors. Enabling capability is simply the project's existing CI pipeline or test runner — "the value comes from the process (test after every wave, spot-check critical files), not from additional tools."

**Phase 5 (Ship) in detail.** Update the changelog if not already done, push the branch, merge if CI passes.

### Wave Decomposition
The wave structure is where planning becomes engineering. A poorly decomposed set of waves produces merge conflicts, stale context, and cascading failures; a well-decomposed set produces clean parallel execution with natural validation boundaries.

- **The Dependency Graph.** Waves are ordered by dependency: Wave 0 contains tasks with no dependencies (foundational changes other waves build on); Wave 1 depends on Wave 0's outputs; Wave 2 depends on Wave 1; and so on. The dependency is directional and strict — no task in wave N may depend on a task in wave N+1. The most common pattern is **foundation-before-migration**: type definitions, protocol changes, and method signatures go in Wave 0; code that uses those new interfaces goes in Wave 1+. Put both in the same wave and agents will try to both define and consume new APIs simultaneously — consumer agents work against a file state that doesn't yet include the definitions.
- **The One-File-One-Agent Rule** (from Chapter 12) shapes wave design more than any other constraint: within a wave, no two agents may edit the same file. Logically independent changes that both touch the same file go in separate waves, or are assigned to a single agent that handles both changes in sequence.
- **Sizing Waves.** Smaller waves (2–4 agents) complete faster and are easier to debug; larger waves (6–10 agents) have higher throughput but are dominated by the slowest agent, and a single failure means triaging more changes.

| Factor | Smaller waves (2–4 agents) | Larger waves (6–10 agents) |
|---|---|---|
| Execution time | 3–5 minutes | 8–12 minutes (slowest agent dominates) |
| Debug difficulty | Low — few changes to inspect | High — more changes interacting |
| Commit granularity | Fine — easy to bisect | Coarse — harder to isolate regressions |
| Overhead | Higher — more validation cycles | Lower — fewer cycles |

  Decision heuristic: **start with smaller waves**; combine tasks into larger waves only when they are genuinely independent (different files, different concerns, no shared state) and when the validation overhead of extra cycles outweighs the debugging advantage.
- **The Self-Sufficiency Test.** Before finalizing a wave, apply this test to each task: *can an agent complete this task without asking me a question?* If no — because the task depends on an ambiguous design decision, unclear scope, or undocumented conventions in the target file — the task isn't ready: refine the instructions, split the task, or move it to a later wave where its dependencies are resolved. Tasks that fail this test are **the primary source of mid-wave escalations**; catching them during planning eliminates interruptions during execution.

### PR #394: The Worked Example
The meta-process is abstract until executed. PR #394 — an auth and logging architecture overhaul on **APM** (the author's open-source agent package manager, the implementation of the distribution layer from Chapters 9–10) — is the book's real worked example. Scope: 5 cross-cutting concerns (auth resolver deduplication, verbose logging coverage gaps, CommandLogger migration, unicode symbol cleanup, test coverage) touching 75 files across the entire source tree. (The canonical metrics — files, lines, test counts, agent dispatches, escalations — live in the full APM Overhaul case study; this chapter focuses on *how the phases played out*.)

**Timeline:**

| Phase | Duration | Agents | Outcome |
|---|---|---|---|
| Audit | 3 min | 2 parallel (architecture + logging/UX) | Severity-ranked findings with file-line citations |
| Planning | 5 min | — | 8 iterations; all findings put in scope; 2 teams defined |
| Wave 0 — Foundation | 5 min | 2 parallel | Resolver dedup + symbol definitions. Tests green |
| Waves 1–2 — Core | 8 min | 5 parallel | Verbose logging + CommandLogger migration. Tests green |
| Wave 2b — Recovery | 7 min | 2 replacement | install.py agent stalled (context exhaustion). Split + re-dispatch |
| Wave 3 — Polish | 4 min | 1 | Unicode symbol cleanup. Tests green |
| Ship | 2 min | — | Spot-check, full suite green, CI passed, merged |

**A note on timing:** the ~90-minute total wave execution time reflects an *experienced* practitioner working with mature instrumentation — battle-tested instruction files, established personas, and conventions already externalized into codebase instrumentation (Chapter 9). A first time through will take roughly **3×** as long — the first run is an investment in infrastructure that makes every subsequent run faster.

**The Three Practitioner Roles in Action.** Across wave execution the human intervened exactly **three times**, each mapping to one of the three practitioner roles from Chapter 8: (1) **Architect** (during planning) — decided to include all severity levels in scope rather than deferring moderate findings, weighing priorities, release timeline, and the cost of context-switching back later; (2) **Escalation Handler** (during Wave 2) — diagnosed an agent stall on `install.py` (58 call sites exhausted the context window) and decided to split remaining work across two replacement agents rather than retrying; (3) **Reviewer** (during Wave 2b) — triaged a test failure caused by an ordering issue in the migration (a function call was migrated but its setup code wasn't) and directed a targeted fix. Three interventions, three roles, no overlap — "these are the categories of human judgment that the meta-process surfaces, not eliminates." The full case study documents two additional escalation events (a token type correction and a silent `NameError`) caught during checkpoint verification and mapped to anti-patterns in Chapter 14.

### Checkpoint Discipline
A checkpoint is the pause between waves — **the mechanism that makes the meta-process safe.**

**Why test after every wave.** The alternative (executing all waves and testing only at the end) is faster in the best case and catastrophic in the worst: if wave 3 introduces a regression and you haven't tested since wave 0, you don't know whether it was introduced in wave 1, 2, or 3 — you can't bisect, you can't revert a single wave, you're debugging a composite changeset spanning the entire execution. Testing after every wave makes each wave an independently verifiable unit: if wave 2 breaks a test, you know the regression is in wave 2, inspect that diff, fix it, continue — without touching waves 0 or 1. The cost is real but small: in PR #394, ~2,850 tests at approximately 2 minutes per run, across 5 checkpoints (4 waves + final validation) — roughly 10 minutes of testing total, versus the alternative of debugging a 75-file composite changeset without bisection points, which "would have cost hours."

**Each checkpoint has four components:**
1. **Test gate** — the full suite runs; any failure means the wave is not committed. The failure is triaged: fixed immediately if the cause is obvious, or escalated to the human.
2. **Spot-check** — the human reviews a sample of changes, focused on boundary conditions, pattern compliance, and scope discipline (did the agent handle the edge case? follow existing patterns or invent new ones? change only what was specified?).
3. **Commit** — every wave gets its own commit with a descriptive message, creating a clean, bisectable history.
4. **Plan review** (optional) — the human reviews the remaining plan and adjusts if the current wave revealed something unexpected (a missed dependency, a task that should be split, a wave that should be reordered).

**The ADAPT loop** fires when a checkpoint fails (tests red, an agent stuck, a dependency missed): DETECT (failure or stall) → DIAGNOSE (root cause) → ADJUST (modify plan) → EXECUTE (re-run wave), looping back to DETECT if a new issue surfaces. It is not a fallback — it is a designed part of the process, "the mechanism that handles the irreducible uncertainty of working with non-deterministic systems on complex codebases." In PR #394, the ADAPT loop fired once, during Wave 2, when an agent stalled on `install.py` (58 call sites) — diagnosis: the file was too large for a single agent session; adjustment: split the remaining work across two agents; total cost: 7 minutes. **The key discipline: adaptation is conservative** — you add tasks, split tasks, reorder waves; you do **not** skip validation, you do **not** merge unvalidated work. Checkpoint discipline holds even — especially — when things go wrong.

---

## Important

### Session Management in the Meta-Process Context
Session discipline follows the principles established in Chapter 12, with two additional considerations specific to the meta-process: (1) **Phase transitions are natural reset boundaries** — each wave is a logically complete unit of work, so starting a fresh session per wave means every agent works with a clean context window loaded only with instructions and code relevant to its specific task. The orchestrating session is the exception — it runs long intentionally, accumulating task status, wave history, and decision rationale. (2) **Parallel agents produce cleaner context, not just faster execution** — the meta-process uses parallel, isolated agents rather than a single agent executing tasks sequentially; this is a context-quality decision as much as a throughput decision.

### Adapting the Meta-Process to Scale
The PR #394 case involved multiple waves including one recovery wave — not every change is that large. The meta-process scales in both directions:

**Small changes (fewer than 10 files)** — for focused changes within a single concern (fixing a bug, adding a feature to one module, refactoring a small subsystem), the full wave structure is overhead, so the process compresses: Audit becomes a single expert agent reviewing the relevant files; Plan becomes a mental model (you know the scope, there's one wave); Execution is a single wave with 1–2 agents; Validate and Ship are unchanged. The checkpoint discipline still applies (test before committing) and the planning discipline still applies (know what you're changing before you change it) — **what changes is the formality, not the structure.**

**Large changes (more than 100 files)** — for changes spanning a significant portion of the codebase (a framework migration, cross-cutting security hardening, a major API version bump), the process extends: Audit uses more expert agents (4–6), each covering a different subsystem or concern; Plan requires more waves (6–10) with careful dependency mapping and explicit scope boundaries for each; Execution may use a two-team structure with distinct agent personas (an architecture team for cross-cutting changes, a domain team for concern-specific changes); the ADAPT loop is more likely to fire, and the plan should anticipate it by leaving slack in the wave structure for recovery waves.

The scaling property to preserve: **each wave remains independently verifiable.** If a 200-file change decomposes into 8 waves of 25 files each, each wave is still a self-contained, testable unit — total complexity grows, but the complexity of any single checkpoint does not.

### What the Meta-Process Produces
Followed correctly, the meta-process produces four things manual development typically does not:
- **Bisectable history** — every wave is a separate commit with passing tests; if a bug surfaces after merge, you can bisect to the exact wave that introduced it. Not possible with typical "one giant commit per feature" or "squash everything" approaches.
- **Auditable decisions** — the plan documents what was decided and why; checkpoint records document what happened at each validation point; escalation records document what required human judgment and what the judgment was. A reviewer reading the PR has a complete record of the decision chain, not just the final code.
- **Reproducible process** — the meta-process is the same regardless of who executes it or which tool orchestrates it; a different developer, with the same codebase, context files, and plan, would produce substantially similar output. The book is explicit that **this is a hypothesis, not a tested claim** — controlled reproducibility experiments would strengthen it. "The non-determinism of AI agents is bounded by the determinism of the process around them."
- **Proportional cost** — time spent scales with change scope, not full codebase size, though this has only been verified at the scale documented in the reference case study. The agent works on the files in the plan; the test suite validates the behavior; nothing else matters.

---

## Key Terms

- **Meta-process** — the five-phase execution methodology (Audit, Plan, Wave, Validate, Ship) that sits above any specific AI tool's mechanics and turns PROSE/context-engineering building blocks into shipped code.
- **ADAPT loop** — the recovery mechanism (Detect → Diagnose → Adjust → Execute) that fires when a wave checkpoint fails; a designed part of the process, not a fallback.
- **Checkpoint** — the pause between waves where the test gate, spot-check, commit, and optional plan review happen; the mechanism that makes the meta-process safe.
- **Foundation-before-migration** — the dominant wave-ordering pattern: type/protocol/signature changes go in Wave 0; code consuming those new interfaces goes in later waves.
- **Self-Sufficiency Test** — the planning-time check ("can an agent complete this task without asking me a question?") used to catch under-specified tasks before they become mid-wave escalations.
- **Test gate** — the full test-suite run at each checkpoint; a wave is not committed unless it passes.
- **Bisectable history** — a commit history (one commit per wave, each with passing tests) that allows regressions to be traced to the exact wave that introduced them.
- **PR #394** — the book's worked example: a 75-file, 5-concern auth/logging overhaul on APM, used throughout Chapters 12–14 to ground abstract claims in measured (or author-estimated) numbers.
