# Chapter 12 — Multi-Agent Orchestration

> Source: *The Agentic SDLC Handbook* (Daniel Meppiel, v0.9.2), pp. 155–172.
> A single agent becomes the bottleneck once a change spans too many files and concerns — not from lack of intelligence but lack of bandwidth — and multi-agent orchestration is the discipline of getting the benefits of parallelism without paying the costs of chaos.

---

## Very Important

### When One Agent Is Enough — The Decision Matrix
The overhead of orchestration (partitioning work, managing sessions, resolving conflicts, validating independently) is real. If the task fits comfortably in a single agent's context, the single agent is the better choice. A single agent is sufficient when scope is narrow (< 10 files in a single module), concern is singular (one type of change, not three interleaved), dependencies are linear (no need for parallel work), and the context budget is adequate (all relevant source, instructions, and history fit without exceeding ~60% of window capacity, leaving room for reasoning). A single agent breaks down when multiple concerns intersect, file count exceeds context capacity (more than 15–20 files), or parallelism would reduce wall-clock time significantly.

| Dimension | Single agent | Multiple agents |
|---|---|---|
| Files changed | < 10 | > 15 |
| Concerns | 1 | 2+ |
| File dependencies | Linear | Graph (can parallelize) |
| Required expertise | One domain | Multiple domains |
| Time pressure | Low | Moderate to high |
| Risk of context overload | Low | High |

The boundary at 10–15 files is approximate and experience-derived, not precisely measured — it reflects the practical limit where a single agent's accumulated tool calls, file reads, edit confirmations, and test output begin crowding out the instructions and source code it needs. **When the decision is marginal, err toward a single agent.** Coordination costs are real; multi-agent orchestration is a tool for tasks that exceed single-agent capacity, not a default mode of operation. (The footnote draws an explicit parallel to Brooks's observation in *The Mythical Man-Month*, 1975, that adding developers to a late project makes it later — with the difference that agent coordination overhead is predictable and reducible through better planning, not just better communication.)

### Agent Specialization Patterns
Specialization produces better results than generalization, for the same reason a team of specialists outperforms a team of generalists — it reduces the context each agent needs to carry, so the context it receives is concentrated, not diluted. Three patterns recur:

**Pattern 1 — Writer / Reviewer / Tester.** Separates code production from validation: one agent writes, a second reviews, a third writes/updates tests. Maps directly to the human author/reviewer/QA workflow. The reviewer receives the diff plus the original source — *not* the writer's full conversation history — deliberately, so it evaluates the output on its own merits rather than being anchored by the writer's reasoning. If the writer had a good reason for a decision but the code doesn't reflect it, that's a signal, not an excuse.

**Pattern 2 — Domain Teams.** For cross-cutting changes, organize agents by area of expertise rather than workflow stage; each team owns a concern and all files related to it.

| Aspect | Architecture Team | Domain Expert Team |
|---|---|---|
| Context loaded | Type definitions, module boundaries, pattern catalog, dependency graph | Output conventions, symbol dictionaries, UX guidelines, migration patterns |
| Owns | Type safety fixes, dead code removal, API consolidation | Verbose coverage, logger migration, formatting cleanup |

In the PR #394 auth-logging overhaul (Chapter 13 case study), this two-team structure handled a 75-file change across five concerns; neither team needed the other's context, and both produced output consistent with their specialization. The pattern scales by adding teams (a security concern adds a security team, a documentation concern adds a documentation team) — coordination cost is *between* teams, not within them.

**Pattern 3 — Audit / Execute / Validate.** For exploratory work where scope isn't fully known in advance: read-only audit agents discover what needs to change → a human makes planning decisions → read-write execution agents make the changes. The critical property is the separation between read-only and read-write operations — audit agents can run simultaneously with no risk of interference, examining the same files and overlapping concerns to produce independent assessments. **The human decision between audit and execution is the highest-leverage point in the entire process** — reviewing findings, deciding which to act on, defining scope, and only then allowing write operations. This separation is what makes multi-agent orchestration safe, not just fast.

### Parallelization Strategies
**The One-File-One-Agent Rule** is the most important parallelization rule: within a single execution batch, no two agents may modify the same file. Most agent tooling edits via string matching (find an exact block of text, replace it); if two agents target the same file, the second agent's expected text has already changed and the edit fails silently or produces corrupted output. This is not theoretical — it is **the most common failure mode in parallel agent execution**.

| Pattern | Agent | Files | Risk |
|---|---|---|---|
| Safe | Agent A | resolver.py, dependency_graph.py | None — distinct files |
| Safe | Agent B | install.py | None — distinct files |
| Safe | Agent C | cli.py, commands/init.py | None — distinct files |
| Unsafe | Agent A | install.py (lines 100–200) | Conflict — Agent B's line references invalidated |
| Unsafe | Agent B | install.py (lines 400–500) | Conflict — after Agent A's edits |

Enforcing the rule requires partitioning the file set *before* dispatch. If two concerns both need the same file, route both to a single agent that handles both, or sequence them across separate waves.

**Wave-Based Parallelism** structures execution as a sequence of batches: each batch (wave) runs in parallel, and the entire batch completes — and is tested and committed — before the next one starts. Dependencies between waves are explicit (Wave 1 agents rely on Wave 0's output being committed and tested); within a wave, agents are independent, sharing no files and making no assumptions about each other's progress. Wave sizing matters: a wave with 2–3 agents completes in the time the slowest agent takes (typically 3–5 minutes); a wave with 8 agents still takes 8–10 minutes (slowest agent dominates) but increases the risk of failures blocking the entire wave — **prefer more, smaller waves over fewer, larger ones.**

**Pipeline Parallelism** runs operations in parallel across workflow stages rather than across files — e.g. review agents start on Wave 0's output while execution agents work on Wave 1, or test agents run extended validation on a previous wave while human review happens on the current one. This works only when review/test operations are read-only and execution has no backward dependency on review findings; if a review agent finds a problem in Wave 0, the fix goes into a later wave rather than interrupting Wave 1 (which was planned against the committed Wave 0 output).

### Conflict Resolution — Three Categories
Despite careful partitioning, conflicts arise. They fall into three categories, each with a different resolution strategy:

| Conflict type | What happens | Resolution |
|---|---|---|
| **File conflicts** | Two agents need to modify the same file in the same wave — a planning error, not a runtime error | Merge both tasks into a single agent's scope, or move one task to a later wave. Files that attract changes from multiple concerns are a signal — assign that file to one agent per wave even if it handles multiple concerns there |
| **Semantic conflicts** | Two agents produce output that is independently correct but mutually inconsistent (e.g. Agent A introduces a new error-handling pattern; Agent B, on a different file, follows the old pattern because its instructions referenced the pre-change codebase) | Foundation-before-migration wave ordering: changes that establish new patterns go in early waves; changes that consume those patterns go in later waves, referencing the *committed* output of the earlier wave, not the original codebase. Requires testing and committing after each wave — skip the commit and Wave N+1 agents work against stale context |
| **Design conflicts** | Two agents, each following their specialization's best practices, produce output reflecting genuinely different design philosophies (e.g. architecture agent consolidates error handling centrally; domain agent keeps it local per UX convention) | Escalate to the human — these are trade-offs requiring judgment, not bugs. Priority-ordered principles in the plan resolve most mechanically; when principles don't resolve the conflict, the human decides and documents the rationale |

**Semantic Conflict Recovery — the PR #394 walkthrough.** Wave 0 introduced a new `OperationError` type replacing bare `ValueError`; Wave 1's domain agent, dispatched to migrate logging (not error handling), correctly migrated logging but also added new error paths using the old `ValueError` pattern — it had no visibility into the Wave 0 pattern change. Wave 1's unit tests passed (each file was correct in isolation), but the **integration test suite at the wave checkpoint** caught mixed error types — three integration tests failed with unhandled exception types. Diagnosis took the orchestrator under two minutes: the Wave 1 dispatch prompt loaded logging context but not error-handling context. **Recovery was not to revert Wave 1** (the logging migration was correct) — instead, a targeted Wave 2b dispatched two agents: Agent 2b-A replaced every `ValueError` raise in the migrated files with `OperationError` (six files in context, completed in 3 minutes); Agent 2b-B updated the corresponding test files. Wave 2b committed, all tests passed, execution continued to Wave 3. **The lesson:** wave ordering prevents most semantic conflicts; when one slips through, identify which context was missing from the dispatch, create a small targeted recovery wave carrying the missing context, and fix forward rather than reverting. **The mistake to avoid: redispatching the entire original wave** — the logging migration was 90% correct; a full redo wastes work and risks new issues. Surgical recovery waves are the right response to surgical failures.

The frequency of design conflicts is itself a metric: in PR #394, 3 human interventions occurred across ~25 agent dispatches (all judgment calls the plan couldn't automate, none design conflicts between agents) — an intervention rate of approximately 12%. **15–20% is used as a starting hypothesis for well-planned work** (not validated across multiple teams — a calibration point from the reference case study, not a benchmark). Rates significantly above 20% may indicate underspecified plans; rates below 5% warrant scrutiny — the work may be too simple for multi-agent orchestration, or review may be insufficient.

### The Human as Orchestrator — Decisions and Escalation Protocol
In a multi-agent workflow the human role shifts from producer to orchestrator: not writing the code or reviewing every line, but making the decisions agents cannot make for themselves — scope, priority, trade-offs, and when to stop. **Before execution:** define the plan (which concerns to address/defer, how to partition work across agents and waves, what principles govern trade-offs) — "the highest-leverage activity in the entire process. A well-structured plan with mediocre agents produces better results than a vague plan with excellent agents." **During execution:** monitor progress and handle escalations — diagnosing whether a stuck agent reflects a prompt problem (refine and retry), a scope problem (split the task), or a tooling problem (work around it). **After execution:** spot-check critical changes, verify test results, decide whether output meets acceptance criteria — proportional to risk, not volume (a 2,000-line diff where 1,800 lines are mechanical migration doesn't require reading all 2,000 lines; it requires verifying the migration pattern is correct, that the 200 non-mechanical lines are sound, and that the test suite covers the behavior).

**The four-level escalation protocol:**

| Level | Trigger | Response | Example |
|---|---|---|---|
| L1: Self-heal | Agent hits a test failure it can debug | Agent fixes and continues | Type error in generated code |
| L2: Retry | Agent produces incomplete output | Re-dispatch with refined prompt | Agent missed 3 of 12 files in scope |
| L3: Human decides | Trade-off between competing principles | Human makes design call | UX convention vs. architectural purity |
| L4: Scope change | Finding requires work outside the current plan | Human creates follow-up task | Discovery of a pre-existing bug unrelated to the change |

L1 and L2 are automated; L3 and L4 require human judgment. The goal is to minimize L3/L4 not by suppressing them but by making the plan specific enough that most decisions resolve at L1 or L2. (The footnote notes the taxonomy draws conceptually from the SAE levels of driving automation, SAE J3016, adapted for software engineering — higher autonomy levels require not less human involvement, but different kinds: supervision rather than operation.) In the PR #394 execution, the distribution across ~25 dispatches was roughly two-thirds L1 (autonomous), one-eighth L2 (automated retry), and one-fifth L3/L4 (human decision) — three interventions during wave execution. That ~20% rate is characteristic of a well-planned execution on a non-trivial change; **if your L3+ rate exceeds 25%, the plan needs more specific principles or better task scoping.**

### The Coordination Tax — Honest Numbers
Multi-agent orchestration saves time through parallelism and improves quality through focused context. It also costs time through planning, monitoring, and intervention. In the PR #394 execution, human time broke into four activities: planning and partitioning (~30%), monitoring execution (~20%), handling interventions (~25%), and post-execution review (~25%). Total human time was roughly **45 minutes** against **24 minutes** of agent computation time, with total wave execution time of roughly **90 minutes** (human work overlaps with agent execution). The same change executed sequentially by a single agent was estimated at 60–75 minutes of agent time, but with compounding context degradation after file 20 — expect 2–3 additional rework cycles adding 30–45 minutes, for a total single-agent elapsed time of roughly **90–120 minutes with lower output quality**. **The multi-agent approach did not save total elapsed time on this change — it traded human planning time for agent quality.** The 45 minutes the orchestrator spent coordinating replaced the 30–45 minutes they would have spent debugging context-degraded output, with better results.

**The sweet spot — multi-agent orchestration pays for itself when:**
- File count exceeds 20 across 2+ concerns (below this, planning overhead exceeds the parallelism benefit).
- Concerns partition cleanly (if most files need changes from multiple concerns, file-ownership conflicts cost more than parallelism saves).
- Context degradation is the real bottleneck (deep architectural understanding, not mechanical find-and-replace — the quality benefit of focused context outweighs coordination cost).
- You will do this more than once (the first orchestration on a codebase takes longest — building instruction files, learning partition boundaries, developing wave-sizing intuition; the second takes half the planning time; by the third, dispatch prompts are templates).

The overhead for a **well-planned** multi-agent execution is roughly **40–60%** of total human time spent on coordination rather than direct value work. For a **poorly planned** execution (vague dispatch prompts, unclear file ownership, missing instruction files) — based on contrast with less-structured attempts — this can reach **70–80% or higher**, at which point a single agent with good context would have been faster. **This is not a tool for every change. It is a tool for changes that exceed what a single agent can hold in focus.**

---

## Important

### Concrete Dispatch — What It Actually Looks Like
Before dispatching, the orchestrator prepares three things: the instruction files the agent will load, the exclusive file list it owns, and the task prompt. Example (terminal-based agent, but the pattern applies regardless of tool): two instruction files loaded — `.ai/instructions.md` (project-wide conventions, always loaded) and `.ai/integrations/logging.md` (logging-specific patterns) — paired with a dispatch prompt that names the established reference pattern (`LoggerFactory` from `src/core/logger.py`, "committed and tested"), lists files assigned with exclusive ownership for the wave, states explicit constraints ("Do NOT modify any file not in this list," "Do NOT change function signatures or public APIs," "Preserve all existing behavior"), and ends with a self-validation command (`pytest tests/commands/ -x`) to run before reporting completion. What makes this effective: exclusive file list (no ambiguity/overlap), committed reference (the agent reads actual committed code, not a description), scoped instructions (two files, not twelve — no irrelevant type-system or deployment context), built-in validation (L1 self-heal before reporting done), and explicit constraints stated in the agent's own terms. Total orchestrator time per dispatch: roughly 2–3 minutes to prepare the prompt and file list, plus shared monitoring time across all active agents in the wave.

### Wave Execution Numbers from PR #394
Four waves plus one recovery wave (2b) handled 75 files in roughly 24 minutes of agent computation time; wave sizes ranged from 1 to 5 agents. Parallelism saved approximately 21 minutes compared to an estimated 45 minutes of sequential agent time — this comparison is estimated, not measured (the sequential approach was never actually run); the numbers reflect the author's judgment based on prior single-agent attempts of similar scope. **The real value was in reduced context degradation, not reduced time.**

### Session Management
Every agent dispatch creates a session — a context window with its own conversation history, loaded instructions, and accumulated state. **Session isolation:** each agent's session is independent; Agent A cannot see Agent B's conversation history, edits, or reasoning. This is a feature — it ensures one agent's context degradation doesn't propagate to others. Information flows between agents through *committed artifacts*, not shared sessions: when Agent B needs to build on Agent A's work, it reads the committed files (the same files that passed tests and were validated at the wave checkpoint), not Agent A's internal reasoning or discarded alternatives.

**Session lifetime — three guidelines:** (1) **One task per session** — reusing a session for a second, unrelated task inherits the first task's conversation history as dead weight. (2) **Reset on failure** — if an agent gets stuck (looping on the same error, producing the same incorrect output), terminate and dispatch a fresh session with refined instructions rather than continuing the failed attempt. (3) **State through files, not memory** — anything that must survive across sessions (committed code, plan documents, checkpoint records) must be written to the filesystem; session-internal state (reasoning, intermediate attempts, debugging output) is ephemeral and should be treated as such.

**Cross-session coordination** is maintained by the orchestration layer (human with multiple terminals, or an automated harness) — not by the agent sessions themselves. This coordination state includes task status (pending/in progress/complete/blocked), file ownership (the enforcement mechanism for the one-file-one-agent rule), wave progress, and the escalation log (what was escalated, what decision was made, and why). Agent sessions are stateless workers; the coordination layer is the stateful manager — keeping this separation clean is what makes multi-agent orchestration predictable. Asking an agent to "track which files you've changed and tell the next agent" mixes coordination state into a session and produces fragile, error-prone results. This is the **runtime layer of the agentic computing stack** (Chapter 4): just as an OS manages processes, memory, and I/O for a CPU, the orchestration harness manages sessions, context loading, and file I/O for the LLM — the harness doesn't do the thinking, it creates the conditions under which thinking produces reliable results.

### Putting It Together — The Six-Step Workflow
The chapter's patterns compose into a workflow: (1) **Assess** scope, (2) **Specialize** team, (3) **Partition** files, (4) **Order** waves, (5) **Dispatch**/Execute, (6) **Validate** (test & check) — looping back to Dispatch on pass, or to Diagnose → L3/L4 human decision → Order waves on fail. This is **not a rigid process** — a small change might skip straight from step 1 to dispatching a single agent; a large change might iterate through steps 5–6 four or five times. The structure is a decision framework, not a ceremony. What matters is the discipline behind it: agents are specialized so context is concentrated, files are partitioned so agents don't conflict, waves are ordered so dependencies are satisfied, and the human makes the decisions that require judgment rather than the decisions that require typing.

---

## Key Terms

- **One-File-One-Agent Rule** — within a single execution batch, no two agents may modify the same file; the primary defense against silent edit corruption in parallel execution.
- **Wave** — a batch of agents that run in parallel and must all complete (and be tested/committed) before the next wave starts; the unit of sequencing in multi-agent execution.
- **File conflict** — two agents assigned to modify the same file in the same wave; a planning error, resolved by merging scope or resequencing.
- **Semantic conflict** — two agents each produce independently correct output that is mutually inconsistent because one agent's context didn't include a pattern change the other established.
- **Design conflict** — two specialized agents produce output reflecting genuinely different, defensible design philosophies; requires human judgment, not mechanical resolution.
- **Escalation protocol (L1–L4)** — the four-level taxonomy (self-heal, retry, human decides, scope change) for routing agent problems to the right level of intervention.
- **Session isolation** — the property that each agent's conversation history, edits, and reasoning are invisible to other agents; coordination happens through committed files, not shared memory.
- **Coordination tax** — the human time cost (planning, monitoring, intervening) of running a multi-agent execution; the variable that determines whether orchestration is worth it for a given change.
- **Foundation-before-migration ordering** — sequencing waves so changes that establish new patterns commit before changes that consume those patterns are dispatched.
