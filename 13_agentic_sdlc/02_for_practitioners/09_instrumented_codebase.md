# Chapter 9 — The Instrumented Codebase

> Source: *The Agentic SDLC Handbook* (Daniel Meppiel, v0.9.2), pp. 104–125.
> A reliable agentic codebase is recognizable before you read a line of application code, by the markdown files that externalize tacit team knowledge into seven machine-readable primitive types — this chapter catalogs what to build; Chapter 10 specifies the constraints they implement.

---

## Very Important

### What Instrumentation Means
An **instrumented codebase** is one that has externalized its tacit knowledge into machine-readable artifacts. Every mature project carries two kinds of knowledge: the first is *in the code itself* — types, function signatures, directory structure, test assertions — and any agent can read this. The second is *in the team's heads* — which authentication pattern is current vs. deprecated, why the logging module wraps a custom helper instead of the standard library, what "follows the BaseIntegrator pattern" means in practice, why one directory has different import rules than every other. **An agent cannot read this. It will guess, and it will guess wrong.**

Instrumentation is the practice of converting the second kind of knowledge into structured files that an agent loads as context. The files are markdown; they version-control alongside the code they describe; they're reviewed in pull requests; and they create a **feedback loop**: when an agent makes a mistake, you don't just fix the generated code — you fix the context file that failed to prevent the mistake. The term "primitive" is deliberate: these artifacts are the atomic units of agentic behavior — like functions in code, they do one thing, compose with other context files, and are testable in isolation. Unlike a prompt typed into a chat window and forgotten, primitives persist, accumulate value, and improve through iteration.

### The Seven Primitive Types
Seven categories cover the full range of knowledge an agent needs; each addresses a distinct gap between what's in the code and what an agent needs to know. Not every project needs all seven on day one (the Instrumentation Audit, below, helps decide where to start), but understanding the complete set comes first.

| Primitive | File Format | Purpose |
|---|---|---|
| **Instructions** | `.instructions.md` (frontmatter: `applyTo` glob) | Scoped conventions per file/directory — "when you touch code in this scope, follow these rules" |
| **Agents** | `.agent.md` (frontmatter: `description`, `tools`, `model`) | Specialist personas with domain expertise, calibrated judgment, and explicit tool boundaries |
| **Skills** | Directory with `SKILL.md` (+ optional `examples/`) | Reusable decision frameworks that activate based on code patterns |
| **Prompts** | `.prompt.md` (frontmatter: `mode`, `tools`) | Repeatable, parameterized multi-step workflows — the agentic equivalent of a script or makefile target |
| **Memory** | `.memory.md`, structured by domain | Cross-session knowledge — accumulated decisions, resolved trade-offs, project history |
| **Orchestration** | `.spec.md`, or workflow composition files | Execution-ready specifications that bridge planning and implementation |
| **Hooks** | Tool-specific config (no single portable file format) | Event-driven automated actions on dev events (save, file creation, PR merge) |

Detail per type:

**1. Instructions** — the most granular context artifact. Frontmatter specifies an `applyTo` glob (e.g. `"src/api/**"`) plus a `description`. **Design test:** can you state the scope in one `applyTo` pattern, and does every rule in the file apply to that scope? If rules apply to two unrelated domains, split the file; if you can't express the scope as a glob, the knowledge probably belongs in an agent configuration or a skill instead. **What distinguishes a good instruction file from a bad one is length** — past 40–50 lines it's trying to do too much. The reason is mechanical: every line of instruction competes for attention with the source code the agent needs to read. A 200-line instruction file doesn't give an agent more to work with — it gives it more to get lost in.

**2. Agents** — specialist personas with domain expertise, calibrated judgment, and explicit behavioral boundaries. Frontmatter specifies `description`, `tools` (a whitelist, e.g. `["changes", "codebase", "editFiles", "runCommands", "search", "problems", "testFailure"]`), and `model`. An agent configuration answers "who should work on this?" in terms of expertise, priorities, and constraints — not a human team member. Four elements make a configuration effective:
   1. **Domain expertise** specific enough to constrain decisions ("You specialize in CLI tool design using the Click framework with Rich terminal output" — not "You are an expert Python developer").
   2. **Named patterns** the agent can reference by name in its reasoning (e.g. "BaseIntegrator," "CredentialChain," "CommandLogger") so it produces code that uses them correctly.
   3. **Anti-patterns** — what the agent must *never* do; these encode institutional memory, each item a mistake that happened at least once and cost the team time to fix.
   4. **Tool boundaries** — which tools the agent can invoke; a documentation agent shouldn't execute destructive commands, a frontend agent shouldn't access backend databases. Tool boundaries are **safety boundaries made concrete**.

   Start with **three to five** agent configurations: an architect, a domain expert for core business logic, and a documentation writer cover most tasks. Add configurations only when you observe repeated corrections — that's the signal a new specialization has earned its place.

**3. Skills** — package reusable *decision frameworks* that activate based on code patterns, living in a directory (`SKILL.md` + optional `examples/`). Skills differ from instructions in kind: instructions give rules ("use `_rich_warning()` for warnings"); skills give **decision frameworks** ("every warning must answer 'what should the user do about this?'") that generalize to situations the author didn't anticipate. Rules cover known cases; frameworks cover unknown ones. **Design test for a skill** — three criteria, all must be yes: does this knowledge apply across multiple files? does it require more than a few rules to express? is it triggered by a detectable code pattern? If the knowledge applies to a single directory, it's an instruction; if it's a general disposition, it's an agent configuration.

**4. Prompts** — reusable, parameterized workflows that orchestrate multi-step tasks; the agentic equivalent of a script or makefile target. `.prompt.md` with frontmatter specifying `mode` (e.g. `agent`) and `tools`. Prompts are the bridge between ad-hoc chat and systematic workflows: without them, every developer asking for a code review types a slightly different request and gets slightly different quality. With a prompt file, the process is consistent — same phases, same checks, same output format. **Quality becomes reproducible.**

**5. Memory** — preserves knowledge across sessions, since agents are stateless and every conversation starts from zero. `.memory.md`, structured by domain (e.g. `## Authentication`, `## API Versioning`, `## Performance Decisions`), each entry dated. Memory captures knowledge that doesn't fit in instructions because it isn't a rule, it's context: "use JWT for authentication" is a rule (instruction); "we migrated from sessions to JWT in Q1, and the old `SessionAuth` class is still in the code but deprecated" is context (memory) — an agent that knows only the rule might accidentally use the deprecated class; an agent that also has the memory won't. Memory is **the primitive most likely to drift from reality** — include dates, and review entries quarterly to verify they're still accurate or remove them.

**6. Orchestration** — bridges planning and implementation by defining structured specifications executable by humans or agents with the same precision. `.spec.md` for specifications, decomposing large features into implementation-ready units (problem statement, approach, components, API contracts, validation criteria as checklists). This is what makes the **Reduced Scope** constraint operational: instead of telling an agent "implement rate limiting" and hoping it infers the approach, the spec defines scope, components, contracts, and success criteria upfront. **The agent implements against a specification, not a wish.**

**7. Hooks** — automated actions triggered by development events (file save, new file creation, PR merge), configured via tool-specific mechanisms (VS Code tasks, GitHub Actions triggers, Copilot hooks config) rather than one portable file format. Hooks bridge the gap between passive context and active behavior: examples include auto-running lint on save, triggering test generation on new source files, auto-updating memory files after a merged PR, or running a security-reviewer agent on every change to `src/auth/`. Without hooks, every instrumentation file is passive, waiting to be queried; with hooks, **the instrumented codebase stops being a library of reference material and starts behaving like an active participant in the development workflow.** Start with one or two (a linting check on save, a test prompt on new file creation); add more only when a repeated manual step reveals the need.

### Tool Support and Portability — The Cross-Platform Mapping Table
The seven primitive types are **conceptual categories, not file format specifications** — how each maps to a concrete file depends on which AI coding tool loads it. The table reflects native support as of mid-2025 (the landscape shifts quarterly, but the underlying pattern is stable: every tool reads project-level markdown; the disagreement is about where it lives and what metadata it supports).

| Primitive | GitHub Copilot (VS Code) | Cursor | Claude Code | Windsurf | OpenCode |
|---|---|---|---|---|---|
| **Instructions** | `.instructions.md` + `applyTo` + `copilot-instructions.md` | `.cursor/rules/*.mdc` + glob | `CLAUDE.md` per directory | `.windsurfrules`; cascade rules | `.opencode/instructions.md` |
| **Agents** | `.agent.md` with model + tools | Via rules and agent modes | `/commands` and agent configs | — | — |
| **Skills** | `SKILL.md` dirs with examples | Embed in rules | Embed in `CLAUDE.md` sections | Embed in rules | — |
| **Prompts** | `.prompt.md` with exec mode | Rules or `.cursor/prompts/` | `/commands` definitions | Flows (different model) | — |
| **Memory** | SQLite DB; `.memory.md` portable | Notepads; project-level rules | `CLAUDE.md` sections; persistent memory | In rules | — |
| **Orchestration** | `.spec.md` as context; plan mode | Context; composer plans | Context; plan mode | Loaded as context | Loaded as context |
| **Hooks** | Copilot hooks config; VS Code tasks | Task runners; `.cursor/hooks/` | Hooks (hooks in settings); pre/post commands | — | — |

`"—"` means the tool has no native format for that type. The knowledge is still usable — you embed it in whatever instruction format the tool does support — but **automatic activation and scoping are lost.**

Three observations the book draws from this table:
1. **Instructions are the universal context file.** Every major tool reads markdown from predictable locations and applies it as context. Naming and scoping mechanisms differ (`applyTo` frontmatter vs. glob-based rule files vs. per-directory placement), but the underlying concept transfers without loss. **This is where to invest first, regardless of tooling.**
2. **Agent configurations are the least portable.** Chat modes, model selection, and tool boundaries are defined differently in every tool and don't transfer. The *knowledge* inside them (domain expertise, named patterns, anti-patterns) is just markdown and moves freely — the activation mechanism does not.
3. **The three major tools now support most primitives natively.** GitHub Copilot, Cursor, and Claude Code each support the majority of the seven types, though file formats and native integration depth differ. As of writing, Copilot has the most extensive native format support, though the gap is narrowing rapidly as Cursor and Claude Code add primitive-equivalent features. OpenCode (the newest entrant) currently supports mainly instructions. Windsurf covers instructions and basic context loading. **The methodology is portable across all of them — the knowledge transfers even when the wiring differs.**

For teams using multiple tools or expecting to switch, organize instrumentation files into two tiers:
- **Portable tier** (works everywhere with minor adaptation): instruction *content* as markdown prose; memory files (decisions, deprecations, historical context); orchestration specs (requirements, contracts, validation criteria); skill knowledge (decision frameworks, anti-patterns, examples).
- **Tool-specific tier** (requires per-tool configuration): instruction *scoping* (how rules get matched to files — `applyTo` vs. glob frontmatter vs. directory placement); agent configurations (model selection, tool boundaries, persona activation); prompt execution (how workflows are triggered and parameterized).

**The portable tier is the knowledge — 80% of the value.** The tool-specific tier is the wiring. Switching tools means rewriting the wiring (a few hours of adaptation), not rewriting what your team knows.

### The Instrumentation Audit
Before building any of this, you need a systematic inventory — not of your code, but of the knowledge your code depends on.

- **Step 1 — List your conventions.** 30 minutes with your team, writing down every convention, pattern, and constraint a new engineer would need to learn in their first two weeks. Don't filter, don't organize. Typical yield: **30–60 items.**
- **Step 2 — Classify each item** by where it lives today:

| Location | Meaning | Agent visibility |
|---|---|---|
| In code | Expressed in types, naming, structure | Partially visible — if it's in the context window |
| In docs | Written in a README, wiki, ADR, style guide | Invisible unless explicitly loaded |
| In heads | Known by team members, never written down | Completely invisible |

  **The "in heads" column is your instrumentation debt.** Every item there is a convention an agent will violate because it has no way to know about it.

- **Step 3 — Rank by failure cost:** **Critical** (security vulnerabilities, data corruption, production outages) → **High** (architectural violations that accumulate as technical debt) → **Medium** (convention violations requiring rework in code review) → **Low** (style preferences that don't affect correctness).
- **Step 4 — Map each item to a primitive type:**

| If the knowledge... | It belongs in... |
|---|---|
| Is a rule scoped to specific files/directories | An instruction file |
| Requires specialist expertise or a specific model | An agent configuration |
| Applies across files and needs a decision framework | A skill |
| Defines a repeatable multi-step process | A prompt |
| Records a decision, trade-off, or historical fact | A memory file |
| Specifies a feature with components and success criteria | A specification |
| Defines an automated response to a development event | A hook |

- **Step 5 — Write your starter set:** begin with **3–5 context files** covering your critical items. Don't aim for completeness — the feedback loop (below) will guide you to what's actually needed faster than upfront planning will.

---

## Important

### Directory Structure
The seven primitive types organize into a predictable directory tree, following GitHub Copilot conventions (the most complete native implementation):

```
project/
  .github/
    copilot-instructions.md       # Global project principles
    instructions/
      api.instructions.md         # applyTo: "src/api/**"
      auth.instructions.md        # applyTo: "src/auth/**"
      frontend.instructions.md    # applyTo: "src/ui/**/*.tsx"
      testing.instructions.md     # applyTo: "**/test/**"
      database.instructions.md    # applyTo: "src/db/**"
    agents/
      architect.agent.md          # Structure, patterns, trade-offs
      backend-dev.agent.md        # API implementation, business logic
      security-reviewer.agent.md  # Injection, traversal, credentials
      doc-writer.agent.md         # Documentation consistency
    skills/
      cli-logging-ux/SKILL.md
      error-handling/SKILL.md
      api-middleware/SKILL.md
    prompts/
      code-review.prompt.md
      feature-impl.prompt.md
      bug-investigation.prompt.md
    specs/
      feature-template.spec.md
      api-endpoint.spec.md
  .memory.md                      # Project-level memory
  AGENTS.md                       # Root discovery file
  src/
    api/AGENTS.md                 # API-specific context
    auth/AGENTS.md                # Auth-specific context
```

Three observations: **context files live in `.github/`**, not scattered through the source tree — centralizing the knowledge layer so a developer looking for the project's AI configuration finds it in one place (the exception is `AGENTS.md`, which lives in the directories it describes because it's part of a discovery hierarchy, covered in Chapter 11). **Each primitive type has its own directory** — instructions, agents, skills, prompts, specs, and hooks don't mix, making it straightforward to audit what exists. **The structure is flat within each directory** — resist nested hierarchies; a flat list of 15 files with descriptive names is easier to scan than a three-level tree. Most projects need **8–12 instruction files**; 50 likely signals over-engineering.

### How Primitives Compose
Primitives are not independent — they form a layered system, and an agent's effective context is the composition of all applicable context files for the current task:

```
Global principles (copilot-instructions.md)
  Scoped instructions (*.instructions.md, matched by applyTo)
    Skills (activated by code patterns in the current task)
      Agent configuration (persona, model, tool boundaries)
        Prompt or spec (the specific workflow being executed)
          Memory (accumulated project context)
            Hooks (event-driven triggers, operating across all layers)
```

Worked example: when an agent is asked to modify `src/api/users.py`, the effective context assembles from (1) **global principles** — error handling, security, testing rules that apply everywhere, (2) **API instructions** — the `applyTo: "src/api/**"` file loads; frontend instructions do not, (3) **API middleware skill** — activates because the task involves an API route, (4) **backend-dev agent** — provides persona, model selection, tool constraints, (5) **memory** — API versioning decisions, the deprecated auth class, the rate-limit timeout choice. Each layer adds specificity; none contradicts the layer above — more specific files *refine* general guidance, they don't override it. **If a conflict exists, it indicates a design error in the instrumentation, not a resolution the agent should attempt.** This composition is the **Explicit Hierarchy constraint made concrete** (Chapter 10).

### Before and After (Worked Example)
A mid-size backend service — 80,000 lines of Python, REST API, message-queue consumer, an auth module with technical debt, an ops CLI; five engineers, two years in.

**Before (uninstrumented):** asking an agent to "add a health check endpoint" produces a route using Flask patterns from training data (the project actually uses FastAPI), a raw DB connection instead of the project's `HealthChecker` service, a plain JSON response ignoring the project's standard envelope (`{"status": ..., "data": ..., "meta": ...}`), a test using inline object construction instead of the team's factory pattern, and no registration in the middleware pipeline (the agent doesn't know it exists). Everything compiles, tests pass, **the PR gets three review comments, all "we don't do it that way here," and the reviewer rewrites 60% of the code.**

**After (instrumented):** the same task, with global instructions, API instructions, an API-middleware skill, a backend-dev agent, and a new-endpoint prompt all loading, produces a FastAPI route using the standard response envelope, a health check delegating to `HealthChecker` (which already knows how to verify DB/cache/queue connectivity), correct middleware registration, and a factory-based test following naming and fixture conventions — **with zero review comments about conventions.** The difference is not the model — it's **150 lines of markdown distributed across 8 files.**

**What the numbers look like** (author's experience across instrumented projects, including the reference case study — explicitly marked as directional, not guaranteed):

| Metric | Uninstrumented | Instrumented |
|---|---|---|
| Convention-violating outputs | 40–60% of generated code | Under 10% |
| Review comments per agent PR | 4–8 ("we don't do it that way") | 0–2 (substantive, not stylistic) |
| Agent-generated code requiring rewrite | 30–50% | Under 15% |
| Time from agent output to merge | Hours (review + rework) | Minutes (spot-check) |

### The Feedback Loop
Instrumentation is not a one-time setup — it is a continuous practice, like testing. The diagnostic pattern when an agent produces incorrect output:

```
Failure observed
  -> Root cause: which context file failed?
       Agent too generic?       --> Add domain knowledge to agent config
       Skill rules incomplete?  --> Add the missing case to the skill
       Instructions missing scope? --> Add a scoped instruction file
       No decision framework?   --> Extract a new skill
       Context gap?             --> Update the memory file
       No repeatable process?   --> Create a prompt
```

Four real examples where fixing the instrumentation file fixed the *class* of error permanently:

| Failure | Root cause | Context fix |
|---|---|---|
| Agent used `_rich_info()` directly instead of `logger.progress()` | Skill didn't explicitly ban direct calls | Added "never call `_rich_*` directly in commands" to the CLI skill |
| Agent invented a new collision-detection pattern | Instructions didn't list all base-class methods | Added "use, don't reimplement" table to integrator instructions |
| Agent produced inconsistent Unicode symbols in output | No single source of truth for status symbols | Created `STATUS_SYMBOLS` reference in skill, added to anti-patterns |
| Agent used deprecated `SessionAuth` in new code | Memory file didn't record the deprecation | Added deprecation notice with migration tracking reference |

After 20–30 iterations of this loop, an instrumentation set covers the conventions agents *actually* violate, not the ones theorized about in advance — that practical grounding is what makes an instrumented codebase effective.

### Annotated Session — What the Feedback Loop Looks Like in Production
Excerpts from a real Copilot CLI session (Session `d89b3ccc`, repo `microsoft/apm`, branch `feat/auth-logging-architecture-393`, result **PR #394** — 75 files changed, 207 turns across 2 sessions) — not a reconstruction or sanitized demo. Selected turns:

- **Turn 0 — the messy start.** The developer pastes a raw terminal failure and role-casts the agent in one breath ("Can you think as a world class UX expert and work with an APM logging mechanism implementation expert..."). No YAML plan, no schema — a bug report, a role assignment, and a quality bar, all in natural language.
- **Turn 3 — fleet deployment in six words:** *"Fleet deployed: create an issue to track this Epic and then implement on a new branch."* The agent completed 17 of 25 planned tasks, created a GitHub issue, pushed a feature branch. Real orchestration is natural language with clear intent, not CLI flags.
- **Turns 4–5 — the feedback loop catches a silent semantic failure.** The agent's plan assumed EMU (Enterprise Managed Users) meant `*.ghe.com` hosts — a reasonable but wrong inference. The developer's correction sharpened into an instrumentation fix, not just a code fix: *"This is critical knowledge for the auth expert agent.md to be updated."* This is **Anti-Pattern #6 from Chapter 14 (Silent Semantic Failure)** caught and resolved at the primitive layer.
- **Turn 23 — multi-agent debugging with real terminal output**, tracing the failure chain (`Auth resolved -> Trying unauthenticated -> failed -> retrying with token -> 403 -> trying credential fill -> failed`) to a Basic-auth URL format GitHub rejects for fine-grained PATs; fixed by switching to `git -c http.extraHeader='Authorization: Bearer {token}'`.
- **Turn 50 — escalation to named agents** — restructuring the working group to bring in a custom `.agent.md` software architect and a logging expert as named subagents, rather than writing more code or tests directly.
- **Turn 71 — granular review then delegation** — reviewing at the level of individual log lines, then handing the fix to the specialist agent whose `.agent.md` defines the quality bar.
- **Turn 73 — the instrumentation payoff:** *"you must ensure you update the logging skill so that such architectural concerns and patterns are enshrined — not as unmovable, but as current art and baseline."* Not a code fix — a **primitive fix**, ensuring every future agent (in this session, next month's session, by a different developer) inherits the correction.
- **Turn 52 — the meta-moment.** The developer recognized mid-session that the orchestration pattern being used (waves, task dependencies, checkpoints with panel discussions, specialized agents) "should become a handbook for AI Engineers to leverage" — this book was partly born from the patterns it describes.

Three things the chapter highlights about this session: the developer's value was **domain knowledge, not keystrokes** (across 207 turns and 75 files, they wrote no application code — their contributions were framing, domain corrections, team-composition decisions, and context-file fixes); **every correction became a permanent artifact** (instead of fixing the auth code and moving on, corrections went into `auth-expert.agent.md` and the logging skill — context files that prevent the *class* of error, not just the instance); and **the hardest bugs were semantic, not syntactic** (the EMU/`*.ghe.com` assumption was plausible, consistent with documentation, and wrong — no linter or test catches that unless the test author already knew the right answer, in which case the test is redundant). This is **Anti-Pattern #6: Silent Semantic Failure** — the safety net was a human with domain expertise reviewing agent output with enough context (verbose logs) to see where the reasoning broke.

### Starting Points
A phased adoption plan, since the full seven-primitive structure represents a *mature* instrumented codebase, not a day-one requirement:

- **Week one** — write three files: (1) global instructions (10–15 lines, your non-negotiable principles), (2) one scoped instruction file for your most-edited module, (3) one agent configuration for the task agents perform most often.
- **Week two** — use these on real work. When the agent violates an uncovered convention, add it. When it does something right that surprised you, check whether your instrumentation contributed. Update the memory file with decisions and trade-offs resolved that week.
- **Week three** — extract the first skill, once you notice you've written the same guidance in two different instruction files. Package it with a decision framework. Create the first prompt file for a task you've now asked an agent to do three or more times.
- **Ongoing** — review context files monthly. Remove rules that never trigger. Tighten rules that trigger but don't prevent the failure. Add new rules only in response to observed failures. **Treat instrumentation like a test suite** — it should grow with the codebase, stay accurate, and never contain dead rules.

The directory structure and primitive types can be built by hand for a first project (doing so builds understanding). For subsequent projects, or teams standardizing across repositories, **package-manager-style tooling** reduces the scaffolding work from hours to seconds — the author's own open-source implementation is **APM (Agent Package Manager)**: `apm init` generates `copilot-instructions.md`, one scoped instruction file, and one agent configuration (the Week One starter set); `apm install` pulls shared context files from any Git repository.

---

## Key Terms

- **Instrumented codebase** — a codebase that has externalized its tacit (in-the-team's-heads) knowledge into structured, machine-readable markdown artifacts an agent loads as context.
- **Primitive** — one of seven atomic, composable artifact types (instructions, agents, skills, prompts, memory, orchestration, hooks) that together cover the knowledge gap between what's in the code and what an agent needs to know.
- **Instructions** — `.instructions.md` files scoped to files/directories via `applyTo` glob frontmatter; the most granular and most portable primitive.
- **Skill** — a packaged, reusable decision framework (not just a rule list) that activates when code matches a pattern; distinguished from instructions by generalizing to unanticipated cases.
- **Memory file** — `.memory.md`, dated and domain-structured, preserving decisions/trade-offs/history across stateless agent sessions; the primitive most prone to drifting out of date.
- **Instrumentation audit** — the five-step process (list conventions, classify by location, rank by failure cost, map to primitive type, write a starter set) for deciding what to instrument first.
- **Instrumentation debt** — conventions that exist only "in heads" (never written down), invisible to any agent and guaranteed to be violated eventually.
- **Portable tier vs. tool-specific tier** — the split between instrumentation content (knowledge, transfers across tools with minor adaptation) and instrumentation wiring (activation/scoping mechanisms, tool-specific and non-portable).
- **APM (Agent Package Manager)** — the author's open-source CLI (`apm init`, `apm install`) for scaffolding and sharing instrumentation primitives across repositories, analogous to npm for JavaScript modules.
