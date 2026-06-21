# Chapter 11 — Context Engineering

> Source: *The Agentic SDLC Handbook* (Daniel Meppiel, v0.9.2), pp. 144–154.
> An AI agent never sees your codebase — it sees whatever fits in its context window; context engineering is the discipline of deciding what that is.

---

## Very Important

### The Context Budget
Every AI interaction operates within a fixed capacity. A context window is not a bucket you fill — it is a budget you allocate. The question is never "how much can I spend?" but "what do I spend it on?" A typical agent session on a 128K–200K-token window (common in current frontier models) divides roughly as follows:

| Segment | % | Tokens | What fills it |
|---|---|---|---|
| System prompt | ~6% | ~8K | Model behavior, safety, base tool definitions |
| Instructions & rules | ~16% | ~20K | Your scoped instruction files, agent config |
| Code context | ~47% | ~60K | Source files, type definitions, dependencies |
| Conversation history | ~23% | ~30K | Prior turns, tool output, agent reasoning |
| Working memory | ~8% | ~10K | The agent's space to think and produce output |

These allocations are starting recommendations from the author's practice — observed across Copilot CLI sessions and similar agentic tools, not derived from controlled experiments. You control two segments directly (instructions, code context), influence a third through session discipline (conversation history), while system prompt and working memory are largely fixed by the tool. Your leverage is concentrated: what instructions load, which code is visible, and how long the session runs before you reset.

20K instruction tokens is approximately 800 lines of typical prose markdown — if global instructions alone consume 400 of those lines, you've spent half your instruction budget before a single scoped rule loads. 60K for code context sounds generous until a single mid-sized module (types, implementation, tests) consumes 15K tokens. At scale, every line of instruction competes directly with a line of source code the agent could have seen instead.

Attention within the window is not uniform — information at the beginning and end gets more weight; content in the middle degrades (Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," 2023, accepted NeurIPS 2024). In the author's experience, a focused 40-line instruction file produces more consistent output than a sprawling 400-line one covering the same material. Two practical implications follow: (1) shorter sessions produce better results than longer ones, because conversation history consumes progressively more of the budget; (2) loading instructions that aren't relevant to the current task is not neutral — it actively degrades performance on the task that matters.

The budget model produces a simple decision rule: before adding anything to an agent's context, ask — **does this earn its space?** Every instruction that loads is a line of code that doesn't. Every file that's visible is one that could have been more relevant.

### The Instruction Hierarchy
The most consequential context engineering decision is how you structure your instructions. Chapter 10 specifies the Explicit Hierarchy constraint (§E in the PROSE framework); this chapter covers the engineering side — how to build that hierarchy in a real repository, size each layer for the context budget, and compose it at runtime. The hierarchy has layers from broadest to most specific scope:

1. **Enterprise / Org Policy** — security baselines, compliance rules (e.g. `apm-policy.yml`). Flows down through **tighten-only inheritance**: allow-lists intersect, deny-lists union, enforcement only escalates. No repository can relax it.
2. **Repository Instructions** — project principles: error handling, naming, testing (e.g. `copilot-instructions.md`, root `AGENTS.md`, `applyTo: '**'`).
3. **Directory Scopes** — domain-targeted instructions so auth rules never pollute frontend context (e.g. `backend/AGENTS.md`, `.instructions.md`, `applyTo: 'src/backend/**'`).
4. **File scope** — surgical constraints targeting specific files or file types; encodes the knowledge that would otherwise require reading a file's git history to understand (e.g. `applyTo: "src/integration/**"` with a "use this, never that" table for base-class methods).
5. **Agent configurations and skills** — bind expertise and tool boundaries to specific task types; activate on code-pattern matches to deliver just-in-time playbooks.

**Engineering constraint at every layer: size.** Global instructions should stay under 50 lines — if project-wide principles exceed that, domain-specific guidance is leaking in that belongs in a narrower scope. Directory-scoped instructions should focus only on patterns unique to that domain. The total instruction load for any single agent task should fit within the ~20K-token instruction budget. The hierarchy must be selective: each layer earns its tokens by being relevant to the current task, not merely present in the repository.

**How the hierarchy composes at runtime.** When an agent edits `src/auth/token_resolver.py`, the effective context is: (1) global principles (error handling, security, testing), (2) auth module rules (token management, credential chain), (3) any file-specific constraints matching `src/auth/token_*.py`. When the same agent later edits `src/frontend/dashboard.tsx`, the auth rules unload and frontend rules load — global principles persist. This is the Explicit Hierarchy constraint in action: specificity increases as scope narrows, and irrelevant context is automatically excluded.

The practical benefit is measurable: a project with 300 lines of instructions split across 8 scoped files produces more consistent agent output than a project with 100 lines in a single global file — the scoped version also contains more total content, illustrating both the value of structure *and* of comprehensive coverage. The split version loads 30–50 relevant lines per task; the monolithic version loads all 100 regardless of relevance.

### Agent Configuration
Instructions tell agents *what rules to follow*. Agent configurations tell them *who to be*. The distinction matters because different tasks require different expertise, judgment, and constraints. An agent configuration defines four things:

1. **Role and expertise** — what domain knowledge the agent brings to every task (an architecture agent knows module boundaries and dependency management; a security agent knows injection vectors and credential handling).
2. **Model selection** — which language model runs the agent. Complex architectural decisions may warrant a more capable (and more expensive) model; routine formatting tasks don't.
3. **Behavioral constraints** — what the agent prioritizes when making trade-offs (an agent configured with "KISS — simplest correct solution" makes different choices than one configured with "optimize for performance").
4. **Anti-patterns** — what the agent should never do. Often more valuable than positive instructions, because it prevents the specific mistakes your team has already seen.

Worked example (`.github/agents/python-architect.agent.md`): a YAML front-matter block (`name`, `description`, `model: claude-sonnet-4.5`) followed by a markdown body with a "Design Philosophy" list, a "Patterns You Enforce" list, and a "You Never" list (e.g. never add a new base class when an existing one can be extended; never instantiate a singleton per-request; never bypass the public API). **The "You Never" section is where institutional memory becomes operational** — every item represents a mistake that happened at least once, possibly caught in review, possibly in production. Encoding it in the agent configuration means it doesn't happen again, regardless of which human or AI writes the next change.

For a typical project, **three to five agent configurations cover most tasks**: an architect (structure and patterns), a domain expert (core business logic), a documentation writer, and optionally a security reviewer and a test specialist. Start with fewer. Add agents when you observe the same correction being made repeatedly — that's the signal a new specialization has earned its place.

### Skill Design
Skills occupy the space between instructions and agents. An instruction says "follow this rule." An agent says "be this expert." **A skill says "when you encounter this situation, here's the complete playbook."** A skill is a reusable knowledge package that activates based on code patterns — when an agent touches logging code, the logging skill fires; when it touches authentication flows, the auth skill fires. The agent doesn't choose to activate a skill; the tooling detects the pattern match and loads it.

**The design test for a skill:** does this knowledge apply across multiple files, require more than a few rules to express, and get triggered by a detectable pattern? If all three are true, it's a skill. If the knowledge applies to one file, it belongs in an instruction. If it's a general approach rather than pattern-specific guidance, it belongs in an agent configuration.

A well-designed skill has three sections, illustrated by the "CLI Logging UX Skill" example:
1. **When This Activates** — the trigger pattern (e.g. "Code touches console helpers, `DiagnosticCollector`, `STATUS_SYMBOLS`, or any user-facing terminal output").
2. **Decision Framework** — how to think about the problem, not a fixed rule list. The example includes the "So What?" Test (every warning must answer: what should the user do about this?), the Traffic Light Rule (a color → helper → meaning table: Green/`_rich_success()`/Completed, Yellow/`_rich_warning()`/User action needed, Red/`_rich_error()`/Cannot continue, Blue/`_rich_info()`/Status update), and the Newspaper Test (can the user scan output like headlines?).
3. **Anti-Patterns** — e.g. never use bare `print()` or `click.echo()` without styling; never emit a warning without an actionable suggestion; never mix Rich and colorama in the same output path.

**A decision framework is what distinguishes a skill from a list of rules.** Rules tell you what to do. A decision framework tells you how to think about the problem, which means it generalizes to situations the author didn't anticipate. Skills vs. one-off instructions: if you find yourself writing the same instructional content in three different instruction files, extract it into a skill — the instruction files then reference the skill by name rather than duplicating the content. This mirrors the DRY principle applied to code, for the same reason: when the convention changes, you update one skill file instead of hunting through every instruction that mentions it.

### The Minimal Viable Context
If a full context audit feels like too large an undertaking, start smaller. The minimum viable context for any project is three files:

| File | Scope | Contains |
|---|---|---|
| Global instructions | Project-wide | 5–10 non-negotiable principles (error handling, security, testing) |
| One domain instruction | Your most-edited module | The patterns and constraints specific to where agents will work most |
| One agent configuration | Your most common task type | The expertise and behavioral constraints for the work agents do daily |

Write these three files. Use the agent on a real task. Observe what goes wrong. Fix the context files. Repeat. Within a few iteration cycles you'll have a context architecture shaped by actual failure modes rather than theoretical completeness — and that architecture will be more effective than any comprehensive upfront design.

**Context engineering is not about making agents smarter. It is about making the information available to agents accurate, relevant, and proportional to the task. The model doesn't change. The context does. And that is where the leverage is.**

---

## Important

### Memory and Retrieval
Agents are stateless. Every session starts with an empty context window, and every session's accumulated knowledge vanishes when it ends — a fundamental constraint, not a bug to work around. Three strategies address it:

- **Session context** — within a single session, the agent accumulates information through conversation turns (tool outputs, file reads, corrections). The most natural form of memory, and the most fragile: it degrades as the session grows and disappears when the session ends.
- **Persistent instructions** — the instruction hierarchy itself is a form of persistent memory: it survives across sessions because it lives in files, not conversation history. The feedback loop: observe failure, diagnose root cause, fix the primitive, verify on the next task. Over time this accumulation makes a codebase increasingly AI-ready.
- **External knowledge retrieval** — for codebases too large to fit relevant context in the window, retrieval mechanisms (code search, semantic index, documentation search) bring specific information into context on demand. This is progressive disclosure at the knowledge level. The implementation varies by tool; the principle is consistent — give the agent a way to pull specific knowledge rather than pushing everything.

**When to reset a session** — three concrete triggers: (1) **Stale references** — the agent references a file it read or modified more than 3–4 turns ago; its mental model of that file is now competing with everything that's happened since, and losing. (2) **Error spirals** — more than two error messages pasted in the same session; each paste pulls the agent toward debugging the symptom rather than rethinking the approach. A fresh session with the error described in one sentence often solves in one turn what three debugging turns couldn't. (3) **Conversation length** — the session exceeds roughly 30–40 turns; original instructions are buried under pages of dialogue and effective context has drifted far from where it started. When resetting, carry forward a one-paragraph summary of what was decided and what remains — not the full conversation. **Fresh context beats accumulated drift.**

**Co-location** matters for retrieval. If auth logic lives in `src/auth/` and its documentation lives in `docs/auth-guide.md`, an agent working on auth code may never see the documentation unless explicitly pointed to it. Structure the repository so the knowledge an agent needs is either scoped to the directory it's working in or discoverable through the instruction hierarchy — an architectural choice, not a tooling choice. Rule of thumb for what to externalize vs. keep in instructions: needed on every task in a scope → instructions (always loaded); needed occasionally → retrievable storage (loaded on demand); needed once → the session prompt.

### The Context Audit
Run the instrumentation audit from Chapter 9 if you haven't already — its five steps (list conventions, classify where they live, rank by failure cost, map to primitive types, write a starter set) identify the raw material. The audit reveals three categories of knowledge: conventions already visible in code (partially available to agents), conventions written in documentation (available only if explicitly loaded), and conventions that exist only in the team's memory (completely invisible). The last category — **your context debt** — is where context engineering begins. Each item in it needs a home: a scope in the hierarchy, a loading strategy (always-on instruction vs. on-demand retrieval), and a format agents can act on.

### Before and After — A Worked Example
Task: "Add rate limiting to the `/api/users` endpoint." **Without structured context**, the agent produces a plausible-looking `flask_limiter` integration — but it violates three conventions that exist only in the team's heads: the project uses a custom rate limiter integrated with the metrics pipeline, rate-limit configuration lives in environment variables (not hardcoded), and middleware decorators are applied centrally in `middleware.py`, never inline on routes. **With structured context** — three scoped instruction files totaling 25 lines (a global middleware-registration rule, a directory-scoped `src/api/` rate-limiting rule, and an `api-middleware` skill with a three-step registration pattern) — the agent produces code using `app.rate_limiter.RateLimiter`, reads config via `RateLimiter.from_env("API_RATE_LIMIT_USERS")`, and registers the middleware externally with zero changes to the route file. Same model, same task, different output — the difference is entirely in what the agent knew before it started. **The critical point: none of these failures are model failures.** The model is capable of doing the right thing in every case; it does the wrong thing because it lacks the information. Context engineering closes that gap.

### Common Mistakes
- **Over-stuffing context** — more information feels helpful but is almost always counterproductive. A 400-line instruction file doesn't give the agent "more to work with"; it gives it more to get distracted by. If you can't remember what's in your instruction file without re-reading it, the agent is having the same problem.
- **Flat instructions without scoping** — one global file covering everything (API conventions, frontend patterns, database rules, deployment procedures) forces every session to load every rule, regardless of relevance (the Flat Instructions anti-pattern from Chapter 10).
- **Duplicating knowledge across files** — when the same convention appears in three instruction files, updating one and forgetting the others produces contradictory guidance and inconsistent output. Extract shared knowledge into a skill; reference, don't repeat.
- **Mixing concerns in a single primitive** — a file covering both "how to write error messages" and "how to structure database migrations" serves two unrelated domains; changing one risks destabilizing the other. One concern per primitive — the single-responsibility principle applied to context.
- **Ignoring the feedback loop** — writing instruction files once and never updating them is like writing tests once and never running them. Teams that skip this loop find their context artifacts drift from reality within weeks.
- **Treating context engineering as a one-time setup** — it is a continuous discipline, like testing or code review. Budget time for primitive maintenance the way you budget time for dependency updates: small, regular investments that prevent large, painful corrections.

---

## Key Terms

- **Context budget** — the fixed token capacity of a session, allocated across system prompt, instructions, code context, conversation history, and working memory; not infinitely expandable, must be spent deliberately.
- **Instruction hierarchy** — the layered structure (org policy → repository → directory → file → agent/skill) through which instructions narrow in scope and increase in specificity as they get more local.
- **Tighten-only inheritance** — the rule by which org-policy constraints flow down to repositories: allow-lists intersect, deny-lists union, enforcement only escalates, never relaxes.
- **Agent configuration** — a file defining an agent's role/expertise, model selection, behavioral constraints, and anti-patterns ("You Never" list); answers "who should the agent be?"
- **Skill** — a reusable, pattern-triggered knowledge package containing a decision framework (not just rules) for a recurring situation that spans multiple files.
- **Context debt** — conventions that exist only in team memory, invisible to agents until externalized into the instruction hierarchy or retrievable storage.
- **Session reset** — discarding accumulated conversation history (carrying forward only a short summary) once stale references, error spirals, or conversation length degrade the agent's effective context.
- **Minimal Viable Context** — the three-file starting point (global instructions, one domain instruction, one agent configuration) recommended before attempting a full context audit.
