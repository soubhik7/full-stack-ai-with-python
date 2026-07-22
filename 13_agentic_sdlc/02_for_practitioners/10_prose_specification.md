# Chapter 10 — The PROSE Specification

> Source: *The Agentic SDLC Handbook* (Daniel Meppiel, v0.9.2), pp. 126–143.
> This is the book's reference chapter: the full technical specification of the five PROSE constraints — definition, why it matters, implementation patterns, and anti-patterns for each — closing with three failure stories showing what breaks when a constraint is missing, a worked example applying all five together, and a yes/no compliance checklist.

---

## Very Important

### The Constraint Model
PROSE defines five constraints. Each addresses a fundamental property of language models (finite context, stateless reasoning, probabilistic output) and each induces a desirable property in the system that follows it. **The constraints are independent in definition and interdependent in practice** — a footnote roots this in design-thinking research (Stokes, *"Creativity from Constraints,"* Springer, 2005) on constraints enabling rather than restricting creativity.

| Constraint | Addresses | Induces |
|---|---|---|
| **P**rogressive Disclosure | Context overload | Efficient context utilization |
| **R**educed Scope | Scope creep | Manageable complexity |
| **O**rchestrated Composition | Monolithic collapse | Flexibility, reusability |
| **S**afety Boundaries | Unbounded autonomy | Reliability, verifiability |
| **E**xplicit Hierarchy | Flat guidance | Modularity, domain adaptation |

The rest of the chapter specifies each constraint in turn — definition, implementation patterns, and anti-patterns — then closes with what goes wrong when constraints are missing **in combination**.

### P — Progressive Disclosure
**Definition:** Structure information to reveal complexity progressively. Context arrives *just-in-time* (loaded when the agent needs it for the current task), not *just-in-case* (everything loaded upfront on the assumption it might be relevant).

**Why it matters:** Context is finite and attention degrades under load. The issue is not capacity (context windows keep growing) but attention — information competes for focus, and material far from the active task gets deprioritized. A context window packed with architecture docs, coding standards, API specs, and every source file that *might* be relevant is not a well-informed agent — it is a diluted one. **The signal-to-noise ratio determines quality, not the volume of signal.**

**Implementation patterns:**
- **Markdown links as lazy-loading pointers.** An instruction file references `[authentication patterns](../../docs/auth-patterns.md)` rather than inlining the full guide; the agent loads that content only when the current task involves authentication. The link is a pointer — a declaration that deeper context exists, with a path to reach it. The agent reads the guideline file; if the task involves auth it follows the link, if it involves error handling it follows a different link; **neither loads material for the other.**
- **Descriptive labels for relevance assessment.** Links only work if the agent can judge whether to follow them *before* following them. Bad: `See [docs](./docs/auth.md)`. Good: `See [JWT validation and refresh token rotation patterns](./docs/auth.md) for all endpoints requiring authentication` — the descriptive version gives the agent enough information to assess relevance without loading the content.
- **Skills metadata as capability indexes.** A skill's frontmatter `description` field acts as an index: the agent reads the description, determines relevance, and loads the full skill content only when the task matches (e.g., a `form-validation` skill's description specifies exactly when to activate; an agent on a database migration sees the description and skips it, an agent building a registration form loads the full content).

**Anti-pattern: Context dumping.** The most common violation — a single instruction file inlining the full coding standards, complete API reference, authentication guide, and error taxonomy all in one document, wasting context capacity on material irrelevant to most tasks. Concretely:

*Before* — everything inlined in one file: 2,000 words of coding standards + 1,500 words of auth patterns + 800 words of error handling + 1,200 words of deployment procedures + 600 words of accessibility requirements, all in `# Project Rules`.

*After* — pointers with descriptive labels: the same `# Project Rules` file becomes five linked bullets (`[Coding standards and naming conventions](./standards/coding.md)`, `[Authentication patterns and token lifecycle](./standards/auth.md)`, etc.) — **same information, a fraction of the context cost per task.**

### R — Reduced Scope
**Definition:** Match task size to context capacity. Complex work is decomposed into tasks sized to fit available context; each sub-task operates with fresh context and focused scope.

**The sizing heuristic:** the best-sized task is one the agent can complete **without asking a follow-up question.** If the agent needs to ask "which error format should I use?" or "where is the existing session logic?", either the task is too large (decompose it) or the context is insufficient (add the missing information) — both are scope problems. **If you find yourself adding context mid-session to keep the agent on track, the scope was wrong from the start.**

Three worked examples of tasks that were too big, and how they got split:
- *"Add JWT authentication"* — too big: spans token schema, middleware, refresh endpoint, tests, and frontend. **Split into five sessions** (one per component), each with fresh context and a single deliverable.
- *"Fix the login bug and update the session tests"* — two domains in one session; the bug fix accumulates context that degrades the quality of the test updates. **Split:** fix in one session, test updates in a follow-up with the fix already committed.
- *"Refactor the payment service to use the new API client"* — the agent needs the old client, the new client, every call site, and the test suite simultaneously. **Split by call site:** each session migrates one module, with only the old→new API mapping and that module's source files in context.

**Implementation patterns:**
- **Phase decomposition** — separate planning from implementation from testing; each phase gets a fresh context window with only what's relevant to that phase:
  ```
  Phase 1: Diagnose   — Input: error logs, relevant source files   — Output: root cause analysis with 2–3 candidate fixes
  Phase 2: Implement  — Input: chosen fix from Phase 1, relevant source files — Output: code changes
  Phase 3: Validate   — Input: changed files, test suite           — Output: test results, regression check
  ```
  The diagnosis phase doesn't carry implementation context; the validation phase doesn't carry diagnosis context. **Quality remains consistent across phases because attention is never split.**
- **Session splitting across domains** — when a change spans frontend/backend/database, use separate sessions for each; a backend agent does not need the frontend component tree in its context, a database migration agent does not need UI routing logic.

**Anti-pattern: Scope creep.** A session starts with a focused task ("fix the login timeout bug"); mid-session, additional requests accumulate — "also update the session management tests," "while you're at it, refactor the token refresh logic," "and can you add the new `/logout` endpoint?" By the fourth request, the agent operates with the accumulated context of four different tasks, attention split across all of them. **The quality of the newest, least-contextualized task (the logout endpoint) is significantly lower than the quality of the original bug fix.** Fix: each request becomes its own session — the cost is four sessions instead of one; the benefit is **four tasks done well instead of four tasks done poorly.**

### O — Orchestrated Composition
**Definition:** Favor small, chainable primitives over monolithic frameworks. Build complex behaviors by composing simple, well-defined units.

**Why it matters:** a 3,000-word mega-prompt covering role definition, coding standards, error handling, testing requirements, security rules, documentation format, and output structure is not a well-specified agent — it is an unpredictable one. Small changes to any section produce unexpected changes in behavior across all sections, because the model processes the entire block as a single context. **Composition preserves clarity** — each primitive is small enough to understand, test, and debug independently; complex behavior emerges from combining primitives, not from making any single unit more complex.

**Implementation patterns:**
- **Primitive types as atomic units** — instruction files, skills, agents, prompts, and specifications each have a focused purpose, e.g.:
  ```
  .github/
    instructions/
      frontend.instructions.md   # applyTo: "**/*.{tsx,jsx}"
      backend.instructions.md    # applyTo: "**/*.py"
      testing.instructions.md    # applyTo: "**/test/**"
    chatmodes/
      architect.chatmode.md      # Planning - cannot execute
      backend-dev.chatmode.md    # Implementation - scoped tools
    prompts/
      feature-impl.prompt.md     # Multi-step workflow
    skills/
      form-validation/SKILL.md   # Auto-activated by relevance
  ```
  Each file has a single responsibility — frontend instructions contain no backend rules, the architect agent has no implementation tools, the feature workflow composes building blocks without duplicating their content.
- **Workflows as compositions** — a prompt file orchestrates multiple instruction files into a sequence (review the spec → analyze existing patterns for consistency → implement following active instructions → run validation → **STOP** and present a summary for human review). The workflow does not restate coding standards (the instruction files handle that) or redefine the agent's role (the agent configuration handles that) — **it composes existing building blocks into a sequence.**

**Anti-pattern: Monolithic prompt.** All guidance in a single block — role, rules, examples, constraints, output format, everything in one prompt (e.g. a single block dictating PEP 8, type hints, the `BaseIntegrator` pattern, banning `os.walk`, mandating `_rich_error()`, requiring pytest fixtures, Google-style docstrings, JSON output, a `src/`-only file boundary, and a pre-commit linter rule — all at once). **This works until it doesn't:** when the agent produces incorrect output, which instruction failed? Which rule contradicted which other rule? A monolithic prompt is impossible to debug because every instruction interacts with every other instruction in ways the model resolves internally, without explanation. **Fix:** each concern becomes its own primitive file — Python conventions in `python.instructions.md`, integrator architecture in `integrators.instructions.md` (scoped via `applyTo: "src/**/integration/**"`), testing standards in `testing.instructions.md`. Each is independently testable, independently debuggable, and independently versioned.

### S — Safety Boundaries
**Definition:** Every agent operates within explicit boundaries: what tools are available (**capability**), what context is loaded (**knowledge**), and what requires human approval (**authority**).

**Why it matters:** Chapter 1 established that output is probabilistic — variance is inherent, not a bug. Safety Boundaries is *how you constrain it*. When a non-deterministic system has unbounded authority, the variance in its outputs translates directly into variance in its effects. **Boundaries do not reduce an agent's usefulness — they constrain its blast radius.** An agent that can modify backend code but not frontend assets, that can run tests but not deploy to production, that must pause for approval before deleting files, is **both more useful and more trustworthy** than one with no restrictions.

**Standard role boundaries** — four common agent roles with concrete capability, knowledge, and authority boundaries (adapt the specific tools per platform, but the *shape* of each boundary should not change):

| Role | Tools (capability) | Knowledge (scope) | Authority (approval gates) |
|---|---|---|---|
| **Code writer** | `editFiles`, `runCommands`, `search`, `readFiles`, `testRunner` | Files matching its domain (`applyTo`/directory scope). Cannot see infra config, CI/CD, or deploy scripts. | Must **STOP** before: modifying public API signatures, changing database schemas, touching auth logic. |
| **Reviewer** | `readFiles`, `search`, `runCommands` (read-only: lint, test, type-check) | Full repository read access. No write tools. | No approval gates — reviewer cannot change anything. Output is commentary only. |
| **Test runner** | `runCommands`, `readFiles`, `testRunner` | Test files + source files under test. No access to deploy config, secrets, or CI definitions. | Must **STOP** before: deleting test fixtures, modifying shared test infrastructure, skipping tests. |
| **Deployer** | `runCommands` (deploy scripts only), `readFiles` | Deployment manifests, environment config, release notes. No source code write access. | Must **STOP** before: every deployment action. Human approves each environment promotion. |

A **planning agent** (architect chatmode) is a special case: it gets `readFiles` and `search` only, no write tools at all — its output is a plan, not code; the implementation agent consumes that plan in a separate session.

**Implementation patterns:**
- **Tool whitelists per agent** — each agent configuration declares the specific tools it can access (e.g. `tools: ["editFiles", "runCommands", "search", "testRunner"]`). The backend agent cannot access deployment tools, cannot modify CI/CD configuration — its capability boundary is explicit and auditable.
- **Validation gates requiring human approval** — critical decisions (architectural changes, security-sensitive modifications, data migrations) require the agent to stop and present its plan before proceeding:
  ```
  ## Validation Gate
  Before modifying any authentication logic:
  1. Present the proposed changes with rationale
  2. **STOP** and wait for explicit human approval
  3. Do not proceed until approval is received
  ```
  The gate is part of the *instruction*, not an external enforcement mechanism — the agent's context includes the requirement to pause.
- **Knowledge scoping** — instructions load only when the agent is working on matching files; backend security rules don't load when the agent edits CSS, frontend accessibility requirements don't load during database migrations. Note: the `applyTo` pattern serves double duty — primarily an Explicit Hierarchy mechanism (defining *which rules apply where*), but it also functions as a knowledge boundary, constraining *what the agent knows* during a task. Explicit Hierarchy defines how `applyTo` works; Safety Boundaries explains why constraining knowledge matters for safety.
- **Deterministic tools as truth anchors** — when an agent claims a test passes, a boundary-constrained system requires it to *actually run the test* and report the deterministic result. Code execution, API calls, file-system operations are deterministic tools that ground probabilistic generation in verifiable reality.

**Anti-pattern: Unbounded agent.** An agent with access to every tool, every file, and no approval requirements (`tools: ["*"]`, no validation gates) can modify production configuration, delete test fixtures, rewrite CI pipelines, and access credentials, all without human oversight. **When (not if) it produces an unexpected output, the blast radius is the entire repository.** Fix: start with the role table above, find the closest match, copy its boundaries, and tighten from there — **the minimum set of tools is always fewer than you think.**

### E — Explicit Hierarchy
**Definition:** Instructions form a hierarchy from global to local. Local context inherits from and may override global context. Agents resolve context by walking from the most specific scope to the most general.

**Why it matters:** different domains require different guidance, but they also share common ground. Hierarchy solves both problems — global rules establish consistency (naming conventions, commit message format, documentation standards); local rules enable specialization (the auth module gets security-specific instructions, the frontend gets accessibility-specific instructions, the database layer gets migration-specific instructions). **Each domain inherits the global rules and adds or overrides with local ones.**

**Implementation patterns:**
- **Directory-scoped context files** — the `AGENTS.md` standard uses directory placement to define scope: a file at the project root applies everywhere, a file in `frontend/` applies only to frontend code, a file in `frontend/components/` applies only to components:
  ```
  project/
    AGENTS.md                  # Global: naming, commits, documentation
    frontend/
      AGENTS.md                # Frontend: React patterns, accessibility
      components/
        AGENTS.md               # Components: prop conventions, testing
    backend/
      AGENTS.md                # Backend: API design, error handling
      auth/
        AGENTS.md               # Auth: security patterns, token handling
  ```
  An agent editing `backend/auth/token.py` resolves context by walking *up*: `auth/AGENTS.md` + `backend/AGENTS.md` + root `AGENTS.md`. An agent editing `frontend/components/Button.tsx` resolves: `components/AGENTS.md` + `frontend/AGENTS.md` + root `AGENTS.md`. **Neither loads the other's domain-specific rules.**
- **Pattern-scoped instruction files** — `applyTo` frontmatter targets instructions by file glob pattern, achieving hierarchical specificity without requiring directory-level files (e.g., general Python rules at `applyTo: "**/*.py"`, stricter API-surface rules at `applyTo: "src/api/**/*.py"`, most-specific auth-module security rules at `applyTo: "src/api/auth/**/*.py"`). More specific patterns override or extend less specific ones — the auth module inherits general Python rules and general API rules, then adds its own security requirements.
- **Compilation for portability** — instruction files authored in tool-specific formats (`.instructions.md` with `applyTo`) can be *compiled* into hierarchical `AGENTS.md` files for universal portability. **The source of truth is the authored instructions; the compiled output is the portable delivery format** — this separation means the hierarchy works regardless of which AI coding tool the developer uses. Package managers for agent primitives (such as APM) automate the scaffolding and sharing of instruction hierarchies across repositories and teams — **the constraint (hierarchical context) drives the tooling, not the reverse.**

**Anti-pattern: Flat instructions.** A single instruction file at the project root containing every rule for every domain (Python type hints, React accessibility, JWT rotation, database migration conventions, CSS BEM naming, all in one `# Project Instructions` file). Every agent, regardless of what it's working on, loads all of this — the Python backend agent processes CSS naming conventions, the database migration agent processes React accessibility guidelines. **Attention is wasted; worse, rules from unrelated domains can interfere** — the agent might apply "use modules" CSS guidance to Python module organization, producing unexpected results. **Fix:** split by scope — Python rules in a Python-scoped file, React rules in a frontend-scoped file, auth rules in an auth-scoped file — each agent loads only what applies to its current task.

### When Constraints Are Missing: Three Failure Stories
The five constraints are defined independently but produce their strongest effects **in combination** — each pair closes a gap that neither constraint addresses alone. (Disclosed as fictional composites distilled from the author's consulting observations, distorted to protect confidentiality — illustrating real patterns, not specific engagements.)

1. **Hierarchy without Progressive Disclosure** — a fintech startup (12 engineers, 18 months into their product) built a meticulous instruction hierarchy: global rules at the root, domain rules per service, module-specific rules for payments and auth. Textbook Explicit Hierarchy. But every instruction file was *self-contained* — the auth instructions inlined the full OWASP token reference (1,400 words), the complete session management spec (900 words), and the rate-limiting policy (600 words). When an agent worked on a simple token-validation bugfix, it loaded **nearly 3,000 words of guidance for a 15-line change** — and "solved" the bug by also implementing an uninvited rate-limiting improvement that introduced a subtle race condition in the refresh flow. **The hierarchy got the right rules to the agent — it got all of them at once, and the agent could not distinguish the relevant from the adjacent.**

2. **Reduced Scope without Composition** — a platform team (8 engineers, 40+ microservices) was disciplined about task sizing: every agent session had a single focused objective, fresh context, clean scope. Textbook Reduced Scope. But they had no composition layer — each task carried its own copy of coding standards, error-handling patterns, and testing requirements, pasted into the prompt rather than referenced from shared context files. When the team updated their error format from string messages to structured error objects, they updated the canonical doc but **missed the pasted copies in four prompt templates** — for two weeks, agents working on different services produced two different error formats depending on which prompt they received. **The scope constraint kept each task reliable in isolation; without composition, there was no single source of truth to keep them consistent with each other.**

3. **Safety Boundaries without Reduced Scope** — an enterprise team (200+ engineers, regulated healthcare domain) enforced strict boundaries: tool whitelists per agent, mandatory approval gates for security-sensitive files, knowledge scoping via `applyTo`. Textbook Safety Boundaries. But they routinely dispatched broad tasks — "implement the full OAuth2 integration," spanning token exchange, PKCE flow, session management, and error handling in one session. The agent operated within its tool boundaries but **accumulated so much context over the long session that its attention degraded.** Late in the session it generated a refresh-token endpoint storing tokens in a client-accessible cookie — a pattern explicitly forbidden by the security instructions loaded at session start, now buried under 40,000 tokens of accumulated implementation context. **The safety boundaries constrained what the agent *could* do; without reduced scope, they could not ensure the agent still *remembered* what it should not do.**

**The general pattern:** each constraint addresses one failure mode, and each partner constraint catches the failures the first one cannot see (visualized in the book's Figure 10.1, "Relationships between the five PROSE constraints": **O**rchestrated Composition provides the composable units that enable **P**rogressive Disclosure and is itself the single source of truth that constrains **R**educed Scope; **P** also controls attention into **E**xplicit Hierarchy and loads what fits **R**'s scope; **E** defines the boundaries that **S**afety Boundaries enforces, while **S** in turn constrains **R**'s safety scope).

### Applying the Constraints: A Worked Example
A team adding JWT authentication to an existing Express.js application. Without PROSE constraints this is one task given to one agent with all project documentation loaded. With PROSE constraints, the same work produces **five focused sessions**, each with explicit boundaries and purpose-built context.

**The instruction hierarchy:**
- **Root instructions** (`AGENTS.md`) — global rules every agent inherits: TypeScript strict mode, Conventional Commits format, JSDoc requirements, plus links to `[error taxonomy and response formatting](./docs/errors.md)` and `[testing strategy and fixture conventions](./docs/testing.md)`.
- **Backend instructions** (`backend/AGENTS.md`, inherits root) — API-layer rules: Express handlers must use the `asyncHandler` wrapper, request bodies validated with Zod, database access only through repository classes, link to `[API design patterns and pagination](./docs/api-design.md)`.
- **Auth module instructions** (`.github/instructions/auth.instructions.md`, `applyTo: "src/auth/**"`) — the security-specific rules: token standards (RS256 signing, never symmetric HS256 for access tokens, 15-minute access TTL / 7-day refresh TTL, single-use rotating refresh tokens), security constraints (never log token values — only `jti` claim IDs; never store tokens in `localStorage`; rate limit `/auth/refresh` to 10 requests/minute/user), links to `[OWASP token best practices](../../docs/security/owasp-tokens.md)` and `[existing session management](../../src/services/session.ts)`, and a **Validation Gate** requiring the agent to present proposed changes with security rationale and **STOP** for explicit approval before modifying token-signing or refresh-rotation logic.

An agent editing `src/auth/middleware.ts` loads this file plus the backend and root instructions. An agent editing `src/billing/invoice.ts` never sees it.

**The agent configuration** — the backend implementation agent is scoped to write code in its domain and run tests, nothing else: `tools: ["editFiles", "runCommands", "search", "readFiles", "testRunner"]`; explicit boundaries state it may create/edit files only under `src/auth/` and `tests/auth/`, may read any file for reference, may run tests filtered to `auth`, **must NOT** modify files outside its scope, touch CI/CD or deployment config, or install new dependencies without presenting the rationale first; workflow: read task + relevant files, check active instructions, implement, run and verify tests, **STOP** and present a summary for human review.

**The task decomposition** — five sessions, each with a fresh context window:

| Session | Task | Key context loaded | Deliverable |
|---|---|---|---|
| 1 | Design token schema and Zod validation types | Auth instructions, existing user model, JWT library docs | `src/auth/schemas.ts` |
| 2 | Implement auth middleware (verify + decode) | Auth instructions, token schemas from Session 1, Express middleware patterns | `src/auth/middleware.ts` |
| 3 | Build refresh endpoint with token rotation | Auth instructions, token schemas, session service reference | `src/auth/routes/refresh.ts` |
| 4 | Write integration tests for auth flow | Auth instructions, test conventions, all auth source files | `tests/auth/` test suite |
| 5 | Update login form to use new auth endpoints | Frontend instructions (different agent), API contract from Sessions 2–3 | `src/components/LoginForm.tsx` |

Session 5 uses a **different agent** (frontend developer) with different instructions, different tools, and no access to `src/auth/` internals — it receives only the API contract (endpoint URLs, request/response shapes), not the implementation details.

**Where a constraint prevented a failure:** during Session 2, the agent implementing auth middleware — following patterns it had seen in the codebase — started to also update the frontend API client to include new auth headers. **The safety boundary stopped it**: its tool whitelist restricted file edits to `src/auth/` and `tests/auth/`; it could not modify `src/api/client.ts`. The agent reported the suggested frontend change in its summary, and the team addressed it in Session 5 with the frontend agent that had the right context. Without the boundary, the backend agent would have modified the API client using only backend context — missing the frontend's existing interceptor pattern, retry logic, and error-handling conventions that the frontend instructions specify. **The fix would have "worked" in isolation and broken the frontend's error recovery flow.**

### Compliance Checklist
A yes/no checklist for evaluating whether a current setup satisfies the PROSE constraints. Each question is specific enough to answer definitively.

| # | Constraint | Question |
|---|---|---|
| P1 | Progressive Disclosure | Does every instruction file over 100 lines use links (not inline content) for subsidiary topics? |
| P2 | Progressive Disclosure | Does every cross-reference link include a description of what the target contains (not just a filename)? |
| R1 | Reduced Scope | Can you state each agent task in one sentence with a single deliverable? |
| R2 | Reduced Scope | Do multi-step workflows start each phase with a fresh context (no accumulated session state)? |
| O1 | Orchestrated Composition | Does each instruction file address exactly one concern (check: could you name the file after its single topic)? |
| O2 | Orchestrated Composition | Do workflow prompts reference shared instruction files by link rather than pasting their content? |
| S1 | Safety Boundaries | Does every agent configuration have an explicit `tools` list (no wildcards)? |
| S2 | Safety Boundaries | Is there a **STOP** gate before every operation that modifies auth, database schemas, or production config? |
| S3 | Safety Boundaries | Does every agent have a file-path boundary (explicit list of directories it may modify)? |
| E1 | Explicit Hierarchy | Do instructions exist at three or more specificity levels (e.g., root → domain → module)? |
| E2 | Explicit Hierarchy | Can you add module-specific rules without editing any file above that module's scope? |

**Prioritization for fixing gaps:**
- **If you fail S1, S2, or S3** — start there regardless of total score. Safety gaps have the highest blast radius and are the fastest to close (one YAML change per agent).
- **If you fail E1 or E2** — address these next. Hierarchy is the fastest architectural change to implement (create two scoped files and you have three levels).
- **If you fail P1, P2, R1, R2, O1, or O2** — these are discipline and refactoring issues. Prioritize whichever constraint maps to the failure mode you are currently experiencing: scope creep → fix R1/R2; context dilution → fix P1/P2; debugging nightmares → fix O1/O2.

The chapter closes by framing the five constraints as **the specification**; the chapters that follow (context engineering in Chapter 11, multi-agent orchestration in Chapter 12, the execution meta-process in Chapter 13) are **the implementation**. Every technique in those chapters traces back to one or more constraints defined here — when a technique works, it is because it satisfies the relevant constraint; when it fails, the constraint it violates tells you where to look.

---

## Important

### One Implementation of PROSE Scaffolding
`pip3 install apm-cli && apm init` — the author's own open-source APM tool scaffolds PROSE-compliant primitives directly. The primitives in this chapter can equally be created manually following the file templates given throughout (instruction files with `applyTo` frontmatter, agent configs with explicit `tools` lists, `AGENTS.md` hierarchies, etc.) — the tooling is a shortcut, not a requirement.

### The Footnote on Specification vs. Formal Methods
A closing footnote draws an explicit parallel: the shift from human-readable to machine-readable specifications mirrors the historical evolution from informal requirements documents to formal methods in safety-critical systems engineering. The stated difference is that agentic specifications must be **both** — readable by the human reviewer **and** parseable by the agent consumer — simultaneously, which is part of why file format and structure (frontmatter, glob scoping, link-based references) matter as much as content.

---

## Key Terms

- **PROSE** — Progressive Disclosure, Reduced Scope, Orchestrated Composition, Safety Boundaries, Explicit Hierarchy; the five architectural constraints specified in full in this chapter.
- **Progressive Disclosure (P)** — context loads just-in-time rather than just-in-case; implemented via markdown links as lazy-loading pointers with descriptive labels.
- **Reduced Scope (R)** — task size is matched to context capacity; sized correctly when the agent can finish without asking a follow-up question.
- **Orchestrated Composition (O)** — complex behavior built from small, single-responsibility, independently debuggable primitives rather than monolithic mega-prompts.
- **Safety Boundaries (S)** — explicit limits on an agent's capability (tools), knowledge (context scope), and authority (human-approval gates); constrains blast radius, not usefulness.
- **Explicit Hierarchy (E)** — instructions resolve from most-specific to most-general scope (e.g., `AGENTS.md` directory walking, `applyTo` pattern specificity); local context inherits from and may extend global context.
- **Validation gate** — an instruction-embedded **STOP** requirement forcing an agent to present a plan and wait for explicit human approval before a high-risk action.
- **Silent semantic failure** — a plausible, documentation-consistent, but wrong inference an agent makes that no linter or naive test catches (referenced here, named fully as Anti-Pattern #6 in Chapter 14).
- **Compliance checklist** — the eleven yes/no questions (P1–P2, R1–R2, O1–O2, S1–S3, E1–E2) for auditing whether an existing setup satisfies PROSE, prioritized Safety > Hierarchy > the rest.
