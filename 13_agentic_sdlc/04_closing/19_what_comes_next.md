# Chapter 19 — What Comes Next

> Source: *The Agentic SDLC Handbook* (Daniel Meppiel, v0.9.2), pp. 225–231.
> The closing chapter applies the book's own "three-tier honesty" framework (available now / emerging / directional) to predict the field's trajectory, then turns it back on the book itself — including admitting where its central assumption is most likely wrong.

---

## Very Important

### The Three-Horizon Timeline
The chapter's central structuring device — every prediction is bucketed by *how sure the author is*, not just by *when it will happen*.

| Horizon | Window | Claims |
|---|---|---|
| **Near-term** | 0–12 months | Tool-using agents become the default interaction mode; multi-agent orchestration moves from research to mainstream platforms; the agentic computing stack crystallizes through independent convergence |
| **Medium-term** | 1–3 years | Agent governance becomes a first-class engineering discipline; the "writing code" / "describing intent" boundary blurs — specification becomes the core skill |
| **Long-term** | 3–5 years | Full lifecycle agent participation (requirements → design → code → test → review → deploy → operate → iterate) becomes achievable; context infrastructure becomes as foundational as CI/CD |

**Independent convergence as a strategic signal:** by mid-2025, at least three uncoordinated efforts — Anthropic's `plugin.json`, GitHub's Agentic Workflows, and open-source frameworks like Squad and Spec-Kit — arrived at the same layered architecture (manifest-based primitive distribution + framework-layer composition + CI/CD-native execution). The book treats this the way HTTP → REST → Rails/Express → npm/pip → applications emerged: *each layer arrives when practitioners need it, not when a standards body decrees it.* When independent implementations converge on the same architecture, the architecture is real — which is the book's argument for why organizations should invest in the **primitive layer (Chapter 9)** now: it's the layer most likely to stay stable as the framework layer above it keeps changing.

### What Will Not Change (The Structural Claims)
The five claims the author is *most* confident about, specifically because they're argued to be properties of the problem, not of current technology — and they map directly back to the five PROSE constraints from Chapter 1:
1. **Context will remain finite and fragile** — there will always be a limit to how much information an agent can effectively consider; this is a property of the problem, not the current model generation.
2. **Output will remain probabilistic** — models will get better, not deterministic; reliability must be architected through constraints and validation, never assumed from model quality.
3. **Explicit knowledge will remain more valuable than implicit knowledge** — agents don't read minds; organizations that externalize knowledge outperform those that don't.
4. **Human judgment will remain the bottleneck and the differentiator** — the scarce resource is defining what should be built, evaluating whether it was built correctly, and deciding what to do when it wasn't.
5. **Composition will remain necessary** — no single agent will hold an entire large system in focus; tools for composition will improve, but the need for it won't diminish.

### Three-Tier Honesty Applied to the Chapter's Own Claims
The book's signature intellectual-honesty device, applied reflexively to its own predictions — worth internalizing as a model for evaluating *any* claim in the book, not just this chapter's:

| Claim | Tier | Confidence |
|---|---|---|
| Tool-using agents become the default interaction mode | Available now | High — shipping in multiple platforms |
| Multi-agent orchestration enters mainstream tooling | Available now | High — shipping in multiple tools |
| Agent governance becomes a distinct discipline | Emerging | Medium — need is clear, tooling is not |
| Specification replaces implementation as the core skill | Emerging | Medium — direction clear, timeline uncertain |
| Agentic computing stack layers consolidate | Emerging | Medium — convergence visible, standardization not |
| Full lifecycle agent coverage becomes operational | Directional | Low-to-medium — plausible, not inevitable |
| Context infrastructure becomes as foundational as CI/CD | Directional | Medium — trajectory clear, timeline 5+ years |
| The five core constraints hold | Structural | High — properties of the problem |

**Calibration guidance:** invest confidently in the "available now" tier, prepare for "emerging," stay aware of "directional" without betting the organization on its specific timeline.

### When NOT to Use Agentic Workflows
A direct, load-bearing counter-balance to the rest of the book — four scenarios where applying the methodology actively makes things worse, not better:
- **The task requires fewer than 50 lines of change** — if you can hold the full scope in your head, persona design, wave planning, and checkpoint discipline aren't worth the overhead. Just write the code.
- **The domain knowledge is entirely implicit** — if conventions/constraints/trade-offs can't be externalized (political context, unwritten relationships, organizational history resistant to documentation), agents produce plausible but wrong output. Instrument the codebase first (Chapter 9), *then* apply agents.
- **The cost of failure is low and iteration is cheap** — for throwaway scripts, prototyping, exploratory work, a single unorchestrated prompt is faster and sufficient. The methodology exists for production-grade work where reliability matters.
- **The work is inherently sequential and creative** — naming things, choosing abstractions, defining API contracts are judgment-dense tasks where agent suggestions help but orchestrated composition adds nothing. Use agents as sounding boards, not orchestrated fleets.

The book adds a fifth, more concrete warning drawn from its own case study: **the platform fights automation** — the Growth Engine case study (Ch18) documents three automated approaches to a UI automation problem, each hitting the same wall (an undocumented, hostile platform internal). When a platform's internals are undocumented and hostile to manipulation, the right move is to escalate to a human with a precise checklist rather than attempting a fourth automated approach.

### What the Author Probably Got Wrong
The book's own pre-emptive self-critique — five named risks to its central assumptions, presented as genuine uncertainty rather than hedging:
1. **The pace of capability improvement may outrun governance** — the book assumes organizations have time to build governance infrastructure before agent capabilities demand it; historically, capability has outrun organizational maturity for every technology shift.
2. **The emphasis on human-in-the-loop may prove too conservative** — for internal tooling/prototyping/throwaway infrastructure, fully autonomous workflows may become practical sooner than the book suggests; "always review" is safer but may leave efficiency on the table where the cost of failure is low.
3. **The multi-agent orchestration model may evolve past human orchestrators** — the book assumes a human planner dispatching specialist agents; future orchestration may involve agents that plan their own decomposition and negotiate resources, making the human-as-orchestrator model a transitional pattern rather than an enduring one.
4. **The documentation burden may not pay for itself** — if productivity gains from agentic development are modest (15–20%† rather than the 2–3x some vendors claim), the time spent creating and maintaining context infrastructure could consume most of the gains; the break-even calculation is less obviously favorable than the book implies.
5. **(The uncomfortable one) The author may be overestimating the durability of human judgment as the differentiator** — the entire methodology is built around the assumption that human judgment is a structural moat. The author flags the motivated-reasoning risk explicitly: if models develop genuine architectural reasoning rather than pattern-matching, the "human judgment moat" may be temporal, not structural — and this is "exactly the kind of assumption that deserves the most scrutiny and the least certainty."

---

## Important

### Near-Term Detail: Safety Boundaries Become More Critical, Not Less
As tool-using agents (file operations, terminal commands, API calls, test runs) become the default rather than text-generation-only interaction, the risk profile inverts: a model that generates bad *text* wastes review time, but a model that *executes* bad commands corrupts state. Guardrails that felt conservative in a text-generation world become essential in a tool-execution world.

### Medium-Term Detail: Agent Governance as an Emerging Discipline
Today, governance of agent output rides on existing processes (PRs, CI, manual approval) — workable at current volumes. As output scales and multi-agent orchestration becomes common, the book predicts dedicated infrastructure will emerge: audit trails for agent decisions, policy engines enforcing constraints *at execution time* rather than review time, and cost controls managing token spend across teams — the same way CI/CD became its own category over the past decade.

### Your First Week: What to Do Starting Monday
A concrete, five-day action plan bridging both audiences (practitioners run Days 1–5; leaders additionally run a parallel readiness assessment):

| Day | Action | Deliverable |
|---|---|---|
| 1 | Audit your team's most-frequently-changed module (not the biggest one) using the Ch9 methodology — find implicit knowledge an agent would violate | A list of 5–10 implicit conventions an agent would break on its first task |
| 2 | Write instruction files for the top three conventions from Day 1's audit (one organizational standard, one architectural constraint, one domain rule), per Ch9/Ch10 format | Three instruction files committed to the repo |
| 3 | Run a real sprint task twice — once without the new context files, once with them | A before-and-after comparison with specific examples |
| 4 | Review Day 3's comparison honestly; revise files that didn't land — this is the Ch11 calibration loop | Revised instruction files based on observed agent behavior |
| 5 | Show the team a 15-minute before-and-after demo at standup, then plan which modules get instrumented next and who owns which files | Team agreement on next steps and ownership |
| Leaders, additionally | Start with the Ch7 readiness assessment; fund a structured pilot (not blanket licenses); protect investment in context infrastructure — it has the highest long-term return and lowest short-term visibility, which is exactly why it's the most likely line item to get cut | — |

### The Closing Argument
Explicit parallel to REST's actual historical arc: *REST did not make HTTP better — it gave engineers constraints to reason about distributed systems. Twenty-five years later the constraints still hold even though every specific technology from that era has been replaced.* The book's stated aspiration for PROSE is the same: durable reasoning tools for a field that won't stop changing. Closing line: **"The methodology is the floor, not the ceiling. Build on it."**

---

## Key Terms

- **Three-tier honesty framework** — the book's own claim-classification system: *available now* (shipping, verifiable), *emerging* (clear need, immature tooling), *directional* (the author's opinion about where the field is heading, explicitly not a forecast).
- **Three-horizon timeline** — near-term (0–12 mo), medium-term (1–3 yr), long-term (3–5 yr) bucketing used to structure every prediction in this chapter.
- **Agentic computing stack** — the layered architecture (manifest-based primitive distribution, framework-layer composition, CI/CD-native execution) that the author argues is crystallizing through independent, uncoordinated convergence across vendors (first introduced in Chapter 4).
- **Context infrastructure** — the context files, instruction hierarchies, and knowledge bases that make agents effective; predicted to become as foundational as CI/CD within 3–5 years.
