# Chapter 6 — Team Structures for AI-Augmented Delivery

> Source: *The Agentic SDLC Handbook* (Daniel Meppiel, v0.9.2), pp. 71–81.
> Agentic development does not eliminate engineering roles — it restructures how teams coordinate, what skills matter, and where humans add irreplaceable value, and this chapter is the organizational framework for designing that shift deliberately rather than letting it happen by accident.

---

## Very Important

### What Shifts, What Stays — The Time Allocation Table
Agent adoption shifts the *proportion* of time engineers spend on different activities; it does not change which activities require human judgment. The chapter's central evidence table (composite from early enterprise adopters and the author's observations; pre-agentic ranges draw on industry surveys 2019–2023, agentic-era figures marked **†** are projections, not survey data):

| Activity | Pre-Agentic | With Agentic Tools (Projected †) | Direction |
|---|---|---|---|
| Writing new code | 30–35% | 10–15% † | Sharply down — agents handle first-draft generation |
| Reading and understanding code | 20–25% | 15–20% † | Slightly down — agents assist navigation and explanation |
| Code review | 10–15% | 20–25% † | Up — early adopter teams report that reviewing agent output is qualitatively harder |
| Specification and design | 10–15% | 20–25% † | Up — better specs produce better agent output |
| Debugging and incident response | 15–20% | 10–15% † | Slightly down, but the bugs that remain are subtler |
| Context engineering | 0% | 10–15% † | New — building and maintaining the context layer |
| Meetings and coordination | 10–15% | 10–15% | Roughly unchanged |

Three patterns matter:
1. **Code production shrinks as a share of engineering work.** Agentic tools accelerate the part of an engineer's time that was already a minority. This is why "10x faster coding" does not translate to "10x faster delivery" — the bottleneck moves to review, specification, and design.
2. **Review becomes harder, not easier.** Agent-generated code is the output of a statistical process — syntactically correct, test-passing, and potentially hiding subtle misunderstandings of intent (Chapter 5's governance lens). The team implication: review skill becomes a core competency, not a chore to minimize.
3. **Context engineering appears as a new activity category.** Someone has to build and maintain the instructions, conventions, and agent configurations that determine whether agents produce useful output or confident nonsense.

### The 10x Team, Not the 10x Developer
The "10x developer" idea has persisted for decades. Agentic development makes it obsolete — not because individual skill stops mattering, but because **the leverage point shifts from individual output to team capability**. An engineer working alone with an AI agent can produce code faster, but output quality depends on the context the agent received, which depends on the team's documentation discipline, which depends on the organization's investment in context engineering. A brilliant developer with poor team context will get worse results than a competent developer with excellent team context — **the multiplier is in the system, not the individual.**

Teams that extract the most value from agentic tools share four properties:
- **Explicit knowledge.** Conventions documented, not tribal. Architecture decisions recorded, not remembered. The context layer (Chapter 4) is built and maintained as an organizational asset.
- **Strong review culture.** Reviewing agent output is a skill. Teams where review is valued, structured, and staffed appropriately catch defects individual developers miss; teams where review is a speed bump produce fragile systems.
- **Clear specification habits.** Agent output quality correlates strongly with specification quality. Teams that invest in clear, scoped specifications get better agent results than teams that write vague tickets and expect agents to fill the gaps.
- **Feedback loops.** Teams that capture what went wrong (which agent outputs required rework, which context gaps caused failures) and feed corrections back into the context layer improve continuously. Teams that treat each agent interaction as independent repeat the same failures.

The practical implication for leaders: **invest in team capability, not individual heroics.** A team of solid engineers with a well-maintained context layer will outperform a team of exceptional engineers working in a knowledge vacuum.

### How Roles Evolve
Agentic development does not create a clean break between "before" and "after" roles — it shifts the emphasis within existing roles.

- **Senior engineers: from implementers to context architects.** A senior engineer's value was always more about judgment than typing speed; agentic tools make this explicit. Senior engineers in an agentic team spend less time writing code and more on: **architecture and design** (defining the boundaries, patterns, and constraints agents must respect — always part of the role, now a larger share); **context engineering** (translating architectural knowledge into explicit artifacts — instructions, conventions, agent configurations — that encode the team's judgment for agent consumption; "the new craft skill for senior engineers"); **review and escalation** (evaluating agent output against architectural intent, catching subtle violations that pass automated checks, handling the cases agents cannot — these cases are harder, not fewer); and **mentoring** (teaching juniors how to evaluate agent output, write effective specifications, and build judgment agents cannot replace). The shift is from "the person who writes the hardest code" to "the person who shapes the system that writes the code." Senior engineers who resist this and insist on writing everything themselves become a bottleneck; those who embrace it become force multipliers.
- **Junior engineers: from code writers to AI-augmented contributors — the most sensitive shift.** If agents handle the tasks juniors used to learn on (simple bug fixes, small features, boilerplate), the learning pathway narrows. The solution is not banning agents from junior workflows — it is redesigning how juniors develop skills. A junior in an agentic team should be: **reviewing agent output, not just writing code** (review builds the same judgment as writing, arguably faster, because the reviewer sees more patterns in less time — structured review assignments under senior supervision are "the most efficient skill-building tool available"); **writing specifications for agent tasks** (forces the junior to think through the problem before code exists); **diagnosing agent failures** (understanding *why* output is wrong builds deeper understanding than writing correct code from scratch — e.g., "the agent violated the repository pattern because it doesn't understand our dependency injection setup" teaches architecture); and **building and maintaining context artifacts** (contributing to documentation and conventions requires understanding the codebase at a level that builds genuine expertise). See "The Junior Pipeline" below for structured models.
- **Tech leads: from task assigners to orchestrators.** The role shifts from assigning tasks to people toward orchestrating work across humans and agents — deciding which tasks suit agent execution, which require human implementation, and which need a hybrid approach. This judgment is substantial: a task that looks simple may touch code paths agents consistently mishandle, and a task that looks complex may decompose into agent-friendly subtasks. Tech leads also become the primary feedback-loop owners, tracking which context gaps cause repeated agent failures and prioritizing context improvements.

### Team Topologies That Work — and Don't
The chapter applies **Team Topologies** (the framework by Matthew Skelton and Manuel Pais) as a lens for how agentic development changes team structures.

**What works:**
- **Stream-aligned teams with embedded context engineering** — the most effective and simplest pattern. Existing product teams add context engineering to their responsibilities; the team that owns the code also owns the context layer for that code. This preserves domain knowledge proximity — the people who understand the system best encode that understanding for agents. Context engineering is a team responsibility, not a separate function.
- **A platform team that provides shared context infrastructure.** Organization-wide conventions, common patterns, and cross-cutting context assets (authentication patterns, logging standards, API design guidelines) are better maintained centrally than duplicated across teams. A platform team (or developer experience team) owns these shared assets; stream-aligned teams consume them and add domain-specific context on top. This mirrors the **Explicit Hierarchy** constraint from Chapter 1: global rules flow down, domain specifics stay local.
- **Enabling teams for adoption support.** During the transition period (which Chapter 7 plans in detail), a small enabling team that coaches other teams through adoption, maintains best-practices documentation, and provides hands-on support is the most effective accelerant. This team has a **finite lifespan** — once adoption is mature, its responsibilities fold into the platform team or dissolve.

**What doesn't work:**
- **A centralized "AI team" that handles all agent interactions.** A specialized group becomes the bottleneck for all agent work — a coordination tax that eliminates the speed advantage, plus a knowledge gap because the AI team doesn't understand each product domain deeply enough to write good context.
- **Splitting "human code" and "agent code" into separate workflows.** Parallel tracks where human engineers handle "important" code and agents handle "routine" code fail because the boundary between important and routine is not stable, agent code still requires human review and integration, and it creates a two-class system that undermines team cohesion. **All code is the team's code, regardless of who or what produced it.**
- **Replacing team roles with agents.** Reducing headcount on the assumption agents will cover the gap. Teams that lose senior engineers because "the AI can do that now" lose the judgment required to evaluate agent output and maintain architectural coherence — the result is a team that produces more code and less working software.

### The Junior Pipeline — Three Apprenticeship Models
If agents handle the tasks that traditionally built junior engineering skills, how do juniors develop? The book explicitly flags these as **informed hypotheses, not proven patterns** — no organization has run any of these for a full cycle (12+ months) with measured outcomes; they draw on early signals, apprenticeship research from adjacent fields, and structured reasoning about which skill-building mechanisms transfer to an agent-augmented environment.

| Model | Approach | What It Builds | Risk |
|---|---|---|---|
| **A: Review-intensive apprenticeship** | Juniors spend 60–70% of their first year reviewing agent-generated code under senior supervision; they write code only for tasks specifically selected to build skills review alone cannot develop. | Pattern recognition, failure-mode understanding, architectural awareness. | Can feel passive; requires disciplined senior oversight and deliberate hands-on coding assignments. |
| **B: Agent-assisted learning with scaffolded complexity** | Juniors use agents as learning tools — generating solutions, then analyzing and improving the output. Tasks start simple and increase in complexity, with the senior engineer designing the progression; early tasks have clear right answers, later tasks require trade-off analysis agents cannot resolve. | Critical evaluation skill. | Without structure, juniors accept agent output uncritically — the scaffolding must actually exist, not just be assumed. |
| **C: Specification-first roles** | Juniors focus upstream: writing specifications, defining acceptance criteria, decomposing requirements into agent-friendly tasks. Code review and debugging responsibilities increase over time. | Specification and design discipline — increasingly valuable, and produces real team output immediately. | Delays hands-on coding experience; some skills require building things, not just specifying them. |

No model is sufficient alone — **Model A builds judgment, Model B builds critical evaluation, Model C builds specification discipline.** A structured first year combines all three, with proportions shifting as the junior's capability grows. Whether this combination produces engineers as capable as those trained through traditional paths is an explicitly open question: "we don't know yet. Plan for these models, measure rigorously, and adjust."

---

## Important

### Skill Matrix Evolution
The skills that differentiate engineers are shifting, with hiring, retention, and development implications.

| Skill | Pre-Agentic Value | Agentic Value | Direction |
|---|---|---|---|
| Syntax and language fluency | High — daily necessity | Low — agents handle this | Declining |
| Algorithm and data structure mastery | Medium — interviews, specific domains | Low to medium — agents implement known algorithms | Declining for implementation, stable for design |
| System design and architecture | High | Very high — the primary human differentiator | Increasing |
| Code review and evaluation | Medium — supporting skill | High — core daily activity | Increasing |
| Technical writing and specification | Low to medium — often neglected | High — specification quality drives agent output quality | Sharply increasing |
| Context engineering | Did not exist | High — new foundational skill | New |
| Debugging and root cause analysis | High | High — agent-generated bugs are subtler | Stable, but harder |
| Domain knowledge | High | Very high — agents cannot learn what is not documented | Increasing |
| Collaboration and communication | Medium | High — coordination with agents adds a new dimension | Increasing |

**Hiring implications.** Screen for: systems thinking, technical communication (can the candidate explain a design decision in writing, not just verbally?), evaluation skill (can they identify subtle flaws in code they didn't write?), comfort with ambiguity, and learning velocity. Stop requiring: whiteboard algorithm implementation, syntax trivia, memorized API knowledge — these were always imperfect proxies for engineering capability and are now increasingly poor ones, because agents eliminate the tasks they supposedly measure. Interview changes: include a **review exercise** (give candidates agent-generated code with subtle defects, evaluate how they identify and explain the problems) and a **specification exercise** (give candidates an ambiguous requirement, evaluate how they decompose it into a clear, implementable specification).

**Retention risks.** Two emerge during the transition: **senior engineers who feel deskilled** — engineers whose identity is tied to writing code may perceive agentic tools as devaluing their expertise, when the reality is the opposite (their judgment is more valuable than ever; only the *form* of their contribution changes) — address this directly by showing that context engineering and architectural guidance are expressions of the same expertise, applied differently. **Junior engineers who feel replaceable** — the "AI replacing developers" discourse lands hardest on the newest members of the profession; if an organization is not actively investing in junior development using the apprenticeship models above, junior engineers correctly conclude their growth path is unclear and leave. This is not just an empathy argument: "the seniors of 2030 are the juniors you invest in today."

### New Roles
Two roles emerge that did not exist before agentic development. Neither requires hiring new people in most cases — they are specializations existing team members grow into:
- **Context engineer.** Responsible for building, maintaining, and optimizing the context layer that shapes agent behavior. In small teams this is a hat worn by a senior engineer; in larger organizations it becomes a dedicated role, often within a platform or developer experience team. The output is not code — it is the knowledge infrastructure that makes agent-produced code reliable. Requires deep codebase knowledge, strong technical writing skills, and a systematic approach to testing whether context changes improve agent output.
- **Agent operations specialist.** In organizations running orchestrated SDLC workflows (agents participating in issue triage, code review, testing, or deployment), someone needs to monitor agent behavior, tune configurations, and manage cost and rate limits. This role overlaps platform engineering and SRE; it is emerging, not established, and most organizations will not need it until they reach Phase 4 maturity (Chapter 2).

Neither role should be created by fiat — both should emerge from demonstrated need. A small, stable context layer doesn't need a dedicated context engineer; no autonomous agent workflows means no need for an agent operations specialist. **Create the role when the work exists, not when the job title sounds innovative.**

### Staffing Models
Team size and composition change under agentic development. The direction is consistent: smaller teams, more senior in composition, with higher leverage per person (pre-agentic norms reflect Forsgren/Humble/Kim's *Accelerate* and Amazon's "two-pizza team" rule of ~6–8 people; agentic-era figures are projections from early-adopter reports, marked **†**).

| Team profile | Pre-Agentic | Agentic (Mature) † | Notes |
|---|---|---|---|
| Typical team size | 6–10 engineers | 4–7 engineers † | Fewer people, higher output per person |
| Senior-to-junior ratio | 1:2 to 1:3 | 1:1 to 2:1 † | More senior judgment required for review and context |
| Context engineering allocation | 0% | 10–20% of team capacity † | Ongoing investment, not a one-time cost |
| Review time allocation | 15–20% of team capacity | 25–35% of team capacity † | Agent output requires more review, not less |

Two caveats the book stresses: these ratios are **directional, not prescriptive** — they depend on codebase complexity, agent maturity, and domain risk (a payments system with strict regulatory requirements needs a higher senior ratio than internal tooling); and **smaller does not mean fewer total engineers** — the economic argument is not "we need fewer engineers," it is "we need the same or fewer engineers to do more, with a different mix of skills." The staffing question is about composition and capability, not reduction.

**Getting from here to there** — three transition paths, typically combined:
- **Path A: Hire senior, hold junior headcount.** Bias new hires toward senior profiles with architecture and review skills as the team grows or backfills attrition; the ratio shifts naturally over roughly 12–18 months. Trade-off: senior engineers are expensive and scarce — slow but low-disruption, best for teams with low attrition and stable headcount.
- **Path B: Accelerate high-potential juniors.** Identify juniors with strong systems thinking and learning velocity; give them structured context engineering responsibilities, senior-supervised review rotations, and explicit mentorship; reclassify based on demonstrated capability, not tenure. Trade-off: requires real mentorship investment from seniors (typically 10–15% of their time, per early-adopter estimates), and not every junior will make the transition.
- **Path C: Attrit and rebalance.** Don't backfill junior departures one-for-one; when a junior leaves, evaluate whether to refill at the same level or convert to a senior hire. Rebalances naturally over roughly 12–24 months. Trade-off: depends on attrition rates you cannot control — if attrition is low, the rebalance stalls.

Most organizations combine all three. The key is deliberateness: track the ratio quarterly, make hiring decisions that move toward the target, and communicate openly about where roles are heading. "The worst outcome is an accidental rebalance where juniors leave because they see no growth path and seniors burn out because they are covering the gap."

### Team Assessment Worksheet
A self-scoring tool (1–5 per dimension) to evaluate current team structure against the chapter's patterns — meant to identify specific gaps needing attention, not to generate an overall grade.

| Dimension | Question | Score (1–5) interpretation |
|---|---|---|
| Knowledge explicitness | What percentage of the team's working knowledge is documented vs. tribal? | 1 = almost all tribal; 5 = comprehensive docs |
| Review capability | Can the team review agent-generated code effectively — catching subtle architectural violations, not just syntax errors? | 1 = no experience; 5 = structured review process |
| Specification discipline | Do work items contain enough detail for an agent to produce useful output without extensive clarification? | 1 = vague tickets; 5 = clear, scoped specs |
| Senior presence | Is there sufficient senior judgment to evaluate agent output on every critical path? | 1 = no senior coverage; 5 = senior review on all critical work |
| Junior development | Does the team have a structured path for juniors to build engineering skills in an agentic environment? | 1 = no plan; 5 = active apprenticeship models |
| Context ownership | Is someone accountable for the quality of the team's context layer? | 1 = nobody; 5 = explicit ownership with maintenance |
| Feedback loops | Does the team systematically capture agent failures and feed corrections back into the context layer? | 1 = no feedback loop; 5 = weekly context improvement cycle |
| Psychological safety | Can team members admit when agent-assisted work fails, without blame? | 1 = blame culture; 5 = learning culture |

**Interpreting results:** scores of 1–2 on any dimension indicate a gap that will actively undermine agentic adoption — address before expanding agent usage; knowledge explicitness and senior presence are the two dimensions that unblock everything else. Scores of 3 indicate basic capability that works for pilot-level adoption but won't scale — plan to invest during the expansion phase (Chapter 7). Scores of 4–5 indicate readiness — these are the dimensions where the team can be a model for others. A pattern the author says is frequently observed in early assessments: **high marks on senior presence and psychological safety, low marks on knowledge explicitness and context ownership** — this is normal, and reflects teams with strong people and weak infrastructure, which is exactly what agentic development exposes. The worksheet is not a one-time exercise; reassess quarterly during the first year of agentic adoption, since the dimensions needing attention shift as the team matures (early focus on knowledge explicitness and review capability; later focus shifts to junior development and feedback loops).

---

## Key Terms

- **Context engineering** — the new activity category of building, maintaining, and optimizing the instructions, conventions, and agent configurations that shape agent output; appears at 0% pre-agentic, 10–15% projected with agentic tools.
- **10x team** — the chapter's reframe of the "10x developer" idea: the leverage point in agentic development is team capability (explicit knowledge, review culture, specification habits, feedback loops), not individual output.
- **Team Topologies** — Matthew Skelton and Manuel Pais's framework for team structure, applied here to classify which patterns (stream-aligned + embedded context engineering, platform team, enabling team) work for agentic adoption and which (centralized AI team, human/agent code split, role replacement) don't.
- **Context engineer** — emerging role responsible for the team's or organization's context layer; a specialization, not necessarily a new hire.
- **Agent operations specialist** — emerging role overlapping platform engineering/SRE, needed only at Phase 4 maturity for orchestrated agent workflows.
- **Junior pipeline problem** — the risk that agents absorbing tasks juniors used to learn on narrows the traditional skill-development pathway; addressed via three apprenticeship models (review-intensive, agent-assisted scaffolded learning, specification-first).
