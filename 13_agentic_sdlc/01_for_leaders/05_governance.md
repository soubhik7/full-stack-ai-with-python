# Chapter 5 — Governance for AI-Assisted Delivery

> Source: *The Agentic SDLC Handbook* (Daniel Meppiel, v0.9.2), pp. 57–70.
> Software governance has always assumed a human at every decision point; this chapter extends that governance — rather than replacing it — to cover AI agents that write code, open pull requests, and trigger deployments.

---

## Very Important

### The Governance Gap
Compliance frameworks (SOC 2, ISO 27001:2013/2022, PCI DSS) all require demonstrating that authorized individuals made deliberate choices about what code runs in production. AI agents break that assumption: an agent that writes code, opens a PR, responds to review feedback, and triggers a deployment pipeline is a **participant** in the SDLC, not a tool like a linter or compiler — those produce deterministic output from deterministic input, while an agent interprets instructions, makes judgment calls, and produces different output from the same input on different runs.

The gap shows up in three places:
- **Audit trails end at the human-agent boundary.** Version control records that a developer authored a commit — not that the developer delegated the work to an agent, what instructions it received, what context it consumed, what alternatives it considered and rejected, or how much of the final code the developer actually reviewed versus rubber-stamped.
- **Approval workflows assume reviewers understand the code.** Agent code is the output of a statistical process optimizing for statistically likely token sequences, which correlates with plausibility but does not guarantee correctness. It can be syntactically correct, pass all tests, and still contain subtle misunderstandings of intent. Review processes designed for human code are necessary but insufficient for agent code.
- **Security boundaries were designed for human threat models.** Data classification, network access rules, and secret management assume a human employee constrained by training, judgment, and legal accountability. An agent with access to your codebase, CI/CD pipeline, and cloud credentials operates under whatever constraints you explicitly configure — what you don't restrict, the agent will eventually touch.

None of this means agents are ungovernable. It means the existing governance framework needs **extension, not replacement**.

### Governance Readiness Checklist
Governance for AI-assisted delivery spans six capability areas, each on a maturity spectrum from None → Basic → Enterprise. The checklist is designed for self-assessment: locate your organization on each row, then prioritize the gaps that carry the most risk in your context.

| # | Capability | None | Basic | Enterprise |
|---|---|---|---|---|
| 1 | **Audit trails** | No record of which code was agent-generated. Commits attributed to the prompting developer with no distinction. | Agent contributions tagged in commit metadata or PR labels. Prompt history retained for a defined period. | Full provenance chain: instruction given, context consumed, output produced, human review decision, and rationale — all queryable and linked to compliance artifacts. |
| 2 | **Agent access controls** | Agents run with the developer's full credentials. No distinction between human and agent access scope. | Agents operate under scoped tokens with reduced permissions. File system and network access restricted to declared boundaries. | Least-privilege agent identities with per-task credential issuance, automatic expiration, and separate audit logging for agent actions. |
| 3 | **Approval workflows** | Standard code review applies identically to human and agent code. No additional scrutiny for agent output. | Agent-generated PRs are flagged for enhanced review. Critical paths (auth, payments, data access) require human sign-off regardless of author. | Risk-tiered review: agent output touching sensitive systems routed through security-aware reviewers with checklist-based verification. Approval latency tracked as a metric. |
| 4 | **Data boundary enforcement** | No controls on what data agents can access during code generation. Proprietary code, secrets, and customer data may enter agent context. | Agents restricted from accessing production data and secrets. Code sent to external models reviewed against data classification policy. | Data loss prevention integrated into agent workflows. Context filters prevent classified data from entering model prompts. Residency requirements enforced per jurisdiction. |
| 5 | **Cost controls** | No visibility into agent-related compute or API spend. Costs absorbed into general cloud bills. | Per-team or per-project token budgets. Alerts on unusual consumption. Monthly cost reporting. | Real-time cost attribution per agent task. Automated circuit breakers on runaway sessions. Cost-per-feature tracking integrated into project planning. |
| 6 | **Compliance reporting** | Cannot demonstrate to an auditor how agent-generated code is governed. Compliance posture unknown. | Periodic manual reports on agent usage, access scope, and review rates. Policies documented but enforcement is process-dependent. | Automated compliance dashboards. Agent governance artifacts generated alongside code. Audit-ready evidence exportable on demand. Policy enforcement is systemic, not procedural. |

Most organizations at Phase 3 (agentic coding, per Chapter 2) will find themselves at "None" or "Basic" in at least four of these six areas — that is expected. The checklist's purpose is not to achieve "Enterprise" everywhere; it is to ensure you are not at "None" in any area that carries material risk for your business. **A note on the Enterprise column:** some of its requirements (full provenance chains, real-time cost attribution, automated compliance dashboards) exceed what current-generation tooling delivers out of the box as of mid-2025 — treat it as directional, not immediate. **Where to start:** audit trails and agent access controls are the two capabilities that unblock everything else; without knowing what agents did and limiting what they can do, the other four capabilities have no foundation.

### Compliance Framework Mapping
A checklist without regulatory context is a conversation starter, not a decision tool. The matrix maps each capability row to the compliance frameworks where it is critical, so leaders can prioritize based on regulatory scope.

| # | Capability | SOC 2 | ISO 27001 | PCI DSS | HIPAA | EU AI Act |
|---|---|---|---|---|---|---|
| 1 | Audit trails | **Critical** — CC8.1 (change management), CC7.2 (monitoring) | **Critical** — A.12.4 (logging and monitoring) | **Critical** — Req. 10 (track and monitor access) | **Critical** — §164.312(b) (audit controls) | **Critical** — Art. 12 (record-keeping) |
| 2 | Agent access controls | **Critical** — CC6.1, CC6.3 (logical access, least privilege) | **Critical** — A.9.2, A.9.4 (access management, access control) | **Critical** — Req. 7, Req. 8 (restrict access, identify users) | **Critical** — §164.312(a) (access control) | Relevant — Art. 14 (human oversight) |
| 3 | Approval workflows | **Critical** — CC8.1 (change management) | Relevant — A.14.2 (secure development) | **Critical** — Req. 6 (secure systems) | Relevant — §164.308(a)(5) (security awareness) | **Critical** — Art. 14 (human oversight of high-risk AI) |
| 4 | Data boundary enforcement | **Critical** — CC6.7 (data transmission), C1.1 (confidentiality) | **Critical** — A.13.2 (information transfer) | **Critical** — Req. 3, Req. 4 (protect stored data, encrypt transmission) | **Critical** — §164.312(e) (transmission security) | Relevant — Art. 10 (data governance) |
| 5 | Cost controls | Relevant — CC3.1 (risk assessment) | Relevant — A.12.1 (operational planning) | Not directly scoped | Not directly scoped | Not directly scoped |
| 6 | Compliance reporting | **Critical** — CC4.1 (monitoring activities) | **Critical** — A.18.2 (compliance review) | **Critical** — Req. 12 (security policy) | **Critical** — §164.308(a)(8) (evaluation) | **Critical** — Art. 13 (transparency) |

**How to read this:** if you are SOC 2-scoped, rows 1, 2, 3, 4, and 6 are critical — you will face audit findings if any of these are at "None." If you handle payment data under PCI DSS, rows 1, 2, 3, and 4 are your floor. If you ship to EU markets and your product touches high-risk categories, the EU AI Act makes rows 1, 3, and 6 non-negotiable. Start where your regulatory exposure intersects with your lowest maturity.

### Risk Taxonomy
Agent-introduced risk falls into six categories, each with specific mechanisms, concrete manifestations, mitigations, and identifiable owners. The taxonomy is not theoretical — these are risks organizations adopting agentic development are encountering now.

| Category | Risk | Example | Mitigation | Owner |
|---|---|---|---|---|
| **IP & data exposure** | Proprietary code sent to external model | Agent context includes auth module source; developer uses cloud-hosted model without enterprise data agreement | Enforce enterprise-tier agreements with training opt-out. Deploy context filters. Maintain data classification policy covering agent workflows. | Security / Legal |
| | Training data reproduced in output | Agent generates a near-exact copy of a GPL-licensed implementation, merged without review | Integrate license-scanning tools into CI. Flag agent-generated code for IP review in sensitive components. | Legal / Engineering |
| **Quality degradation** | Plausible incorrectness | Agent implements a data pipeline that passes all tests but silently drops null values the business logic depends on | Require property-based or invariant tests for agent-generated code in critical paths. Review output against ADRs. | Engineering leads |
| | Convention drift | Fifty agent-generated files use three different error-handling patterns; none match the team standard | Encode conventions as structured context (instruction files, linters, architectural rules) that agents consume during generation. | Tech leads / Architects |
| **Dependency & concentration** | Model outage | Primary model provider has a 4-hour outage during a release sprint; team cannot complete agent-assisted tasks | Maintain fallback model configurations. Ensure critical workflows degrade gracefully to human-only execution. Test fallback quarterly. | Platform / Engineering |
| | Vendor lock-in | Organization has 2,000 tool-specific instruction files; switching tools requires rewriting all of them | Use portable, vendor-neutral formats for context artifacts. Separate content from format. | Architecture / Platform |
| **Knowledge atrophy** | Debugging skill loss | Junior engineers cannot diagnose a production issue because they never debugged code without agent assistance | Require regular unassisted development exercises. Pair juniors with agent output for review practice. | Engineering managers |
| | Architectural reasoning decay | Team cannot redesign a subsystem because no one has practiced trade-off decisions outside agent-provided constraints | Rotate architecture review responsibilities. Include constraint-design tasks in sprint work. | Architecture / CTO |
| **Regulatory liability** | Implicit compliance violation | Agent generates a logging module that captures user IP addresses and geolocation where this requires explicit consent | Define compliance constraints as explicit agent context for regulated code paths. Require compliance-aware review. | Legal / Security |
| | Accountability gap | Regulator asks who decided to store customer data in a specific format; the decision was made by an agent in a 50-file PR | Maintain decision logs for agent-generated code in regulated areas. Include "compliance-relevant choices" in PR review checklists. | Engineering leads / Legal |
| **Supply chain & context integrity** | Prompt injection via dependency | A transitive dependency README includes hidden instructions causing the agent to exfiltrate environment variables | Restrict agent context to vetted, first-party sources for sensitive operations. Apply context sanitization. | Security / Platform |
| | Compromised instruction files | Attacker subtly modifies an agent instruction file via PR, causing generated auth code to include a backdoor pattern | Apply code review and change-management controls to instruction files with the same rigor as production code. | Security / Engineering leads |

Three of these six categories introduce genuinely novel failure modes worth understanding in depth:

- **Quality degradation — the silent failure mode.** A weak model fails obviously: the code doesn't work. A strong model with poor context fails insidiously: the code works, passes tests, and silently violates architectural invariants no test covers. **Plausible incorrectness** is code that reads well and compiles cleanly but misunderstands intent — for example, a function returning correct results for all test cases but with O(n²) complexity where O(n) was required, or a database query that's correct but bypasses the caching layer. **Hallucinated dependencies** are references to APIs or methods that don't exist or have been deprecated; when the hallucination happens to compile, the failure is deferred to production. **Convention drift** is code that works but doesn't belong — inconsistent error handling, non-standard logging, creative-but-wrong module structure — minor individually, but at scale it degrades the codebase coherence a team needs to navigate and modify code confidently.
- **Knowledge atrophy — the aviation parallel.** The least discussed and most consequential long-term risk: when agents handle tasks humans used to perform, humans get less practice at those tasks, and over months and years the team's collective ability to perform them unassisted erodes. This is not hypothetical — it follows patterns documented in aviation (pilots who rely on autopilot are measurably less proficient at manual flying, which is why the industry mandates manual-flying requirements) and financial analysis (analysts who rely on automated models are less able to identify model failures, which is why regulatory frameworks require human understanding, not just human approval). In software, the specific atrophy risks are **debugging skills** (if agents write the code and fix the bugs, juniors never develop debugging intuition), **architectural reasoning** (engineers get less practice reasoning about trade-offs outside agent-provided constraints), and **review depth** (if reviewers habitually approve agent code that passes tests, the skill of reading for intent atrophies). The mitigation is not to avoid agents — it is to design deliberate practice into the development process, the way aviation designs mandatory manual-flying requirements into pilot training.
- **Supply chain and context integrity — the new attack surface.** Agents consume context — instruction files, documentation, configuration, code from dependencies — and that context is an attack surface. **Prompt injection via context** means an agent reading repository files, fetching documentation, or consuming dependency metadata can be influenced by adversarial content planted in those sources; this is not speculative — prompt injection is an active area of security research and a documented attack vector against LLM-integrated systems. **Compromised instruction files** are especially dangerous because agent instruction files are code that governs code — if an attacker gains write access (compromised dependency, supply chain attack, malicious contribution), they can influence every line of agent-generated code without modifying a single source file.

A flagged organizational-policy note: AI-assisted governance has a limitation no model improvement will resolve — **organizational policy awareness lives nowhere in training data**. An agent can enforce coding standards from a rules file and run automated policy checks in CI, but it cannot know that the organization's legal team requires review for any feature touching PII, or that a PR linking a personal asset from a corporate repository creates a compliance risk — unless that policy is explicitly encoded in the context layer. This is why governance primitives (Chapter 9) must include organizational policies, not just technical standards. The book cites a "Growth Engine case study" where fifteen agent personas across seven expert panels missed a compliance constraint that a human caught in seconds.

### Regulatory Landscape
The chapter surveys frameworks most likely to affect engineering organizations using AI agents in production (explicitly not legal advice):

- **EU AI Act** — entered into force August 2024, phased enforcement through 2027. Classifies AI systems by risk tier; code-generating agents are not by default high-risk, but the software they produce may be. If agents generate code for systems the Act classifies as high-risk (medical devices, critical infrastructure, safety components), governance requirements extend to the development process, including how the code was generated. Key requirements: transparency obligations (users must know when interacting with AI), record-keeping requirements (logs of AI system behavior), human oversight provisions (meaningful human control over AI outputs).
- **SOC 2** — evaluates controls for security, availability, processing integrity, confidentiality, and privacy. If your organization undergoes SOC 2 audits, the auditor will eventually ask how AI-generated code changes are governed — "the question is when, not whether." Relevant controls span change management (how agent-generated changes are authorized and reviewed), access management (what systems and data agents can reach), and monitoring (how agent behavior is logged and reviewed).
- **Data residency** — model API calls transmit code to infrastructure operated by the model provider. For organizations subject to data residency requirements (GDPR, sector-specific rules, or contractual obligation), where agent context is processed matters. Most major providers offer regional deployment options at enterprise tiers; verify that tooling routes data through compliant infrastructure and document the verification.

A consolidated framework table:

| Framework | Relevance to agent-assisted development | Key requirement | Recommended posture |
|---|---|---|---|
| EU AI Act | Software built by agents may inherit risk classification of the deployed system. | Transparency, record-keeping, human oversight for high-risk applications. | Map your products to risk tiers. Evaluate whether your agent governance satisfies the tier's requirements. |
| SOC 2 | Auditors will ask about change management for agent-generated code. | Demonstrable controls for authorization, review, and monitoring of all code changes. | Extend existing change management controls to cover agent-generated changes explicitly. Build audit trail capability. |
| GDPR / Data residency | Agent context may be transmitted to model provider infrastructure in different jurisdictions. | Data processing must comply with residency and transfer requirements. | Verify model API routing. Use enterprise agreements with data processing addenda. Document compliance. |
| PCI DSS | Agents generating code that handles payment data must operate within PCI scope. | Restrict agent access to cardholder data environments. Log all agent interactions with payment systems. | Include agent access in your PCI scope assessment. Apply the same controls as human developer access. |
| HIPAA | Agents generating code for health data systems must comply with PHI protections. | Agent context must not include protected health information unless compliant safeguards are in place. | Exclude PHI from agent context. Use on-premises or BAA-covered model deployments for health data systems. |

---

## Important

### Board Reporting Template
Leaders need a one-page format to communicate AI agent adoption status to executive and board audiences, covering what is happening, what it costs, what the risks are, and what decisions are needed. The book distinguishes a status snapshot (a status email) from a governance artifact (shows where you are, where you are going, and whether you are on track) — the template includes a **target** and a **trend** for every metric row, because without them the board cannot distinguish progress from noise.

**AI-Assisted Development — Quarterly Status**

| Section | Metric | Current (example) | Target | Trend |
|---|---|---|---|---|
| Adoption | Developers using agent tools | 120 of 400 (30%) | 80% by Q4 | ↑ from 18% last quarter |
| Adoption | PRs with agent-generated code | 22% | 40% by Q4 | ↑ from 12% |
| Adoption | Phase maturity | Phase 2 (conversational) | Phase 3 (agentic) by year-end | Advanced from Phase 1 in Q1 |
| Value | Cycle time (agent-assisted vs. baseline) | −18% on eligible tasks | −25% | ↑ improving (was −11%) |
| Value | Deployment frequency | 3.2/week | 4/week | → flat |
| Value | Developer satisfaction (survey) | 7.4/10 | 7.5 | ↑ from 6.8 |
| Cost | Tool licensing | $42K/quarter | $50K | → stable |
| Cost | Model API / token spend | $28K/quarter | $35K | ↑ from $19K (adoption growth) |
| Cost | Total cost of ownership | $85K/quarter | $100K | ↑ tracking to plan |
| Risk | Governance readiness (lowest capability) | Basic in 4/6 areas | Basic in 6/6 | ↑ was None in 3/6 |
| Risk | Open audit findings (agent-related) | 2 open | 0 | ↓ from 5 |
| Risk | Agent-related incidents | 1 this quarter | 0 | → flat |
| Risk | Data boundary compliance | Compliant | Maintain | → stable |
| Risk | Insurance / liability coverage | E&O and cyber reviewed; agent clause pending | Agent-specific coverage confirmed | In progress |
| Decisions needed | — | Budget approval for next quarter. Data classification policy update requiring board awareness. Vendor contract renewal. Risk acceptance for identified gaps. | — | — |

The template is deliberately brief — board reporting should communicate status and surface decisions, not educate the audience on how agents work. The trend column is the most important: it tells the board whether the investment is producing directional progress or whether intervention is needed.

### From Rules to Runway
Governance has an image problem: engineers associate it with bureaucracy (approval queues that slow delivery, compliance checklists that exist for auditors rather than developers). If AI governance is positioned as another layer of restriction, adoption will route around it. The book's reframe draws a parallel to automated testing: before comprehensive test suites were standard, every deployment required extensive manual verification — "governance" (testing) slowed individual changes, but organizations with strong test suites deploy *more* frequently, not less, because each deployment carries lower risk and requires less manual scrutiny.

Agent governance works the same way: an organization with clear audit trails, scoped agent permissions, and risk-tiered review can give agents more autonomy in low-risk areas, because the controls exist to catch problems in high-risk ones. Without governance, every agent interaction carries ambiguous risk — cautious organizations restrict agent use broadly, and incautious organizations expose themselves to risks they cannot quantify. **The governance checklist is a floor, not a ceiling** — it creates the conditions for teams to adopt agents aggressively where risk is managed, rather than timidly everywhere because risk is unknown.

### The APM Project's Concrete Implementations
The book ties the governance principles above to real, CI-enforceable infrastructure built in the author's APM (Agent Package Manager) project, as evidence the framework is not purely theoretical:
- **Lock file audit trails** pin every agent configuration to exact commit SHAs with full dependency provenance, producing SOC 2-ready evidence from standard `git log` queries.
- **Policy inheritance chains** (Enterprise → Organization → Repository) ensure security baselines cascade automatically; child policies can only tighten constraints, never relax them.
- **CI enforcement gates** run 22 automated checks (6 baseline + 16 organizational policy) and block deployments that violate policy — no human gatekeeper required.
- **Content scanning** detects hidden Unicode attacks (bidirectional overrides, tag characters, variation selectors) before files reach agent-readable directories, addressing the prompt supply chain threat at the pre-deployment stage.

The generalizing pattern: governance primitives that can be expressed as CI checks should be; the ones that cannot (organizational policy, legal review triggers, risk classification) must be encoded as explicit context for agents and humans alike.

### Chapter Checklist
A 12-item action list the chapter closes with: (1) conduct a governance readiness self-assessment across the six areas, using the compliance mapping to prioritize by regulatory scope; (2) prioritize audit trails and agent access controls if currently at "None" in either; (3) classify agent-introduced risks across all six taxonomy categories and assign owners; (4) map products to relevant regulatory frameworks and evaluate gaps specific to agent-assisted development; (5) review agent instruction files and context sources for supply chain integrity, applying change-management controls; (6) establish a board reporting cadence using the template (with targets and trends); (7) review the code review process to verify it accounts for agent-generated failure modes, including implicit compliance decisions; (8) document the data boundary policy for agent workflows and verify enforcement is systemic, not procedural; (9) design deliberate practice into the development process to mitigate knowledge atrophy; (10) test the fallback — verify the team can sustain delivery if agent assistance is unavailable for 48 hours; (11) confirm E&O and cyber insurance policies address agent-generated code, raising the question with the CFO before the board does; (12) schedule a quarterly governance review, since agent capabilities and regulatory requirements both move fast.

---

## Key Terms

- **Governance gap** — the mismatch between governance frameworks built for human-only decision points and the reality of AI agents participating in the SDLC.
- **Plausible incorrectness** — agent-generated code that reads well, compiles, and passes tests while silently misunderstanding intent or violating an architectural invariant no test covers.
- **Knowledge atrophy** — the long-term erosion of a team's unassisted skill (debugging, architectural reasoning, review depth) as agents absorb the tasks that used to build that skill.
- **Context poisoning / prompt injection via context** — adversarial instructions planted in dependencies, documentation, or repository files that influence agent behavior when consumed as context.
- **Governance floor, not ceiling** — the chapter's framing that a governance baseline enables faster, more aggressive agent adoption in low-risk areas rather than restricting it everywhere.
- **APM (Agent Package Manager)** — the author's open-source tool, referenced here for its CI-enforceable governance primitives (lock file audit trails, policy inheritance, content scanning).
