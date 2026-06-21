# Chapter 17 — The Publishing Pipeline

> Source: *The Agentic SDLC Handbook* (Daniel Meppiel, v0.9.2), pp. 214–219.
> A case study in infrastructure automation: turning a 15-chapter Markdown manuscript into a multi-format, auto-deployed publication exposes the "Almost Done" trap as clearly as code ever does.

---

## Very Important

### The Problem and Strategy
**Scope:** manuscript-to-deployment pipeline — HTML, PDF, EPUB output + CI/CD.
**Duration:** multiple waves within a single extended session.
**Theme:** infrastructure automation and the "Almost Done" cascade.

The problem, stated plainly: you have a 15-chapter technical handbook in Markdown; you want readers to find it as a fast-loading website, a downloadable PDF, and an EPUB for e-readers — all auto-deployed from a single source. The publishing industry offers dozens of toolchains, each trading off cost, control, update friction, and reader reach. As the book frames the division of labor: **the author knew what the book should *feel* like; the agents knew how to make it *work*.**

### Publishing Strategy: Three Rounds of Deliberation
The first wave wasn't code — it was research, and it shows human judgment actively steering agent recommendations rather than rubber-stamping them:

1. **Round 1.** Orchestrator's initial recommendation: Open Core model — free web, paid PDF/EPUB. Author pushed back immediately: *"Why not Amazon ebooks? Why just PDF in a repo?"*
2. **Round 2.** A research agent investigated Quarto, Leanpub, GitBook, and mdBook. Revised recommendation: Leanpub + Amazon KDP + a GitHub source repo. Closer, but the author pushed again: *"What's state of the art for free tech books with regular updates?"*
3. **Round 3 (final).** **Quarto + GitHub Pages** — the pattern used by *R for Data Science* and *Python for Data Analysis*. One source repository, Markdown-native authoring, multi-format output, CI/CD deployment, zero hosting cost.

The book is explicit about what this three-round deliberation demonstrates: **human judgment remains the bottleneck and differentiator** under real conditions. The agents surfaced options and trade-offs; the human chose the publishing philosophy.

### The "Almost Done" Trap: PDF Rendering Cascade
What happened during PDF conversion is presented as a **textbook instance of Anti-pattern #15: The "Almost Done" Trap** — each fix revealed the next issue, creating a cascade where "one more fix" repeated five times:

| Issue | Symptom | Fix |
|---|---|---|
| 1 — Narrow text | `scrbook` document class has wide binding margins designed for physical books | Switched to `scrreprt` with explicit geometry (25mm left/right, 30mm top/bottom) |
| 2 — Missing Mermaid diagrams | All 15 chapters used GitHub-flavored ` ```mermaid ` fences; Quarto requires ` ```{mermaid} ` for PDF rendering via headless Chromium | Bulk `sed` conversion across every chapter; PDF grew from 686KB to 3.8MB as rendered diagram PNGs appeared |
| 3 — Duplicate heading numbers | Manual numbers in headings (`### 1. The Monolithic Prompt`) clashed with Quarto's auto-numbering, producing `15.2.1 1.` in the PDF | Stripped manual numbers from 25 headings across two chapters |
| 4 — Margin overflow | Diagrams and code blocks exceeded the text area | Added `fig-width: 6.5`, `code-overflow: wrap`, LaTeX packages (`fvextra`, `float`); then discovered 6.5in exceeded the 6.3in usable width — reduced to 5.5in and created `preamble.tex` with `\small` font for long table environments |
| 5 — ASCII art table | Chapter 4 contained a complex ASCII-art SDLC-phases table that rendered as a code block in the PDF | Converted to a proper 9-column Markdown pipe table |

Issue 2 is explicitly tagged as **Anti-pattern #10: Not Fixing the Primitives** — the fence syntax was wrong at the source level across every single chapter, not a one-off glitch. The book's general lesson: **each fix was technically correct; the trap is that correctness at one layer exposes incorrectness at the next.** The orchestrator's discipline was to treat every fix as its own micro-wave with its own checkpoint — never batching speculative fixes. The book's prescribed pattern:

1. Make one fix.
2. Render/build to see the result.
3. Commit before touching anything else.
4. Only then investigate the next issue.

This prevents speculative batching — fixing three things at once before you know whether fix #1 changes the landscape for fixes #2 and #3.

### CI/CD: Four Iterations to Stability
The deployment pipeline went through its own iteration gauntlet:

| Iteration | Failure | Fix |
|---|---|---|
| 1 | Mermaid→PNG via headless Chromium hung on CI | Added `quarto install chromium --no-prompt` |
| 2 | Custom LaTeX packages (DejaVu fonts, `fvextra`) hung CI | Simplified PDF config, removed custom fonts |
| 3 | `render: "html"` param to publish action — `_book` dir not found | Split into explicit `quarto render --to html` then `quarto publish` with `render: false` |
| 4 | CI HTML-only deploy overwrote PDF/EPUB on `gh-pages` | Added a "Restore PDF and EPUB from `gh-pages`" step using `git show` |

The final architecture deliberately splits rendering by environment: CI optimizes for **speed** (HTML-only, 49 seconds); local rendering optimizes for **completeness** (all formats, ~15 minutes). The 15-minute local render is called out as a practical constraint — Chromium-based Mermaid rendering is too heavy for a fast CI feedback loop, so the architecture makes the trade-off explicit rather than hiding it.

### What This Case Study Shows
The publishing pipeline compressed weeks of toolchain research, format debugging, and CI/CD configuration into focused waves of agent execution guided by human judgment.

**What held true.** The structural properties from the APM Overhaul case study held again here. The chapter's **novel test**: the "Almost Done" cascade demonstrated that **checkpoint discipline prevents compounding rendering failures as effectively as it prevents compounding code failures** — the same discipline, a different failure domain.

**The division of labor was consistent throughout:** every design decision was human; every technical execution was agent.

| Metric | Value |
|---|---|
| Output formats | 3 (HTML, PDF, EPUB) |
| Mermaid diagrams rendered | 37 |
| PDF/EPUB rendering fixes (cascade) | 5 |
| CI/CD iterations | 4 |
| Download UX iterations | 7 |
| Expert panel deliberations | 3 |
| CI deploy time | 49 seconds |

---

## Important

### The Conversion Wave
A single general-purpose agent converted the entire manuscript in one wave:
- Created `_quarto.yml` with a four-part book structure (Foundation, Leaders, Practitioners, Closing).
- Converted 15 `.md` files to `.qmd` via `git mv` — preserving git history at 99% similarity.
- Added YAML frontmatter to each chapter.
- Created `index.qmd` with a reading guide and download callouts.
- Created `.github/workflows/publish.yml` for CI auto-deploy.

Result: HTML render of 16 files, zero errors, 25 Mermaid diagrams confirmed. The site went live on the `gh-pages` branch within the same wave.

### Download UX: Seven Iterations of Human Refinement
The download experience went through seven rounds, all driven by the human author noticing friction:
1. Quarto's built-in `downloads: [pdf, epub]` — toolbar icon too small, easily missed.
2. Fixed duplicate titles and "0.1" section-numbering clashes between manual and Quarto-generated headings.
3. Simplified copy to: *"Free to read online. Free to download."*
4–7. Middle iterations were variations on the same theme, each fixed by the agent in under a minute.

The book calls this **the collaboration pattern at its clearest: the human refines intent; the agent refines implementation.**

### Licensing: Expert Panel Deliberation
A three-expert panel (IP Attorney, Publishing Strategist, Growth Hacker) evaluated licensing options. The recommendation was unanimous: **CC BY-NC-ND 4.0**. The rationale mapped to four desired properties: invites sharing (reach), requires attribution (authorship), prohibits commercial use (protects the author's brand), prohibits derivatives (preserves one canonical version). Implementation touched four files: `LICENSE`, `README.md`, `index.qmd`, and the `_quarto.yml` footer.

---

## Key Terms

- **The "Almost Done" Trap (Anti-pattern #15)** — fixing one rendering/build issue only to expose the next, in a repeating cascade; the chapter's central failure mode.
- **Quarto** — the Markdown-native publishing engine chosen (over Leanpub/KDP/GitBook/mdBook) for one-source, multi-format (HTML/PDF/EPUB), CI/CD-deployed output.
- **Micro-wave checkpointing** — treating each individual cascade fix as its own wave with its own commit and render/build verification, instead of batching speculative fixes.
- **CC BY-NC-ND 4.0** — the license unanimously recommended by the three-expert licensing panel: share freely, attribute, no commercial use, no derivatives.
- **CI/local split** — the final deployment design where CI renders HTML-only for speed (49s) and local rendering produces all formats for completeness (~15 min).
