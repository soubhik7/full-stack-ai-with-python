# Animated draw.io Diagrams

Editable [diagrams.net](https://www.diagrams.net) (`.drawio`) versions of every diagram in this
chapter, with animated edges (flowing dashes show the direction data moves) — a companion to the
static [Mermaid](https://mermaid.js.org) diagrams embedded directly in each sub-chapter's
`README.md`. Mermaid renders inline with no extra tooling (GitHub, VS Code preview, most Markdown
viewers); these `.drawio` files are for when you want to present, edit, re-theme, or step through
a diagram interactively.

## How to view the animation

The flowing-dash animation is a feature of the draw.io **editor/viewer**, not a property of the
file that survives a static export (PNG/SVG/PDF exports are frozen frames). To see it move:

- **VS Code** — install the [Draw.io Integration](https://marketplace.visualstudio.com/items?itemName=hediet.vscode-drawio) extension, then open any `.drawio` file directly.
- **Desktop app** — [draw.io desktop](https://github.com/jgraph/drawio-desktop/releases), File → Open.
- **Browser** — [app.diagrams.net](https://app.diagrams.net) → File → Open From → Device.

All 11 diagrams use the same style convention: solid violet edges = primary data flow, teal =
storage/persistence, ember/orange = risk or invalidation, faint grey dashed = a loop-back or
secondary relationship. Every edge is animated (`flowAnimation=1` in the style); double-click any
edge and open **Edit Style** to see or change it.

## Diagram index

| File | Mirrors the diagram in | What it shows |
|---|---|---|
| `00-memory-layer-flow.drawio` | [`00_landscape/README.md`](../../00_landscape/README.md) | Raw event stream → write path → memory store → read path → context window → response loop |
| `00-five-memory-operations.drawio` | [`00_landscape/README.md`](../../00_landscape/README.md) | Store → Retrieve → Update → Compress → Forget, with the forget→retrieve feedback loop |
| `01-memory-taxonomy-cycle.drawio` | [`01_memory_taxonomy/README.md`](../../01_memory_taxonomy/README.md) | Episodic → Semantic → Procedural → Episodic cycle |
| `02-os-tiered-memory.drawio` | [`02_architectures/README.md`](../../02_architectures/README.md) | Letta/MemGPT's core / recall / archival tiers (RAM / cache / disk analogy) |
| `02-temporal-knowledge-graph.drawio` | [`02_architectures/README.md`](../../02_architectures/README.md) | Graphiti/Zep's bi-temporal edge invalidation (Berlin/Paris example) |
| `02-hybrid-vector-graph-kv.drawio` | [`02_architectures/README.md`](../../02_architectures/README.md) | Mem0-style fusion across vector, graph, and key-value indexes |
| `03-architecture-quadrant.drawio` | [`03_frameworks_comparison/README.md`](../../03_frameworks_comparison/README.md) | Mem0 / HippocampAI / Zep / Letta positioned on the two architecture axes |
| `03-decision-path.drawio` | [`03_frameworks_comparison/README.md`](../../03_frameworks_comparison/README.md) | The rule-of-thumb decision tree for picking a framework |
| `04-memory-poisoning-attack.drawio` | [`04_security_and_interop/README.md`](../../04_security_and_interop/README.md) | The persistent attack chain — poison once, exploit every future session |
| `04-defense-layers.drawio` | [`04_security_and_interop/README.md`](../../04_security_and_interop/README.md) | Ingest → store → runtime defense stages (OWASP ASI06) |
| `04-interop-protocol-stack.drawio` | [`04_security_and_interop/README.md`](../../04_security_and_interop/README.md) | MCP / A2A / ACP and the emerging memory-interoperability layer |

## Regenerating a static image

To export a frozen (non-animated) PNG/SVG/PDF from any file, e.g. for a slide deck:

```bash
# draw.io desktop CLI (headless export)
drawio --export --format svg --output out.svg 00-memory-layer-flow.drawio
drawio --export --format pdf --output out.pdf 00-memory-layer-flow.drawio
```
