import json
import re
from pathlib import Path

notebook_path = Path("/Users/soubhik/AI/full-stack-ai-with-python/R&D/LLM/01_LLM from Scratch with PyTorch/model.ipynb")
mhtml_path = Path("/Users/soubhik/Downloads/No Libraries, No Shortcuts_ LLM from Scratch with PyTorch _ by Ashish Abraham _ Towards AI.mhtml")
report_path = Path("/Users/soubhik/AI/full-stack-ai-with-python/scripts/parity_report.txt")

if not notebook_path.exists():
    print(f"Notebook not found: {notebook_path}")
    raise SystemExit(1)
if not mhtml_path.exists():
    print(f"MHTML not found: {mhtml_path}")
    raise SystemExit(1)

nb = json.loads(notebook_path.read_text(encoding='utf-8'))
md_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'markdown']
headings = []
for cell in md_cells:
    for line in cell.get('source', []):
        m = re.match(r"^(#{1,6})\s*(.*)", line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            if text:
                headings.append((level, text))

mhtml_text = mhtml_path.read_text(encoding='utf-8', errors='ignore')
text_lower = mhtml_text.lower()

results = []
for lvl, h in headings:
    found = h.lower() in text_lower
    results.append({'level': lvl, 'heading': h, 'found_in_article': found})

found_count = sum(1 for r in results if r['found_in_article'])
missing = [r for r in results if not r['found_in_article']]

with report_path.open('w', encoding='utf-8') as f:
    f.write('Parity report: Notebook headings vs. article\n')
    f.write(f'Total headings in notebook: {len(results)}\n')
    f.write(f'Found in article: {found_count}\n')
    f.write(f'Missing from article (or not exact match): {len(missing)}\n\n')
    f.write('Details:\n')
    for r in results:
        f.write(f"Level {r['level']}: {r['heading']} -> {'FOUND' if r['found_in_article'] else 'MISSING'}\n")

print('Report written to', report_path)
print(f'Total headings: {len(results)}, Found: {found_count}, Missing: {len(missing)}')
if missing:
    print('First missing headings:')
    for r in missing[:10]:
        print('-', r['heading'])
