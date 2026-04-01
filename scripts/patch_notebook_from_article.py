import json
import re
from pathlib import Path

notebook_path = Path("/Users/soubhik/AI/full-stack-ai-with-python/R&D/LLM/01_LLM from Scratch with PyTorch/model.ipynb")
mhtml_path = Path("/Users/soubhik/Downloads/No Libraries, No Shortcuts_ LLM from Scratch with PyTorch _ by Ashish Abraham _ Towards AI.mhtml")

nb = json.loads(notebook_path.read_text(encoding='utf-8'))
md_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'markdown']
existing_headings = set()
for cell in md_cells:
    for line in cell.get('source', []):
        m = re.match(r"^(#{1,6})\s*(.*)", line)
        if m:
            existing_headings.add(m.group(2).strip().lower())

mhtml_text = mhtml_path.read_text(encoding='utf-8', errors='ignore')
# find headings and following content
pattern = re.compile(r"<(h[1-6])[^>]*>(.*?)</\1>(.*?)(?=<h[1-6]|$)", re.S | re.I)
sections = []
for m in pattern.finditer(mhtml_text):
    tag = m.group(1).lower()
    heading_html = m.group(2)
    content_html = m.group(3)
    # strip tags for heading
    heading_text = re.sub(r'<[^>]+>', '', heading_html).strip()
    # simple convert content_html to plaintext by removing tags
    content_text = re.sub(r'<[^>]+>', '', content_html).strip()
    if heading_text:
        sections.append((tag, heading_text, content_text))

# find which sections are not present in notebook
missing_sections = [s for s in sections if s[1].strip().lower() not in existing_headings]
print(f"Found {len(sections)} article sections, {len(missing_sections)} are missing from notebook")

# Append missing sections as markdown cells at end of notebook
if missing_sections:
    for tag, heading, content in missing_sections:
        md_lines = []
        level = int(tag[1]) if tag and tag[1].isdigit() else 2
        md_lines.append('#' * level + ' ' + heading + '\n')
        # split content into paragraphs
        paras = [p.strip() for p in re.split(r'\n{2,}|\r\n{2,}', content) if p.strip()]
        for p in paras[:10]:  # limit to first 10 paragraphs to avoid huge inserts
            md_lines.append(p + '\n\n')
        cell = {
            'cell_type': 'markdown',
            'metadata': {},
            'source': md_lines
        }
        nb['cells'].append(cell)
    # backup original
    backup = notebook_path.with_suffix('.backup.ipynb')
    backup.write_text(json.dumps(json.loads(notebook_path.read_text(encoding='utf-8')), indent=1), encoding='utf-8')
    # write updated
    notebook_path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
    print(f"Appended {len(missing_sections)} markdown cells to notebook and backed up original to {backup}")
else:
    print("No missing sections to append")
