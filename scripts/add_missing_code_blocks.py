import re
import json
from pathlib import Path

notebook_path = Path("/Users/soubhik/AI/full-stack-ai-with-python/R&D/LLM/01_LLM from Scratch with PyTorch/model.ipynb")
mhtml_path = Path("/Users/soubhik/Downloads/No Libraries, No Shortcuts_ LLM from Scratch with PyTorch _ by Ashish Abraham _ Towards AI.mhtml")

nb = json.loads(notebook_path.read_text(encoding='utf-8'))
# collect existing code cells content
nb_code = ["\n".join(cell.get('source', [])) for cell in nb.get('cells', []) if cell.get('cell_type')=='code']
# build heading->index map
heading_map = {}
for idx, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type')!='markdown':
        continue
    for line in cell.get('source', []):
        m = re.match(r"^(#{1,6})\s*(.*)", line)
        if m:
            heading_text = m.group(2).strip().lower()
            if heading_text and heading_text not in heading_map:
                heading_map[heading_text] = idx

mhtml = mhtml_path.read_text(encoding='utf-8', errors='ignore')
# find sections with headings and code (<pre>)
pattern = re.compile(r"<(h[1-6])[^>]*>(.*?)</\1>(.*?)(?=<h[1-6]|$)", re.S | re.I)
sections = []
for m in pattern.finditer(mhtml):
    heading_html = m.group(2)
    content_html = m.group(3)
    heading_text = re.sub(r'<[^>]+>', '', heading_html).strip()
    # code blocks
    code_blocks = re.findall(r'<pre[^>]*>(.*?)</pre>', content_html, re.S | re.I)
    # strip tags from code blocks
    code_blocks = [re.sub(r'<[^>]+>', '', cb).strip() for cb in code_blocks if cb.strip()]
    if heading_text and code_blocks:
        sections.append((heading_text, code_blocks))

added = 0
inserted_positions = []
for heading, code_list in sections:
    for code in code_list:
        # normalize snippet
        snippet = '\n'.join([line for line in code.splitlines() if line.strip()])
        if not snippet:
            continue
        found = False
        for existing in nb_code:
            if snippet.splitlines()[0].strip() in existing:
                found = True
                break
        if found:
            continue
        # prepare code cell
        code_lines = [line + "\n" for line in snippet.splitlines()]
        new_cell = {
            'cell_type': 'code',
            'metadata': {'language': 'python'},
            'source': code_lines
        }
        key = heading.strip().lower()
        if key in heading_map:
            insert_idx = heading_map[key] + 1
            nb['cells'].insert(insert_idx, new_cell)
            # update heading_map
            for k, v in list(heading_map.items()):
                if v >= insert_idx:
                    heading_map[k] = v + 1
            inserted_positions.append(insert_idx)
        else:
            nb['cells'].append(new_cell)
            inserted_positions.append(len(nb['cells'])-1)
        nb_code.append('\n'.join(code_lines))
        added += 1

if added:
    backup = notebook_path.with_suffix('.pre_code_add.backup.ipynb')
    backup.write_text(json.dumps(json.loads(notebook_path.read_text(encoding='utf-8')), indent=1), encoding='utf-8')
    notebook_path.write_text(json.dumps(nb, indent=1), encoding='utf-8')

print(f"Inserted {added} missing code blocks into the notebook.")
if added:
    print('Inserted positions (cell indices):', inserted_positions[:20])
