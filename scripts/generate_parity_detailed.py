import re
import json
from pathlib import Path

notebook_path = Path("/Users/soubhik/AI/full-stack-ai-with-python/R&D/LLM/01_LLM from Scratch with PyTorch/model.ipynb")
mhtml_path = Path("/Users/soubhik/Downloads/No Libraries, No Shortcuts_ LLM from Scratch with PyTorch _ by Ashish Abraham _ Towards AI.mhtml")
output_md = Path("/Users/soubhik/AI/full-stack-ai-with-python/scripts/parity_detailed.md")

nb = json.loads(notebook_path.read_text(encoding='utf-8'))
nb_md = []
nb_code = []
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'markdown':
        nb_md.append('\n'.join(cell.get('source', [])))
    elif cell.get('cell_type') == 'code':
        nb_code.append('\n'.join(cell.get('source', [])))

mhtml = mhtml_path.read_text(encoding='utf-8', errors='ignore')
# Extract headings and content blocks
pattern = re.compile(r"<(h[1-6])[^>]*>(.*?)</\1>(.*?)(?=<h[1-6]|$)", re.S | re.I)
sections = []
for m in pattern.finditer(mhtml):
    tag = m.group(1).lower()
    heading_html = m.group(2)
    content_html = m.group(3)
    heading_text = re.sub(r'<[^>]+>', '', heading_html).strip()
    content_text = re.sub(r'<[^>]+>', '', content_html).strip()
    # find image srcs in content_html
    imgs = re.findall(r'src=["\']([^"\']+)["\']', content_html)
    # find code blocks (<pre><code> or code tags)
    code_blocks = [re.sub(r'<[^>]+>', '', cb).strip() for cb in re.findall(r'<pre[^>]*>(.*?)</pre>', content_html, re.S | re.I)]
    # detect inline math $...$
    inline_math = re.findall(r'\$(.+?)\$', content_text)
    display_math = re.findall(r'\$\$(.+?)\$\$', content_text, re.S)
    sections.append({
        'tag': tag,
        'heading': heading_text,
        'content': content_text,
        'images': imgs,
        'code_blocks': code_blocks,
        'inline_math': inline_math,
        'display_math': display_math,
    })

# Helper checks
def found_in_notebook_heading(h):
    hnorm = re.sub(r"\s+"," ", h.strip().lower())
    for md in nb_md:
        if hnorm in md.lower():
            return True
    return False

def find_image_filename(url):
    return url.split('/')[-1].split('?')[0]

def notebook_has_image(filename):
    for md in nb_md:
        if filename in md:
            return True
    # check filesystem
    img_path = Path(notebook_path.parent) / 'images' / filename
    if img_path.exists():
        return True
    return False

def code_block_exists(block):
    bnorm = re.sub(r"\s+"," ", block.strip())
    for c in nb_code:
        if bnorm[:80] and bnorm[:80] in re.sub(r"\s+"," ", c):
            return True
    return False

# Generate report
lines = ["# Parity Detailed Report\n"]
lines.append(f"Notebook: {notebook_path}\nArticle (MHTML): {mhtml_path}\n")
lines.append(f"Total article sections found: {len(sections)}\n")
for i, s in enumerate(sections, 1):
    lines.append(f"## {i}. {s['heading']} ({s['tag']})\n")
    in_nb = found_in_notebook_heading(s['heading'])
    lines.append(f"- **Heading in notebook:** {'YES' if in_nb else 'NO'}\n")
    # images
    if s['images']:
        lines.append(f"- **Images in section ({len(s['images'])}):**\n")
        for img in s['images']:
            fname = find_image_filename(img)
            has = notebook_has_image(fname)
            lines.append(f"  - {fname} -> {'FOUND' if has else 'MISSING'}\n")
    else:
        lines.append("- **Images in section:** None\n")
    # code blocks
    if s['code_blocks']:
        lines.append(f"- **Code blocks in section ({len(s['code_blocks'])}):**\n")
        for cb in s['code_blocks']:
            exists = code_block_exists(cb)
            snippet = cb.strip().splitlines()[0][:120]
            lines.append(f"  - snippet: `{snippet}` -> {'FOUND' if exists else 'MISSING'}\n")
    else:
        lines.append("- **Code blocks in section:** None\n")
    # equations
    math_count = len(s['inline_math']) + len(s['display_math'])
    if math_count:
        lines.append(f"- **Equations in section:** {math_count} (inline: {len(s['inline_math'])}, display: {len(s['display_math'])})\n")
        # quick check in notebook
        math_found = False
        for m in s['inline_math'] + s['display_math']:
            if m.strip() and any(m.strip() in md for md in nb_md):
                math_found = True
                break
        lines.append(f"  - **Equation present in notebook:** {'YES' if math_found else 'NO'}\n")
    else:
        lines.append("- **Equations in section:** None\n")
    lines.append('\n')

output_md.write_text('\n'.join(lines), encoding='utf-8')
print('Detailed parity report written to', output_md)
