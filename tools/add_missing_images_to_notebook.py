import re
import json
import ssl
from pathlib import Path
from urllib.request import Request, urlopen

notebook_path = Path("/Users/soubhik/AI/full-stack-ai-with-python/R&D/LLM/01_LLM from Scratch with PyTorch/model.ipynb")
mhtml_path = Path("/Users/soubhik/Downloads/No Libraries, No Shortcuts_ LLM from Scratch with PyTorch _ by Ashish Abraham _ Towards AI.mhtml")
images_dir = notebook_path.parent / 'images'
images_dir.mkdir(parents=True, exist_ok=True)

nb = json.loads(notebook_path.read_text(encoding='utf-8'))
# build heading -> index map
heading_map = {}
for idx, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') != 'markdown':
        continue
    for line in cell.get('source', []):
        m = re.match(r"^(#{1,6})\s*(.*)", line)
        if m:
            heading_text = m.group(2).strip().lower()
            if heading_text and heading_text not in heading_map:
                heading_map[heading_text] = idx

mhtml = mhtml_path.read_text(encoding='utf-8', errors='ignore')
# extract sections with images
pattern = re.compile(r"<(h[1-6])[^>]*>(.*?)</\1>(.*?)(?=<h[1-6]|$)", re.S | re.I)
sections = []
for m in pattern.finditer(mhtml):
    heading_html = m.group(2)
    content_html = m.group(3)
    heading_text = re.sub(r'<[^>]+>', '', heading_html).strip()
    imgs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', content_html)
    if heading_text and imgs:
        sections.append((heading_text, imgs))

# helper
def filename_from_url(url):
    fn = url.split('/')[-1].split('?')[0]
    return fn

def notebook_has_image(fn):
    # check markdown cells
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'markdown':
            continue
        for line in cell.get('source', []):
            if fn in line:
                return True
    # check filesystem
    if (images_dir / fn).exists():
        return True
    return False

# SSL context
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

added = 0
failed = []
for heading, imgs in sections:
    for url in imgs:
        fn = filename_from_url(url)
        if notebook_has_image(fn):
            continue
        # download
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=30, context=context) as resp:
                data = resp.read()
            out_path = images_dir / fn
            with open(out_path, 'wb') as f:
                f.write(data)
        except Exception as e:
            failed.append((url, str(e)))
            continue
        # create markdown cell to insert after heading if exists
        alt = fn.replace('-', ' ').replace('_', ' ').split('.')[0]
        md_source = [f"![{alt}](images/{fn})\n"]
        new_cell = {
            'cell_type': 'markdown',
            'metadata': {'language': 'markdown'},
            'source': md_source
        }
        inserted = False
        key = heading.strip().lower()
        if key in heading_map:
            # insert after the heading cell index
            insert_idx = heading_map[key] + 1
            nb['cells'].insert(insert_idx, new_cell)
            # update heading_map indices for subsequent headings
            for k, v in list(heading_map.items()):
                if v >= insert_idx:
                    heading_map[k] = v + 1
            inserted = True
        else:
            nb['cells'].append(new_cell)
        added += 1

# backup and write
backup = notebook_path.with_suffix('.pre_images_add.backup.ipynb')
backup.write_text(json.dumps(json.loads(notebook_path.read_text(encoding='utf-8')), indent=1), encoding='utf-8')
notebook_path.write_text(json.dumps(nb, indent=1), encoding='utf-8')

print(f"Downloaded and inserted {added} images. {len(failed)} failures.")
if failed:
    for u, e in failed[:10]:
        print('FAILED:', u, e)
