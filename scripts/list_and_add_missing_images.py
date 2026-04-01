import re
import json
import ssl
from pathlib import Path
from urllib.request import Request, urlopen

notebook_path = Path("/Users/soubhik/AI/full-stack-ai-with-python/R&D/LLM/01_LLM from Scratch with PyTorch/model.ipynb")
mhtml_path = Path("/Users/soubhik/Downloads/No Libraries, No Shortcuts_ LLM from Scratch with PyTorch _ by Ashish Abraham _ Towards AI.mhtml")
images_dir = notebook_path.parent / 'images'
images_dir.mkdir(parents=True, exist_ok=True)

mhtml = mhtml_path.read_text(encoding='utf-8', errors='ignore')
# find all img src occurrences
img_urls = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', mhtml)
# also find URLs in CSS/background
img_urls += re.findall(r'url\((https?[^)]+)\)', mhtml)
# normalize
img_urls = [u.strip() for u in img_urls if u.strip()]
# unique preserve order
seen = set()
unique_urls = []
for u in img_urls:
    if u not in seen:
        seen.add(u)
        unique_urls.append(u)

nb = json.loads(notebook_path.read_text(encoding='utf-8'))
nb_md = []
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'markdown':
        nb_md.append('\n'.join(cell.get('source', [])).lower())

def filename_from_url(url):
    fn = url.split('/')[-1].split('?')[0]
    return fn

missing = []
for u in unique_urls:
    fn = filename_from_url(u)
    if not fn:
        continue
    in_nb = any(fn in md for md in nb_md)
    on_disk = (images_dir / fn).exists()
    if not in_nb and not on_disk:
        missing.append((u, fn))

# write short list
out_list = Path('/Users/soubhik/AI/full-stack-ai-with-python/scripts/missing_image_urls.txt')
with out_list.open('w', encoding='utf-8') as f:
    for u, fn in missing:
        f.write(u + '\n')

# download and insert
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

nb_changed = False
added = []
failed = []
for u, fn in missing:
    try:
        req = Request(u, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30, context=context) as resp:
            data = resp.read()
        out_path = images_dir / fn
        with open(out_path, 'wb') as f:
            f.write(data)
        added.append((u, fn))
    except Exception as e:
        failed.append((u, str(e)))

# append markdown cells for each downloaded image
if added:
    for u, fn in added:
        alt = fn.rsplit('.',1)[0].replace('-', ' ').replace('_',' ')
        cell = {
            'cell_type': 'markdown',
            'metadata': {'language': 'markdown'},
            'source': [f'![{alt}](images/{fn})\n']
        }
        nb['cells'].append(cell)
    nb_changed = True

if nb_changed:
    backup = notebook_path.with_suffix('.pre_add_images.backup.ipynb')
    backup.write_text(json.dumps(json.loads(notebook_path.read_text(encoding='utf-8')), indent=1), encoding='utf-8')
    notebook_path.write_text(json.dumps(nb, indent=1), encoding='utf-8')

print('Missing images found:', len(missing))
print('Downloaded & added:', len(added))
if failed:
    print('Failed downloads:', len(failed))
    for u,e in failed[:10]:
        print('FAILED:', u, e)
print('List saved to', out_list)
