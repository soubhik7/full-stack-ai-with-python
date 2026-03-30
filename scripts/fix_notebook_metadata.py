import json
from pathlib import Path

nb_path = Path("/Users/soubhik/AI/full-stack-ai-with-python/R&D/LLM/01_LLM from Scratch with PyTorch/model.ipynb")
nb = json.loads(nb_path.read_text(encoding='utf-8'))
changed = 0
for cell in nb.get('cells', []):
    meta = cell.get('metadata')
    if not isinstance(meta, dict):
        cell['metadata'] = {}
        meta = cell['metadata']
    if 'language' not in meta:
        meta['language'] = 'markdown' if cell.get('cell_type') == 'markdown' else 'python'
        changed += 1

if changed:
    backup = nb_path.with_suffix('.pre_meta_fix.backup.ipynb')
    backup.write_text(json.dumps(json.loads(nb_path.read_text(encoding='utf-8')), indent=1), encoding='utf-8')
    nb_path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
    print(f"Updated metadata.language for {changed} cells. Backup written to {backup}")
else:
    print("No metadata changes needed")
