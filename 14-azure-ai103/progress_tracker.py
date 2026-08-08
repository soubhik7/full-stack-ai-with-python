"""Concept progress tracker for Chapter 14 (Azure AI-103).

Used by `00_PROGRESS_TRACKER.ipynb` at the chapter root. It live-scans the
numbered section folders (01_responses_api_basics/ … 08_document_intelligence_capstone/)
for every concept notebook, so newly added concepts appear automatically —
nothing here is a hardcoded list.

A concept counts as **visited** when its notebook has at least one executed
code cell (a non-null `execution_count` or saved outputs). Because students
work through concepts in any order — and may later clear a notebook's
outputs — every detection is persisted to `concept_progress.json` next to
this file, so a concept stays marked once seen. `mark_done()` / `mark_undone()`
allow manual overrides (e.g. for a concept studied via its `.py` script only).

Typical use (from the tracker notebook):

    from progress_tracker import refresh, show, mark_done, mark_undone
    refresh()   # re-scan all sections, absorb newly executed notebooks
    show()      # render the per-section checklist
"""

import json
from datetime import datetime, timezone
from pathlib import Path

CHAPTER_ROOT = Path(__file__).resolve().parent
PROGRESS_FILE = CHAPTER_ROOT / "concept_progress.json"

# Directories that never contain concept notebooks.
_SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".venv", "venv",
              "site-packages", "node_modules", "bin", "obj", ".git"}


def discover():
    """Return every concept notebook in the numbered sections, sorted.

    Each record: {"rel": path-from-chapter-root (str), "section": top folder,
    "name": stem, "script": paired .py rel path or None}.
    """
    concepts = []
    sections = sorted(p for p in CHAPTER_ROOT.iterdir()
                      if p.is_dir() and p.name[0].isdigit())
    for section in sections:
        for nb in sorted(section.rglob("*.ipynb")):
            if _SKIP_DIRS & set(nb.relative_to(CHAPTER_ROOT).parts):
                continue
            # Paired script: sibling NN_name.py, or ../NN_name.py for note/ dirs
            script = None
            for cand in (nb.with_suffix(".py"), nb.parent.parent / (nb.stem + ".py")):
                if cand.exists():
                    script = str(cand.relative_to(CHAPTER_ROOT))
                    break
            concepts.append({"rel": str(nb.relative_to(CHAPTER_ROOT)),
                             "section": section.name,
                             "name": nb.stem,
                             "script": script})
    return concepts


def _has_executed_cells(nb_path):
    try:
        nb = json.loads(nb_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return any(cell.get("execution_count") or cell.get("outputs")
               for cell in nb.get("cells", []) if cell.get("cell_type") == "code")


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}


def _save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")


def refresh():
    """Re-scan every concept; newly executed notebooks become 'done' (auto).

    Never un-marks: clearing a notebook's outputs doesn't erase the fact
    that the concept was visited.
    """
    progress = load_progress()
    for concept in discover():
        rel = concept["rel"]
        if rel not in progress and _has_executed_cells(CHAPTER_ROOT / rel):
            progress[rel] = {"status": "done", "marked": "auto",
                             "when": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _save_progress(progress)
    return progress


def _match(pattern):
    matches = [c for c in discover() if pattern.lower() in c["rel"].lower()]
    if not matches:
        print(f"No concept matches {pattern!r}. Try e.g. mark_done('05_web_search').")
    elif len(matches) > 1:
        print(f"{pattern!r} is ambiguous — matches:")
        for m in matches:
            print("  ", m["rel"])
        matches = []
    return matches


def mark_done(pattern):
    """Manually mark one concept done, matched by any unique path substring."""
    for concept in _match(pattern):
        progress = load_progress()
        progress[concept["rel"]] = {"status": "done", "marked": "manual",
                                    "when": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        _save_progress(progress)
        print("✅ marked done:", concept["rel"])


def mark_undone(pattern):
    """Remove a concept's done mark (both manual and auto)."""
    for concept in _match(pattern):
        progress = load_progress()
        if progress.pop(concept["rel"], None):
            _save_progress(progress)
            print("⬜ unmarked:", concept["rel"])
        else:
            print("already unmarked:", concept["rel"])


def _bar(done, total, width=20):
    filled = round(width * done / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def show():
    """Render the checklist — Markdown when in Jupyter, plain text otherwise."""
    progress = load_progress()
    concepts = discover()
    total_done = sum(1 for c in concepts if c["rel"] in progress)

    lines = [f"## Chapter 14 concept progress — {total_done}/{len(concepts)} "
             f"({100 * total_done // len(concepts) if concepts else 0}%)", ""]
    for section in sorted({c["section"] for c in concepts}):
        section_concepts = [c for c in concepts if c["section"] == section]
        done = sum(1 for c in section_concepts if c["rel"] in progress)
        lines.append(f"### {section} — {done}/{len(section_concepts)}  `{_bar(done, len(section_concepts))}`")
        for c in section_concepts:
            entry = progress.get(c["rel"])
            box = "✅" if entry else "⬜"
            how = f" *(marked {entry['marked']} {entry['when'][:10]})*" if entry else ""
            lines.append(f"- {box} [{c['rel']}]({c['rel']}){how}")
        lines.append("")
    text = "\n".join(lines)

    try:
        from IPython import get_ipython
        in_jupyter = get_ipython() is not None
    except ImportError:
        in_jupyter = False

    if in_jupyter:
        from IPython.display import Markdown, display
        display(Markdown(text))
    else:
        print(text.replace("### ", "").replace("## ", ""))


if __name__ == "__main__":
    refresh()
    show()
