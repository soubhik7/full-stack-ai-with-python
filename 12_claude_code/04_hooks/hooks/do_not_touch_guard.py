#!/usr/bin/env python3
"""PreToolUse hook — enforces this repo's CLAUDE.md "Do Not Touch" list.

Claude Code runs hooks with cwd == the project root, so paths below are
resolved against the repo root, not against this script's own location.
"""
import json
import sys
from pathlib import Path

DO_NOT_TOUCH = [
    "reference/",
    "01_python/",
    "02_math_for_ml/",
    "04_deep_learning/05_coursera_assignments/",  # CLAUDE.md scopes this to W*/ subfolders; blocking the whole tree is simpler and still correct
    "09_projects/04_made_with_ml/",
    "venv/",
]


def main() -> None:
    payload = json.load(sys.stdin)
    file_path = payload.get("tool_input", {}).get("file_path")
    if not file_path:
        sys.exit(0)  # not a file-editing tool call, nothing to check

    repo_root = Path.cwd().resolve()
    try:
        rel_path = Path(file_path).resolve().relative_to(repo_root).as_posix()
    except ValueError:
        sys.exit(0)  # path is outside the repo, not our concern

    for guarded in DO_NOT_TOUCH:
        if rel_path.startswith(guarded):
            print(
                f"Blocked: '{rel_path}' is under the Do Not Touch list in CLAUDE.md.",
                file=sys.stderr,
            )
            sys.exit(2)  # exit 2 = deny the tool call; stderr is surfaced to Claude

    sys.exit(0)


if __name__ == "__main__":
    main()
