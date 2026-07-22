# tools/

Dev utility scripts for this repo. Most are one-off notebook-maintenance
helpers (see root `CLAUDE.md` for the full list). This file documents the
one that needs setup + run instructions.

## udemy_transcript_fetcher.py

Archives Udemy Business lecture transcripts as local Markdown notes, using
your own logged-in browser session (no private API, no auth bypass).

### Setup (one-time)

```bash
cd /Users/soubhik/AI/full-stack-ai-with-python
source venv/bin/activate
pip3 install -r requirements.txt   # installs playwright
python -m playwright install chromium
```

### Run

```bash
source venv/bin/activate
python tools/udemy_transcript_fetcher.py --course-url "https://ibm-learning.udemy.com/course/ai-102-microsoft-certified-azure-ai-engineer-associate-d/learn/lecture/56792919" --debug
```

Same command, split across lines for readability (copy the whole block at
once — don't edit or comment out individual lines, since a `#` on a
continued line ends the command early in zsh/bash):

```bash
source venv/bin/activate
python tools/udemy_transcript_fetcher.py \
  --course-url "https://ibm-learning.udemy.com/course/ai-102-microsoft-certified-azure-ai-engineer-associate-d/learn/lecture/56792919" \
  --debug
```

- Use `source venv/bin/activate` first — the venv Python, not the system
  `python3`, has Playwright installed.
- `--course-url` is required: any lecture link from the course works, the
  script derives the course's `/learn` URL from it.
- Keep `--debug` on for your first run against a course: if curriculum or
  transcript extraction comes back empty, it dumps the raw page HTML next
  to the output so the selectors can be corrected via browser devtools
  (Udemy's markup wasn't inspectable ahead of time, so selectors are
  best-effort).

### What happens when you run it

1. A real Chromium window opens.
2. Log in via SSO in that window.
3. Back in the terminal, press **Enter** once you can see the course
   curriculum sidebar.
4. The script walks every lecture, opens its transcript panel, and writes
   one `.md` file per lecture plus a `README.md` index.

### Other flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--output-dir` | `~/Documents/udemy-notes` | Where notes are written — kept **outside this repo** since transcripts are licensed course content, not repo content |
| `--session-dir` | `~/.cache/udemy_scraper_session` | Persists your login so you don't have to SSO in again next run |
| `--headless` | off | Skip the visible browser window (only useful after the first successful logged-in run) |
| `--delay` | `1.5` | Base seconds paused between lectures (politeness delay) |

### Output layout

```
~/Documents/udemy-notes/<course-slug>/
├── README.md                      # index of all lectures
├── 01_<section-slug>/
│   ├── 001_<lecture-slug>.md
│   └── 002_<lecture-slug>.md
└── 02_<section-slug>/
    └── 003_<lecture-slug>.md
```

Each lecture `.md` has a small frontmatter block (title, section, source
URL, fetch timestamp) followed by the transcript text. Re-running the
script skips lectures whose `.md` file already exists, so an interrupted
run can just be re-invoked.
