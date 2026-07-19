"""
Archive Udemy Business lecture transcripts as local Markdown notes.

Personal-use tool: automates what you could do by hand (open each lecture,
click "Transcript", copy the text) using your own logged-in browser session.
It does not touch any private/internal API and does not bypass SSO/DRM.

Udemy's course-player markup can change and this script can't be tested
against an authenticated page ahead of time, so the CSS/data-purpose
selectors below are best-effort. If curriculum or transcript extraction
comes back empty, re-run with --debug and inspect the dumped HTML to
correct the selectors (browser devtools -> right-click the transcript
panel / curriculum sidebar -> Inspect).

Output is written to a directory OUTSIDE this repo by default, since
transcript content is IBM/Udemy-licensed training material, not something
that belongs in a public git repo.

Usage:
    python tools/udemy_transcript_fetcher.py \
        --course-url "https://ibm-learning.udemy.com/course/<slug>/learn/lecture/<id>" \
        --output-dir ~/Documents/udemy-notes

First run opens a real browser window. Log in via IBM SSO, then press
Enter in the terminal when the course curriculum is visible. The session
is saved to --session-dir so future runs skip the login step.
"""

import argparse
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import Page, TimeoutError as PWTimeoutError, sync_playwright

CURRICULUM_LINK_SELECTOR = 'a[href*="/learn/lecture/"]'
CURRICULUM_READY_SELECTOR = (
    '[data-purpose="curriculum-item-link"], [data-purpose*="curriculum"], '
    f"{CURRICULUM_LINK_SELECTOR}"
)
TRANSCRIPT_TOGGLE_SELECTORS = [
    'button[data-purpose="transcript-toggle"]',
    '[data-purpose="transcript-toggle"]',
    'button[aria-label*="Transcript" i]',
]
TRANSCRIPT_READY_SELECTOR = '[data-purpose="transcript-panel"], [data-purpose*="cue-text"]'
TRANSCRIPT_CUE_SELECTOR = '[data-purpose="transcript-cue"], [data-purpose*="cue-text"]'


def slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "untitled"


def parse_course_slug(course_url: str) -> str:
    parts = [p for p in urlparse(course_url).path.split("/") if p]
    if "course" in parts:
        idx = parts.index("course")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "udemy-course"


def course_learn_url(course_url: str) -> str:
    parsed = urlparse(course_url)
    if "/learn/" in parsed.path:
        base = parsed.path.split("/learn/")[0] + "/learn"
    else:
        base = parsed.path.rstrip("/") + "/learn"
    return f"{parsed.scheme}://{parsed.netloc}{base}"


def wait_for_login(page: Page, learn_url: str) -> None:
    page.goto(learn_url, wait_until="domcontentloaded")
    try:
        page.wait_for_selector(CURRICULUM_READY_SELECTOR, timeout=8000)
        return
    except PWTimeoutError:
        pass
    print("\nNot logged in yet (or curriculum sidebar not detected).")
    print("Complete the IBM SSO login in the opened browser window.")
    input("Once you can see the course curriculum, press Enter here to continue...")
    page.wait_for_selector(CURRICULUM_READY_SELECTOR, timeout=15000)


def get_curriculum(page: Page) -> list[dict]:
    items = page.eval_on_selector_all(
        CURRICULUM_LINK_SELECTOR,
        """(els) => els.map((el) => {
            const sectionEl = el.closest('[data-purpose^="curriculum-section"]') || el.closest('section');
            const heading = sectionEl
                ? sectionEl.querySelector('[data-purpose="section-heading"], h2, h3')
                : null;
            return {
                href: el.href,
                title: (el.textContent || '').trim(),
                section: heading ? (heading.textContent || '').trim() : 'section',
            };
        })""",
    )
    seen = set()
    curriculum = []
    for item in items:
        if not item["href"] or item["href"] in seen:
            continue
        seen.add(item["href"])
        item["order"] = len(curriculum) + 1
        curriculum.append(item)
    return curriculum


def open_transcript_panel(page: Page) -> bool:
    for selector in TRANSCRIPT_TOGGLE_SELECTORS:
        try:
            btn = page.query_selector(selector)
        except Exception:
            continue
        if btn:
            btn.click()
            page.wait_for_timeout(500)
            return True
    return False


def extract_transcript(page: Page) -> str:
    if not open_transcript_panel(page):
        return ""
    try:
        page.wait_for_selector(TRANSCRIPT_READY_SELECTOR, timeout=5000)
    except PWTimeoutError:
        return ""
    cues = page.eval_on_selector_all(
        TRANSCRIPT_CUE_SELECTOR,
        "(els) => els.map((el) => (el.textContent || '').trim())",
    )
    return "\n".join(c for c in cues if c)


def dump_debug_html(page: Page, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page.content())
    print(f"  [debug] dumped page HTML to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--course-url", required=True, help="Any lecture URL from the course (e.g. the .../learn/lecture/<id> link)")
    parser.add_argument("--output-dir", default=str(Path.home() / "Documents" / "udemy-notes"), help="Directory OUTSIDE this repo to write notes into")
    parser.add_argument("--session-dir", default=str(Path.home() / ".cache" / "udemy_scraper_session"), help="Directory to persist the browser login session")
    parser.add_argument("--headless", action="store_true", help="Run without a visible browser window (only useful after the first logged-in run)")
    parser.add_argument("--debug", action="store_true", help="Dump raw page HTML when curriculum/transcript extraction fails")
    parser.add_argument("--delay", type=float, default=1.5, help="Base seconds to wait between lectures (politeness delay)")
    args = parser.parse_args()

    course_slug = parse_course_slug(args.course_url)
    learn_url = course_learn_url(args.course_url)
    output_dir = Path(args.output_dir).expanduser() / course_slug
    session_dir = Path(args.session_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(session_dir),
            headless=args.headless,
            viewport={"width": 1400, "height": 900},
        )
        page = context.pages[0] if context.pages else context.new_page()

        wait_for_login(page, learn_url)

        curriculum = get_curriculum(page)
        if not curriculum:
            print("No curriculum items found — the page structure may differ from the expected selectors.")
            if args.debug:
                dump_debug_html(page, output_dir / "_debug_curriculum_page.html")
            context.close()
            sys.exit(1)

        print(f"Found {len(curriculum)} lecture(s) in curriculum.")

        section_order: dict[str, int] = {}
        index_lines = [f"# {course_slug}", "", f"Fetched: {datetime.now().isoformat(timespec='seconds')}", ""]

        for item in curriculum:
            section = item["section"] or "section"
            if section not in section_order:
                section_order[section] = len(section_order) + 1
            sec_dir = output_dir / f"{section_order[section]:02d}_{slugify(section)}"
            sec_dir.mkdir(parents=True, exist_ok=True)

            lecture_slug = slugify(item["title"]) or f"lecture-{item['order']}"
            md_path = sec_dir / f"{item['order']:03d}_{lecture_slug}.md"
            rel_path = md_path.relative_to(output_dir)

            if md_path.exists():
                print(f"[skip] {rel_path}")
                index_lines.append(f"- [{item['title']}]({rel_path})")
                continue

            print(f"[fetch] {item['title']}")
            transcript = ""
            try:
                page.goto(item["href"], wait_until="domcontentloaded")
                page.wait_for_timeout(1500)
                transcript = extract_transcript(page)
            except Exception as e:
                print(f"  error: {e}")
                if args.debug:
                    dump_debug_html(page, sec_dir / f"{item['order']:03d}_{lecture_slug}_debug.html")

            if not transcript and args.debug:
                dump_debug_html(page, sec_dir / f"{item['order']:03d}_{lecture_slug}_debug.html")

            body = "\n".join([
                "---",
                f"title: {item['title']}",
                f"section: {section}",
                f"source_url: {item['href']}",
                f"fetched_at: {datetime.now().isoformat(timespec='seconds')}",
                "---",
                "",
                transcript if transcript else "_(no transcript found for this lecture)_",
                "",
            ])
            md_path.write_text(body)
            index_lines.append(f"- [{item['title']}]({rel_path})")

            time.sleep(args.delay + random.uniform(0, 0.75))

        (output_dir / "README.md").write_text("\n".join(index_lines) + "\n")
        context.close()

    print(f"\nDone. Notes written to: {output_dir}")


if __name__ == "__main__":
    main()
