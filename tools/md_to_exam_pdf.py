"""Render one of this repo's exam-prep Markdown docs (EXAM_NOTES.md /
EXAM_PRACTICE_QUESTIONS.md style: GitHub-flavored tables + <details><summary>
answer blocks) into a clean PDF using fpdf2 + python-markdown.

Usage: python tools/md_to_exam_pdf.py <input.md> <output.pdf> "<Title>"
"""
import re
import sys

import markdown
from fpdf import FPDF
from fpdf.fonts import FontFace

ARROW_SUBS = {
    "→": " -> ",   # →
    "↔": " <-> ",  # ↔
    "←": " <- ",   # ←
    "–": "-",       # – en dash
    "—": " - ",     # — em dash
    "§": "Sec. ",   # § (Standard-14 fonts mangle this on text extraction)
    "·": "-",       # · middle dot used as a header separator
}

WINDOWS_FONT_DIR = "C:/Windows/Fonts"

# Emoji / pictograph / variation-selector ranges - stripped entirely (core
# PDF fonts have no glyphs for them; they're purely decorative here).
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "☀-➿"
    "️"
    "]+"
)


def normalize_text(text: str) -> str:
    for char, repl in ARROW_SUBS.items():
        text = text.replace(char, repl)
    text = EMOJI_RE.sub("", text)
    # Collapse the double space left behind where a leading emoji used to sit.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def details_to_markdown(text: str) -> str:
    """Turn <details><summary><b>Answer</b></summary>...</details> into a
    plain markdown paragraph so python-markdown will process the **bold**
    etc. inside it (raw HTML blocks are otherwise passed through untouched)."""
    text = re.sub(
        r"<details>\s*<summary><b>Answer</b></summary>",
        "\n\n**Answer:**\n\n",
        text,
    )
    text = text.replace("</details>", "\n")
    return text


def build_pdf(md_path: str, pdf_path: str, title: str) -> None:
    with open(md_path, encoding="utf-8") as f:
        raw = f.read()

    raw = normalize_text(raw)
    raw = details_to_markdown(raw)

    html = markdown.markdown(
        raw, extensions=["tables", "sane_lists", "fenced_code"]
    )

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)
    pdf.set_title(title)

    pdf.add_font("Arial", "", f"{WINDOWS_FONT_DIR}/arial.ttf")
    pdf.add_font("Arial", "B", f"{WINDOWS_FONT_DIR}/arialbd.ttf")
    pdf.add_font("Arial", "I", f"{WINDOWS_FONT_DIR}/ariali.ttf")
    pdf.add_font("Arial", "BI", f"{WINDOWS_FONT_DIR}/arialbi.ttf")
    pdf.set_font("Arial", "", 11)

    pdf.add_page()

    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(20, 40, 90)
    pdf.multi_cell(0, 10, title)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 11)
    pdf.ln(4)

    tag_styles = {
        "h1": FontFace(family="Arial", color=(20, 40, 90), emphasis="B", size_pt=18),
        "h2": FontFace(family="Arial", color=(30, 60, 120), emphasis="B", size_pt=15),
        "h3": FontFace(family="Arial", color=(50, 80, 140), emphasis="B", size_pt=13),
        "h4": FontFace(family="Arial", color=(70, 70, 70), emphasis="B", size_pt=11.5),
        "blockquote": FontFace(family="Arial", color=(90, 90, 90)),
    }

    pdf.write_html(
        html,
        tag_styles=tag_styles,
        table_line_separators=True,
    )

    pdf.output(pdf_path)
    print(f"wrote {pdf_path} ({pdf.pages_count} pages)")


if __name__ == "__main__":
    build_pdf(sys.argv[1], sys.argv[2], sys.argv[3])
