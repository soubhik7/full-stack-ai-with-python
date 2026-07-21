"""Daily digest email: an LLM-written narrative over the day's watchlist
snapshot, sent via Gmail SMTP.

The disclaimer banner is hard-coded into the HTML template — it always
appears regardless of what the model outputs. The system prompt requires the
model to ground every claim in the numbers it's given and forbids promising
outcomes, but an LLM narrative is still a model's synthesis of the data, not
verified analysis — see this app's README for the yfinance-accuracy caveat
this was built after finding.
"""
from __future__ import annotations

import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from google import genai

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

DISCLAIMER = (
    "This is an AI-generated summary of public market data (Groww live quotes, "
    "Yahoo Finance fundamentals, NSE filings, Google News), not investment advice "
    "from a registered advisor. Fundamentals data has known accuracy limitations "
    "(see the app's README) and markets carry real risk of loss. Verify "
    "independently before acting."
)

_SYSTEM_PROMPT = """You are a research-notes assistant summarizing a personal stock \
watchlist for the account owner, who reads this email daily.

You will receive a JSON array of company snapshots — pricing, valuation, technicals, \
fundamentals, recent news, and corporate announcements, one object per company.

For each company, write 2-4 sentences citing specific numbers from the data (P/E, \
RSI, technical_summary, recent growth figures, analyst target, notable news) — never \
invent a number that isn't in the data. End each with a one-line lean: Bullish, \
Neutral, or Bearish, and a one-sentence reason.

Then write a short "Today's highlights" section (2-3 sentences) naming which \
companies in the list look most worth a closer look today based on the data, and why \
— but always frame this as "worth reviewing," never as a guarantee of profit or a \
directive to buy. Flag if any company's data looks internally inconsistent (e.g. a \
P/B that doesn't reconcile with price and book value) rather than ignoring it.

Keep the whole thing skimmable: use the company name as a heading per section. Do not \
add your own disclaimer — one is already appended separately."""


def build_llm_recommendation(rows: list[dict]) -> str:
    """Best-effort LLM narrative. Raises on failure — caller decides the fallback."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    compact_rows = [
        {k: v for k, v in row.items() if k != "business_summary"} for row in rows
    ]
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=str(compact_rows),
        config={"system_instruction": _SYSTEM_PROMPT, "temperature": 0.3},
    )
    return response.text


def _row_table_html(rows: list[dict]) -> str:
    columns = [
        "trading_symbol",
        "last_price",
        "day_change_perc",
        "technical_summary",
        "trailing_pe",
        "roe",
        "analyst_recommendation",
    ]
    header = "".join(f"<th>{c}</th>" for c in columns)
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{row.get(c, '')}</td>" for c in columns) + "</tr>"
        for row in rows
    )
    return f"<table border='1' cellpadding='6' cellspacing='0'><thead><tr>{header}</tr></thead><tbody>{body_rows}</tbody></table>"


def build_email_html(
    rows: list[dict], narrative: str | None, narrative_error: str | None = None
) -> str:
    narrative_html = (
        narrative.replace("\n", "<br>")
        if narrative
        else f"<p><em>LLM narrative unavailable this run ({narrative_error}). Raw data table below.</em></p>"
    )
    return f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 720px;">
      <div style="background:#fff3cd; border:1px solid #ffe69c; padding:12px; border-radius:8px; margin-bottom:16px;">
        <strong>Disclaimer:</strong> {DISCLAIMER}
      </div>
      <h2>Daily Stock Watchlist Digest — {date.today().isoformat()}</h2>
      {narrative_html}
      <h3>Raw data snapshot</h3>
      {_row_table_html(rows)}
    </div>
    """


def send_email(subject: str, html_body: str) -> None:
    from_addr = os.environ["EMAIL_FROM"]
    app_password = os.environ["EMAIL_APP_PASSWORD"]
    to_addr = os.environ["EMAIL_TO"]

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = from_addr
    message["To"] = to_addr
    message.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_addr, app_password)
        server.sendmail(from_addr, [to_addr], message.as_string())
