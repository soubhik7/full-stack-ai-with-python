"""Dynamic news headlines per company, via Google News' public RSS feed.

No API key needed. Per Google's own feed copyright notice this is "made
available solely for the purpose of rendering Google News results within a
personal feed reader for personal, non-commercial use" — fine for this
research tool, not for redistribution.

Sentiment is a crude keyword heuristic over headline text, not real NLP — it
exists to give a quick directional read, not a verdict.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

POSITIVE_WORDS = [
    "surge", "surges", "rally", "rallies", "jump", "jumps", "soar", "soars",
    "beat", "beats", "upgrade", "upgraded", "buyback", "record profit",
    "record high", "outperform", "wins order", "bags order", "expansion",
    "profit rises", "revenue growth", "strong growth", "bullish",
]
NEGATIVE_WORDS = [
    "plunge", "plunges", "crash", "falls", "fall", "downgrade", "downgraded",
    "probe", "fraud", "scam", "loss", "misses", "miss estimates", "layoff",
    "layoffs", "default", "resign", "resigns", "penalty", "fine", "raid",
    "bearish", "sell-off", "selloff", "decline",
]


def get_news(company_name: str, max_items: int = 6, lookback_days: int = 45) -> list[dict]:
    """Best-effort recent news fetch. Returns [] (never raises) on any failure."""
    query = f'"{company_name}" stock NSE'
    try:
        resp = requests.get(
            GOOGLE_NEWS_RSS_URL,
            params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
            timeout=10,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError):
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    items = []
    for item in root.findall(".//item"):
        title_el = item.find("title")
        source_el = item.find("source")
        link_el = item.find("link")
        pubdate_el = item.find("pubDate")
        if title_el is None or not title_el.text:
            continue

        try:
            published = parsedate_to_datetime(pubdate_el.text) if pubdate_el is not None else None
        except (TypeError, ValueError):
            published = None
        if published is not None:
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if published < cutoff:
                continue

        source_name = source_el.text if source_el is not None and source_el.text else ""
        title = title_el.text
        suffix = f" - {source_name}"
        if source_name and title.endswith(suffix):
            title = title[: -len(suffix)]

        items.append(
            {
                "title": title,
                "source": source_name,
                "published": published.strftime("%Y-%m-%d") if published else "",
                "link": link_el.text if link_el is not None else "",
            }
        )
        if len(items) >= max_items:
            break

    return items


def score_sentiment(news_items: list[dict]) -> dict:
    """Keyword-count heuristic over headline titles — a rough signal, not real NLP sentiment."""
    positive = negative = 0
    for item in news_items:
        text = item["title"].lower()
        positive += sum(1 for word in POSITIVE_WORDS if word in text)
        negative += sum(1 for word in NEGATIVE_WORDS if word in text)
    return {"positive": positive, "negative": negative}


def format_for_csv(news_items: list[dict], limit: int = 5) -> str:
    return " | ".join(f"{n['title']} ({n['source']}, {n['published']})" for n in news_items[:limit])
