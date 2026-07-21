"""Corporate announcements (board meetings, results dates, dividends, ...)
from NSE India's public corporate-announcements API.

No API key needed, but NSE fronts its API with bot protection that expects
cookies from a prior visit to the site — a plain request can get blocked, so
this warms up a session against the homepage first. Best-effort throughout:
returns [] rather than raising if NSE blocks or rate-limits the request.
"""
from __future__ import annotations

import requests

BASE_URL = "https://www.nseindia.com"
ANNOUNCEMENTS_URL = f"{BASE_URL}/api/corporate-announcements"
SHAREHOLDING_URL = f"{BASE_URL}/api/corporate-share-holdings-master"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": f"{BASE_URL}/companies-listing/corporate-filings-announcements",
}


def _warmed_session() -> requests.Session | None:
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get(BASE_URL, timeout=10)
        return session
    except requests.RequestException:
        return None


def get_corporate_announcements(trading_symbol: str, max_items: int = 3) -> list[dict]:
    """Best-effort recent announcements fetch. Returns [] (never raises) on any failure."""
    session = _warmed_session()
    if session is None:
        return []

    try:
        resp = session.get(
            ANNOUNCEMENTS_URL,
            params={"index": "equities", "symbol": trading_symbol},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return []

    if not isinstance(payload, list):
        return []

    items = []
    for entry in payload[:max_items]:
        items.append(
            {
                "date": entry.get("an_dt", ""),
                "description": entry.get("desc") or entry.get("attchmntText") or "",
                "attachment": entry.get("attchmntFile", ""),
            }
        )
    return items


def format_for_csv(announcements: list[dict]) -> str:
    return " | ".join(f"{a['date']}: {a['description']}" for a in announcements)


def get_shareholding_pattern(trading_symbol: str) -> dict:
    """Best-effort promoter vs. public shareholding split from NSE's latest filing.

    NSE's public API only gives this broad split, not the granular FII/DII/
    mutual-fund breakdown Groww's app shows — that's Groww's own licensed
    data, not available from any free public source. Returns
    {"promoter_holding_pct": None, "public_holding_pct": None, "shareholding_as_of": None}
    on any failure.
    """
    empty = {
        "promoter_holding_pct": None,
        "public_holding_pct": None,
        "shareholding_as_of": None,
    }
    session = _warmed_session()
    if session is None:
        return empty

    try:
        resp = session.get(
            SHAREHOLDING_URL,
            params={"index": "equities", "symbol": trading_symbol},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return empty

    if not isinstance(payload, list) or not payload:
        return empty

    latest = max(payload, key=lambda entry: entry.get("date", ""))
    try:
        return {
            "promoter_holding_pct": float(latest["pr_and_prgrp"]),
            "public_holding_pct": float(latest["public_val"]),
            "shareholding_as_of": latest.get("date"),
        }
    except (KeyError, TypeError, ValueError):
        return empty
