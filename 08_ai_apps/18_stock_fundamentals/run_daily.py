#!/usr/bin/env python3
"""Headless daily run: fetch every watchlisted company, snapshot to CSV,
upsert the vector store, email a digest. Meant to run from GitHub Actions
(see .github/workflows/stock-watchlist-daily.yml) on a cron schedule, but
runs the same way locally: `python run_daily.py`.

Exits non-zero only on setup failures (no watchlist, no data fetched at
all) — a single company failing to fetch is logged and skipped, matching
every client module's best-effort design elsewhere in this app.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import emailer
import groww_client
import vector_store
from fetch import fetch_company_row

APP_DIR = Path(__file__).parent
WATCHLIST_PATH = APP_DIR / "watchlist.json"
DAILY_DATA_DIR = APP_DIR / "data" / "daily"
LATEST_CSV_PATH = APP_DIR / "data" / "watchlist_latest.csv"


def load_watchlist_symbols() -> list[str]:
    if not WATCHLIST_PATH.is_file():
        return []
    return json.loads(WATCHLIST_PATH.read_text()).get("symbols", [])


def main() -> int:
    load_dotenv(APP_DIR / ".env")
    load_dotenv(APP_DIR.parent.parent / ".env")  # repo-root .env, for shared keys

    symbols = load_watchlist_symbols()
    if not symbols:
        print(
            "watchlist.json is empty — nothing to fetch. Add symbols via the UI "
            "('Save as watchlist') or edit watchlist.json directly."
        )
        return 1

    print(f"Fetching {len(symbols)} watchlisted companies: {', '.join(symbols)}")
    instruments_df = groww_client.load_instruments()

    rows = []
    for symbol in symbols:
        print(f"  fetching {symbol}...")
        row = fetch_company_row(symbol, instruments_df)
        if row is None:
            print(f"  WARNING: {symbol} not found in instrument list, skipping")
            continue
        rows.append(row)
        try:
            vector_store.upsert_company(row)
        except Exception as e:
            print(f"  WARNING: vector store upsert failed for {symbol}: {e}")

    if not rows:
        print("No data fetched for any watchlisted symbol — aborting.")
        return 1

    try:
        removed = vector_store.reconcile(symbols)
    except Exception as e:
        removed = []
        print(f"WARNING: vector store reconcile failed: {e}")
    if removed:
        print(
            f"Removed from vector store (no longer in watchlist): {', '.join(removed)}"
        )

    DAILY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    daily_path = DAILY_DATA_DIR / f"{today}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(daily_path, index=False)
    df.to_csv(LATEST_CSV_PATH, index=False)
    print(f"Wrote {len(rows)} rows to {daily_path} and {LATEST_CSV_PATH}")

    narrative, narrative_error = None, None
    try:
        narrative = emailer.build_llm_recommendation(rows)
    except Exception as e:
        narrative_error = str(e)
        print(f"WARNING: LLM narrative generation failed: {e}")

    html = emailer.build_email_html(rows, narrative, narrative_error)
    try:
        emailer.send_email(f"Daily Stock Watchlist Digest — {today}", html)
        print("Email sent.")
    except Exception as e:
        print(f"WARNING: email send failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
