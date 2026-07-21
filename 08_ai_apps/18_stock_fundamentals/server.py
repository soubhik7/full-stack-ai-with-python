"""Stock fundamentals fetcher.

Live company search (Groww instrument list) + per-selection fetch of Groww
live-quote data, yfinance fundamentals, technicals, news, and NSE filings,
saved to a timestamped CSV under data/. Selections can be saved as a
watchlist.json, which run_daily.py (headless automation, see that file and
the GitHub Actions workflow) reads to run the same fetch unattended.

Run:
    cd 08_ai_apps/18_stock_fundamentals
    uvicorn server:app --reload
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import groww_client
from fetch import fetch_company_row

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
STATIC_DIR = Path(__file__).parent / "static"
WATCHLIST_PATH = Path(__file__).parent / "watchlist.json"

state: dict = {"instruments": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["instruments"] = groww_client.load_instruments()
    yield


app = FastAPI(title="Stock Fundamentals Fetcher", lifespan=lifespan)


class FetchRequest(BaseModel):
    symbols: list[str]


class WatchlistRequest(BaseModel):
    symbols: list[str]


@app.get("/api/search")
def search(q: str = Query(""), limit: int = Query(20, le=50)):
    df = state["instruments"]
    return groww_client.search_companies(df, q, limit=limit)


@app.post("/api/fetch")
def fetch(req: FetchRequest):
    if not req.symbols:
        raise HTTPException(400, "No symbols provided")

    df = state["instruments"]
    rows = [row for symbol in req.symbols if (row := fetch_company_row(symbol, df))]

    if not rows:
        raise HTTPException(404, "None of the requested symbols were found")

    DATA_DIR.mkdir(exist_ok=True)
    filename = f"companies_fundamentals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(rows).to_csv(DATA_DIR / filename, index=False)

    return {"filename": filename, "rows": rows}


@app.get("/api/download/{filename}")
def download(filename: str):
    safe_name = Path(filename).name
    path = DATA_DIR / safe_name
    if path.suffix != ".csv" or not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="text/csv", filename=safe_name)


@app.get("/api/watchlist")
def get_watchlist():
    if not WATCHLIST_PATH.is_file():
        return {"symbols": []}
    symbols = json.loads(WATCHLIST_PATH.read_text()).get("symbols", [])

    df = state["instruments"]
    matches = df[df["trading_symbol"].isin(symbols)]
    by_symbol = {
        row["trading_symbol"]: row for row in matches.to_dict(orient="records")
    }
    # preserve saved order; skip symbols no longer in the instrument list
    return [by_symbol[s] for s in symbols if s in by_symbol]


@app.post("/api/watchlist")
def save_watchlist(req: WatchlistRequest):
    WATCHLIST_PATH.write_text(json.dumps({"symbols": req.symbols}, indent=2) + "\n")
    return {"saved": len(req.symbols)}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
