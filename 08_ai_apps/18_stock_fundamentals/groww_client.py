"""Client for Groww's Trade API (https://groww.in/trade-api/docs).

Groww's Trade API is a brokerage/trading API, not a fundamentals API — it has
no EPS/ROE/cash-flow endpoints. What it does give us, with no auth required:

  * the full tradable-instruments CSV (used here to power live company search)

And, with a valid access token, the live quote endpoint (price, day change,
52-week high/low, market cap, bid/ask, circuit limits) and daily historical
candles (used by technicals.py for moving averages / RSI / returns).
Fundamentals (EPS, ROE, cash flow, business summary, etc.) come from
fundamentals_client.py instead.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

INSTRUMENT_CSV_URL = "https://growwapi-assets.groww.in/instruments/instrument.csv"
QUOTE_URL = "https://api.groww.in/v1/live-data/quote"
HISTORICAL_CANDLE_URL = "https://api.groww.in/v1/historical/candle/range"
TOKEN_URL = "https://api.groww.in/v1/token/api/access"

DATA_DIR = Path(__file__).parent / "data"
INSTRUMENT_CACHE_PATH = DATA_DIR / "instruments_cache.csv"
INSTRUMENT_CACHE_TTL_SECONDS = 24 * 60 * 60

_token_cache: dict = {"token": None, "expires_at": 0}


def _cache_is_fresh() -> bool:
    if not INSTRUMENT_CACHE_PATH.exists():
        return False
    age = time.time() - INSTRUMENT_CACHE_PATH.stat().st_mtime
    return age < INSTRUMENT_CACHE_TTL_SECONDS


def load_instruments() -> pd.DataFrame:
    """Return NSE, cash-segment, plain-equity instruments (name/symbol/isin).

    Downloads Groww's public instrument CSV (no auth needed) and caches it
    locally for a day. Filtered to segment=CASH, instrument_type=EQ,
    series=EQ to drop derivatives, bonds/NCDs, and index rows.
    """
    DATA_DIR.mkdir(exist_ok=True)
    if not _cache_is_fresh():
        resp = requests.get(INSTRUMENT_CSV_URL, timeout=30)
        resp.raise_for_status()
        INSTRUMENT_CACHE_PATH.write_bytes(resp.content)

    df = pd.read_csv(INSTRUMENT_CACHE_PATH, low_memory=False)
    equities = df[
        (df["segment"] == "CASH")
        & (df["instrument_type"] == "EQ")
        & (df["series"] == "EQ")
    ].copy()
    equities = equities[["exchange", "trading_symbol", "groww_symbol", "name", "isin"]]
    equities["name"] = equities["name"].fillna("")
    return equities.reset_index(drop=True)


def search_companies(df: pd.DataFrame, query: str, limit: int = 20) -> list[dict]:
    """Live substring search over company name / trading symbol, name-startswith first."""
    query = query.strip().lower()
    if not query:
        return []

    name_lower = df["name"].str.lower()
    symbol_lower = df["trading_symbol"].str.lower()
    mask = name_lower.str.contains(query, na=False) | symbol_lower.str.contains(
        query, na=False
    )
    matches = df[mask].copy()
    if matches.empty:
        return []

    matches["_rank"] = (~name_lower[mask].str.startswith(query)).astype(int)
    matches = matches.sort_values(["_rank", "name"]).head(limit)
    return matches.drop(columns="_rank").to_dict(orient="records")


def _generate_token_via_key_secret() -> str | None:
    api_key = os.getenv("GROWW_API_KEY")
    api_secret = os.getenv("GROWW_API_SECRET")
    if not (api_key and api_secret):
        return None

    timestamp = str(int(time.time()))
    checksum = hashlib.sha256((api_secret + timestamp).encode()).hexdigest()
    resp = requests.post(
        TOKEN_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"key_type": "approval", "checksum": checksum, "timestamp": timestamp},
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json().get("payload", resp.json())
    token = payload.get("token")
    expiry = payload.get("expiry")  # epoch seconds, if provided
    _token_cache["token"] = token
    _token_cache["expires_at"] = float(expiry) if expiry else time.time() + 3600
    return token


def get_access_token() -> str | None:
    """Static GROWW_ACCESS_TOKEN takes priority; else auto-generate via key+secret."""
    static_token = os.getenv("GROWW_ACCESS_TOKEN")
    if static_token:
        return static_token

    if _token_cache["token"] and time.time() < _token_cache["expires_at"] - 30:
        return _token_cache["token"]

    return _generate_token_via_key_secret()


_OHLC_NUMBER_RE = re.compile(r"(open|high|low|close)\s*:\s*([\d.]+)")


def _parse_ohlc(raw) -> dict:
    """Groww's quote endpoint returns 'ohlc' as a non-JSON string like
    "{open: 149.50,high: 150.50,low: 148.50,close: 149.50}" — parse it by hand."""
    if not isinstance(raw, str):
        return {}
    return {key: float(value) for key, value in _OHLC_NUMBER_RE.findall(raw)}


def _auth_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "X-API-VERSION": "1.0",
    }


def get_quote(
    trading_symbol: str, exchange: str = "NSE", segment: str = "CASH"
) -> dict | None:
    """Best-effort live quote fetch. Returns None (never raises) if unauthenticated or on error."""
    token = get_access_token()
    if not token:
        return None

    try:
        resp = requests.get(
            QUOTE_URL,
            params={
                "exchange": exchange,
                "segment": segment,
                "trading_symbol": trading_symbol,
            },
            headers=_auth_headers(token),
            timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        payload = body.get("payload", body)
        ohlc = _parse_ohlc(payload.get("ohlc"))
        payload["day_open"] = ohlc.get("open")
        payload["day_high"] = ohlc.get("high")
        payload["day_low"] = ohlc.get("low")
        return payload
    except requests.RequestException:
        return None


def get_historical_candles(
    trading_symbol: str,
    exchange: str = "NSE",
    segment: str = "CASH",
    days: int = 500,
) -> pd.DataFrame | None:
    """Best-effort daily OHLCV history for the last `days` days, for technical indicators.

    Returns a DataFrame with columns [date, open, high, low, close, volume],
    or None if unauthenticated or on error — never raises.
    """
    token = get_access_token()
    if not token:
        return None

    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=days)
    try:
        resp = requests.get(
            HISTORICAL_CANDLE_URL,
            params={
                "exchange": exchange,
                "segment": segment,
                "trading_symbol": trading_symbol,
                "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end.strftime("%Y-%m-%d %H:%M:%S"),
                "interval_in_minutes": 1440,
            },
            headers=_auth_headers(token),
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
        payload = body.get("payload", body)
        candles = payload.get("candles")
        if not candles:
            return None
        df = pd.DataFrame(
            candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
        return df[["date", "open", "high", "low", "close", "volume"]]
    except (requests.RequestException, ValueError, KeyError):
        return None
