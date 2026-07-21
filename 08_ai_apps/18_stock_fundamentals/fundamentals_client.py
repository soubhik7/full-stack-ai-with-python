"""Fundamentals (EPS, ROE, cash flow, business summary, ...) via yfinance.

Groww's Trade API doesn't expose fundamentals at all — see groww_client.py's
module docstring. yfinance is free, needs no API key, and maps cleanly onto
NSE/BSE tickers (SYMBOL.NS / SYMBOL.BO), so it fills that gap.
"""
from __future__ import annotations

import re

import pandas as pd
import yfinance as yf

_EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}

_FOUNDED_RE = re.compile(
    r"\b(?:founded|incorporated|established)\s+in\s+(\d{4})\b", re.IGNORECASE
)
_CEO_TITLE_KEYWORDS = ("ceo", "chief executive", "managing director", "md")

EVENTS_FIELDS = [
    "ceo_name",
    "founded_year",
    "latest_quarter_revenue",
    "latest_quarter_revenue_yoy_growth_pct",
    "latest_quarter_profit",
    "latest_quarter_profit_yoy_growth_pct",
    "revenue_cagr_3y_pct",
    "profit_cagr_3y_pct",
    "last_dividend_amount",
    "last_dividend_ex_date",
    "next_earnings_date",
]

FUNDAMENTALS_FIELDS = [
    # Business
    "sector",
    "industry",
    "longBusinessSummary",
    "fullTimeEmployees",
    # Valuation
    "marketCap",
    "enterpriseValue",
    "trailingPE",
    "forwardPE",
    "pegRatio",
    "priceToSalesTrailing12Months",
    "priceToBook",
    "enterpriseToRevenue",
    "enterpriseToEbitda",
    # Per-share / profitability
    "trailingEps",
    "forwardEps",
    "bookValue",
    "returnOnEquity",
    "returnOnAssets",
    "grossMargins",
    "operatingMargins",
    "ebitdaMargins",
    "profitMargins",
    # Growth
    "revenueGrowth",
    "earningsGrowth",
    "earningsQuarterlyGrowth",
    # Financial health
    "debtToEquity",
    "currentRatio",
    "quickRatio",
    "totalRevenue",
    "netIncomeToCommon",
    "operatingCashflow",
    "freeCashflow",
    # Dividends
    "dividendYield",
    "payoutRatio",
    # Risk / momentum
    "beta",
    "fiftyDayAverage",
    "twoHundredDayAverage",
    "fiftyTwoWeekChange",
    "shortRatio",
    # Ownership
    "heldPercentInsiders",
    "heldPercentInstitutions",
    # Analyst sentiment (Yahoo Finance's own analyst coverage, not our opinion)
    "recommendationKey",
    "numberOfAnalystOpinions",
    "targetMeanPrice",
    "targetHighPrice",
    "targetLowPrice",
]


def get_fundamentals(trading_symbol: str, exchange: str = "NSE") -> dict:
    """Best-effort fundamentals fetch. Returns a dict with None values on failure."""
    suffix = _EXCHANGE_SUFFIX.get(exchange, ".NS")
    result = {field: None for field in FUNDAMENTALS_FIELDS}

    try:
        ticker = yf.Ticker(f"{trading_symbol}{suffix}")
        info = ticker.info or {}
        for field in FUNDAMENTALS_FIELDS:
            result[field] = info.get(field)
    except Exception:
        pass

    return result


def _extract_ceo_name(officers: list[dict]) -> str | None:
    for officer in officers:
        title = (officer.get("title") or "").lower()
        if any(keyword in title for keyword in _CEO_TITLE_KEYWORDS):
            return officer.get("name")
    return officers[0].get("name") if officers else None


def _yoy_growth(series, latest_idx: int = 0, lag: int = 4) -> float | None:
    """% change vs. the same period `lag` entries back (quarters -> YoY at lag=4)."""
    if series is None or len(series) <= latest_idx + lag:
        return None
    latest = series.iloc[latest_idx]
    prior = series.iloc[latest_idx + lag]
    if not prior or pd.isna(latest) or pd.isna(prior):
        return None
    return round((latest - prior) / abs(prior) * 100, 2)


def _cagr(series, years: int = 3) -> float | None:
    if series is None or len(series) <= years:
        return None
    latest = series.iloc[0]
    past = series.iloc[years]
    if not past or past < 0 or pd.isna(latest) or pd.isna(past):
        return None
    return round(((latest / past) ** (1 / years) - 1) * 100, 2)


def get_growth_and_events(trading_symbol: str, exchange: str = "NSE") -> dict:
    """Best-effort: CEO, founding year, quarterly/yearly revenue & profit trend
    with growth, dividend history, and next known earnings date.

    Returns a dict with None values on failure — never raises.
    """
    suffix = _EXCHANGE_SUFFIX.get(exchange, ".NS")
    result = {field: None for field in EVENTS_FIELDS}

    try:
        ticker = yf.Ticker(f"{trading_symbol}{suffix}")
        info = ticker.info or {}
    except Exception:
        info = {}

    result["ceo_name"] = _extract_ceo_name(info.get("companyOfficers") or [])
    summary = info.get("longBusinessSummary") or ""
    match = _FOUNDED_RE.search(summary)
    result["founded_year"] = int(match.group(1)) if match else None

    try:
        qf = ticker.quarterly_financials
        if qf is not None and "Total Revenue" in qf.index:
            revenue_row = qf.loc["Total Revenue"]
            result["latest_quarter_revenue"] = float(revenue_row.iloc[0])
            result["latest_quarter_revenue_yoy_growth_pct"] = _yoy_growth(revenue_row)
        if qf is not None and "Net Income" in qf.index:
            profit_row = qf.loc["Net Income"]
            result["latest_quarter_profit"] = float(profit_row.iloc[0])
            result["latest_quarter_profit_yoy_growth_pct"] = _yoy_growth(profit_row)
    except Exception:
        pass

    try:
        yearly = ticker.financials
        if yearly is not None and "Total Revenue" in yearly.index:
            result["revenue_cagr_3y_pct"] = _cagr(yearly.loc["Total Revenue"])
        if yearly is not None and "Net Income" in yearly.index:
            result["profit_cagr_3y_pct"] = _cagr(yearly.loc["Net Income"])
    except Exception:
        pass

    try:
        dividends = ticker.dividends
        if dividends is not None and len(dividends) > 0:
            result["last_dividend_amount"] = float(dividends.iloc[-1])
            result["last_dividend_ex_date"] = dividends.index[-1].strftime("%Y-%m-%d")
    except Exception:
        pass

    try:
        calendar = ticker.calendar or {}
        earnings_dates = calendar.get("Earnings Date")
        if earnings_dates:
            result["next_earnings_date"] = str(earnings_dates[0])
    except Exception:
        pass

    return result
