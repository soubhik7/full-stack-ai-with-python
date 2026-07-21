"""Per-company data fetch: blends every source in this app into one row.

Shared by server.py (interactive UI) and run_daily.py (headless automation)
so both paths build the exact same row shape from a single place.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

import fundamentals_client
import groww_client
import news_client
import nse_client
import technicals

_FORMULA_LEAD_CHARS = ("=", "+", "-", "@", "\t", "\r")


def _defuse_formula(value):
    """Neutralize CSV-formula-injection risk in text pulled from external sources
    (news headlines, announcement text) before it lands in a spreadsheet cell."""
    if isinstance(value, str) and value.startswith(_FORMULA_LEAD_CHARS):
        return "'" + value
    return value


def fetch_company_row(symbol: str, instruments_df: pd.DataFrame) -> dict | None:
    """Fetch and blend all data sources for one company. None if symbol isn't found."""
    match = instruments_df[instruments_df["trading_symbol"] == symbol]
    if match.empty:
        return None
    instrument = match.iloc[0]
    exchange = instrument["exchange"]

    quote = groww_client.get_quote(symbol, exchange=exchange) or {}
    fundamentals = fundamentals_client.get_fundamentals(symbol, exchange=exchange)
    events = fundamentals_client.get_growth_and_events(symbol, exchange=exchange)
    tech = technicals.get_technicals(symbol, exchange=exchange)
    news_items = news_client.get_news(instrument["name"])
    sentiment = news_client.score_sentiment(news_items)
    announcements = nse_client.get_corporate_announcements(symbol)
    shareholding = nse_client.get_shareholding_pattern(symbol)
    time.sleep(0.3)  # be polite to yfinance/Google News/NSE

    previous_close = None
    if quote.get("last_price") is not None and quote.get("day_change") is not None:
        previous_close = round(quote["last_price"] - quote["day_change"], 2)

    row = {
        "trading_symbol": symbol,
        "groww_symbol": instrument["groww_symbol"],
        "company_name": instrument["name"],
        "exchange": exchange,
        "isin": instrument["isin"],
        "sector": fundamentals["sector"],
        "industry": fundamentals["industry"],
        "business_summary": fundamentals["longBusinessSummary"],
        "employees": fundamentals["fullTimeEmployees"],
        "ceo_name": events["ceo_name"],
        "founded_year": events["founded_year"],
        # Live quote (Groww, needs auth)
        "last_price": quote.get("last_price"),
        "average_price": quote.get("average_price"),
        "day_open": quote.get("day_open"),
        "day_high": quote.get("day_high"),
        "day_low": quote.get("day_low"),
        "previous_close": previous_close,
        "day_change": quote.get("day_change"),
        "day_change_perc": quote.get("day_change_perc"),
        "bid_price": quote.get("bid_price"),
        "bid_quantity": quote.get("bid_quantity"),
        "week_52_high": quote.get("week_52_high"),
        "week_52_low": quote.get("week_52_low"),
        "upper_circuit_limit": quote.get("upper_circuit_limit"),
        "lower_circuit_limit": quote.get("lower_circuit_limit"),
        "volume": quote.get("volume"),
        # Technicals (Groww historical candles if authenticated, else yfinance)
        "price_source": tech["price_source"],
        **{f"sma_{p}": tech[f"sma_{p}"] for p in technicals.MA_PERIODS},
        **{f"ema_{p}": tech[f"ema_{p}"] for p in technicals.MA_PERIODS},
        "above_sma_50": tech["above_sma_50"],
        "above_sma_200": tech["above_sma_200"],
        "golden_cross": tech["golden_cross"],
        "rsi_14": tech["rsi_14"],
        "rsi_verdict": tech["rsi_verdict"],
        "macd": tech["macd"],
        "macd_signal": tech["macd_signal"],
        "macd_histogram": tech["macd_histogram"],
        "macd_verdict": tech["macd_verdict"],
        "pivot": tech["pivot"],
        "resistance_1": tech["resistance_1"],
        "resistance_2": tech["resistance_2"],
        "resistance_3": tech["resistance_3"],
        "support_1": tech["support_1"],
        "support_2": tech["support_2"],
        "support_3": tech["support_3"],
        "return_1m_pct": tech["return_1m_pct"],
        "return_3m_pct": tech["return_3m_pct"],
        "return_6m_pct": tech["return_6m_pct"],
        "return_1y_pct": tech["return_1y_pct"],
        "annualized_volatility_pct": tech["annualized_volatility_pct"],
        "technical_summary": tech["technical_summary"],
        "bullish_signal_count": tech["bullish_signal_count"],
        "bearish_signal_count": tech["bearish_signal_count"],
        # Valuation
        "market_cap": quote.get("market_cap") or fundamentals["marketCap"],
        "enterprise_value": fundamentals["enterpriseValue"],
        "trailing_pe": fundamentals["trailingPE"],
        "forward_pe": fundamentals["forwardPE"],
        "peg_ratio": fundamentals["pegRatio"],
        "price_to_sales": fundamentals["priceToSalesTrailing12Months"],
        "price_to_book": fundamentals["priceToBook"],
        "ev_to_revenue": fundamentals["enterpriseToRevenue"],
        "ev_to_ebitda": fundamentals["enterpriseToEbitda"],
        # Per-share / profitability
        "trailing_eps": fundamentals["trailingEps"],
        "forward_eps": fundamentals["forwardEps"],
        "book_value": fundamentals["bookValue"],
        "roe": fundamentals["returnOnEquity"],
        "roa": fundamentals["returnOnAssets"],
        "gross_margin": fundamentals["grossMargins"],
        "operating_margin": fundamentals["operatingMargins"],
        "ebitda_margin": fundamentals["ebitdaMargins"],
        "profit_margin": fundamentals["profitMargins"],
        # Growth
        "revenue_growth": fundamentals["revenueGrowth"],
        "earnings_growth": fundamentals["earningsGrowth"],
        "earnings_quarterly_growth": fundamentals["earningsQuarterlyGrowth"],
        # Financial health
        "debt_to_equity": fundamentals["debtToEquity"],
        "current_ratio": fundamentals["currentRatio"],
        "quick_ratio": fundamentals["quickRatio"],
        "total_revenue": fundamentals["totalRevenue"],
        "net_income": fundamentals["netIncomeToCommon"],
        "operating_cash_flow": fundamentals["operatingCashflow"],
        "free_cash_flow": fundamentals["freeCashflow"],
        # Quarterly/yearly financial trend (Groww's "Financial performance" chart)
        "latest_quarter_revenue": events["latest_quarter_revenue"],
        "latest_quarter_revenue_yoy_growth_pct": events[
            "latest_quarter_revenue_yoy_growth_pct"
        ],
        "latest_quarter_profit": events["latest_quarter_profit"],
        "latest_quarter_profit_yoy_growth_pct": events[
            "latest_quarter_profit_yoy_growth_pct"
        ],
        "revenue_cagr_3y_pct": events["revenue_cagr_3y_pct"],
        "profit_cagr_3y_pct": events["profit_cagr_3y_pct"],
        # Dividends & events
        "dividend_yield": fundamentals["dividendYield"],
        "payout_ratio": fundamentals["payoutRatio"],
        "last_dividend_amount": events["last_dividend_amount"],
        "last_dividend_ex_date": events["last_dividend_ex_date"],
        "next_earnings_date": events["next_earnings_date"],
        # Risk
        "beta": fundamentals["beta"],
        "fifty_two_week_change": fundamentals["fiftyTwoWeekChange"],
        "short_ratio": fundamentals["shortRatio"],
        # Ownership
        "held_percent_insiders": fundamentals["heldPercentInsiders"],
        "held_percent_institutions": fundamentals["heldPercentInstitutions"],
        "promoter_holding_pct": shareholding["promoter_holding_pct"],
        "public_holding_pct": shareholding["public_holding_pct"],
        "shareholding_as_of": shareholding["shareholding_as_of"],
        # Analyst sentiment (Yahoo Finance's own coverage, shown as-is — not investment advice)
        "analyst_recommendation": fundamentals["recommendationKey"],
        "analyst_count": fundamentals["numberOfAnalystOpinions"],
        "analyst_target_mean": fundamentals["targetMeanPrice"],
        "analyst_target_high": fundamentals["targetHighPrice"],
        "analyst_target_low": fundamentals["targetLowPrice"],
        # News & corporate actions (fetched fresh per request)
        "recent_news": news_client.format_for_csv(news_items),
        "news_positive_signals": sentiment["positive"],
        "news_negative_signals": sentiment["negative"],
        "recent_corporate_announcements": nse_client.format_for_csv(announcements),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return {k: _defuse_formula(v) for k, v in row.items()}
