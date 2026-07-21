"""Price-history-derived technicals: moving averages, RSI, MACD, pivot
points (support/resistance), trailing returns, and a rule-based summary.

Prefers Groww's authenticated historical-candle endpoint (official exchange
data); falls back to yfinance history when no Groww token is configured, so
this still works with zero setup.

The bullish/bearish "technical_summary" is a simple, transparent vote count
over standard indicators (price vs. moving averages, MACD crossover, RSI
zone) — not a black-box score. It's a mechanical read of trend/momentum, not
investment advice.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

import groww_client

MA_PERIODS = [5, 10, 20, 50, 100, 200]

TECHNICAL_FIELDS = (
    [f"sma_{p}" for p in MA_PERIODS]
    + [f"ema_{p}" for p in MA_PERIODS]
    + [
        "above_sma_50",
        "above_sma_200",
        "golden_cross",
        "rsi_14",
        "rsi_verdict",
        "macd",
        "macd_signal",
        "macd_histogram",
        "macd_verdict",
        "pivot",
        "resistance_1",
        "resistance_2",
        "resistance_3",
        "support_1",
        "support_2",
        "support_3",
        "return_1m_pct",
        "return_3m_pct",
        "return_6m_pct",
        "return_1y_pct",
        "annualized_volatility_pct",
        "technical_summary",
        "bullish_signal_count",
        "bearish_signal_count",
        "price_source",
    ]
)

_EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}


def _rsi(closes: pd.Series, period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _rsi_verdict(rsi: float | None) -> str | None:
    if rsi is None:
        return None
    if rsi >= 70:
        return "overbought"
    if rsi <= 30:
        return "oversold"
    return "neutral"


def _macd(closes: pd.Series) -> tuple[float, float, float] | tuple[None, None, None]:
    if len(closes) < 35:
        return None, None, None
    ema_12 = closes.ewm(span=12, adjust=False).mean()
    ema_26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return (
        round(macd_line.iloc[-1], 2),
        round(signal_line.iloc[-1], 2),
        round(histogram.iloc[-1], 2),
    )


def _pivot_points(candles: pd.DataFrame) -> dict:
    """Classic (floor trader) pivot points from the most recent completed daily candle."""
    if (
        candles is None
        or len(candles) < 2
        or not {"high", "low", "close"} <= set(candles.columns)
    ):
        return {}
    last = candles.iloc[-1]
    high, low, close = last["high"], last["low"], last["close"]
    pivot = (high + low + close) / 3
    return {
        "pivot": round(pivot, 2),
        "resistance_1": round(2 * pivot - low, 2),
        "resistance_2": round(pivot + (high - low), 2),
        "resistance_3": round(high + 2 * (pivot - low), 2),
        "support_1": round(2 * pivot - high, 2),
        "support_2": round(pivot - (high - low), 2),
        "support_3": round(low - 2 * (high - pivot), 2),
    }


def _trailing_return(closes: pd.Series, trading_days: int) -> float | None:
    if len(closes) <= trading_days:
        return None
    past = closes.iloc[-trading_days - 1]
    latest = closes.iloc[-1]
    if not past:
        return None
    return round((latest - past) / past * 100, 2)


def _from_yfinance(trading_symbol: str, exchange: str) -> pd.DataFrame | None:
    suffix = _EXCHANGE_SUFFIX.get(exchange, ".NS")
    try:
        hist = yf.Ticker(f"{trading_symbol}{suffix}").history(
            period="2y", interval="1d"
        )
        if hist.empty:
            return None
        hist = hist.reset_index().rename(
            columns={"Date": "date", "High": "high", "Low": "low", "Close": "close"}
        )
        return hist[["date", "high", "low", "close"]]
    except Exception:
        return None


def get_technicals(trading_symbol: str, exchange: str = "NSE") -> dict:
    """Best-effort technicals. Returns a dict with None values on failure."""
    result = {field: None for field in TECHNICAL_FIELDS}

    candles = groww_client.get_historical_candles(trading_symbol, exchange=exchange)
    if candles is not None and len(candles) >= 20:
        result["price_source"] = "groww"
    else:
        candles = _from_yfinance(trading_symbol, exchange)
        if candles is None or len(candles) < 20:
            return result
        result["price_source"] = "yfinance"

    closes = candles["close"]
    latest_close = closes.iloc[-1]

    smas, emas = {}, {}
    for period in MA_PERIODS:
        if len(closes) >= period:
            smas[period] = closes.tail(period).mean()
            result[f"sma_{period}"] = round(smas[period], 2)
        emas[period] = closes.ewm(span=period, adjust=False).mean().iloc[-1]
        result[f"ema_{period}"] = round(emas[period], 2)

    sma_50, sma_200 = smas.get(50), smas.get(200)
    result["above_sma_50"] = bool(latest_close > sma_50) if sma_50 is not None else None
    result["above_sma_200"] = (
        bool(latest_close > sma_200) if sma_200 is not None else None
    )
    result["golden_cross"] = (
        bool(sma_50 > sma_200) if sma_50 is not None and sma_200 is not None else None
    )

    rsi = _rsi(closes)
    result["rsi_14"] = rsi
    result["rsi_verdict"] = _rsi_verdict(rsi)

    macd, macd_signal, macd_hist = _macd(closes)
    result["macd"], result["macd_signal"], result["macd_histogram"] = (
        macd,
        macd_signal,
        macd_hist,
    )
    if macd is not None and macd_signal is not None:
        result["macd_verdict"] = "bullish" if macd > macd_signal else "bearish"

    result.update(_pivot_points(candles))

    result["return_1m_pct"] = _trailing_return(closes, 21)
    result["return_3m_pct"] = _trailing_return(closes, 63)
    result["return_6m_pct"] = _trailing_return(closes, 126)
    result["return_1y_pct"] = _trailing_return(closes, 252)

    daily_returns = closes.pct_change().dropna()
    if len(daily_returns) >= 20:
        result["annualized_volatility_pct"] = round(
            daily_returns.std() * (252**0.5) * 100, 2
        )

    bullish = bearish = 0
    for period in MA_PERIODS:
        for avg in (result.get(f"sma_{period}"), result.get(f"ema_{period}")):
            if avg is None:
                continue
            bullish += latest_close > avg
            bearish += latest_close < avg
    if result["macd_verdict"] == "bullish":
        bullish += 1
    elif result["macd_verdict"] == "bearish":
        bearish += 1
    if rsi is not None:
        if rsi >= 50:
            bullish += 1
        else:
            bearish += 1

    result["bullish_signal_count"] = int(bullish)
    result["bearish_signal_count"] = int(bearish)
    total = bullish + bearish
    if total:
        ratio = bullish / total
        if ratio >= 0.8:
            result["technical_summary"] = "strongly bullish"
        elif ratio >= 0.6:
            result["technical_summary"] = "bullish"
        elif ratio >= 0.4:
            result["technical_summary"] = "neutral"
        elif ratio >= 0.2:
            result["technical_summary"] = "bearish"
        else:
            result["technical_summary"] = "strongly bearish"

    return result
