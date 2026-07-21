# 18 — Stock Fundamentals Fetcher

**Pattern:** FastAPI backend + live-search frontend, blended data sources, CSV export

Search NSE-listed companies as you type, select the ones you want, and fetch
a broad research snapshot — valuation, growth, profitability, financial
health, dividends, ownership, analyst targets, and price technicals — into a
timestamped CSV under `data/`.

> **Not investment advice.** This tool surfaces raw public data (Groww's own
> market data, plus Yahoo Finance fundamentals/analyst coverage via
> yfinance) so you can do your own research. It doesn't generate buy/sell
> recommendations.

## Why two data sources

[Groww's Trade API](https://groww.in/trade-api/docs) is a **brokerage/trading
API** — instruments, orders, portfolio, margin, live/historical market data.
It has no endpoint for company fundamentals: no EPS, ROE, cash flow, or
business summary anywhere in its docs. So this app blends two sources:

| Data | Source | Auth needed? |
|------|--------|--------------|
| Company search (name, symbol, ISIN) | Groww instrument CSV | No |
| Live price, day change, bid, 52w high/low, circuit limits, volume, market cap | Groww `/v1/live-data/quote` | Yes |
| Daily price history → SMA-50/200, RSI-14, trailing returns, volatility | Groww `/v1/historical/candle/range`, falls back to yfinance if no token | No (better with auth) |
| Valuation, growth, profitability, financial health, dividends, ownership, analyst targets, business summary | [yfinance](https://pypi.org/project/yfinance/) | No |
| Recent news headlines + a crude keyword sentiment count | Google News RSS (`news_client.py`) | No |
| Corporate announcements (board meetings, results dates, filings) | NSE India's public announcements API (`nse_client.py`) | No |
| CEO, founding year, quarterly/yearly revenue & profit trend + growth, dividend history, next earnings date | yfinance (`get_growth_and_events`) | No |
| Promoter vs. public shareholding split | NSE India's shareholding-pattern API (`nse_client.py`) | No |
| MACD, pivot points (support/resistance), expanded moving averages, rule-based bullish/bearish summary | Computed in `technicals.py` from the same price history above | No |

Not fetched, on purpose: the granular FII/DII/mutual-fund shareholding
breakdown and the "which mutual funds hold this stock" list Groww's app
shows — that's Groww's own licensed data (from a paid vendor), not available
from any free public source I could find. Same for F&O data (options chain,
open interest, put/call ratio) — that's a derivatives-trading concern, out of
scope for a fundamentals/investing research tool.

Company search, fundamentals, and technicals all work with **zero
configuration** (technicals fall back to Yahoo Finance price history).
Groww auth adds official exchange data: live bid/price/circuit-limit columns,
and Groww's own daily candles for the technical indicators instead of
Yahoo's.

## Setup

Run from the repo root (`full-stack-ai-with-python/`):

```bash
source venv/bin/activate
pip install -r requirements.txt   # fastapi, uvicorn, pandas, requests, yfinance
cd 08_ai_apps/18_stock_fundamentals
cp .env.example .env
```

To enable Groww auth (live quotes + official historical candles), edit `.env`
with one of:

- **Access Token** (simplest): Groww app → Settings → Trading APIs → Generate
  API keys → Access Token. Paste it as `GROWW_ACCESS_TOKEN`. Expires daily at
  6:00 AM IST — regenerate each day you use the app.
- **API Key + Secret**: from the Groww Cloud API Keys page. Set
  `GROWW_API_KEY` and `GROWW_API_SECRET`; the app computes the required
  SHA-256 checksum and auto-generates a token per run. Still needs daily
  approval on the Groww Cloud API Keys page.

Leave `.env` empty and the app still runs fully — those columns are just
blank/sourced from Yahoo instead.

## Run

```bash
cd 08_ai_apps/18_stock_fundamentals
uvicorn server:app --reload
```

Open http://127.0.0.1:8000 — type a company name or symbol, click matches to
select them, then **Fetch data & save to CSV**. The CSV is written to
`data/companies_fundamentals_<timestamp>.csv` and offered as a download link.

## What's in the CSV

- **Business**: sector, industry, employee count, business summary, CEO/MD
  name, founding year
- **Live quote** (Groww, needs auth): last price, average price, today's
  open/high/low, previous close, day change, bid price/quantity, 52-week
  high/low, upper/lower circuit limits, volume
- **Technicals**: SMA & EMA at 5/10/20/50/100/200 days, above/below flags,
  golden cross, RSI-14 (+ verdict), MACD/signal/histogram (+ verdict),
  classic pivot points (pivot + 3 resistance/support levels), 1m/3m/6m/1y
  trailing returns, annualized volatility, a rule-based bullish/bearish
  summary with signal counts, and which price source was used (`groww` or
  `yfinance`)
- **Valuation**: market cap, enterprise value, trailing/forward P/E, PEG,
  price/sales, price/book, EV/revenue, EV/EBITDA
- **Profitability**: EPS (trailing/forward), book value, ROE, ROA, gross/
  operating/EBITDA/net margins
- **Growth**: revenue growth, earnings growth (annual + quarterly), latest
  quarter revenue/profit with YoY growth, 3-year revenue/profit CAGR
- **Financial health**: debt/equity, current ratio, quick ratio, revenue,
  net income, operating & free cash flow
- **Dividends & events**: yield, payout ratio, last dividend amount/ex-date,
  next known earnings date
- **Risk**: beta, 52-week change, short ratio
- **Ownership**: % held by insiders (yfinance), % held by institutions
  (yfinance), promoter/public shareholding split (NSE, as-of date included)
- **Analyst coverage** (Yahoo's own, not ours): recommendation, analyst
  count, mean/high/low price targets
- **News & corporate actions** (fetched fresh per request, not cached):
  recent headlines with source/date, a rough positive/negative keyword
  count over those headlines, and recent NSE corporate announcements
  (board meetings, results dates, filings)

## Notes

- The instrument list is Groww's full NSE cash-equity CSV, cached locally for
  24 hours (`data/instruments_cache.csv`).
- yfinance fields are best-effort — Yahoo Finance doesn't have complete data
  for every small-cap/micro-cap NSE stock, so some columns may be empty.
- **Cross-check yfinance's valuation figures before trusting them.** Spot-checked
  against Groww's own numbers for TVS Motor: live price matched exactly, and
  CEO/founding year/dividend history/promoter holding/earnings date all
  matched closely — but book value, P/E, and P/B were meaningfully off
  (₹70 vs. Groww's ₹201 book value for the same stock, same day). Yahoo's
  fundamentals data for Indian equities can lag or miss corporate actions.
  Treat price/technicals/events columns as reliable; treat valuation-ratio
  columns as a starting point to verify elsewhere, not ground truth.
- If you're feeding this CSV into an LLM for analysis: say so explicitly in
  the prompt and ask it to flag any figures that look internally
  inconsistent (e.g. P/B not matching price÷book value) rather than taking
  every column at face value.
- News and announcements are fetched live on every request, not cached —
  each is best-effort and degrades to an empty string if the source is
  unreachable or rate-limits the request; it never fails the whole fetch.
- The keyword-based news sentiment count is a rough heuristic, not real NLP
  sentiment analysis — treat it as a nudge to go read the headlines, not a
  signal on its own.
- The Google News RSS feed is, per Google's own terms on the feed, for
  "personal, non-commercial use" — fine for this research tool.
- `data/*.csv` is gitignored — each fetch produces a fresh, local file.
