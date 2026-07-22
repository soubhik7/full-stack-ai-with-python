# 18 — Stock Fundamentals Fetcher

**Pattern:** FastAPI backend + live-search frontend, blended data sources, CSV export

Search NSE-listed companies as you type, select the ones you want, and fetch
a broad research snapshot — valuation, growth, profitability, financial
health, dividends, ownership, analyst targets, and price technicals — into a
timestamped CSV under `data/`.

> **Not investment advice.** The interactive app (this section) surfaces raw
> public data so you can do your own research — it doesn't generate buy/sell
> recommendations. The optional [daily automation](#daily-automation-github-actions)
> below *does* generate an LLM narrative recommendation, by explicit choice
> when this was built — see that section for what that means and its risks
> before enabling it.

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
pip3 install -r requirements.txt   # fastapi, uvicorn, pandas, requests, yfinance
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
Click **Save as watchlist** to persist your current selection to
`watchlist.json` — that's the file the daily automation below reads.

## Daily automation (GitHub Actions)

`run_daily.py` is a headless script — no server needed — that: fetches every
company in `watchlist.json`, writes a dated CSV snapshot, upserts each
company into a committed vector store, and emails you an LLM-written digest.
`.github/workflows/stock-watchlist-daily.yml` runs it on a cron schedule
(default: 08:00 IST, Mon–Fri) and commits the new data back to the repo.

**This is off by default.** The workflow file exists in the repo, but it only
runs once you add the secrets below in your GitHub repo's Settings → Secrets
and variables → Actions. Test it manually first via the Actions tab → "Daily
Stock Watchlist Digest" → **Run workflow**, before trusting the schedule.

### 1. Build a watchlist

Either use **Save as watchlist** in the running app, or edit
`watchlist.json` directly:

```json
{ "symbols": ["RELIANCE", "TCS", "INFY"] }
```

Symbols are Groww trading symbols (same ones `/api/search` returns).

### 2. Add repo secrets

| Secret | Required? | Where to get it |
|--------|-----------|------------------|
| `GOOGLE_API_KEY` | Yes | [Google AI Studio](https://aistudio.google.com/apikey) — powers both the vector store embeddings and the LLM narrative (Gemini) |
| `EMAIL_FROM` | Yes | A Gmail address you control |
| `EMAIL_APP_PASSWORD` | Yes | [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) — a 16-character App Password, **not** your normal Gmail password (needs 2-Step Verification enabled on that account) |
| `EMAIL_TO` | Yes | Where the digest gets sent — can be the same as `EMAIL_FROM` |
| `GROWW_ACCESS_TOKEN` | No | Live quotes + official candles. **Expires daily at 6 AM IST** — a token secret you set once will go stale; the API Key + Secret flow below is the only one that keeps working unattended |
| `GROWW_API_KEY` / `GROWW_API_SECRET` | No | Auto-refreshes daily, so this is the one that actually works for an unattended schedule. Still needs manual daily approval on the Groww Cloud API Keys page — read Groww's docs on what that approval step means for automation before relying on it |

Without any Groww secret, the workflow still runs fully — quote columns
(price, bid, circuit limits) are blank, and technicals fall back to Yahoo
Finance price history, exactly like the interactive app.

### 3. What gets committed back to the repo, daily

- `data/daily/<YYYY-MM-DD>.csv` — that day's full snapshot (append-only archive)
- `data/watchlist_latest.csv` — always overwritten with the latest snapshot
- `vector_store/` — Chroma's persisted files (SQLite + data), **upserted** per
  company (not appended) — see below

**Repo growth**: `vector_store/`'s files change every run, and git doesn't
diff binary SQLite files efficiently — the repo's `.git` history will grow
daily and won't shrink on its own. Fine for a personal/learning repo over
normal timeframes; if it matters long-term, periodically squash history or
move the vector store to an external DB instead of committing it.

### How the vector store lifecycle works ("update/add/refine/delete")

One document per company, keyed by trading symbol, in `vector_store.py`:

- **Add / update / refine** — every run, each watchlisted company's document
  is **upserted** (`vector_store.upsert_company`): re-embedded with that
  day's numbers, replacing the old vector. There's one current document per
  company, not a growing pile of daily snapshots — the *history* of changes
  lives in the dated CSVs, not in the vector store.
- **Delete** — `vector_store.reconcile()` runs after every fetch and deletes
  any company whose vector exists but is no longer in `watchlist.json` — so
  removing a symbol from your watchlist and re-running cleans it up
  automatically.

This is a Chroma collection, not RAG wired into anything yet — `vector_store.query()`
is there for you (or a future app) to build semantic search / RAG over the
watchlist later, per your original ask about feeding this into an LLM.

### About the recommendation email

You explicitly chose an LLM-written narrative over a plain data digest when
this was built, so `emailer.py` calls Gemini with the day's full watchlist
snapshot and a system prompt that requires it to cite specific numbers from
the data, forbids inventing figures or promising profit, and asks it to flag
internally-inconsistent data rather than gloss over it. A disclaimer banner
is hard-coded into the email template — it always appears regardless of what
the model writes.

None of that makes it reliable. It's a language model's synthesis of data
that — per the yfinance-accuracy note below — is itself sometimes wrong. Read
the digest as "here's what an AI noticed in today's numbers," not as
research from a professional analyst, and definitely not as a guarantee. If
the LLM call fails for any reason, the email still sends with just the raw
data table and a note that the narrative was unavailable — see `run_daily.py`.

### Testing it without waiting for the cron

```bash
cd 08_ai_apps/18_stock_fundamentals
python run_daily.py
```

Runs the exact same code the workflow runs, using your local `.env` (this
app's own `.env` first, then the repo-root `.env` for any shared keys).
Sends a real email if `EMAIL_*` secrets are set — expect that.

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
- `data/*.csv` is gitignored for ad-hoc UI fetches (each one produces a fresh
  local file); the daily automation's own outputs are deliberately tracked —
  see [Daily automation](#daily-automation-github-actions) above.
- **Secrets security**: GitHub Actions secrets are encrypted and redacted
  from logs automatically — the workflow YAML itself (visible if the repo is
  public) never contains a credential value, only secret *names*. If you use
  the Groww API Key + Secret flow, be aware that's a live trading-account
  credential sitting in your GitHub repo's secret store; use whatever scope
  restriction Groww's dashboard offers and treat repo/org access control
  accordingly.
