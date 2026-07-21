"""Persistent vector store of the latest snapshot per watchlisted company.

Backed by Chroma (`chromadb`), persisted to vector_store/ so it can be
committed to the repo alongside the CSVs. One document per company, keyed by
trading_symbol — each day's run_daily.py *upserts* that document (so
"refine" == re-embed with the latest numbers) rather than appending a new
one, and reconcile() deletes any company dropped from the watchlist. The
history of *changes* lives in the dated CSVs under data/daily/, not here.

Embeddings: Google's gemini-embedding-001, needs GOOGLE_API_KEY (same key
used by emailer.py's narrative generation — one credential for both).
"""
from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

VECTOR_STORE_DIR = Path(__file__).parent / "vector_store"
COLLECTION_NAME = "stock_fundamentals"
EMBEDDING_MODEL = "gemini-embedding-001"

# Chroma metadata values must be str/int/float/bool — no None, no dict/list.
_METADATA_TEXT_FIELDS = {
    "business_summary",
    "recent_news",
    "recent_corporate_announcements",
}


def _get_collection():
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    embedding_fn = embedding_functions.GoogleGenaiEmbeddingFunction(
        model_name=EMBEDDING_MODEL, api_key_env_var="GOOGLE_API_KEY"
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )


def _sanitize_metadata(row: dict) -> dict:
    """Drop Nones (unsupported by Chroma) and truncate long free-text fields."""
    metadata = {}
    for key, value in row.items():
        if value is None:
            continue
        if key in _METADATA_TEXT_FIELDS and isinstance(value, str):
            metadata[key] = value[:2000]
        elif isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            metadata[key] = str(value)
    return metadata


def _build_document_text(row: dict) -> str:
    """Human-readable summary of the row — this is what gets embedded and
    what a future RAG query would retrieve/match against."""
    parts = [
        f"{row.get('company_name')} ({row.get('trading_symbol')}, {row.get('exchange')})",
        f"Sector: {row.get('sector')}. Industry: {row.get('industry')}.",
        row.get("business_summary") or "",
        (
            f"Last price {row.get('last_price')}, day change {row.get('day_change_perc')}%. "
            f"P/E {row.get('trailing_pe')}, P/B {row.get('price_to_book')}, "
            f"ROE {row.get('roe')}, debt/equity {row.get('debt_to_equity')}."
        ),
        (
            f"Technical summary: {row.get('technical_summary')} "
            f"(RSI {row.get('rsi_14')} {row.get('rsi_verdict')}, MACD {row.get('macd_verdict')})."
        ),
        f"Analyst recommendation: {row.get('analyst_recommendation')}.",
        f"Recent news: {row.get('recent_news')}" if row.get("recent_news") else "",
        (
            f"Recent corporate announcements: {row.get('recent_corporate_announcements')}"
            if row.get("recent_corporate_announcements")
            else ""
        ),
    ]
    return "\n".join(p for p in parts if p)


def upsert_company(row: dict) -> None:
    """Add or refine (re-embed) this company's document with its latest snapshot."""
    collection = _get_collection()
    collection.upsert(
        ids=[row["trading_symbol"]],
        documents=[_build_document_text(row)],
        metadatas=[_sanitize_metadata(row)],
    )


def delete_company(symbol: str) -> None:
    collection = _get_collection()
    collection.delete(ids=[symbol])


def reconcile(current_symbols: list[str]) -> list[str]:
    """Delete vector-store entries for companies no longer in the watchlist.
    Returns the list of symbols that were removed."""
    collection = _get_collection()
    existing_ids = set(collection.get(include=[])["ids"])
    stale = existing_ids - set(current_symbols)
    if stale:
        collection.delete(ids=list(stale))
    return sorted(stale)


def query(text: str, n_results: int = 5) -> dict:
    """Semantic search over the current watchlist snapshot — for future RAG use."""
    collection = _get_collection()
    return collection.query(query_texts=[text], n_results=n_results)
