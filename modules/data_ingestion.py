"""
Module 1 — Data Ingestion Layer
Pulls and caches: OHLCV (yfinance), news (Marketaux/RSS), macro (FRED),
geopolitical tension (GDELT), FX rates (frankfurter.app).
All data stored in local SQLite cache with TTL-based expiry.
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser
import numpy as np
import pandas as pd
import requests
import yfinance as yf

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CROSS_ASSETS,
    DATA_DIR,
    FRED_API_KEY,
    FRED_SERIES,
    FX_API_URL,
    GDELT_API_URL,
    MARKETAUX_API_KEY,
    PORTFOLIO,
    RANDOM_SEED,
    RSS_FEEDS,
    SQLITE_DB,
    TICKERS,
    WATCHLIST,
)

logger = logging.getLogger(__name__)
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PortfolioWorldSimulation/1.0)"}


# ---------------------------------------------------------------------------
# SQLite cache helpers
# ---------------------------------------------------------------------------

class DataCache:
    """SQLite-backed cache with TTL support."""

    def __init__(self, db_path: Path = SQLITE_DB):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    ttl_seconds REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    adj_close REAL, volume REAL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS headlines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    headline TEXT NOT NULL,
                    source TEXT,
                    published TEXT,
                    url TEXT,
                    fetched_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_data (
                    series_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    value REAL,
                    PRIMARY KEY (series_id, date)
                )
            """)
            conn.commit()

    def get(self, key: str) -> Optional[Any]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value, timestamp, ttl_seconds FROM cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        value, ts, ttl = row
        if time.time() - ts > ttl:
            return None
        return json.loads(value)

    def set(self, key: str, value: Any, ttl_seconds: float = 3600):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, timestamp, ttl_seconds) "
                "VALUES (?, ?, ?, ?)",
                (key, json.dumps(value, default=str), time.time(), ttl_seconds),
            )
            conn.commit()

    def store_ohlcv(self, ticker: str, df: pd.DataFrame):
        with self._conn() as conn:
            for _, row in df.iterrows():
                conn.execute(
                    "INSERT OR REPLACE INTO ohlcv "
                    "(ticker, date, open, high, low, close, adj_close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ticker,
                        str(row.name.date()) if hasattr(row.name, 'date') else str(row.name),
                        float(row.get("Open", 0)),
                        float(row.get("High", 0)),
                        float(row.get("Low", 0)),
                        float(row.get("Close", 0)),
                        float(row.get("Adj Close", row.get("Close", 0))),
                        float(row.get("Volume", 0)),
                    ),
                )
            conn.commit()

    def load_ohlcv(self, ticker: str) -> Optional[pd.DataFrame]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT date, open, high, low, close, adj_close, volume "
                "FROM ohlcv WHERE ticker = ? ORDER BY date",
                (ticker,),
            ).fetchall()
        if not rows:
            return None
        df = pd.DataFrame(
            rows, columns=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        )
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df

    def store_headlines(self, ticker: str, headlines: List[Dict]):
        with self._conn() as conn:
            for h in headlines:
                conn.execute(
                    "INSERT INTO headlines (ticker, headline, source, published, url, fetched_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        ticker,
                        h.get("title", ""),
                        h.get("source", ""),
                        h.get("published", ""),
                        h.get("url", ""),
                        time.time(),
                    ),
                )
            conn.commit()

    def load_headlines(self, ticker: str, days: int = 30) -> List[Dict]:
        cutoff = time.time() - days * 86400
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT headline, source, published, url FROM headlines "
                "WHERE ticker = ? AND fetched_at > ? ORDER BY fetched_at DESC",
                (ticker, cutoff),
            ).fetchall()
        return [
            {"title": r[0], "source": r[1], "published": r[2], "url": r[3]}
            for r in rows
        ]

    def store_macro(self, series_id: str, df: pd.DataFrame):
        with self._conn() as conn:
            for date_val, row in df.items():
                if pd.notna(row):
                    conn.execute(
                        "INSERT OR REPLACE INTO macro_data (series_id, date, value) "
                        "VALUES (?, ?, ?)",
                        (series_id, str(date_val.date()) if hasattr(date_val, 'date') else str(date_val), float(row)),
                    )
            conn.commit()

    def load_macro(self, series_id: str) -> Optional[pd.Series]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT date, value FROM macro_data WHERE series_id = ? ORDER BY date",
                (series_id,),
            ).fetchall()
        if not rows:
            return None
        s = pd.Series(
            [r[1] for r in rows],
            index=pd.to_datetime([r[0] for r in rows]),
            name=series_id,
        )
        return s


# ---------------------------------------------------------------------------
# Data Ingestion Functions
# ---------------------------------------------------------------------------

_cache = DataCache()


def fetch_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance with SQLite fallback."""
    try:
        logger.info(f"Fetching OHLCV for {ticker} from yfinance...")
        data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        if data is not None and not data.empty:
            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            _cache.store_ohlcv(ticker, data)
            logger.info(f"  ✓ {ticker}: {len(data)} rows cached")
            return data
    except Exception as e:
        logger.warning(f"  ✗ yfinance failed for {ticker}: {e}")

    # Fallback to cache
    cached = _cache.load_ohlcv(ticker)
    if cached is not None:
        logger.info(f"  ↺ Using cached OHLCV for {ticker}: {len(cached)} rows")
        return cached
    logger.error(f"  ✗ No data available for {ticker}")
    return pd.DataFrame()


def fetch_all_ohlcv() -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for all portfolio tickers + cross-asset tickers."""
    all_tickers = TICKERS + list(CROSS_ASSETS.values())
    result = {}
    for t in all_tickers:
        result[t] = fetch_ohlcv(t)
    return result


def fetch_fx_rate() -> float:
    """Fetch live USD/EUR rate from frankfurter.app."""
    cache_key = "fx_usd_eur"
    cached = _cache.get(cache_key)
    if cached is not None:
        return float(cached)
    try:
        resp = requests.get(FX_API_URL, timeout=10)
        resp.raise_for_status()
        rate = resp.json()["rates"]["EUR"]
        _cache.set(cache_key, rate, ttl_seconds=3600)
        logger.info(f"  ✓ USD/EUR rate: {rate}")
        return float(rate)
    except Exception as e:
        logger.warning(f"  ✗ FX fetch failed: {e}. Using default 0.92")
        return 0.92


def _normalise_headline(title: str, source: str, published: str, url: str) -> Dict:
    return {
        "title": title or "",
        "source": source or "",
        "published": published or "",
        "url": url or "",
    }


def fetch_news_marketaux(ticker: str, days: int = 30, limit: int = 10) -> List[Dict]:
    """Fetch ticker news from Marketaux API (if key is available)."""
    if not MARKETAUX_API_KEY or MARKETAUX_API_KEY == "your_marketaux_api_key_here":
        return []

    published_after = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M")
    try:
        resp = requests.get(
            "https://api.marketaux.com/v1/news/all",
            params={
                "symbols": ticker,
                "published_after": published_after,
                "language": "en",
                "limit": min(limit, 50),
                "filter_entities": "true",
                "group_similar": "true",
                "api_token": MARKETAUX_API_KEY,
            },
            timeout=15,
            headers=REQUEST_HEADERS,
        )
        resp.raise_for_status()
        articles = resp.json().get("data", [])
        headlines = []
        for a in articles:
            title = a.get("title") or a.get("snippet") or a.get("description") or ""
            if not title:
                continue
            source = a.get("source") or a.get("domain") or "Marketaux"
            headlines.append(_normalise_headline(
                title=title,
                source=source,
                published=a.get("published_at", ""),
                url=a.get("url", ""),
            ))
        return headlines
    except Exception as e:
        logger.warning(f"  ✗ Marketaux failed for {ticker}: {e}")
        return []


def fetch_news_rss(ticker: str) -> List[Dict]:
    """Fetch news from RSS feeds (free, no key required)."""
    headlines = []
    keywords = [ticker.lower()] + [kw.lower() for kw in WATCHLIST.get(ticker, [])]
    for feed_url_template in RSS_FEEDS:
        url = feed_url_template.format(ticker=ticker)
        try:
            feed = feedparser.parse(url, request_headers=REQUEST_HEADERS)
            for entry in feed.entries[:10]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                haystack = f"{title} {summary}".lower()
                if not any(keyword in haystack for keyword in keywords):
                    continue
                headlines.append(_normalise_headline(
                    title=title,
                    source=feed.feed.get("title", "RSS"),
                    published=entry.get("published", "") or entry.get("updated", ""),
                    url=entry.get("link", ""),
                ))
        except Exception as e:
            logger.warning(f"  ✗ RSS feed failed ({url}): {e}")
    return headlines


def fetch_news(ticker: str) -> List[Dict]:
    """Fetch news for a ticker using Marketaux (if available) + RSS feeds.
    Results are cached in SQLite."""
    cache_key = f"news_{ticker}_{datetime.now().strftime('%Y%m%d')}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    headlines = []
    # Try Marketaux first
    marketaux_articles = fetch_news_marketaux(ticker)
    headlines.extend(marketaux_articles)
    # Always supplement with RSS
    rss_articles = fetch_news_rss(ticker)
    headlines.extend(rss_articles)

    # Deduplicate by title
    seen = set()
    unique = []
    for h in headlines:
        title_key = h["title"].strip().lower()
        if title_key and title_key not in seen:
            seen.add(title_key)
            unique.append(h)
    headlines = unique

    if headlines:
        _cache.store_headlines(ticker, headlines)
        _cache.set(cache_key, headlines, ttl_seconds=43200)  # 12h TTL
        logger.info(f"  ✓ {ticker}: {len(headlines)} headlines fetched")
    else:
        # Fallback to cached headlines
        cached_headlines = _cache.load_headlines(ticker)
        if cached_headlines:
            logger.info(f"  ↺ Using {len(cached_headlines)} cached headlines for {ticker}")
            return cached_headlines
        logger.warning(f"  ✗ No headlines for {ticker}")

    return headlines


def fetch_all_news() -> Dict[str, List[Dict]]:
    """Fetch news for all portfolio tickers."""
    return {t: fetch_news(t) for t in TICKERS}


def fetch_macro_fred() -> Dict[str, pd.Series]:
    """Fetch macro indicators from FRED API."""
    result = {}
    if not FRED_API_KEY or FRED_API_KEY == "your_fred_api_key_here":
        logger.warning("  ✗ No FRED API key. Falling back to cache / synthetic data.")
        rng = np.random.RandomState(RANDOM_SEED)
        for name, series_id in FRED_SERIES.items():
            cached = _cache.load_macro(series_id)
            if cached is not None:
                result[name] = cached
            else:
                # Generate synthetic placeholder
                dates = pd.date_range(end=datetime.now(), periods=252 * 5, freq="B")
                if "rate" in name or "yield" in name or "funds" in name:
                    result[name] = pd.Series(
                        rng.normal(4.5, 0.5, len(dates)).cumsum() * 0.001 + 4.0,
                        index=dates, name=series_id,
                    )
                elif "cpi" in name:
                    result[name] = pd.Series(
                        np.linspace(260, 315, len(dates)) + rng.normal(0, 0.5, len(dates)),
                        index=dates, name=series_id,
                    )
                elif "unemployment" in name:
                    result[name] = pd.Series(
                        rng.normal(3.8, 0.3, len(dates)).clip(3.0, 6.0),
                        index=dates, name=series_id,
                    )
                else:
                    result[name] = pd.Series(
                        rng.normal(0, 1, len(dates)),
                        index=dates, name=series_id,
                    )
        return result

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        for name, series_id in FRED_SERIES.items():
            try:
                data = fred.get_series(series_id, observation_start="2019-01-01")
                if data is not None and not data.empty:
                    _cache.store_macro(series_id, data)
                    result[name] = data
                    logger.info(f"  ✓ FRED {name}: {len(data)} observations")
                else:
                    cached = _cache.load_macro(series_id)
                    if cached is not None:
                        result[name] = cached
            except Exception as e:
                logger.warning(f"  ✗ FRED {name} failed: {e}")
                cached = _cache.load_macro(series_id)
                if cached is not None:
                    result[name] = cached
    except ImportError:
        logger.warning("  ✗ fredapi not installed")

    return result


def fetch_gdelt_tension(keywords: Optional[List[str]] = None) -> pd.DataFrame:
    """Fetch geopolitical tension / conflict scores from GDELT."""
    cache_key = f"gdelt_{datetime.now().strftime('%Y%m%d')}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return pd.DataFrame(cached)

    if keywords is None:
        keywords = [
            "military conflict", "war", "sanctions", "trade war",
            "chip ban", "TSMC", "export control", "nuclear",
        ]

    try:
        dfs = []
        for kw in keywords[:3]:  # Limit to avoid rate limiting
            params = {
                "query": kw,
                "mode": "TimelineVolInfo",
                "format": "json",
                "timespan": "3m",
            }
            resp = requests.get(GDELT_API_URL, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if "timeline" in data and len(data["timeline"]) > 0:
                    series_data = data["timeline"][0].get("data", [])
                    for point in series_data:
                        dfs.append({
                            "date": point.get("date", ""),
                            "keyword": kw,
                            "volume": point.get("value", 0),
                        })
        if dfs:
            df = pd.DataFrame(dfs)
            _cache.set(cache_key, df.to_dict(orient="records"), ttl_seconds=86400)
            logger.info(f"  ✓ GDELT: {len(df)} data points")
            return df
    except Exception as e:
        logger.warning(f"  ✗ GDELT fetch failed: {e}")

    # Return empty DataFrame with expected columns
    return pd.DataFrame(columns=["date", "keyword", "volume"])


def fetch_semiconductor_news() -> List[Dict]:
    """Fetch semiconductor export restriction news via RSS keyword filter."""
    keywords = ["chip ban", "TSMC", "export control", "semiconductor restriction",
                 "Nvidia ban", "China chip"]
    headlines = []
    for feed_url in RSS_FEEDS:
        try:
            url = feed_url.format(ticker="NVDA")
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.get("title", "").lower()
                if any(kw.lower() in title for kw in keywords):
                    headlines.append({
                        "title": entry.get("title", ""),
                        "source": feed.feed.get("title", "RSS"),
                        "published": entry.get("published", ""),
                        "url": entry.get("link", ""),
                    })
        except Exception:
            pass
    return headlines


# ---------------------------------------------------------------------------
# Master ingestion function
# ---------------------------------------------------------------------------

def run_ingestion() -> Dict[str, Any]:
    """Run full data ingestion pipeline. Returns all fetched data."""
    logger.info("=" * 60)
    logger.info("MODULE 1 — Data Ingestion Layer")
    logger.info("=" * 60)

    # 1. OHLCV data
    logger.info("\n[1/6] Fetching OHLCV data (5yr)...")
    ohlcv = fetch_all_ohlcv()

    # 2. FX rate
    logger.info("\n[2/6] Fetching USD/EUR exchange rate...")
    fx_rate = fetch_fx_rate()

    # 3. News
    logger.info("\n[3/6] Fetching news headlines...")
    news = fetch_all_news()

    # 4. Macro data
    logger.info("\n[4/6] Fetching macro indicators (FRED)...")
    macro = fetch_macro_fred()

    # 5. GDELT geopolitical tension
    logger.info("\n[5/6] Fetching GDELT geopolitical tension...")
    gdelt = fetch_gdelt_tension()

    # 6. Semiconductor export restriction news
    logger.info("\n[6/6] Fetching semiconductor restriction news...")
    semi_news = fetch_semiconductor_news()

    result = {
        "ohlcv": ohlcv,
        "fx_rate": fx_rate,
        "news": news,
        "macro": macro,
        "gdelt": gdelt,
        "semi_news": semi_news,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info("\n✓ Data ingestion complete.")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = run_ingestion()
    print(f"\nTickers with OHLCV data: {[t for t, df in data['ohlcv'].items() if not df.empty]}")
    print(f"USD/EUR rate: {data['fx_rate']}")
    print(f"Headlines per ticker: {({t: len(h) for t, h in data['news'].items()})}")
    print(f"Macro series: {list(data['macro'].keys())}")
