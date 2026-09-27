"""
Module 2 — Sentiment & News Analysis Engine
Scores headlines with FinBERT, builds rolling sentiment indices,
detects sector-level shocks, and matches watchlist keywords.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FINBERT_MODEL, SEMI_TICKERS, TICKERS, WATCHLIST

logger = logging.getLogger(__name__)
NEUTRAL_SENTIMENT = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}


class SentimentEngine:
    """FinBERT-based sentiment analysis engine for financial headlines."""

    def __init__(self, model_name: str = FINBERT_MODEL):
        self.model_name = model_name
        self._pipeline = None
        self._loaded = False

    def _load_model(self):
        """Load FinBERT model (lazy — only on first use)."""
        if self._loaded:
            return
        logger.info(f"Loading FinBERT model: {self.model_name}...")
        try:
            device = 0 if torch.cuda.is_available() else -1
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=device,
                top_k=None,  # Return all scores
                truncation=True,
                max_length=512,
            )
            self._loaded = True
            logger.info(f"  ✓ FinBERT loaded (device: {'CUDA' if device == 0 else 'CPU'})")
        except Exception as e:
            logger.error(f"  ✗ Failed to load FinBERT: {e}")
            self._pipeline = None
            self._loaded = True  # Don't retry

    def score_headline(self, headline: str) -> Dict[str, float]:
        """Score a single headline → {positive, negative, neutral}."""
        self._load_model()
        if self._pipeline is None:
            return dict(NEUTRAL_SENTIMENT)
        try:
            results = self._pipeline(headline[:512])
            scores = {}
            if results and isinstance(results[0], list):
                for item in results[0]:
                    scores[item["label"].lower()] = item["score"]
            elif results:
                for item in results:
                    scores[item["label"].lower()] = item["score"]
            # Ensure all keys present
            for key in ("positive", "negative", "neutral"):
                scores.setdefault(key, 0.0)
            return scores
        except Exception as e:
            logger.warning(f"  ✗ Scoring failed for headline: {e}")
            return dict(NEUTRAL_SENTIMENT)

    def score_headlines_batch(self, headlines: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """Score multiple headlines in batches for efficiency."""
        self._load_model()
        if self._pipeline is None:
            return [dict(NEUTRAL_SENTIMENT) for _ in headlines]

        all_scores = []
        for i in range(0, len(headlines), batch_size):
            batch = [h[:512] for h in headlines[i:i + batch_size]]
            try:
                results = self._pipeline(batch)
                for result in results:
                    scores = {}
                    if isinstance(result, list):
                        for item in result:
                            scores[item["label"].lower()] = item["score"]
                    else:
                        scores[result["label"].lower()] = result["score"]
                    for key in ("positive", "negative", "neutral"):
                        scores.setdefault(key, 0.0)
                    all_scores.append(scores)
            except Exception as e:
                logger.warning(f"  ✗ Batch scoring failed: {e}")
                all_scores.extend(dict(NEUTRAL_SENTIMENT) for _ in batch)
        return all_scores


def match_headlines_to_tickers(headlines: List[Dict], watchlist: Dict[str, List[str]] = WATCHLIST) -> Dict[str, List[Dict]]:
    """Match headlines to tickers based on watchlist keywords."""
    matched = defaultdict(list)
    for h in headlines:
        title_lower = h.get("title", "").lower()
        for ticker, keywords in watchlist.items():
            if any(kw.lower() in title_lower for kw in keywords):
                matched[ticker].append(h)
    return dict(matched)


def build_rolling_sentiment(
    scored_headlines: List[Dict],
    window_days: int = 7,
    halflife_days: float = 3.0,
) -> pd.DataFrame:
    """Build a 7-day exponentially-weighted rolling sentiment index.

    Args:
        scored_headlines: List of dicts with 'published', 'positive', 'negative', 'neutral'.
        window_days: Rolling window in days.
        halflife_days: Exponential decay halflife (recent news counts more).

    Returns:
        DataFrame with daily sentiment scores (positive, negative, neutral, composite).
    """
    if not scored_headlines:
        return pd.DataFrame(columns=["positive", "negative", "neutral", "composite"])

    records = []
    for h in scored_headlines:
        pub = h.get("published", "")
        try:
            if isinstance(pub, str) and pub:
                dt = pd.to_datetime(pub, utc=True)
            else:
                continue
        except Exception:
            continue
        records.append({
            "date": dt.normalize(),
            "positive": h.get("positive", 0.0),
            "negative": h.get("negative", 0.0),
            "neutral": h.get("neutral", 1.0),
        })

    if not records:
        return pd.DataFrame(columns=["positive", "negative", "neutral", "composite"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")[["positive", "negative", "neutral"]].mean()
    daily = daily.sort_index()

    # Exponentially weighted moving average
    ewm = daily.ewm(halflife=f"{halflife_days}D", times=daily.index).mean()

    # Composite score: positive - negative (range: -1 to +1)
    ewm["composite"] = ewm["positive"] - ewm["negative"]

    return ewm


def detect_sector_shocks(
    all_scored_headlines: Dict[str, List[Dict]],
    sector_keywords: Dict[str, List[str]] = None,
    threshold: int = 3,
    window_hours: int = 24,
) -> Dict[str, bool]:
    """Detect sector-level sentiment shocks.

    If 3+ negative headlines about 'semiconductors' appear in 24h →
    flag correlated sentiment drag on NVDA + AVGO + MU.

    Returns:
        Dict mapping sector names to shock status.
    """
    if sector_keywords is None:
        sector_keywords = {
            "semiconductor": ["semiconductor", "chip", "nvidia", "broadcom", "micron",
                              "AI chip", "GPU", "DRAM", "HBM", "NAND", "TSMC"],
            "defense": ["defense", "military", "pentagon", "weapon", "war"],
            "pharma": ["pharmaceutical", "drug pricing", "FDA", "GLP-1"],
            "energy": ["nuclear energy", "data center power", "grid"],
            "gold": ["gold price", "mining", "safe haven"],
            "banking": ["banking", "federal reserve", "interest rate"],
        }

    shocks = {}
    now = pd.Timestamp.utcnow()
    cutoff = now - timedelta(hours=window_hours)

    # Collect all headlines across tickers
    all_headlines = []
    for ticker, headlines in all_scored_headlines.items():
        all_headlines.extend(headlines)

    for sector, keywords in sector_keywords.items():
        negative_count = 0
        for h in all_headlines:
            title_lower = h.get("title", "").lower()
            if not any(kw.lower() in title_lower for kw in keywords):
                continue

            # Check recency
            pub = h.get("published", "")
            try:
                dt = pd.to_datetime(pub, utc=True)
                if dt < cutoff:
                    continue
            except Exception:
                continue  # Can't parse date, skip recency check

            # Check if negative
            neg_score = h.get("negative", 0)
            if neg_score > 0.5:
                negative_count += 1

        shocks[sector] = negative_count >= threshold

    return shocks


def apply_sector_shock_drag(
    sentiment_scores: Dict[str, float],
    shocks: Dict[str, bool],
    drag_factor: float = 0.15,
) -> Dict[str, float]:
    """Apply correlated sentiment drag when sector shocks are detected.

    If semiconductor shock is active, drag NVDA, AVGO, MU sentiment down.
    """
    adjusted = dict(sentiment_scores)

    sector_ticker_map = {
        "semiconductor": SEMI_TICKERS,
        "defense": ["NOC"],
        "pharma": ["LLY"],
        "energy": ["VST"],
        "gold": ["NEM"],
        "banking": ["JPM"],
    }

    for sector, is_shocked in shocks.items():
        if not is_shocked:
            continue
        affected_tickers = sector_ticker_map.get(sector, [])
        for ticker in affected_tickers:
            if ticker in adjusted:
                adjusted[ticker] = float(np.clip(adjusted[ticker] - drag_factor, -1.0, 1.0))
                logger.info(f"  ⚠ Sector shock [{sector}] → {ticker} sentiment dragged by {drag_factor}")

    return adjusted


# ---------------------------------------------------------------------------
# Master sentiment analysis function
# ---------------------------------------------------------------------------

def run_sentiment_analysis(
    news_data: Dict[str, List[Dict]],
) -> Dict[str, Any]:
    """Run full sentiment analysis pipeline.

    Args:
        news_data: Dict mapping ticker → list of headline dicts.

    Returns:
        Dict with scored headlines, rolling sentiment, shocks, and composite scores.
    """
    logger.info("=" * 60)
    logger.info("MODULE 2 — Sentiment & News Analysis Engine")
    logger.info("=" * 60)

    engine = SentimentEngine()

    # Score all headlines per ticker
    scored_by_ticker = {}
    for ticker in TICKERS:
        headlines = news_data.get(ticker, [])
        if not headlines:
            scored_by_ticker[ticker] = []
            logger.info(f"  {ticker}: No headlines to score")
            continue

        titles = [h.get("title", "") for h in headlines if h.get("title")]
        if not titles:
            scored_by_ticker[ticker] = []
            continue

        logger.info(f"  Scoring {len(titles)} headlines for {ticker}...")
        scores = engine.score_headlines_batch(titles)

        # Merge scores back into headline dicts
        scored = []
        for h, s in zip(headlines, scores):
            merged = {**h, **s}
            scored.append(merged)
        scored_by_ticker[ticker] = scored
        logger.info(f"  ✓ {ticker}: avg sentiment = "
                     f"pos:{np.mean([s['positive'] for s in scores]):.3f} "
                     f"neg:{np.mean([s['negative'] for s in scores]):.3f}")

    # Build rolling sentiment per ticker
    logger.info("\n  Building rolling sentiment indices...")
    rolling_sentiment = {}
    for ticker in TICKERS:
        rolling = build_rolling_sentiment(scored_by_ticker.get(ticker, []))
        rolling_sentiment[ticker] = rolling

    # Compute current composite score per ticker
    composite_scores = {}
    for ticker in TICKERS:
        rs = rolling_sentiment[ticker]
        if not rs.empty and "composite" in rs.columns:
            composite_scores[ticker] = float(rs["composite"].iloc[-1])
        else:
            composite_scores[ticker] = 0.0

    # Detect sector shocks
    logger.info("\n  Checking for sector-level sentiment shocks...")
    shocks = detect_sector_shocks(scored_by_ticker)
    active_shocks = [s for s, v in shocks.items() if v]
    if active_shocks:
        logger.info(f"  ⚠ Active sector shocks: {active_shocks}")
    else:
        logger.info(f"  ✓ No sector shocks detected")

    # Apply shock drag
    adjusted_scores = apply_sector_shock_drag(composite_scores, shocks)

    result = {
        "scored_headlines": scored_by_ticker,
        "rolling_sentiment": rolling_sentiment,
        "composite_scores": adjusted_scores,
        "raw_composite_scores": composite_scores,
        "sector_shocks": shocks,
        "engine": engine,
    }

    logger.info("\n✓ Sentiment analysis complete.")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Quick test with dummy data
    dummy_news = {
        ticker: [
            {"title": f"{ticker} stock rises on strong earnings", "published": datetime.utcnow().isoformat(), "source": "Test"},
            {"title": f"{ticker} faces regulatory headwinds", "published": datetime.utcnow().isoformat(), "source": "Test"},
        ]
        for ticker in TICKERS
    }
    result = run_sentiment_analysis(dummy_news)
    print(f"\nComposite scores: {result['composite_scores']}")
    print(f"Sector shocks: {result['sector_shocks']}")
