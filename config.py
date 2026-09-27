"""
config.py — Central configuration for the Stock Portfolio World Simulation Engine.
Hardcodes portfolio, risk profile, agent archetypes, watchlists, and geopolitical events.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(exist_ok=True)

SQLITE_DB = DATA_DIR / "cache.db"

# ---------------------------------------------------------------------------
# API Keys (optional — system degrades gracefully without them)
# ---------------------------------------------------------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY", "")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Portfolio Definition
# ---------------------------------------------------------------------------
# Example portfolio: set your own holdings and amounts here.
PORTFOLIO = {
    "currency": "EUR",
    "total_invested": 10000.00,
    "holdings": {
        "NVDA": {"weight": 0.15, "invested_eur": 1500.00, "sector": "Semiconductors / AI Infrastructure"},
        "AVGO": {"weight": 0.15, "invested_eur": 1500.00, "sector": "Semiconductors / Networking"},
        "LLY":  {"weight": 0.12, "invested_eur": 1200.00, "sector": "Pharmaceuticals / GLP-1 Biotech"},
        "VST":  {"weight": 0.12, "invested_eur": 1200.00, "sector": "Energy / Nuclear Power"},
        "NOC":  {"weight": 0.12, "invested_eur": 1200.00, "sector": "Defense / Aerospace"},
        "MU":   {"weight": 0.12, "invested_eur": 1200.00, "sector": "Semiconductors / Memory"},
        "JPM":  {"weight": 0.12, "invested_eur": 1200.00, "sector": "Financials / Banking"},
        "NEM":  {"weight": 0.10, "invested_eur": 1000.00, "sector": "Gold Mining / Safe Haven"},
    },
}

TICKERS = list(PORTFOLIO["holdings"].keys())

# Sector colour mapping for dashboard
SECTOR_COLOURS = {
    "Semiconductors / AI Infrastructure": "#3B82F6",
    "Semiconductors / Networking":        "#60A5FA",
    "Semiconductors / Memory":            "#93C5FD",
    "Pharmaceuticals / GLP-1 Biotech":    "#22C55E",
    "Energy / Nuclear Power":             "#F97316",
    "Defense / Aerospace":                "#EF4444",
    "Financials / Banking":               "#A855F7",
    "Gold Mining / Safe Haven":           "#EAB308",
}

# ---------------------------------------------------------------------------
# Ticker watchlist keywords for news matching
# ---------------------------------------------------------------------------
WATCHLIST = {
    "NVDA": ["Nvidia", "CUDA", "AI chips", "H100", "Blackwell", "Jensen Huang"],
    "AVGO": ["Broadcom", "VMware", "AI networking", "custom ASIC"],
    "LLY":  ["Eli Lilly", "tirzepatide", "Mounjaro", "Zepbound", "GLP-1", "FDA", "drug pricing"],
    "VST":  ["Vistra", "nuclear energy", "data center power", "Texas grid", "ERCOT"],
    "NOC":  ["Northrop Grumman", "defense budget", "B-21", "GBSD", "Pentagon", "military spending"],
    "MU":   ["Micron", "DRAM", "HBM", "memory chips", "China NAND ban"],
    "JPM":  ["JPMorgan", "Federal Reserve", "interest rates", "banking", "Jamie Dimon", "credit"],
    "NEM":  ["Newmont", "gold price", "XAU", "safe haven", "mining"],
}

# Semiconductor tickers (for concentration risk checks)
SEMI_TICKERS = ["NVDA", "AVGO", "MU"]

# ---------------------------------------------------------------------------
# Cross-asset tickers for model features
# ---------------------------------------------------------------------------
CROSS_ASSETS = {
    "gold":  "GC=F",
    "oil":   "CL=F",
    "sox":   "^SOX",
    "usd":   "DX-Y.NYB",
    "ten_yr":"^TNX",
}

# ---------------------------------------------------------------------------
# FRED macro series
# ---------------------------------------------------------------------------
FRED_SERIES = {
    "fed_funds":      "FEDFUNDS",
    "ten_yr_yield":   "DGS10",
    "cpi_yoy":        "CPIAUCSL",
    "unemployment":   "UNRATE",
    "gdp":            "GDP",
}

# ---------------------------------------------------------------------------
# Agent archetypes
# ---------------------------------------------------------------------------
AGENT_ARCHETYPES = {
    "retail_fomo": {
        "fraction": 0.40,
        "risk_tolerance_alpha": 2.0,
        "risk_tolerance_beta": 5.0,
        "memory_horizon": 7,
        "reaction_delay": 0,
        "herding_coefficient": 0.6,
        "description": "News-reactive, momentum-chasing, panic sells on -5% days",
    },
    "retail_passive": {
        "fraction": 0.20,
        "risk_tolerance_alpha": 3.0,
        "risk_tolerance_beta": 3.0,
        "memory_horizon": 30,
        "reaction_delay": 5,
        "herding_coefficient": 0.3,
        "description": "Slow, monthly rebalancers, low volatility sensitivity",
    },
    "institutional_fundamental": {
        "fraction": 0.15,
        "risk_tolerance_alpha": 5.0,
        "risk_tolerance_beta": 2.0,
        "memory_horizon": 90,
        "reaction_delay": 3,
        "herding_coefficient": 0.1,
        "description": "DCF-driven, mean-reverting, ignores short-term noise",
    },
    "quant_algo": {
        "fraction": 0.10,
        "risk_tolerance_alpha": 4.0,
        "risk_tolerance_beta": 3.0,
        "memory_horizon": 2,
        "reaction_delay": 0,
        "herding_coefficient": 0.15,
        "description": "Momentum + mean-reversion hybrid, reacts instantly",
    },
    "hedge_fund_macro": {
        "fraction": 0.08,
        "risk_tolerance_alpha": 5.0,
        "risk_tolerance_beta": 2.0,
        "memory_horizon": 60,
        "reaction_delay": 1,
        "herding_coefficient": 0.2,
        "description": "Trades macro signals: Fed, geopolitics, FX",
    },
    "central_bank_sovereign": {
        "fraction": 0.04,
        "risk_tolerance_alpha": 8.0,
        "risk_tolerance_beta": 2.0,
        "memory_horizon": 180,
        "reaction_delay": 5,
        "herding_coefficient": 0.05,
        "description": "Policy actors; gold buyers, bond holders",
    },
    "corporate_insider": {
        "fraction": 0.03,
        "risk_tolerance_alpha": 6.0,
        "risk_tolerance_beta": 2.0,
        "memory_horizon": 60,
        "reaction_delay": 2,
        "herding_coefficient": 0.08,
        "description": "Earnings-driven; trades around LLY, NVDA, JPM earnings dates",
    },
}

# ---------------------------------------------------------------------------
# Geopolitical & Macro Events
# ---------------------------------------------------------------------------
GEOPOLITICAL_EVENTS = {
    "iran_us_escalation": {
        "name": "Iran–US Military Escalation",
        "emoji": "🇮🇷🇺🇸",
        "probability_per_quarter": 0.25,
        "duration_days": (30, 120),
        "portfolio_impact": {
            "NOC":  {"direction": 1,  "low": 0.15, "high": 0.35},
            "NEM":  {"direction": 1,  "low": 0.10, "high": 0.25},
            "VST":  {"direction": 1,  "low": 0.05, "high": 0.20},
            "JPM":  {"direction": -1, "low": 0.05, "high": 0.15},
            "NVDA": {"direction": -1, "low": 0.08, "high": 0.20},
            "AVGO": {"direction": -1, "low": 0.05, "high": 0.15},
            "MU":   {"direction": -1, "low": 0.10, "high": 0.25},
            "LLY":  {"direction": 0,  "low": -0.05, "high": 0.05},
        },
    },
    "ai_bubble_burst": {
        "name": "AI Bubble Burst",
        "emoji": "🤖",
        "probability_per_quarter": 0.15,
        "trigger_condition": "NVDA_PE > 40 AND AVGO_PE > 35",
        "duration_days": (60, 365),
        "portfolio_impact": {
            "NVDA": {"direction": -1, "low": 0.35, "high": 0.60},
            "AVGO": {"direction": -1, "low": 0.25, "high": 0.50},
            "MU":   {"direction": -1, "low": 0.30, "high": 0.55},
            "VST":  {"direction": -1, "low": 0.10, "high": 0.25},
            "LLY":  {"direction": 0,  "low": -0.05, "high": 0.10},
            "JPM":  {"direction": -1, "low": 0.05, "high": 0.15},
            "NOC":  {"direction": 0,  "low": -0.05, "high": 0.08},
            "NEM":  {"direction": 1,  "low": 0.05, "high": 0.20},
        },
    },
    "trump_tariff_escalation": {
        "name": "Trump Tariff Escalation (China Semiconductors)",
        "emoji": "🇺🇸",
        "probability_per_quarter": 0.35,
        "duration_days": (90, 270),
        "portfolio_impact": {
            "NVDA": {"direction": -1, "low": 0.10, "high": 0.30},
            "MU":   {"direction": -1, "low": 0.15, "high": 0.35},
            "AVGO": {"direction": -1, "low": 0.08, "high": 0.20},
            "NOC":  {"direction": 1,  "low": 0.05, "high": 0.15},
            "JPM":  {"direction": -1, "low": 0.03, "high": 0.10},
            "LLY":  {"direction": 0,  "low": -0.03, "high": 0.05},
            "VST":  {"direction": 0,  "low": -0.03, "high": 0.05},
            "NEM":  {"direction": 1,  "low": 0.03, "high": 0.12},
        },
    },
    "israel_iran_war": {
        "name": "Israel/Iran Regional War Expansion",
        "emoji": "🇮🇱",
        "probability_per_quarter": 0.20,
        "duration_days": (30, 180),
        "portfolio_impact": {
            "NOC":  {"direction": 1,  "low": 0.20, "high": 0.40},
            "NEM":  {"direction": 1,  "low": 0.15, "high": 0.30},
            "VST":  {"direction": 1,  "low": 0.10, "high": 0.25},
            "JPM":  {"direction": -1, "low": 0.08, "high": 0.18},
            "NVDA": {"direction": -1, "low": 0.05, "high": 0.15},
            "AVGO": {"direction": -1, "low": 0.05, "high": 0.12},
            "MU":   {"direction": -1, "low": 0.08, "high": 0.20},
            "LLY":  {"direction": 0,  "low": -0.03, "high": 0.05},
        },
    },
    "drug_price_regulation": {
        "name": "US Drug Price Regulation (Medicare Negotiation Expansion)",
        "emoji": "💊",
        "probability_per_quarter": 0.30,
        "duration_days": (180, 540),
        "portfolio_impact": {
            "LLY":  {"direction": -1, "low": 0.15, "high": 0.40},
            "JPM":  {"direction": 0,  "low": -0.02, "high": 0.03},
            "NEM":  {"direction": 0,  "low": -0.02, "high": 0.03},
            "NOC":  {"direction": 0,  "low": -0.01, "high": 0.02},
            "NVDA": {"direction": 0,  "low": -0.01, "high": 0.02},
            "AVGO": {"direction": 0,  "low": -0.01, "high": 0.02},
            "MU":   {"direction": 0,  "low": -0.01, "high": 0.02},
            "VST":  {"direction": 0,  "low": -0.01, "high": 0.02},
        },
    },
    "nuclear_energy_boom": {
        "name": "US Nuclear Energy Boom (Policy Tailwind)",
        "emoji": "⚡",
        "probability_per_quarter": 0.40,
        "duration_days": (180, 730),
        "portfolio_impact": {
            "VST":  {"direction": 1,  "low": 0.20, "high": 0.50},
            "NEM":  {"direction": 0,  "low": -0.02, "high": 0.05},
            "NOC":  {"direction": 1,  "low": 0.03, "high": 0.08},
            "JPM":  {"direction": 1,  "low": 0.02, "high": 0.06},
            "NVDA": {"direction": 0,  "low": -0.01, "high": 0.04},
            "AVGO": {"direction": 0,  "low": -0.01, "high": 0.04},
            "MU":   {"direction": 0,  "low": -0.01, "high": 0.02},
            "LLY":  {"direction": 0,  "low": -0.01, "high": 0.02},
        },
    },
    "fed_surprise_rate_cut": {
        "name": "Fed Surprise Rate Cut (≥50bps)",
        "emoji": "🏦",
        "probability_per_quarter": 0.20,
        "duration_days": (30, 90),
        "portfolio_impact": {
            "JPM":  {"direction": -1, "low": 0.05, "high": 0.15},
            "NVDA": {"direction": 1,  "low": 0.08, "high": 0.20},
            "AVGO": {"direction": 1,  "low": 0.06, "high": 0.15},
            "MU":   {"direction": 1,  "low": 0.08, "high": 0.18},
            "LLY":  {"direction": 1,  "low": 0.03, "high": 0.10},
            "NEM":  {"direction": 1,  "low": 0.05, "high": 0.15},
            "VST":  {"direction": 1,  "low": 0.04, "high": 0.12},
            "NOC":  {"direction": 1,  "low": 0.02, "high": 0.06},
        },
    },
}

# ---------------------------------------------------------------------------
# Risk Profile Notes (used in analytics & reports)
# ---------------------------------------------------------------------------
RISK_PROFILE = {
    "semiconductor_concentration": {
        "tickers": ["NVDA", "AVGO", "MU"],
        "total_weight": 0.42,
        "note": "42% of portfolio. High correlation during AI sentiment shifts. "
                "All exposed to US–China chip export restrictions.",
    },
    "geopolitical_hedge": {
        "tickers": ["NOC", "NEM"],
        "total_weight": 0.22,
        "note": "22% of portfolio. Benefits from escalating conflict scenarios — "
                "natural hedge against geopolitical risk.",
    },
    "ai_narrative_dependency": {
        "tickers": ["NVDA", "AVGO", "MU"],
        "total_weight": 0.42,
        "note": "Priced at elevated multiples. Valuations contingent on AI capex "
                "cycles continuing. AI bubble burst would devastate 42% of portfolio.",
    },
    "macro_sensitivity": {
        "note": "JPM is rate-sensitive (benefits from higher-for-longer). "
                "LLY is macro-insensitive but FDA/political drug pricing risk is high. "
                "VST benefits from AI data center power demand (correlated with NVDA narrative).",
    },
    "currency_risk": {
        "note": "All tickers USD-denominated. EUR/USD fluctuations directly impact "
                "real returns for this investor.",
    },
}

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------
DEFAULT_N_AGENTS = 50_000
DEFAULT_MC_PATHS = 10_000
DEFAULT_HORIZONS = [30, 60, 90, 365]  # T+30, T+60, T+90, T+365 days
FINBERT_MODEL = "ProsusAI/finbert"
FX_API_URL = "https://api.frankfurter.app/latest?from=USD&to=EUR"

# ---------------------------------------------------------------------------
# News RSS feeds (fallback when GNews key unavailable)
# ---------------------------------------------------------------------------
RSS_FEEDS = [
    "https://finviz.com/rss.ashx?t={ticker}",
    "https://feeds.benzinga.com/benzinga/news",
    "https://feeds.feedburner.com/seekingalpha-market_currents",
    "https://feeds.feedburner.com/marketwatch/topstories/",
]

# GDELT API endpoint
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
