# 🌍 Stock Portfolio World Simulation Engine

A production-ready, modular stock portfolio forecasting system combining **machine learning**, **agent-based simulation**, and **real-world data ingestion** — all running locally with zero paid APIs.

## 📋 Features

- **8-ticker EUR portfolio** with hardcoded sector-aware risk profiles
- **FinBERT sentiment analysis** on live financial news
- **LSTM + Temporal Fusion Transformer** per-ticker forecasting
- **50,000-agent** market simulation across 7 behavioural archetypes
- **7 geopolitical event** stochastic scenarios (Iran, AI bubble, tariffs, etc.)
- **Monte Carlo analytics**: VaR, CVaR, Sharpe, efficient frontier
- **Streamlit dashboard** with dark theme, interactive charts, PDF export

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Set API Keys

```bash
cp .env.example .env
# Edit .env with your FRED_API_KEY and MARKETAUX_API_KEY
# The system works without them via graceful fallback
```

### 3. Run the Pipeline

```bash
# Quick test (reduced agents, fast ML training)
python main.py --dry-run

# Full pipeline (takes 15-30 min on first run)
python main.py

# Custom parameters
python main.py --agents 10000 --mc-paths 5000 --horizons 30 60 90
```

### 4. Launch Dashboard

```bash
python -m streamlit run modules/dashboard.py
```

## 🏗️ Architecture

```
Financial World Simulation/
├── main.py                      # Orchestrator
├── config.py                    # Portfolio, events, constants
├── requirements.txt
├── .env.example
├── data/cache.db                # SQLite cache (auto-created)
├── models/                      # Saved model weights (auto-created)
├── reports/                     # PDF reports (auto-created)
└── modules/
    ├── data_ingestion.py        # yfinance, FRED, Marketaux, RSS, FX
    ├── sentiment_engine.py      # FinBERT scoring, rolling sentiment
    ├── ml_forecasting.py        # LSTM + TFT, 22 technical indicators
    ├── agent_simulation.py      # 50K vectorised agents, 7 archetypes
    ├── event_generator.py       # 7 geopolitical event scenarios
    ├── portfolio_analytics.py   # MC simulation, VaR, efficient frontier
    └── dashboard.py             # Streamlit UI with Plotly charts
```

## 💼 Portfolio (example — edit `config.py`)

| Ticker | Weight | Sector | Invested (EUR) |
|--------|--------|--------|----------------|
| NVDA | 15% | Semiconductors / AI Infrastructure | €1,500 |
| AVGO | 15% | Semiconductors / Networking | €1,500 |
| LLY | 12% | Pharmaceuticals / GLP-1 Biotech | €1,200 |
| VST | 12% | Energy / Nuclear Power | €1,200 |
| NOC | 12% | Defense / Aerospace | €1,200 |
| MU | 12% | Semiconductors / Memory | €1,200 |
| JPM | 12% | Financials / Banking | €1,200 |
| NEM | 10% | Gold Mining / Safe Haven | €1,000 |
| **TOTAL** | **100%** | | **€10,000** |

## 🌐 Simulated Events

1. 🇮🇷🇺🇸 Iran–US Military Escalation (25%/qtr)
2. 🤖 AI Bubble Burst (15%/qtr, conditional on P/E)
3. 🇺🇸 Trump Tariff Escalation (35%/qtr)
4. 🇮🇱 Israel/Iran Regional War (20%/qtr)
5. 💊 US Drug Price Regulation (30%/qtr)
6. ⚡ US Nuclear Energy Boom (40%/qtr)
7. 🏦 Fed Surprise Rate Cut ≥50bps (20%/qtr)

## 🔑 API Keys

| API | Purpose | Required? |
|-----|---------|-----------|
| [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) | Macro indicators (CPI, rates) | Optional — falls back to synthetic |
| [Marketaux](https://www.marketaux.com/) | Financial news | Optional — falls back to RSS |
| [frankfurter.app](https://frankfurter.app/) | USD/EUR FX rate | No key needed |
| [yfinance](https://pypi.org/project/yfinance/) | OHLCV data | No key needed |
| [GDELT](https://www.gdeltproject.org/) | Geopolitical tension | No key needed |

## ⚠️ Notes

- **First run** downloads FinBERT (~500MB) from HuggingFace
- **Full run** (50K agents + 10K MC paths) takes 15-30 min on CPU
- Use `--dry-run` for fast iteration (1K agents, 500 MC paths)
- All data is cached in SQLite; subsequent runs are faster
- Models are saved to `models/` and reused automatically on later runs when the feature shape still matches
- News now uses Marketaux when a key is present, then falls back to Finviz/Benzinga/Seeking Alpha/MarketWatch RSS
- For the best first-run experience, warm the cache with `python main.py --dry-run` before launching the dashboard
- `365`-day runs are supported, but should be treated as scenario ranges rather than precise forecasts
- The simulator now shrinks long-horizon forecasts, caps volatility/shocks, and limits rebalance turnover to keep outputs more realistic
- On Windows, launch the dashboard with `python -m streamlit run modules/dashboard.py` if `streamlit` is not on your PATH
- First run trains and saves LSTM/TFT checkpoints; reruns should be much faster because the dashboard and CLI now reuse them automatically
- Forecasts now blend a historical baseline with ML outputs, and each ticker logs simple validation metrics like directional accuracy and MAE for the selected horizons
- The dashboard caches ingestion, sentiment, event generation, forecasting, agent simulation, and analytics results to speed up repeated runs with the same settings
- The dashboard now includes a forecast-validation table and a benchmark comparison panel against buy-and-hold and historical-bootstrap baselines
