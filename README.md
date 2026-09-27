# Financial World Simulation

A local pipeline that tries to reason about a stock portfolio the way several different tools would: it reads financial news sentiment, forecasts each ticker's price with two neural network architectures, simulates a crowd of trading agents reacting to those forecasts, and runs a Monte Carlo risk analysis over the result. Everything is viewable in a Streamlit dashboard.

I wanted to see whether I could combine a few separate ideas I'd been reading about (sentiment analysis on news, sequence models for price forecasting, agent-based modeling, and Monte Carlo risk metrics) into one pipeline that all feeds into the same portfolio, instead of building each one as an isolated toy example.

## What it does

The portfolio itself is just an example. `config.py` ships with a fictional allocation across eight tickers spanning semiconductors, pharma, defense, energy, banking, and gold, clearly labeled as an example to edit with your own holdings. The point of the repo isn't that specific allocation, it's the pipeline that runs on top of whatever allocation you put there.

Running `main.py` walks through six stages. Data ingestion pulls OHLCV price history from `yfinance`, macro series from FRED, financial news from Marketaux (if you have a key) or RSS feeds otherwise, geopolitical tension scores from GDELT, and the USD/EUR rate from frankfurter.app, all cached in a local SQLite database so repeated runs are faster. Sentiment analysis scores headlines with FinBERT (`ProsusAI/finbert`, downloaded from Hugging Face on first use) and builds a rolling sentiment index per ticker. Event generation stochastically triggers geopolitical and macro scenarios, such as rate cuts, tariff escalation, or an AI-bubble-burst scenario tied to P/E ratios, using per-ticker impact ranges defined in `config.py`. ML forecasting trains an LSTM and a simplified Temporal Fusion Transformer per ticker and blends their outputs with a historical baseline. Agent-based simulation runs a population of trading agents split across seven behavioral archetypes, trading based on sentiment, momentum, and the triggered events. Portfolio analytics then runs a Monte Carlo simulation over the combined signals to get Value-at-Risk, Conditional VaR, Sharpe ratio, an efficient frontier, and rebalancing suggestions.

The Streamlit dashboard (`modules/dashboard.py`) visualizes all of this: a donut chart of the allocation, fan charts of the Monte Carlo paths per horizon, an event impact waterfall and heatmap, an efficient frontier plot, a rebalancing panel, and a PDF export of the summary (via `reportlab`, if installed).

## How it works

### Sentiment

`sentiment_engine.py` loads FinBERT lazily the first time it's needed and scores each headline in batches. Headlines are matched to tickers using a keyword watchlist (for example, "Jensen Huang" and "Blackwell" for NVDA), then aggregated into a rolling sentiment score per ticker over time.

### Forecasting

`ml_forecasting.py` builds two PyTorch models per ticker on top of about twenty technical indicators, plus sentiment, macro, and cross-asset features such as gold, oil, the SOX index, the dollar index, and the 10-year yield.

`LSTMForecaster` is a 2-layer LSTM with dropout left active at inference time, so running it multiple times gives a Monte Carlo dropout estimate of uncertainty rather than a single point forecast. `SimplifiedTFT` is a smaller version of a Temporal Fusion Transformer, with a variable-selection network that learns which input features matter most and gated residual blocks, outputting several forecast quantiles directly.

The two models' outputs get ensembled together, and the pipeline logs simple validation metrics per horizon (directional accuracy, mean absolute error) so you can see how well a given ticker's forecast actually did against history rather than just trusting the number. Longer horizons are deliberately shrunk toward the historical baseline and volatility is capped, because a 365-day neural network price forecast on noisy financial data is not something I'd take literally. I treat these as scenario ranges, not predictions.

### Agent-based simulation

`agent_simulation.py` assigns each of up to 50,000 simulated agents to one of seven archetypes defined in `config.py`: retail investors chasing momentum, slower passive retail, fundamentals-driven institutions, a fast quant/algo trader, macro-focused hedge funds, sovereign or central-bank-style actors, and insiders who trade around earnings. Each archetype has its own risk tolerance distribution, memory horizon, reaction delay, and herding coefficient, so the same sentiment shock produces different behavior across the population instead of everyone reacting identically.

### Risk analytics

`portfolio_analytics.py` runs a Monte Carlo simulation over the combined forecast and agent signals and computes VaR and CVaR at the 95% and 99% confidence levels, an annualized Sharpe ratio, maximum drawdown, and an efficient frontier via `PyPortfolioOpt`. It also checks for concentration risk: in the example portfolio, the three semiconductor names make up a large fraction of the allocation, and the analytics flag that correlation explicitly rather than treating each ticker as independent.

## Running it

```bash
pip install -r requirements.txt
```

API keys are optional. Copy `.env.example` to `.env` and fill in `FRED_API_KEY` and `MARKETAUX_API_KEY` if you have them. Without them, the pipeline falls back to synthetic macro data and RSS-only news, so it still runs end to end.

```bash
# Quick test with reduced agents and Monte Carlo paths
python main.py --dry-run

# Full pipeline (defaults: 50,000 agents, 10,000 Monte Carlo paths, 30/60/90/365-day horizons)
python main.py

# Custom parameters
python main.py --agents 10000 --mc-paths 5000 --horizons 30 60 90
```

Then launch the dashboard:

```bash
python -m streamlit run modules/dashboard.py
```

The first full run downloads FinBERT (a few hundred MB from Hugging Face) and trains the LSTM/TFT models per ticker, which takes a while on CPU. Later runs reuse the saved checkpoints and cached data, so they're much faster. `--dry-run` is meant for iterating on the code without waiting for the full pipeline each time.

## What I'd flag as limitations

The price forecasts should not be read as predictions anyone should trade on. They're built from public OHLCV history, news sentiment, and a handful of macro series, over a market that reacts to far more than that. The agent-based simulation is a simplification of how real markets aggregate information; the archetypes and their parameters are hand-tuned guesses at plausible behavior, not calibrated against real trading data. Geopolitical event probabilities in `config.py` are also just estimates I set myself, not derived from any model. I built this as a way to combine several techniques in one pipeline and see how they interact, not as a forecasting tool to rely on.
