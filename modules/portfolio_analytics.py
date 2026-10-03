"""
Module 6 — Portfolio Analytics & Optimisation Layer
Monte Carlo simulation, VaR/CVaR, Sharpe, drawdown, scenario analysis,
concentration risk, rebalancing, and efficient frontier.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEFAULT_HORIZONS,
    DEFAULT_MC_PATHS,
    GEOPOLITICAL_EVENTS,
    PORTFOLIO,
    RANDOM_SEED,
    RISK_PROFILE,
    SEMI_TICKERS,
    TICKERS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monte Carlo Portfolio Simulation
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """Runs Monte Carlo simulations combining ML forecasts, agent dynamics, and events."""

    def __init__(self, n_paths: int = DEFAULT_MC_PATHS, seed: int = RANDOM_SEED):
        self.n_paths = n_paths
        self.rng = np.random.RandomState(seed)

    def simulate_paths(
        self,
        ticker: str,
        initial_price: float,
        forecast_data: Optional[Dict] = None,
        agent_sim_data: Optional[Dict] = None,
        event_scenario: Optional[Dict] = None,
        horizon: int = 90,
        historical_vol: float = 0.02,
    ) -> np.ndarray:
        """Generate Monte Carlo price paths for a single ticker.

        Combines:
        - ML forecast distribution (mean & std from LSTM/TFT)
        - Agent simulation price pressure
        - Event shock sampling

        Returns:
            Array of shape (n_paths, horizon+1) — simulated price paths.
        """
        paths = np.zeros((self.n_paths, horizon + 1))
        paths[:, 0] = initial_price

        # Extract forecast parameters
        if forecast_data and horizon in forecast_data.get("forecasts", {}):
            fc = forecast_data["forecasts"][horizon]
            ensemble = fc.get("ensemble_percentiles", {})
            forecast_target = ensemble.get("p50", initial_price)
            forecast_std = fc.get("lstm_std", initial_price * historical_vol * np.sqrt(horizon))
            annual_vol = fc.get("annualised_volatility", historical_vol * np.sqrt(252))
        else:
            forecast_target = initial_price
            forecast_std = initial_price * historical_vol * np.sqrt(horizon)
            annual_vol = historical_vol * np.sqrt(252)

        forecast_target, annual_vol = stabilise_forecast_inputs(
            initial_price=initial_price,
            forecast_target=float(forecast_target),
            annual_vol=float(annual_vol),
            historical_vol=historical_vol,
            horizon=horizon,
        )

        daily_vol = annual_vol / np.sqrt(252)

        # Daily drift from forecast in log-return space
        if initial_price > 0:
            total_log_return = np.log(np.clip(forecast_target / initial_price, 1e-6, 10.0))
            daily_drift = total_log_return / horizon
        else:
            daily_drift = 0

        # Agent price pressure (if available)
        agent_bias = 0.0
        if agent_sim_data:
            final_agent_price = agent_sim_data.get("prices", [initial_price])[-1]
            agent_return = (final_agent_price / initial_price - 1) if initial_price > 0 else 0
            agent_bias = float(np.clip(agent_return / horizon * 0.2, -0.01, 0.01))

        # Event shock (if active)
        event_shock_mean = 0.0
        event_shock_std = 0.0
        if event_scenario:
            for event_key, event_data in event_scenario.items():
                if isinstance(event_data, dict):
                    impacts = event_data.get("impacts", {})
                    if ticker in impacts:
                        imp = impacts[ticker]
                        if isinstance(imp, dict):
                            event_shock_mean += imp.get("p50", 0.0)
                            event_shock_std += abs(imp.get("p90", 0.0) - imp.get("p10", 0.0)) / 4

        # Simulate paths using Geometric Brownian Motion
        for t in range(1, horizon + 1):
            # Combined drift
            mu = daily_drift + agent_bias
            mu = float(np.clip(mu, -0.05, 0.05))

            # Random innovations
            z = self.rng.standard_normal(self.n_paths)
            daily_log_return = mu + daily_vol * z

            # Apply event shock at random point during event
            if event_shock_mean != 0:
                event_day = self.rng.randint(1, max(2, horizon // 3))
                if t == event_day:
                    shock = self.rng.normal(event_shock_mean, max(event_shock_std, 0.02), self.n_paths)
                    daily_log_return += np.clip(shock, -0.5, 0.5)

            daily_log_return = np.clip(daily_log_return, -0.25, 0.25)
            paths[:, t] = paths[:, t - 1] * np.exp(daily_log_return)
            paths[:, t] = np.maximum(paths[:, t], 0.01)

        return paths


def stabilise_forecast_inputs(
    initial_price: float,
    forecast_target: float,
    annual_vol: float,
    historical_vol: float,
    horizon: int,
) -> Tuple[float, float]:
    """Bound forecast inputs before they enter long-horizon Monte Carlo."""
    if initial_price <= 0:
        return max(forecast_target, 0.01), max(annual_vol, 0.05)

    daily_vol_floor = max(historical_vol, 0.01)
    max_total_log_move = max(4.0 * daily_vol_floor * np.sqrt(horizon), 0.30)

    safe_target = float(np.clip(forecast_target, 0.01, initial_price * np.exp(max_total_log_move)))
    safe_target = float(max(safe_target, initial_price * np.exp(-max_total_log_move)))

    annual_vol_floor = max(daily_vol_floor * np.sqrt(252), 0.10)
    annual_vol_cap = min(max(3.0 * daily_vol_floor * np.sqrt(252), 0.35), 1.50)
    safe_annual_vol = float(np.clip(annual_vol, annual_vol_floor, annual_vol_cap))

    return safe_target, safe_annual_vol


# ---------------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------------

def compute_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Value at Risk at given confidence level."""
    return float(-np.percentile(returns, (1 - confidence) * 100))


def compute_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall)."""
    var = compute_var(returns, confidence)
    tail = returns[returns <= -var]
    return float(-np.mean(tail)) if len(tail) > 0 else var


def compute_max_drawdown(price_path: np.ndarray) -> float:
    """Maximum drawdown of a price path."""
    peak = np.maximum.accumulate(price_path)
    drawdown = (peak - price_path) / np.where(peak != 0, peak, 1)
    return float(np.max(drawdown))


def compute_sharpe(returns: np.ndarray, risk_free_annual: float = 0.04) -> float:
    """Annualised Sharpe ratio."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    daily_rf = risk_free_annual / 252
    excess = returns - daily_rf
    std = np.std(excess)
    # a constant series can have a std of ~1e-18 from rounding, which would blow the ratio up
    if std <= 1e-12 * max(1.0, float(np.max(np.abs(excess)))):
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def _summarise_distribution(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=float)
    return {
        "expected_return_eur": float(np.mean(values)),
        "var_95_eur": float(compute_var(values, 0.95)),
        "cvar_95_eur": float(compute_cvar(values, 0.95)),
        "percentiles_eur": {
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
        },
    }


def _compute_benchmarks(
    ohlcv_data: Dict[str, pd.DataFrame],
    holdings: Dict[str, Dict[str, Any]],
    horizons: List[int],
    n_paths: int,
    seed: int,
) -> Dict[int, Dict[str, Any]]:
    rng = np.random.RandomState(seed + 99)
    benchmarks: Dict[int, Dict[str, Any]] = {}

    for h in horizons:
        buy_hold_pnl = np.zeros(n_paths, dtype=float)
        bootstrap_pnl = np.zeros(n_paths, dtype=float)

        for ticker in TICKERS:
            df = ohlcv_data.get(ticker, pd.DataFrame())
            if df.empty or "Close" not in df.columns:
                continue

            invested = float(holdings[ticker]["invested_eur"])
            close = df["Close"].dropna().astype(float)
            daily_returns = close.pct_change().dropna().to_numpy(dtype=float)
            if len(daily_returns) == 0:
                continue

            horizon_returns = (close.shift(-h) / close - 1.0).dropna().to_numpy(dtype=float)
            if len(horizon_returns) == 0:
                sampled_horizon = np.full(n_paths, np.mean(daily_returns) * h)
            else:
                sampled_horizon = rng.choice(horizon_returns, size=n_paths, replace=True)
            sampled_horizon = np.clip(sampled_horizon, -0.85, 1.5)
            buy_hold_pnl += invested * sampled_horizon

            sampled_daily = rng.choice(daily_returns, size=(n_paths, h), replace=True)
            sampled_daily = np.clip(sampled_daily, -0.20, 0.20)
            bootstrap_returns = np.prod(1.0 + sampled_daily, axis=1) - 1.0
            bootstrap_returns = np.clip(bootstrap_returns, -0.90, 2.0)
            bootstrap_pnl += invested * bootstrap_returns

        benchmarks[h] = {
            "buy_and_hold": _summarise_distribution(buy_hold_pnl),
            "historical_bootstrap": _summarise_distribution(bootstrap_pnl),
        }

    return benchmarks


# ---------------------------------------------------------------------------
# Portfolio Analytics
# ---------------------------------------------------------------------------

def run_portfolio_analytics(
    ohlcv_data: Dict[str, pd.DataFrame],
    forecast_data: Dict[str, Any],
    agent_sim_data: Dict[str, Any],
    event_data: Dict[str, Any],
    fx_rate: float = 0.92,
    n_paths: int = DEFAULT_MC_PATHS,
    horizons: List[int] = None,
    seed: int = RANDOM_SEED,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run full portfolio analytics.

    Returns comprehensive analytics dict with:
    - Per-ticker return distributions
    - Portfolio-level metrics
    - Scenario impact table
    - Concentration risk analysis
    - Rebalancing recommendations
    - Efficient frontier data
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    if dry_run:
        n_paths = min(n_paths, 500)

    logger.info("=" * 60)
    logger.info("MODULE 6 — Portfolio Analytics & Optimisation")
    logger.info("=" * 60)
    logger.info(f"  MC paths: {n_paths:,} | Horizons: {horizons} | FX: {fx_rate:.4f}")

    mc = MonteCarloEngine(n_paths=n_paths, seed=seed)
    scenario_table = event_data.get("scenario_table", {})
    holdings = PORTFOLIO["holdings"]

    # 1. Per-ticker MC simulation
    logger.info("\n  [1/6] Running Monte Carlo simulations per ticker...")
    ticker_paths = {}
    ticker_metrics = {}

    for ticker in TICKERS:
        df = ohlcv_data.get(ticker, pd.DataFrame())
        if df.empty:
            continue

        initial_price = float(df["Close"].iloc[-1])
        returns = df["Close"].pct_change().dropna()
        hist_vol = float(returns.std()) if len(returns) > 20 else 0.02

        fc = forecast_data.get(ticker) if forecast_data else None
        agent = agent_sim_data.get(ticker) if agent_sim_data else None

        horizon_results = {}
        for h in horizons:
            paths = mc.simulate_paths(
                ticker=ticker,
                initial_price=initial_price,
                forecast_data=fc,
                agent_sim_data=agent,
                event_scenario=scenario_table,
                horizon=h,
                historical_vol=hist_vol,
            )

            final_prices = paths[:, -1]
            returns_pct = (final_prices / initial_price - 1)
            returns_eur = returns_pct * holdings[ticker]["invested_eur"]

            horizon_results[h] = {
                "paths": paths,
                "final_prices": final_prices,
                "returns_pct": returns_pct,
                "returns_eur": returns_eur,
                "percentiles": {
                    "p10": float(np.percentile(final_prices, 10)),
                    "p25": float(np.percentile(final_prices, 25)),
                    "p50": float(np.percentile(final_prices, 50)),
                    "p75": float(np.percentile(final_prices, 75)),
                    "p90": float(np.percentile(final_prices, 90)),
                },
                "return_percentiles_eur": {
                    "p10": float(np.percentile(returns_eur, 10)),
                    "p25": float(np.percentile(returns_eur, 25)),
                    "p50": float(np.percentile(returns_eur, 50)),
                    "p75": float(np.percentile(returns_eur, 75)),
                    "p90": float(np.percentile(returns_eur, 90)),
                },
                "expected_return_eur": float(np.mean(returns_eur)),
                "var_95": compute_var(returns_pct, 0.95),
                "var_99": compute_var(returns_pct, 0.99),
                "cvar_95": compute_cvar(returns_pct, 0.95),
                "initial_price": initial_price,
            }

        ticker_paths[ticker] = horizon_results

        # Compute ticker-level metrics from T+90
        t90 = horizon_results.get(90, horizon_results.get(horizons[-1], {}))
        if t90:
            all_returns = t90["returns_pct"]
            max_dd = np.array([compute_max_drawdown(p) for p in t90["paths"]])
            daily_returns = np.diff(t90["paths"], axis=1) / np.where(t90["paths"][:, :-1] != 0, t90["paths"][:, :-1], 1.0)
            sharpe = compute_sharpe(daily_returns.reshape(-1))

            ticker_metrics[ticker] = {
                "expected_return": float(np.mean(all_returns)),
                "volatility": float(np.std(all_returns)),
                "var_95": t90["var_95"],
                "var_99": t90["var_99"],
                "cvar_95": t90["cvar_95"],
                "max_drawdown_mean": float(np.mean(max_dd)),
                "max_drawdown_p95": float(np.percentile(max_dd, 95)),
                "sharpe": sharpe,
            }
            logger.info(f"  ✓ {ticker}: E[r]={np.mean(all_returns)*100:+.1f}% | "
                         f"VaR95={t90['var_95']*100:.1f}% | Sharpe={sharpe:.2f}")

    # 2. Portfolio-level aggregation
    logger.info("\n  [2/6] Computing portfolio-level metrics...")
    portfolio_metrics = {}
    for h in horizons:
        portfolio_returns_eur = np.zeros(n_paths)
        portfolio_paths = np.full((n_paths, h + 1), PORTFOLIO["total_invested"], dtype=float)
        for ticker in TICKERS:
            if ticker in ticker_paths and h in ticker_paths[ticker]:
                portfolio_returns_eur += ticker_paths[ticker][h]["returns_eur"]
                invested = holdings[ticker]["invested_eur"]
                initial_price = ticker_paths[ticker][h]["initial_price"]
                scaled_paths = invested * ticker_paths[ticker][h]["paths"] / max(initial_price, 0.01)
                portfolio_paths += scaled_paths - invested

        portfolio_returns_pct = portfolio_returns_eur / PORTFOLIO["total_invested"]
        portfolio_daily_returns = np.diff(portfolio_paths, axis=1) / np.where(portfolio_paths[:, :-1] != 0, portfolio_paths[:, :-1], 1.0)
        portfolio_sharpe = compute_sharpe(portfolio_daily_returns.reshape(-1))
        portfolio_max_drawdown = np.array([compute_max_drawdown(path) for path in portfolio_paths])

        portfolio_metrics[h] = {
            "expected_return_eur": float(np.mean(portfolio_returns_eur)),
            "expected_return_pct": float(np.mean(portfolio_returns_pct)),
            "var_95_eur": float(compute_var(portfolio_returns_eur, 0.95)),
            "var_99_eur": float(compute_var(portfolio_returns_eur, 0.99)),
            "cvar_95_eur": float(compute_cvar(portfolio_returns_eur, 0.95)),
            "var_95_pct": compute_var(portfolio_returns_pct, 0.95),
            "var_99_pct": compute_var(portfolio_returns_pct, 0.99),
            "sharpe": portfolio_sharpe,
            "max_drawdown_mean": float(np.mean(portfolio_max_drawdown)),
            "max_drawdown_p95": float(np.percentile(portfolio_max_drawdown, 95)),
            "return_distribution": portfolio_returns_eur,
            "percentiles_eur": {
                "p10": float(np.percentile(portfolio_returns_eur, 10)),
                "p25": float(np.percentile(portfolio_returns_eur, 25)),
                "p50": float(np.percentile(portfolio_returns_eur, 50)),
                "p75": float(np.percentile(portfolio_returns_eur, 75)),
                "p90": float(np.percentile(portfolio_returns_eur, 90)),
            },
        }
        logger.info(f"  T+{h}: E[PnL]=€{np.mean(portfolio_returns_eur):+.2f} | "
                     f"VaR95=€{compute_var(portfolio_returns_eur, 0.95):.2f} | "
                     f"CVaR95=€{compute_cvar(portfolio_returns_eur, 0.95):.2f}")

    # 3. Scenario impact table
    logger.info("\n  [3/6] Computing scenario impact table...")
    scenario_impact = {}
    for event_key, event_data_entry in scenario_table.items():
        event_portfolio_pnl = np.zeros(n_paths)
        ticker_impact = {}
        for ticker in TICKERS:
            imp = event_data_entry.get("impacts", {}).get(ticker, {})
            p10 = imp.get("p10", 0)
            p50 = imp.get("p50", 0)
            p90 = imp.get("p90", 0)

            invested = holdings[ticker]["invested_eur"]
            ticker_impact[ticker] = {
                "p10_eur": p10 * invested * fx_rate,
                "p50_eur": p50 * invested * fx_rate,
                "p90_eur": p90 * invested * fx_rate,
            }
            event_portfolio_pnl += mc.rng.uniform(p10, p90, n_paths) * invested * fx_rate

        scenario_impact[event_key] = {
            "name": event_data_entry.get("name", event_key),
            "emoji": event_data_entry.get("emoji", ""),
            "probability": event_data_entry.get("probability", 0),
            "ticker_impact": ticker_impact,
            "portfolio_pnl": {
                "p10": float(np.percentile(event_portfolio_pnl, 10)),
                "p50": float(np.percentile(event_portfolio_pnl, 50)),
                "p90": float(np.percentile(event_portfolio_pnl, 90)),
            },
        }

    # 4. Concentration risk
    logger.info("\n  [4/6] Analysing concentration risk...")
    semi_returns = []
    concentration_horizon = horizons[-1] if horizons else 90
    for ticker in SEMI_TICKERS:
        if ticker in ticker_paths and concentration_horizon in ticker_paths[ticker]:
            semi_returns.append(ticker_paths[ticker][concentration_horizon]["returns_pct"])
    
    concentration_warning = False
    semi_correlation = 0.0
    if len(semi_returns) >= 2:
        corr_matrix = np.corrcoef(np.array(semi_returns))
        # Average pairwise correlation
        n = len(semi_returns)
        upper_tri = corr_matrix[np.triu_indices(n, k=1)]
        semi_correlation = float(np.mean(upper_tri))
        concentration_warning = semi_correlation > 0.85
        logger.info(f"  Semiconductor cluster correlation: {semi_correlation:.3f} "
                     f"{'⚠ WARNING: >0.85' if concentration_warning else '✓ OK'}")

    # 5. Rebalancing recommendations
    logger.info("\n  [5/6] Generating rebalancing recommendations...")
    rebalancing = _compute_rebalancing(ticker_metrics, holdings)

    # 6. Efficient frontier
    logger.info("\n  [6/6] Computing efficient frontier...")
    efficient_frontier = _compute_efficient_frontier(ohlcv_data, fx_rate)
    benchmarks = _compute_benchmarks(ohlcv_data, holdings, horizons, n_paths, seed)

    result = {
        "ticker_paths": ticker_paths,
        "ticker_metrics": ticker_metrics,
        "portfolio_metrics": portfolio_metrics,
        "scenario_impact": scenario_impact,
        "concentration": {
            "warning": concentration_warning,
            "semi_correlation": semi_correlation,
            "semi_weight": sum(holdings[t]["weight"] for t in SEMI_TICKERS),
        },
        "rebalancing": rebalancing,
        "efficient_frontier": efficient_frontier,
        "benchmarks": benchmarks,
        "fx_rate": fx_rate,
    }

    logger.info("\n✓ Portfolio analytics complete.")
    return result


def _compute_rebalancing(
    ticker_metrics: Dict[str, Dict],
    holdings: Dict[str, Dict],
) -> Dict[str, Any]:
    """Suggest rebalancing based on risk-adjusted returns."""
    total = PORTFOLIO["total_invested"]
    recommendations = {}

    # Simple risk-parity-inspired rebalancing
    if not ticker_metrics:
        return {"recommendations": {}, "suggested_weights": {}}

    # Score each ticker using shrunk risk-adjusted returns.
    scores = {}
    for ticker, metrics in ticker_metrics.items():
        expected_return = float(np.clip(metrics.get("expected_return", 0.0), -0.35, 0.35))
        sharpe = float(np.clip(metrics.get("sharpe", 0.0), -2.0, 2.0))
        var95 = float(max(metrics.get("var_95", 0.10), 0.05))
        score = 0.55 * expected_return + 0.35 * sharpe - 0.25 * var95
        scores[ticker] = float(np.clip(score, -0.5, 0.5))

    raw_scores = np.array([scores.get(t, 0.0) for t in TICKERS], dtype=float)
    if np.allclose(raw_scores, raw_scores[0]):
        target_weights = np.array([holdings[t]["weight"] for t in TICKERS], dtype=float)
    else:
        temperature = 4.0
        shifted = np.exp((raw_scores - raw_scores.max()) * temperature)
        target_weights = shifted / shifted.sum()

    current_weights = np.array([holdings[t]["weight"] for t in TICKERS], dtype=float)
    blended_weights = 0.75 * current_weights + 0.25 * target_weights
    bounded_weights = np.clip(blended_weights, 0.05, 0.22)
    bounded_weights /= bounded_weights.sum()

    max_step = 0.05
    deltas = np.clip(bounded_weights - current_weights, -max_step, max_step)
    suggested_weights = current_weights + deltas
    suggested_weights = np.clip(suggested_weights, 0.03, 0.25)
    suggested_weights /= suggested_weights.sum()
    suggested_weights = {ticker: float(weight) for ticker, weight in zip(TICKERS, suggested_weights)}

    for ticker in TICKERS:
        current_w = float(holdings[ticker]["weight"])
        suggested_w = float(suggested_weights.get(ticker, current_w))
        delta = suggested_w - current_w
        delta_eur = delta * total

        action = "hold"
        if delta > 0.02:
            action = "overweight"
        elif delta < -0.02:
            action = "underweight"

        recommendations[ticker] = {
            "current_weight": current_w,
            "suggested_weight": round(suggested_w, 4),
            "delta_weight": round(delta, 4),
            "delta_eur": round(delta_eur, 2),
            "action": action,
        }

    return {
        "recommendations": recommendations,
        "suggested_weights": suggested_weights,
    }


def _compute_efficient_frontier(
    ohlcv_data: Dict[str, pd.DataFrame],
    fx_rate: float,
    n_portfolios: int = 5000,
) -> Dict[str, Any]:
    """Compute efficient frontier using historical returns."""
    # Build returns matrix
    returns_dict = {}
    for ticker in TICKERS:
        df = ohlcv_data.get(ticker, pd.DataFrame())
        if not df.empty and "Close" in df.columns:
            ret = df["Close"].pct_change().dropna()
            returns_dict[ticker] = ret

    if len(returns_dict) < 2:
        return {"portfolios": [], "current_portfolio": {}}

    # Align dates
    returns_df = pd.DataFrame(returns_dict).dropna()
    if len(returns_df) < 30:
        return {"portfolios": [], "current_portfolio": {}}

    mean_returns = returns_df.mean() * 252  # Annualised
    cov_matrix = returns_df.cov() * 252

    # Try pypfopt first
    try:
        from pypfopt import EfficientFrontier as EF
        from pypfopt import risk_models, expected_returns

        mu = expected_returns.mean_historical_return(
            pd.DataFrame({t: ohlcv_data[t]["Close"] for t in TICKERS if t in ohlcv_data and not ohlcv_data[t].empty})
        )
        S = risk_models.sample_cov(
            pd.DataFrame({t: ohlcv_data[t]["Close"] for t in TICKERS if t in ohlcv_data and not ohlcv_data[t].empty})
        )

        frontier_portfolios = []

        # Target Sharpe ratios
        for target_sharpe in [1.0, 1.5, 2.0]:
            try:
                ef = EF(mu, S)
                ef.max_sharpe(risk_free_rate=0.04)
                weights = ef.clean_weights()
                perf = ef.portfolio_performance(risk_free_rate=0.04)
                frontier_portfolios.append({
                    "target_sharpe": target_sharpe,
                    "weights": dict(weights),
                    "expected_return": perf[0],
                    "volatility": perf[1],
                    "sharpe": perf[2],
                })
            except Exception:
                pass

        # Min volatility portfolio
        try:
            ef = EF(mu, S)
            ef.min_volatility()
            weights = ef.clean_weights()
            perf = ef.portfolio_performance(risk_free_rate=0.04)
            frontier_portfolios.append({
                "target_sharpe": 0,
                "weights": dict(weights),
                "expected_return": perf[0],
                "volatility": perf[1],
                "sharpe": perf[2],
                "label": "Min Volatility",
            })
        except Exception:
            pass

        # Current portfolio performance
        current_weights = np.array([PORTFOLIO["holdings"][t]["weight"] for t in TICKERS if t in mu.index])
        current_tickers = [t for t in TICKERS if t in mu.index]
        if len(current_weights) > 0:
            current_return = float(np.dot(current_weights, mu[current_tickers].values))
            current_vol = float(np.sqrt(
                np.dot(current_weights, np.dot(S.loc[current_tickers, current_tickers].values, current_weights))
            ))
            current_sharpe = (current_return - 0.04) / current_vol if current_vol > 0 else 0

            current_portfolio = {
                "weights": {t: PORTFOLIO["holdings"][t]["weight"] for t in TICKERS},
                "expected_return": current_return,
                "volatility": current_vol,
                "sharpe": current_sharpe,
            }
        else:
            current_portfolio = {}

        # Random portfolios for frontier visualization
        random_portfolios = []
        rng = np.random.RandomState(RANDOM_SEED)
        n_assets = len(current_tickers)
        for _ in range(n_portfolios):
            w = rng.dirichlet(np.ones(n_assets))
            ret = float(np.dot(w, mu[current_tickers].values))
            vol = float(np.sqrt(np.dot(w, np.dot(S.loc[current_tickers, current_tickers].values, w))))
            sr = (ret - 0.04) / vol if vol > 0 else 0
            random_portfolios.append({"return": ret, "volatility": vol, "sharpe": sr})

        return {
            "frontier_portfolios": frontier_portfolios,
            "current_portfolio": current_portfolio,
            "random_portfolios": random_portfolios,
            "mean_returns": mean_returns.to_dict(),
            "cov_matrix": cov_matrix.to_dict(),
        }

    except ImportError:
        logger.warning("  ✗ pypfopt not installed — using basic frontier computation")

        # Fallback: random sampling
        random_portfolios = []
        rng = np.random.RandomState(RANDOM_SEED)
        n_assets = len(TICKERS)
        tickers_with_data = [t for t in TICKERS if t in returns_dict]

        for _ in range(n_portfolios):
            w = rng.dirichlet(np.ones(len(tickers_with_data)))
            ret = float(np.dot(w, mean_returns[tickers_with_data].values))
            vol = float(np.sqrt(np.dot(w, np.dot(cov_matrix.loc[tickers_with_data, tickers_with_data].values, w))))
            sr = (ret - 0.04) / vol if vol > 0 else 0
            random_portfolios.append({"return": ret, "volatility": vol, "sharpe": sr})

        return {
            "frontier_portfolios": [],
            "current_portfolio": {},
            "random_portfolios": random_portfolios,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Module 6 — Portfolio Analytics loaded. Run via main.py for full pipeline.")
