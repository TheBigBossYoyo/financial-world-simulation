"""
Module 4 — Agent-Based World Simulation
Vectorised agent-based model: 50,000 agents across 7 archetypes generate
buy/sell orders that produce emergent price pressure each timestep.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AGENT_ARCHETYPES, DEFAULT_N_AGENTS, RANDOM_SEED, TICKERS

logger = logging.getLogger(__name__)


class AgentPopulation:
    """Vectorised agent population — all agent state stored in NumPy arrays."""

    def __init__(self, n_agents: int = DEFAULT_N_AGENTS, seed: int = RANDOM_SEED):
        self.n = n_agents
        self.rng = np.random.RandomState(seed)
        self._initialise_agents()

    def _initialise_agents(self):
        """Assign archetypes and sample agent parameters from distributions."""
        archetypes = list(AGENT_ARCHETYPES.keys())
        fractions = [AGENT_ARCHETYPES[a]["fraction"] for a in archetypes]

        # Assign archetype index to each agent
        counts = [int(f * self.n) for f in fractions]
        # Adjust rounding to match total
        counts[-1] = self.n - sum(counts[:-1])
        self.archetype_idx = np.concatenate([
            np.full(c, i, dtype=np.int32) for i, c in enumerate(counts)
        ])
        self.archetype_names = archetypes

        # Sample parameters per agent based on archetype distributions
        self.risk_tolerance = np.zeros(self.n)
        self.memory_horizon = np.zeros(self.n, dtype=np.int32)
        self.reaction_delay = np.zeros(self.n, dtype=np.int32)
        self.herding_coeff = np.zeros(self.n)

        for i, name in enumerate(archetypes):
            cfg = AGENT_ARCHETYPES[name]
            mask = self.archetype_idx == i

            alpha = cfg["risk_tolerance_alpha"]
            beta = cfg["risk_tolerance_beta"]
            self.risk_tolerance[mask] = self.rng.beta(alpha, beta, size=mask.sum())

            self.memory_horizon[mask] = cfg["memory_horizon"]
            self.reaction_delay[mask] = cfg["reaction_delay"]
            self.herding_coeff[mask] = cfg["herding_coefficient"]

        # Agent cash & position tracking (per ticker)
        self.positions = {}  # ticker -> array of positions
        for ticker in TICKERS:
            self.positions[ticker] = np.zeros(self.n)

        # Track last action timestamp (for reaction delay)
        self.last_action_step = np.full(self.n, -999, dtype=np.int32)

        logger.info(f"  ✓ Initialised {self.n:,} agents across {len(archetypes)} archetypes")
        for i, name in enumerate(archetypes):
            count = int((self.archetype_idx == i).sum())
            logger.info(f"    {name}: {count:,} agents ({count/self.n*100:.1f}%)")

    def compute_signals(
        self,
        ticker: str,
        step: int,
        current_price: float,
        price_history: np.ndarray,
        sentiment: float,
        macro_score: float,
        peer_action_ratio: float,
        event_active: bool,
        event_impact: float,
    ) -> np.ndarray:
        """Compute buy/sell signal for every agent (vectorised).

        Returns:
            Array of shape (n_agents,) with values in [-1, +1].
            Positive = buy signal, negative = sell signal.
        """
        signals = np.zeros(self.n)

        # -- Price momentum signal --
        if len(price_history) >= 2:
            # Short-term momentum (each agent looks back by their memory horizon)
            for i, name in enumerate(self.archetype_names):
                mask = self.archetype_idx == i
                horizon = AGENT_ARCHETYPES[name]["memory_horizon"]
                lookback = min(horizon, len(price_history) - 1)
                if lookback > 0:
                    past_price = price_history[-lookback - 1]
                    if past_price > 0:
                        momentum = (current_price - past_price) / past_price
                    else:
                        momentum = 0.0
                else:
                    momentum = 0.0

                # Different archetypes interpret momentum differently
                if name == "retail_fomo":
                    # Momentum chasing — buy when rising, panic sell when dropping
                    signals[mask] += momentum * 3.0
                    if momentum < -0.05:
                        signals[mask] -= 0.5  # Panic sell on -5%+ drops
                elif name == "retail_passive":
                    # Barely reacts to momentum
                    signals[mask] += momentum * 0.3
                elif name == "institutional_fundamental":
                    # Mean-reverting — buy dips, sell rallies
                    signals[mask] -= momentum * 1.5
                elif name == "quant_algo":
                    # Momentum + mean-reversion hybrid
                    short_mom = (current_price / price_history[-2] - 1) if len(price_history) >= 3 else 0
                    long_mom = momentum
                    signals[mask] += short_mom * 2.0 - long_mom * 1.0
                elif name == "hedge_fund_macro":
                    # Macro-driven with some momentum
                    signals[mask] += momentum * 0.8 + macro_score * 2.0
                elif name == "central_bank_sovereign":
                    # Very slow, stability-oriented
                    signals[mask] += macro_score * 0.5
                elif name == "corporate_insider":
                    # Slight momentum awareness + fundamental bias
                    signals[mask] += momentum * 0.5

        # -- Sentiment signal --
        for i, name in enumerate(self.archetype_names):
            mask = self.archetype_idx == i
            if name in ("retail_fomo",):
                signals[mask] += sentiment * 2.0
            elif name in ("retail_passive",):
                signals[mask] += sentiment * 0.5
            elif name in ("institutional_fundamental",):
                signals[mask] += sentiment * 0.3
            elif name in ("quant_algo",):
                signals[mask] += sentiment * 1.0
            elif name in ("hedge_fund_macro",):
                signals[mask] += sentiment * 1.5
            elif name in ("central_bank_sovereign",):
                signals[mask] += sentiment * 0.2
            elif name in ("corporate_insider",):
                signals[mask] += sentiment * 0.8

        # -- Geopolitical event signal --
        if event_active:
            for i, name in enumerate(self.archetype_names):
                mask = self.archetype_idx == i
                if name in ("hedge_fund_macro", "central_bank_sovereign"):
                    signals[mask] += event_impact * 3.0
                elif name in ("retail_fomo",):
                    signals[mask] += event_impact * 2.0
                elif name in ("quant_algo",):
                    signals[mask] += event_impact * 1.5
                else:
                    signals[mask] += event_impact * 0.5

        # -- Herding effect --
        signals += peer_action_ratio * self.herding_coeff

        # -- Risk tolerance modulation --
        # Higher risk tolerance should lead to larger position sizes.
        signals *= (0.5 + self.risk_tolerance)

        # -- Reaction delay filter --
        eligible = (step - self.last_action_step) >= self.reaction_delay
        signals *= eligible.astype(float)

        # -- Add noise (idiosyncratic agent decisions) --
        signals += self.rng.normal(0, 0.1, self.n)

        # Clip to [-1, +1]
        signals = np.clip(signals, -1, 1)

        return signals


class MarketSimulator:
    """Simulates order book dynamics from agent signals → price impact."""

    def __init__(
        self,
        n_agents: int = DEFAULT_N_AGENTS,
        seed: int = RANDOM_SEED,
        price_impact_coeff: float = 0.0008,
        volatility_floor: float = 0.005,
    ):
        self.population = AgentPopulation(n_agents, seed)
        self.price_impact_coeff = price_impact_coeff
        self.volatility_floor = volatility_floor
        self.rng = np.random.RandomState(seed + 1)

    def simulate_ticker(
        self,
        ticker: str,
        initial_price: float,
        n_steps: int,
        sentiment_path: np.ndarray,
        macro_path: np.ndarray,
        event_timeline: np.ndarray,
        event_impacts: np.ndarray,
        historical_volatility: float = 0.02,
    ) -> Dict[str, np.ndarray]:
        """Simulate price path for a single ticker over n_steps.

        Args:
            ticker: Stock ticker symbol.
            initial_price: Starting price.
            n_steps: Number of timesteps (trading days).
            sentiment_path: Array of shape (n_steps,) — daily sentiment.
            macro_path: Array of shape (n_steps,) — macro environment score.
            event_timeline: Boolean array (n_steps,) — is event active.
            event_impacts: Array of shape (n_steps,) — event impact on this ticker.
            historical_volatility: Daily volatility for random walk component.

        Returns:
            Dict with 'prices', 'volumes', 'order_flows'.
        """
        prices = np.zeros(n_steps + 1)
        prices[0] = initial_price
        volumes = np.zeros(n_steps)
        order_flows = np.zeros(n_steps)
        price_history = [initial_price]

        for t in range(n_steps):
            # Get agent signals
            peer_ratio = order_flows[t - 1] / self.population.n if t > 0 else 0.0

            signals = self.population.compute_signals(
                ticker=ticker,
                step=t,
                current_price=prices[t],
                price_history=np.array(price_history),
                sentiment=float(sentiment_path[t]) if t < len(sentiment_path) else 0.0,
                macro_score=float(macro_path[t]) if t < len(macro_path) else 0.0,
                peer_action_ratio=peer_ratio,
                event_active=bool(event_timeline[t]) if t < len(event_timeline) else False,
                event_impact=float(event_impacts[t]) if t < len(event_impacts) else 0.0,
            )

            # Aggregate order flow
            net_flow = np.sum(signals)
            order_flows[t] = net_flow
            volumes[t] = np.sum(np.abs(signals))

            # Convert forces into bounded daily log-return components.
            flow_ratio = net_flow / max(self.population.n, 1)
            impact_return = np.tanh(flow_ratio * 5.0) * self.price_impact_coeff

            vol = min(max(historical_volatility, self.volatility_floor), 0.06)
            random_return = self.rng.normal(0, vol)

            if t < len(event_timeline) and event_timeline[t]:
                event_return = float(np.clip(event_impacts[t], -0.03, 0.03))
            else:
                event_return = 0.0

            daily_log_return = np.clip(impact_return + random_return + event_return, -0.18, 0.18)
            prices[t + 1] = max(prices[t] * np.exp(daily_log_return), 0.01)
            price_history.append(prices[t + 1])

            # Update agent action timestamps
            active_mask = np.abs(signals) > 0.1
            self.population.last_action_step[active_mask] = t

        return {
            "prices": prices,
            "volumes": volumes,
            "order_flows": order_flows,
        }


# ---------------------------------------------------------------------------
# Master simulation function
# ---------------------------------------------------------------------------

def run_agent_simulation(
    ohlcv_data: Dict[str, pd.DataFrame],
    sentiment_scores: Dict[str, float],
    event_timelines: Dict[str, Dict[str, np.ndarray]],
    rolling_sentiment: Optional[Dict[str, pd.DataFrame]] = None,
    macro_data: Optional[Dict[str, pd.Series]] = None,
    n_agents: int = DEFAULT_N_AGENTS,
    n_steps: int = 90,
    seed: int = RANDOM_SEED,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run agent-based simulation for all tickers.

    Args:
        ohlcv_data: Historical OHLCV per ticker.
        sentiment_scores: Current composite sentiment per ticker.
        event_timelines: Per-ticker event timeline & impact arrays.
        n_agents: Number of agents to simulate.
        n_steps: Number of trading days to simulate.
        seed: Random seed.
        dry_run: If True, reduce agents for speed.

    Returns:
        Dict with simulated prices, volumes, order flows per ticker.
    """
    if dry_run:
        n_agents = min(n_agents, 1000)

    logger.info("=" * 60)
    logger.info("MODULE 4 — Agent-Based World Simulation")
    logger.info("=" * 60)
    logger.info(f"  Agents: {n_agents:,} | Steps: {n_steps} | Seed: {seed}")

    simulator = MarketSimulator(n_agents=n_agents, seed=seed)
    results = {}
    macro_path = _build_macro_path(macro_data, n_steps)

    for ticker in TICKERS:
        df = ohlcv_data.get(ticker, pd.DataFrame())
        if df.empty:
            logger.warning(f"  ✗ No data for {ticker}, skipping")
            results[ticker] = None
            continue

        initial_price = float(df["Close"].iloc[-1])

        # Compute historical daily volatility
        returns = df["Close"].pct_change().dropna()
        hist_vol = float(returns.std()) if len(returns) > 20 else 0.02

        sentiment = sentiment_scores.get(ticker, 0.0)
        sentiment_path = _build_sentiment_path(rolling_sentiment, ticker, sentiment, n_steps)

        # Event timelines for this ticker
        ticker_events = event_timelines.get(ticker, {})
        event_timeline = ticker_events.get("timeline", np.zeros(n_steps, dtype=bool))
        event_impacts = ticker_events.get("impacts", np.zeros(n_steps))

        logger.info(f"  Simulating {ticker} | Price: ${initial_price:.2f} | Vol: {hist_vol:.4f}")
        sim = simulator.simulate_ticker(
            ticker=ticker,
            initial_price=initial_price,
            n_steps=n_steps,
            sentiment_path=sentiment_path,
            macro_path=macro_path,
            event_timeline=event_timeline,
            event_impacts=event_impacts,
            historical_volatility=hist_vol,
        )

        results[ticker] = {
            "prices": sim["prices"],
            "volumes": sim["volumes"],
            "order_flows": sim["order_flows"],
            "initial_price": initial_price,
            "historical_volatility": hist_vol,
            "sentiment_path": sentiment_path,
            "macro_path": macro_path,
        }
        logger.info(f"  ✓ {ticker}: ${initial_price:.2f} → ${sim['prices'][-1]:.2f} "
                     f"({(sim['prices'][-1]/initial_price - 1)*100:+.1f}%)")

    logger.info("\n✓ Agent-based simulation complete.")
    return results


def _build_sentiment_path(
    rolling_sentiment: Optional[Dict[str, pd.DataFrame]],
    ticker: str,
    fallback_score: float,
    n_steps: int,
) -> np.ndarray:
    if rolling_sentiment:
        ticker_sentiment = rolling_sentiment.get(ticker)
        if ticker_sentiment is not None and not ticker_sentiment.empty and "composite" in ticker_sentiment.columns:
            values = ticker_sentiment["composite"].tail(n_steps).to_numpy(dtype=float)
            if len(values) < n_steps:
                pad_value = values[0] if len(values) else fallback_score
                values = np.concatenate([np.full(n_steps - len(values), pad_value), values])
            return np.clip(values[-n_steps:], -1.0, 1.0)
    return np.full(n_steps, fallback_score, dtype=float)


def _build_macro_path(macro_data: Optional[Dict[str, pd.Series]], n_steps: int) -> np.ndarray:
    if not macro_data:
        return np.zeros(n_steps, dtype=float)

    components = []
    for series in macro_data.values():
        if series is None or series.empty:
            continue
        history = series.dropna().astype(float).tail(max(90, n_steps + 30))
        if history.empty:
            continue
        diff = history.diff().fillna(0.0)
        std = float(diff.std())
        if std > 0:
            diff = (diff - diff.mean()) / std
        component = diff.tail(n_steps).to_numpy(dtype=float)
        if len(component) < n_steps:
            component = np.concatenate([np.full(n_steps - len(component), component[0]), component])
        components.append(component)

    if not components:
        return np.zeros(n_steps, dtype=float)

    composite = np.mean(np.vstack(components), axis=0)
    return np.clip(composite[-n_steps:], -3.0, 3.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Module 4 — Agent-Based Simulation loaded. Run via main.py for full pipeline.")
