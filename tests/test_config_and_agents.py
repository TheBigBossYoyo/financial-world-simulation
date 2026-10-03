import numpy as np
import pytest

import config
from modules.agent_simulation import AgentPopulation, MarketSimulator


def test_portfolio_weights_sum_to_one():
    weights = [h["weight"] for h in config.PORTFOLIO["holdings"].values()]
    assert sum(weights) == pytest.approx(1.0)
    assert all(w > 0 for w in weights)


def test_invested_amounts_match_weights():
    total = config.PORTFOLIO["total_invested"]
    for name, h in config.PORTFOLIO["holdings"].items():
        assert h["invested_eur"] == pytest.approx(h["weight"] * total), name
    assert sum(h["invested_eur"] for h in config.PORTFOLIO["holdings"].values()) == pytest.approx(total)


def test_tickers_consistent_across_config():
    assert set(config.TICKERS) == set(config.WATCHLIST)
    assert set(config.SEMI_TICKERS) <= set(config.TICKERS)
    for event in config.GEOPOLITICAL_EVENTS.values():
        assert set(event["portfolio_impact"]) == set(config.TICKERS)


def test_archetype_fractions_sum_to_one():
    assert sum(a["fraction"] for a in config.AGENT_ARCHETYPES.values()) == pytest.approx(1.0)


def test_agent_counts_per_archetype():
    n = 10_000
    pop = AgentPopulation(n, seed=3)
    assert len(pop.archetype_idx) == n
    for i, name in enumerate(pop.archetype_names):
        count = int((pop.archetype_idx == i).sum())
        expected = config.AGENT_ARCHETYPES[name]["fraction"] * n
        assert abs(count - expected) <= len(pop.archetype_names), name  # only rounding differences


def test_agent_parameters_valid():
    pop = AgentPopulation(5_000, seed=3)
    assert np.all((pop.risk_tolerance >= 0) & (pop.risk_tolerance <= 1))
    assert np.all(np.isfinite(pop.risk_tolerance))
    assert np.all(pop.memory_horizon > 0)
    assert np.all(pop.reaction_delay >= 0)


def test_population_deterministic_with_seed():
    a, b, c = AgentPopulation(2_000, seed=5), AgentPopulation(2_000, seed=5), AgentPopulation(2_000, seed=6)
    assert np.array_equal(a.risk_tolerance, b.risk_tolerance)
    assert not np.array_equal(a.risk_tolerance, c.risk_tolerance)


def _run(seed):
    n_steps = 40
    sim = MarketSimulator(n_agents=1_000, seed=seed)
    sentiment = np.linspace(-0.3, 0.3, n_steps)
    event = np.zeros(n_steps, dtype=bool)
    event[10:20] = True
    impacts = np.where(event, -0.01, 0.0)
    return sim.simulate_ticker("NVDA", 100.0, n_steps, sentiment, np.zeros(n_steps), event, impacts, 0.02)


def test_simulation_deterministic_and_finite():
    a, b, c = _run(11), _run(11), _run(12)
    for key in ("prices", "volumes", "order_flows"):
        assert np.all(np.isfinite(a[key])), key
        assert np.array_equal(a[key], b[key]), key
    assert not np.array_equal(a["prices"], c["prices"])


def test_simulation_prices_positive_and_start_correct():
    out = _run(1)
    assert out["prices"][0] == 100.0
    assert len(out["prices"]) == 41
    assert np.all(out["prices"] > 0)


def test_signals_bounded():
    pop = AgentPopulation(1_000, seed=2)
    history = np.array([100.0, 101.0, 99.0, 102.0, 98.0])
    s = pop.compute_signals("NVDA", 5, 98.0, history, sentiment=0.8, macro_score=-0.5,
                            peer_action_ratio=0.2, event_active=True, event_impact=-0.5)
    assert s.shape == (1_000,)
    assert np.all(np.abs(s) <= 1.0) and np.all(np.isfinite(s))
