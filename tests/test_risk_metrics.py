import numpy as np
import pytest
from scipy import stats

from modules.portfolio_analytics import (
    MonteCarloEngine,
    compute_cvar,
    compute_max_drawdown,
    compute_sharpe,
    compute_var,
)


@pytest.fixture(scope="module")
def normal_returns():
    rng = np.random.RandomState(0)
    return rng.normal(0.001, 0.02, 400_000), 0.001, 0.02


@pytest.mark.parametrize("confidence", [0.95, 0.99])
def test_var_matches_analytic_normal(normal_returns, confidence):
    x, mu, sigma = normal_returns
    z = stats.norm.ppf(1 - confidence)
    analytic = -(mu + sigma * z)
    assert compute_var(x, confidence) == pytest.approx(analytic, abs=3e-4)


@pytest.mark.parametrize("confidence", [0.95, 0.99])
def test_cvar_matches_analytic_normal(normal_returns, confidence):
    x, mu, sigma = normal_returns
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)
    analytic = -(mu - sigma * stats.norm.pdf(z) / alpha)
    assert compute_cvar(x, confidence) == pytest.approx(analytic, abs=5e-4)


def test_cvar_is_at_least_var(normal_returns):
    x = normal_returns[0]
    assert compute_cvar(x, 0.95) >= compute_var(x, 0.95)


def test_var_on_known_small_sample():
    # 100 evenly spaced returns from -0.50 to 0.49: 5th percentile is -0.4505
    x = np.arange(-50, 50) / 100.0
    assert compute_var(x, 0.95) == pytest.approx(0.4505)


def test_max_drawdown_hand_computed():
    path = np.array([100, 120, 90, 110, 60, 80], dtype=float)
    assert compute_max_drawdown(path) == pytest.approx(0.5)  # 120 -> 60


def test_max_drawdown_monotonic_rise_is_zero():
    assert compute_max_drawdown(np.array([1.0, 2.0, 3.0, 3.0])) == 0.0


def test_sharpe_hand_computed():
    r = np.array([0.02, 0.0, 0.02, 0.0])  # mean 0.01, std 0.01
    assert compute_sharpe(r, risk_free_annual=0.0) == pytest.approx(np.sqrt(252))


def test_sharpe_subtracts_risk_free():
    r = np.array([0.02, 0.0, 0.02, 0.0])
    rf = 0.0252  # 0.0001 per day
    expected = (0.01 - 0.0001) / 0.01 * np.sqrt(252)
    assert compute_sharpe(r, risk_free_annual=rf) == pytest.approx(expected)


def test_sharpe_degenerate_inputs():
    assert compute_sharpe(np.array([])) == 0.0
    assert compute_sharpe(np.full(10, 0.01)) == 0.0


def test_monte_carlo_reproducible_with_seed():
    a = MonteCarloEngine(n_paths=200, seed=7).simulate_paths("NVDA", 100.0, horizon=30)
    b = MonteCarloEngine(n_paths=200, seed=7).simulate_paths("NVDA", 100.0, horizon=30)
    c = MonteCarloEngine(n_paths=200, seed=8).simulate_paths("NVDA", 100.0, horizon=30)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_monte_carlo_shape_and_sanity():
    paths = MonteCarloEngine(n_paths=500, seed=1).simulate_paths("JPM", 50.0, horizon=20, historical_vol=0.015)
    assert paths.shape == (500, 21)
    assert np.all(paths[:, 0] == 50.0)
    assert np.all(np.isfinite(paths)) and np.all(paths > 0)
