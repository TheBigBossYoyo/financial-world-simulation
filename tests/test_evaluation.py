import numpy as np
import pandas as pd
import pytest

import evaluate_forecasts as ev


def test_walk_forward_splits_are_consecutive_and_end_at_data_end():
    splits = ev.make_walk_forward_splits(n_rows=2000, n_folds=3, test_len=250, min_train=500)
    assert splits == [(1250, 1500), (1500, 1750), (1750, 2000)]
    for (_, end_a), (start_b, _) in zip(splits, splits[1:]):
        assert end_a == start_b  # test blocks do not overlap and leave no gap


def test_walk_forward_rejects_too_little_training_data():
    with pytest.raises(ValueError):
        ev.make_walk_forward_splits(n_rows=800, n_folds=3, test_len=250, min_train=500)


def test_forward_returns_hand_computed():
    close = np.array([100.0, 110.0, 99.0, 99.0])
    r = ev.forward_returns(close, 2)
    assert r[0] == pytest.approx(-0.01)
    assert r[1] == pytest.approx(-0.1)
    assert np.isnan(r[2]) and np.isnan(r[3])


def test_random_walk_is_zero():
    assert np.all(ev.baseline_random_walk(5) == 0)


def test_drift_baseline_hand_computed():
    rets = np.array([0.0, 0.02, 0.04, 0.0])  # index 0 is the placeholder
    pred = ev.baseline_drift(rets, np.array([2, 3]), horizon=1)
    assert pred[0] == pytest.approx(0.03)   # mean of 0.02, 0.04
    assert pred[1] == pytest.approx(0.02)   # mean of 0.02, 0.04, 0.0


def test_drift_has_no_lookahead():
    rng = np.random.RandomState(0)
    rets = rng.normal(0.0005, 0.01, 300)
    rets[0] = 0.0
    origins = np.arange(100, 200)
    base = ev.baseline_drift(rets, origins, 5)
    changed = rets.copy()
    changed[200:] = 0.5  # rewrite the future
    assert np.allclose(base, ev.baseline_drift(changed, origins, 5))


def test_ar1_recovers_coefficient():
    rng = np.random.RandomState(1)
    phi_true, n = 0.4, 20_000
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi_true * r[t - 1] + rng.normal(0, 0.01)
    mu, phi = ev.fit_ar1(r)
    assert phi == pytest.approx(phi_true, abs=0.03)
    assert mu == pytest.approx(0.0, abs=1e-3)


def test_ar1_forecast_hand_computed():
    # mu=0, phi=0.5, last return 0.02: h=1 -> 0.01, h=2 -> 0.01 + 0.005
    assert ev.baseline_ar1(0.0, 0.5, np.array([0.02]), 1)[0] == pytest.approx(0.01)
    assert ev.baseline_ar1(0.0, 0.5, np.array([0.02]), 2)[0] == pytest.approx(0.015)


def test_metrics_hand_computed():
    actual = np.array([0.01, -0.02, 0.03, -0.01])
    pred = np.array([0.02, -0.01, -0.01, 0.01])
    m = ev.compute_metrics(actual, pred)
    assert m["n"] == 4
    assert m["mae"] == pytest.approx((0.01 + 0.01 + 0.04 + 0.02) / 4)
    assert m["rmse"] == pytest.approx(np.sqrt((0.0001 + 0.0001 + 0.0016 + 0.0004) / 4))
    assert m["dir_acc"] == pytest.approx(0.5)


def test_metrics_ignore_nan_and_zero_prediction_has_no_direction():
    m = ev.compute_metrics(np.array([0.01, np.nan, -0.01]), np.zeros(3))
    assert m["n"] == 2
    assert np.isnan(m["dir_acc"])


def test_skill_score():
    actual = np.array([0.01, -0.02, 0.03, -0.01])
    assert ev.skill_score(actual, np.zeros(4)) == pytest.approx(0.0)   # random walk vs itself
    assert ev.skill_score(actual, actual) == pytest.approx(1.0)        # perfect forecast
    assert ev.skill_score(actual, actual * -1) < 0                     # worse than the random walk


def test_skill_against_custom_reference():
    actual = np.array([1.0, 2.0, 3.0])
    ref = np.array([0.0, 0.0, 0.0])
    pred = np.array([1.0, 2.0, 2.0])
    assert ev.skill_score(actual, pred, ref) == pytest.approx(1.0 - 1.0 / 14.0)


def test_scalers_are_fit_on_training_window_only():
    rng = np.random.RandomState(2)
    feats = rng.normal(size=(300, 3)).astype(np.float32)
    feats[200:] += 50.0  # a regime change after the cutoff must not leak into the scaler
    close = np.linspace(10, 100, 300)
    fs, ts = ev._fit_scalers(feats, close, cutoff=200)
    assert np.allclose(fs.mean_, feats[:200].mean(axis=0), atol=1e-5)
    assert ts.mean_[0] == pytest.approx(close[:200].mean())


def test_stationary_features_are_scale_free():
    from modules.ml_forecasting import build_features

    rng = np.random.RandomState(3)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                       "Volume": rng.randint(1_000, 2_000, n).astype(float)},
                      index=pd.bdate_range("2020-01-01", periods=n))
    scaled = df.copy()
    for c in ["Open", "High", "Low", "Close"]:
        scaled[c] = df[c] * 10
    a, _ = build_features(df)
    b, _ = build_features(scaled)
    sa, sb = ev.stationary_features(a), ev.stationary_features(b)
    assert sa.shape[1] > 10
    assert np.allclose(sa.drop(columns=["VWAP_dev"]).values, sb.drop(columns=["VWAP_dev"]).values, atol=1e-6)


def test_summarise_on_synthetic_predictions():
    rng = np.random.RandomState(4)
    n = 200
    actual = rng.normal(0, 0.01, n)
    df = pd.DataFrame({"ticker": "X", "fold": 0, "date": pd.bdate_range("2024-01-01", periods=n),
                       "horizon": 1, "actual": actual})
    for name in ev.MODEL_NAMES:
        df[name] = 0.0
    df["ensemble"] = actual  # a perfect model
    s = ev.summarise(df).set_index("model")
    assert s.loc["random_walk", "skill_vs_rw"] == pytest.approx(0.0)
    assert s.loc["ensemble", "skill_vs_rw"] == pytest.approx(1.0)
    assert s.loc["ensemble", "dir_acc"] == pytest.approx(1.0)
