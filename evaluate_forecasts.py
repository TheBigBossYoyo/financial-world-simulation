"""
evaluate_forecasts.py - walk-forward check of the LSTM / simplified TFT forecasters
against naive baselines, using price-only features (no API keys needed).

What it does
------------
For each ticker the history is cut into expanding-window folds. In every fold the
scalers and both networks are fit only on rows before the cutoff, and are then scored
on the next block of trading days that they never saw. The networks use the same
architecture, feature builder and target definition (next-day Close, standardised) as
modules/ml_forecasting.py, and the multi-day forecast is built the same way the
pipeline does it (project_horizon_prices). Their checkpoints in models/ are not touched.

Baselines (all causal):
  random_walk   predicted return = 0 ("tomorrow = today")
  drift         expanding mean of daily returns up to the forecast origin, scaled to h days
  ar1           AR(1) on daily returns, fit on the training window only

Metrics per ticker / horizon / model: MAE, RMSE of the h-day simple return,
directional accuracy, and an MSE skill score against the random walk
(1 - MSE_model / MSE_random_walk; positive means better than the random walk).

Usage
-----
    python evaluate_forecasts.py                      # default config, writes docs/
    python evaluate_forecasts.py --tickers NVDA JPM --epochs 5 --folds 2   # quick run
"""

import argparse
import logging
import warnings
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

logger = logging.getLogger("evaluate_forecasts")
warnings.filterwarnings("ignore", message="dropout option adds dropout")

DEFAULT_TICKERS = ["NVDA", "AVGO", "LLY", "VST", "NOC", "MU", "JPM", "NEM"]
DEFAULT_HORIZONS = [1, 5, 20]
SEQ_LEN = 60
CACHE_DIR = BASE_DIR / "data" / "eval_cache"
MODEL_NAMES = ["random_walk", "drift", "ar1", "lstm", "tft", "ensemble", "lstm_ret", "tft_ret", "ensemble_ret"]
# "lstm/tft/ensemble" follow the pipeline as built (target = next-day Close price).
# "*_ret" are the same networks with a next-day return target and ratio-only features.


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_prices(ticker: str, start: str, end: Optional[str] = None, refresh: bool = False) -> pd.DataFrame:
    """Daily OHLCV from yfinance (split/dividend adjusted), cached as CSV."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{ticker}_{start}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_csv(path)
    return df


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def make_walk_forward_splits(n_rows: int, n_folds: int, test_len: int, min_train: int = 500) -> List[Tuple[int, int]]:
    """Expanding-window folds as (cutoff, test_end) row indices.

    Training uses rows [0, cutoff), testing uses forecast origins in [cutoff, test_end).
    The folds are consecutive and the last one ends at n_rows.
    """
    splits = []
    for k in range(n_folds):
        test_end = n_rows - (n_folds - 1 - k) * test_len
        cutoff = test_end - test_len
        if cutoff < min_train:
            raise ValueError(f"fold {k} would train on only {cutoff} rows (min {min_train})")
        splits.append((cutoff, test_end))
    return splits


def forward_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    """Simple return from t to t+horizon; NaN where t+horizon is beyond the data."""
    close = np.asarray(close, dtype=float)
    out = np.full(len(close), np.nan)
    if horizon < len(close):
        out[: len(close) - horizon] = close[horizon:] / close[: len(close) - horizon] - 1.0
    return out


def daily_returns(close: np.ndarray) -> np.ndarray:
    """Simple daily returns aligned to close (first entry is 0)."""
    close = np.asarray(close, dtype=float)
    r = np.zeros(len(close))
    r[1:] = close[1:] / close[:-1] - 1.0
    return r


def baseline_random_walk(n: int) -> np.ndarray:
    return np.zeros(n)


def baseline_drift(returns: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    """Expanding mean of daily returns up to and including each origin, compounded to h days.

    Only returns[: origin + 1] are used, so there is no look-ahead.
    """
    returns = np.asarray(returns, dtype=float)
    # returns[0] is a placeholder zero, so the mean starts at index 1
    csum = np.cumsum(returns[1:])
    counts = np.arange(1, len(returns))
    mean_to_t = np.concatenate([[0.0], csum / counts])
    mu = mean_to_t[np.asarray(origins)]
    return (1.0 + mu) ** horizon - 1.0


def fit_ar1(train_returns: np.ndarray) -> Tuple[float, float]:
    """OLS fit of r_t - mu = phi * (r_{t-1} - mu) on the training window. Returns (mu, phi)."""
    r = np.asarray(train_returns, dtype=float)
    mu = float(np.mean(r))
    x = r[:-1] - mu
    y = r[1:] - mu
    denom = float(np.dot(x, x))
    phi = float(np.dot(x, y) / denom) if denom > 0 else 0.0
    return mu, float(np.clip(phi, -0.99, 0.99))


def baseline_ar1(mu: float, phi: float, last_returns: np.ndarray, horizon: int) -> np.ndarray:
    """Expected sum of the next h daily returns under AR(1), used as the h-day return forecast."""
    last_returns = np.asarray(last_returns, dtype=float)
    if abs(phi) < 1e-12:
        persist = 0.0
    else:
        persist = phi * (1.0 - phi ** horizon) / (1.0 - phi)
    return horizon * mu + (last_returns - mu) * persist


def compute_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """MAE, RMSE and directional accuracy of predicted vs actual returns.

    Directional accuracy is NaN when the model never predicts a direction
    (all predictions are exactly zero, as for the random walk).
    """
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[mask], pred[mask]
    if len(actual) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "dir_acc": np.nan}
    err = pred - actual
    dir_acc = float(np.mean(np.sign(pred) == np.sign(actual))) if np.any(pred != 0) else np.nan
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "dir_acc": dir_acc,
    }


def skill_score(actual: np.ndarray, pred: np.ndarray, reference: Optional[np.ndarray] = None) -> float:
    """1 - MSE(pred) / MSE(reference). The reference defaults to the zero-return random walk."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ref = np.zeros_like(actual) if reference is None else np.asarray(reference, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred) & np.isfinite(ref)
    mse_ref = float(np.mean((ref[mask] - actual[mask]) ** 2))
    if mse_ref == 0:
        return float("nan")
    mse = float(np.mean((pred[mask] - actual[mask]) ** 2))
    return 1.0 - mse / mse_ref


def stationary_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Scale-free feature subset (ratios, oscillators, returns) for the return-target variant."""
    c = feat_df["Close"]
    out = pd.DataFrame(index=feat_df.index)
    for col in ["RSI", "BB_pctb", "VWAP_dev", "price_sma50_ratio", "log_return", "volatility_20d",
                "volume_ratio", "momentum_10d", "ROC_20d", "stoch_k", "williams_r", "CCI", "ADX"]:
        out[col] = feat_df[col]
    out["MACD_pct"] = feat_df["MACD"] / c
    out["MACD_hist_pct"] = feat_df["MACD_hist"] / c
    out["ATR_pct"] = feat_df["ATR"] / c
    out["price_sma200_ratio"] = c / feat_df["SMA_200"]
    out["sma50_sma200_ratio"] = feat_df["SMA_50"] / feat_df["SMA_200"]
    out["bb_width_pct"] = (feat_df["BB_upper"] - feat_df["BB_lower"]) / c
    return out


# ---------------------------------------------------------------------------
# Neural models, trained on one fold
# ---------------------------------------------------------------------------

def _fit_scalers(features: np.ndarray, close: np.ndarray, cutoff: int):
    """Fit feature and target scalers on rows before the cutoff only."""
    from sklearn.preprocessing import StandardScaler

    fs = StandardScaler().fit(features[:cutoff])
    ts = StandardScaler().fit(close[:cutoff].reshape(-1, 1))
    return fs, ts


def _train_net(model, kind: str, x_train: np.ndarray, y_train: np.ndarray, epochs: int, patience: int,
               batch_size: int = 64, lr: float = 1e-3):
    """Train with a chronological 85/15 split of the training window for early stopping."""
    import copy
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from modules.ml_forecasting import TimeSeriesDataset, quantile_loss

    split = int(len(x_train) * 0.85)
    train_ds = TimeSeriesDataset(x_train[:split], y_train[:split], SEQ_LEN)
    val_ds = TimeSeriesDataset(x_train[split:], y_train[split:], SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    def loss_fn(pred, y):
        if kind == "tft":
            return quantile_loss(pred, y.expand_as(pred), model.quantiles)
        return mse(pred, y)

    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(np.mean([loss_fn(model(xb), yb).item() for xb, yb in val_loader]))
        if val < best:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _predict_next_close(model, kind: str, x_scaled: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Deterministic (dropout off) next-day scaled Close for windows ending at each origin."""
    import torch

    windows = np.stack([x_scaled[o - SEQ_LEN + 1: o + 1] for o in origins]).astype(np.float32)
    outs = []
    with torch.no_grad():
        for i in range(0, len(windows), 256):
            out = model(torch.from_numpy(windows[i:i + 256]))
            outs.append(out[:, 2].numpy() if kind == "tft" else out[:, 0].numpy())  # TFT: median quantile
    return np.concatenate(outs)


def neural_horizon_returns(next_close_pred: np.ndarray, close_at_origin: np.ndarray, horizon: int,
                           hist_vol: float) -> np.ndarray:
    """Turn next-day price predictions into h-day returns the way the pipeline does."""
    from modules.ml_forecasting import project_horizon_prices

    out = np.empty(len(next_close_pred))
    for j, (p1, c0) in enumerate(zip(next_close_pred, close_at_origin)):
        proj = project_horizon_prices(np.array([p1]), float(c0), horizon, hist_vol)[0]
        out[j] = proj / c0 - 1.0
    return out


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------

def evaluate_ticker(ticker: str, ohlcv: pd.DataFrame, horizons: List[int], n_folds: int, test_len: int,
                    epochs: int, patience: int, seed: int) -> pd.DataFrame:
    """Return one row per (fold, origin, horizon) with the actual return and every model's prediction."""
    import torch

    from modules.ml_forecasting import LSTMForecaster, SimplifiedTFT, build_features

    feat_df, feat_cols = build_features(ohlcv, ticker=ticker)
    features = feat_df[feat_cols].values.astype(np.float32)
    close = feat_df["Close"].values.astype(float)
    stat_features = stationary_features(feat_df).values.astype(np.float32)
    dates = feat_df.index
    rets = daily_returns(close)
    splits = make_walk_forward_splits(len(feat_df), n_folds, test_len)
    rows = []

    for fold, (cutoff, test_end) in enumerate(splits):
        t0 = time.time()
        origins = np.arange(cutoff, test_end)
        close_o = close[origins]
        train_r = rets[1:cutoff]
        hist_vol = float(np.std(train_r))
        p1 = {}
        # variant 1: as built (price-level target, all technical features)
        # variant 2: return target, ratio-only features
        for tag, feats, target in (("", features, close), ("_ret", stat_features, rets)):
            fs, ts = _fit_scalers(feats, target, cutoff)
            x_all = np.nan_to_num(fs.transform(feats)).astype(np.float32)  # constant columns have zero variance
            y_all = ts.transform(target.reshape(-1, 1)).flatten().astype(np.float32)
            # training samples only use rows < cutoff (inputs and the next-day target)
            x_tr, y_tr = x_all[:cutoff], y_all[:cutoff]
            for kind, net_cls in (("lstm", LSTMForecaster), ("tft", SimplifiedTFT)):
                torch.manual_seed(seed + fold)
                np.random.seed(seed + fold)
                net = _train_net(net_cls(x_all.shape[1]), kind, x_tr, y_tr, epochs, patience)
                raw = ts.inverse_transform(_predict_next_close(net, kind, x_all, origins).reshape(-1, 1)).flatten()
                p1[kind + tag] = raw if tag == "" else close_o * (1.0 + raw)

        mu, phi = fit_ar1(train_r)

        for h in horizons:
            actual = forward_returns(close, h)[origins]
            r = {k: neural_horizon_returns(v, close_o, h, hist_vol) for k, v in p1.items()}
            preds = {
                "random_walk": baseline_random_walk(len(origins)),
                "drift": baseline_drift(rets, origins, h),
                "ar1": baseline_ar1(mu, phi, rets[origins], h),
                "lstm": r["lstm"],
                "tft": r["tft"],
                "ensemble": 0.8 * r["lstm"] + 0.2 * r["tft"],
                "lstm_ret": r["lstm_ret"],
                "tft_ret": r["tft_ret"],
                "ensemble_ret": 0.8 * r["lstm_ret"] + 0.2 * r["tft_ret"],
            }
            frame = pd.DataFrame({"ticker": ticker, "fold": fold, "date": dates[origins], "horizon": h,
                                  "actual": actual})
            for name, p in preds.items():
                frame[name] = p
            rows.append(frame)
        logger.info("  %s fold %d/%d (train %d rows to %s, test %d origins) done in %.0fs",
                    ticker, fold + 1, n_folds, cutoff, dates[cutoff - 1].date(), len(origins), time.time() - t0)

    return pd.concat(rows, ignore_index=True)


def summarise(preds: pd.DataFrame) -> pd.DataFrame:
    """Pool folds and compute metrics per ticker / horizon / model."""
    out = []
    for (ticker, h), g in preds.groupby(["ticker", "horizon"]):
        g = g.dropna(subset=["actual"])
        for model in MODEL_NAMES:
            m = compute_metrics(g["actual"].values, g[model].values)
            m.update(ticker=ticker, horizon=h, model=model,
                     skill_vs_rw=skill_score(g["actual"].values, g[model].values))
            out.append(m)
    cols = ["ticker", "horizon", "model", "n", "mae", "rmse", "dir_acc", "skill_vs_rw"]
    return pd.DataFrame(out)[cols]


def average_over_tickers(summary: pd.DataFrame) -> pd.DataFrame:
    g = summary.groupby(["horizon", "model"], sort=False)
    avg = g.agg(mae=("mae", "mean"), rmse=("rmse", "mean"), dir_acc=("dir_acc", "mean"),
                skill_vs_rw=("skill_vs_rw", "mean"),
                tickers_beating_rw=("skill_vs_rw", lambda s: int((s > 0).sum())),
                n_tickers=("skill_vs_rw", "size")).reset_index()
    avg["model"] = pd.Categorical(avg["model"], MODEL_NAMES, ordered=True)
    return avg.sort_values(["horizon", "model"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_table(df: pd.DataFrame, cols: List[str], pct_cols=(), digits=4) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    cells.append("n/a")
                elif c in pct_cols:
                    cells.append(f"{100 * v:.1f}%")
                else:
                    cells.append(f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, summary: pd.DataFrame, avg: pd.DataFrame, config: Dict) -> None:
    avg_show = avg.copy()
    avg_show["model"] = avg_show["model"].astype(str)
    avg_show["beats_rw"] = avg_show.apply(
        lambda r: "-" if r["model"] == "random_walk" else f"{int(r['tickers_beating_rw'])}/{int(r['n_tickers'])}", axis=1)
    lines = [
        "# Forecast evaluation results",
        "",
        "Generated by `evaluate_forecasts.py`. Every number below comes from `forecast_eval_results.csv`.",
        "",
        "## Configuration that was run",
        "",
    ]
    for k, v in config.items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "Returns are simple returns over the horizon. MAE and RMSE are in return units (0.01 = 1%).",
        "Skill is 1 - MSE(model)/MSE(random walk), so positive means better than predicting zero.",
        "Directional accuracy is n/a for the random walk because it never predicts a direction.",
        "Test origins are daily, so horizon 5 and 20 targets overlap and the samples are not independent.",
        "",
        "## Average over tickers",
        "",
        _fmt_table(avg_show, ["horizon", "model", "mae", "rmse", "dir_acc", "skill_vs_rw", "beats_rw"],
                   pct_cols=("dir_acc",)),
        "",
        "`beats_rw` counts the tickers where the model had a positive skill score.",
        "",
        "## Per ticker",
        "",
    ]
    for h in sorted(summary["horizon"].unique()):
        lines += [f"### Horizon {h} trading day(s)", "",
                  _fmt_table(summary[summary["horizon"] == h].drop(columns="horizon"),
                             ["ticker", "model", "n", "mae", "rmse", "dir_acc", "skill_vs_rw"],
                             pct_cols=("dir_acc",)), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figures(preds: pd.DataFrame, summary: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    models = [m for m in MODEL_NAMES if m != "random_walk"]
    horizons = sorted(summary["horizon"].unique())
    colours = {"drift": "#8a8a8a", "ar1": "#b8a77a", "lstm": "#3b6ea8", "tft": "#c0642f", "ensemble": "#3a8a5a",
               "lstm_ret": "#7aa6d6", "tft_ret": "#e39a6c", "ensemble_ret": "#7bc496"}

    # Figure 1: skill score vs random walk, per ticker dots and mean bar
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, horizons):
        s = summary[summary["horizon"] == h]
        for i, m in enumerate(models):
            vals = s[s["model"] == m]["skill_vs_rw"].values
            ax.bar(i, np.mean(vals), color=colours[m], alpha=0.75, width=0.6)
            ax.scatter(np.full(len(vals), i), vals, color="black", s=10, zorder=3)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_title(f"{h}-day horizon")
        ax.set_ylabel("skill vs random walk (MSE)")
        # the price-target networks are far off scale, so zoom on the small models and label the rest
        small = s[s["model"].isin(["drift", "ar1", "lstm_ret", "tft_ret", "ensemble_ret"])]["skill_vs_rw"]
        lo, hi = min(small.min(), 0) - 0.05, max(small.max(), 0) + 0.05
        ax.set_ylim(lo, hi)
        for i, m in enumerate(models):
            mean_v = float(np.mean(s[s["model"] == m]["skill_vs_rw"].values))
            if mean_v < lo:
                ax.text(i, lo + 0.01 * (hi - lo), f"{mean_v:.1f}\n(off scale)", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Out-of-sample skill vs random walk (bar: mean over tickers, dots: each ticker; below 0 is worse)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "forecast_skill_vs_random_walk.png", dpi=130)
    plt.close(fig)

    # Figure 2: predicted vs realised returns for the two ensembles, pooled over tickers
    hs = [h for h in (1, 5, 20) if h in horizons][:3]
    rows = [("ensemble", "as built (price target)"), ("ensemble_ret", "return-target variant")]
    fig, axes = plt.subplots(len(rows), len(hs), figsize=(4.2 * len(hs), 3.6 * len(rows)), squeeze=False)
    for ri, (col, label) in enumerate(rows):
        for ax, h in zip(axes[ri], hs):
            g = preds[(preds["horizon"] == h)].dropna(subset=["actual"])
            ax.scatter(g[col], g["actual"], s=3, alpha=0.25, color=colours[col])
            lim = float(np.nanpercentile(np.abs(g["actual"]), 99))
            ax.set_ylim(-lim, lim)
            ax.axhline(0, color="black", lw=0.6)
            ax.axvline(0, color="black", lw=0.6)
            corr = np.corrcoef(g[col], g["actual"])[0, 1]
            ax.set_title(f"{label}, {h}-day: corr {corr:.3f}", fontsize=9)
            ax.set_xlabel("predicted return")
            ax.set_ylabel("realised return")
    fig.suptitle("Ensemble prediction vs what happened (all tickers, out-of-sample)")
    fig.tight_layout()
    fig.savefig(fig_dir / "forecast_pred_vs_actual.png", dpi=130)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    ap.add_argument("--start", default="2013-01-01", help="first date of price history")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--test-len", type=int, default=252, help="trading days per test fold")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refresh", action="store_true", help="re-download prices instead of using the cache")
    ap.add_argument("--out-dir", default=str(BASE_DIR / "docs"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, spans = [], {}
    for t in args.tickers:
        logger.info("%s: loading prices", t)
        ohlcv = load_prices(t, args.start, refresh=args.refresh)
        p = evaluate_ticker(t, ohlcv, args.horizons, args.folds, args.test_len, args.epochs, args.patience, args.seed)
        spans[t] = (p["date"].min().date(), p["date"].max().date())
        frames.append(p)
    preds = pd.concat(frames, ignore_index=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(CACHE_DIR / "predictions.csv.gz", index=False)
    summary = summarise(preds)
    avg = average_over_tickers(summary)
    summary.to_csv(out_dir / "forecast_eval_results.csv", index=False, float_format="%.6f")
    avg.to_csv(out_dir / "forecast_eval_average.csv", index=False, float_format="%.6f")

    first = min(v[0] for v in spans.values())
    last = max(v[1] for v in spans.values())
    config = {
        "tickers": ", ".join(args.tickers),
        "horizons (trading days)": ", ".join(map(str, args.horizons)),
        "price history": f"yfinance daily, adjusted, from {args.start}",
        "test period covered (forecast origins)": f"{first} to {last}",
        "walk-forward": f"{args.folds} expanding-window folds of {args.test_len} trading days each, models refit per fold",
        "training": f"max {args.epochs} epochs, early stopping patience {args.patience} on last 15% of the training window",
        "features": "technical indicators from build_features(), no sentiment/macro/cross-asset (placeholder columns are constant)",
        "scaling": "StandardScaler for features and target fit on the training window of each fold only",
        "networks": "LSTMForecaster and SimplifiedTFT from modules/ml_forecasting.py, default sizes, dropout off at test time",
        "multi-day forecast": "one-step prediction projected with project_horizon_prices(); ensemble = 0.8 LSTM + 0.2 TFT median",
        "model variants": "lstm/tft/ensemble = pipeline as built (target is next-day Close price, all technical features); lstm_ret/tft_ret/ensemble_ret = same networks with next-day return target and ratio-only features",
        "seed": args.seed,
    }
    write_markdown(out_dir / "forecast_eval_results.md", summary, avg, config)
    make_figures(preds, summary, out_dir / "figures")
    logger.info("Wrote results to %s", out_dir)
    print(avg.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
