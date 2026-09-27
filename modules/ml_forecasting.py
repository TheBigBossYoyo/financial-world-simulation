"""
Module 3 — Machine Learning Forecasting Engine
LSTM + simplified Temporal Fusion Transformer per ticker.
Features: 20+ technical indicators, sentiment, macro, cross-asset.
Output: probability distributions via Monte Carlo dropout.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CROSS_ASSETS,
    DEFAULT_HORIZONS,
    MODELS_DIR,
    RANDOM_SEED,
    SEMI_TICKERS,
    TICKERS,
)

logger = logging.getLogger(__name__)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 20+ technical indicators from OHLCV data."""
    out = df.copy()
    close = out["Close"].values.astype(float)
    high = out["High"].values.astype(float)
    low = out["Low"].values.astype(float)
    volume = out["Volume"].values.astype(float)

    # 1. RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
    out["RSI"] = 100 - (100 / (1 + rs))

    # 2-4. MACD (12, 26, 9)
    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9).mean().values
    out["MACD"] = macd_line
    out["MACD_signal"] = signal_line
    out["MACD_hist"] = macd_line - signal_line

    # 5. ATR (14)
    tr1 = high - low
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    out["ATR"] = pd.Series(tr).rolling(14).mean().values

    # 6. OBV
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    out["OBV"] = obv

    # 7-9. Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(20).mean().values
    std20 = pd.Series(close).rolling(20).std().values
    out["BB_upper"] = sma20 + 2 * std20
    out["BB_lower"] = sma20 - 2 * std20
    out["BB_pctb"] = np.where(
        (out["BB_upper"] - out["BB_lower"]) != 0,
        (close - out["BB_lower"]) / (out["BB_upper"] - out["BB_lower"]),
        0.5,
    )

    # 10. VWAP deviation
    typical_price = (high + low + close) / 3
    cum_tp_vol = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)
    vwap = np.where(cum_vol != 0, cum_tp_vol / cum_vol, close)
    out["VWAP_dev"] = (close - vwap) / np.where(vwap != 0, vwap, 1.0)

    # 11-12. SMA 50 & 200
    out["SMA_50"] = pd.Series(close).rolling(50).mean().values
    out["SMA_200"] = pd.Series(close).rolling(200).mean().values

    # 13. Price / SMA_50 ratio
    out["price_sma50_ratio"] = close / np.where(out["SMA_50"] != 0, out["SMA_50"], 1.0)

    # 14. Log returns
    out.loc[:, "log_return"] = np.log(close / np.where(prev_close != 0, prev_close, close))
    out.loc[out.index[0], "log_return"] = 0

    # 15. Realized volatility (20d)
    out["volatility_20d"] = pd.Series(out["log_return"]).rolling(20).std().values

    # 16. Volume ratio
    vol_sma = pd.Series(volume).rolling(20).mean().values
    out["volume_ratio"] = volume / np.where(vol_sma != 0, vol_sma, 1.0)

    # 17. Momentum (10d)
    close_10d = np.roll(close, 10)
    close_10d[:10] = close[:10]
    out.loc[:, "momentum_10d"] = close / np.where(close_10d != 0, close_10d, close) - 1
    out.loc[out.index[:10], "momentum_10d"] = 0

    # 18. Rate of Change (20d)
    close_20d = np.roll(close, 20)
    close_20d[:20] = close[:20]
    out.loc[:, "ROC_20d"] = (close - close_20d) / np.where(close_20d != 0, close_20d, 1.0)
    out.loc[out.index[:20], "ROC_20d"] = 0

    # 19. Stochastic %K (14)
    low14 = pd.Series(low).rolling(14).min().values
    high14 = pd.Series(high).rolling(14).max().values
    denom = high14 - low14
    out["stoch_k"] = np.where(denom != 0, (close - low14) / denom * 100, 50.0)

    # 20. Williams %R
    out["williams_r"] = np.where(denom != 0, (high14 - close) / denom * -100, -50.0)

    # 21. CCI (20)
    tp_series = pd.Series(typical_price)
    sma_tp = tp_series.rolling(20).mean().values
    mad_tp = tp_series.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    out["CCI"] = np.where(mad_tp != 0, (typical_price - sma_tp) / (0.015 * mad_tp), 0.0)

    # 22. Average Directional Index approximation
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0.0
    down_move[0] = 0.0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14 = pd.Series(tr).rolling(14).mean().values
    plus_di = np.where(atr14 != 0, pd.Series(plus_dm).rolling(14).mean().values / atr14 * 100, 0)
    minus_di = np.where(atr14 != 0, pd.Series(minus_dm).rolling(14).mean().values / atr14 * 100, 0)
    dx = np.where(
        (plus_di + minus_di) != 0,
        np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100,
        0,
    )
    out["ADX"] = pd.Series(dx).rolling(14).mean().values

    return out


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(
    ohlcv: pd.DataFrame,
    sentiment_score: float = 0.0,
    sentiment_series: Optional[pd.Series] = None,
    macro_data: Optional[Dict[str, pd.Series]] = None,
    cross_asset_data: Optional[Dict[str, pd.DataFrame]] = None,
    ticker: str = "",
    gdelt_data: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build full feature set for a ticker.

    Returns (feature_df, feature_column_names)
    """
    df = compute_technical_indicators(ohlcv)

    # Add sentiment features
    if sentiment_series is not None and not sentiment_series.empty:
        sentiment_index = pd.DatetimeIndex(pd.to_datetime(sentiment_series.index))
        if sentiment_index.tz is not None:
            sentiment_index = sentiment_index.tz_localize(None)
        aligned_sentiment = pd.Series(sentiment_series.to_numpy(dtype=float), index=sentiment_index)
        aligned_sentiment = aligned_sentiment.sort_index().reindex(df.index, method="ffill").fillna(0.0)
        df["sentiment"] = aligned_sentiment.values
        df["sentiment_change_5d"] = aligned_sentiment.diff(5).fillna(0.0).values
        df["sentiment_ma_5d"] = aligned_sentiment.rolling(5).mean().bfill().fillna(0.0).values
    else:
        df["sentiment"] = sentiment_score
        df["sentiment_change_5d"] = 0.0
        df["sentiment_ma_5d"] = sentiment_score

    # Add macro features (forward-fill to daily)
    if macro_data:
        for name, series in macro_data.items():
            if series is not None and not series.empty:
                s = series.reindex(df.index, method="ffill")
                df[f"macro_{name}"] = s.values if len(s) == len(df) else np.nan

    # Add cross-asset features
    if cross_asset_data:
        # Gold → NEM, Oil → VST/NOC, SOX → NVDA/AVGO/MU
        cross_map = {
            "GC=F": ["NEM"],
            "CL=F": ["VST", "NOC"],
            "^SOX": SEMI_TICKERS,
        }
        for asset_ticker, asset_df in cross_asset_data.items():
            if asset_df is None or asset_df.empty:
                continue
            target_tickers = cross_map.get(asset_ticker, [])
            if ticker in target_tickers or not target_tickers:
                if "Close" in asset_df.columns:
                    asset_close = asset_df["Close"].reindex(df.index, method="ffill")
                    safe_name = asset_ticker.replace("=", "_").replace("^", "").replace(".", "_").replace("-", "_")
                    df[f"cross_{safe_name}"] = asset_close.values if len(asset_close) == len(df) else np.nan
                    # Also add returns
                    asset_ret = asset_close.pct_change()
                    df[f"cross_{safe_name}_ret"] = asset_ret.values if len(asset_ret) == len(df) else np.nan

    # Add GDELT conflict index
    if gdelt_data is not None and not gdelt_data.empty:
        try:
            gdelt_daily = gdelt_data.groupby("date")["volume"].sum()
            gdelt_daily.index = pd.to_datetime(gdelt_daily.index)
            gdelt_reindexed = gdelt_daily.reindex(df.index, method="ffill").fillna(0)
            df["gdelt_tension"] = gdelt_reindexed.values
        except Exception:
            df["gdelt_tension"] = 0.0
    else:
        df["gdelt_tension"] = 0.0

    # Drop rows with NaN (from lookback periods)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Feature columns (exclude OHLCV)
    exclude = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    feature_cols = [c for c in df.columns if c not in exclude]

    return df, feature_cols


def project_horizon_prices(
    predicted_prices: np.ndarray,
    last_price: float,
    horizon: int,
    historical_vol: float,
) -> np.ndarray:
    """Project one-step model prices to longer horizons without numerical blow-ups."""
    if last_price <= 0:
        return np.full_like(predicted_prices, 0.01, dtype=float)

    safe_prices = np.asarray(predicted_prices, dtype=float)
    safe_prices = np.where(np.isfinite(safe_prices), safe_prices, last_price)
    safe_prices = np.clip(safe_prices, 0.01, last_price * 5.0)

    one_step_log_returns = np.log(np.clip(safe_prices / last_price, 1e-6, 5.0))
    daily_log_cap = max(2.0 * historical_vol, 0.015)
    capped_daily_log_returns = np.clip(one_step_log_returns, -daily_log_cap, daily_log_cap)

    shrink = min(1.0, np.sqrt(30.0 / max(horizon, 1)))
    shrunk_daily_log_returns = capped_daily_log_returns * shrink

    horizon_log_cap = min(max(2.5 * historical_vol * np.sqrt(horizon), 0.20), 1.10)
    horizon_log_returns = np.clip(shrunk_daily_log_returns * horizon, -horizon_log_cap, horizon_log_cap)

    return np.maximum(last_price * np.exp(horizon_log_returns), 0.01)


def calibrate_forecast_band(
    percentiles: Dict[str, float],
    last_price: float,
    historical_vol: float,
    horizon: int,
) -> Dict[str, float]:
    """Shrink and bound forecast percentile bands toward realistic horizon ranges."""
    if not percentiles or last_price <= 0:
        return percentiles

    keys = ["p10", "p25", "p50", "p75", "p90"]
    ordered = np.array([float(percentiles.get(k, last_price)) for k in keys], dtype=float)
    ordered = np.maximum.accumulate(ordered)

    base_log = np.log(np.clip(ordered / last_price, 1e-6, 10.0))
    shrink = min(0.8, np.sqrt(45.0 / max(horizon, 1)))
    center = base_log[2] * shrink
    spreads = (base_log - base_log[2]) * shrink

    horizon_cap = min(max(2.2 * historical_vol * np.sqrt(horizon), 0.18), 0.95)
    calibrated = np.clip(center + spreads, -horizon_cap, horizon_cap)
    calibrated_prices = np.maximum(last_price * np.exp(calibrated), 0.01)
    calibrated_prices = np.maximum.accumulate(calibrated_prices)

    return {key: float(value) for key, value in zip(keys, calibrated_prices)}


def historical_baseline_percentiles(df: pd.DataFrame, horizon: int) -> Dict[str, float]:
    """Historical rolling-return baseline for horizon percentiles."""
    close = df["Close"].dropna().astype(float)
    last_price = float(close.iloc[-1])
    if len(close) <= horizon + 20:
        return {key: last_price for key in ["p10", "p25", "p50", "p75", "p90"]}

    horizon_returns = close.shift(-horizon) / close - 1.0
    samples = horizon_returns.dropna().tail(252)
    if samples.empty:
        samples = close.pct_change().dropna().tail(252) * np.sqrt(max(horizon, 1))

    clipped = np.clip(samples.to_numpy(dtype=float), -0.60, 0.80)
    percentiles = np.percentile(clipped, [10, 25, 50, 75, 90])
    prices = np.maximum(last_price * (1.0 + percentiles), 0.01)
    return {key: float(value) for key, value in zip(["p10", "p25", "p50", "p75", "p90"], prices)}


def compute_validation_metrics(df: pd.DataFrame, horizon: int) -> Dict[str, float]:
    """Simple walk-forward baseline validation for realism reporting."""
    close = df["Close"].dropna().astype(float)
    if len(close) <= horizon + 30:
        return {"sample_size": 0, "mae_pct": 0.0, "rmse_pct": 0.0, "directional_accuracy": 0.0}

    future_returns = close.shift(-horizon) / close - 1.0
    trailing_mean = close.pct_change().rolling(60).mean() * horizon

    actual = future_returns.dropna()
    pred = trailing_mean.reindex(actual.index).fillna(0.0)
    window = min(len(actual), 252)
    actual_values = np.clip(actual.tail(window).to_numpy(dtype=float), -1.0, 1.5)
    pred_values = np.clip(pred.tail(window).to_numpy(dtype=float), -0.5, 0.5)

    errors = pred_values - actual_values
    mae_pct = float(np.mean(np.abs(errors))) if len(errors) else 0.0
    rmse_pct = float(np.sqrt(np.mean(errors ** 2))) if len(errors) else 0.0
    directional_accuracy = float(np.mean(np.sign(pred_values) == np.sign(actual_values))) if len(errors) else 0.0
    return {
        "sample_size": int(len(errors)),
        "mae_pct": mae_pct,
        "rmse_pct": rmse_pct,
        "directional_accuracy": directional_accuracy,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """Sliding window dataset for LSTM/TFT training."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 60):
        self.seq_len = seq_len
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.FloatTensor(x), torch.FloatTensor([y])


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """2-layer LSTM with Monte Carlo dropout for uncertainty estimation."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# ---------------------------------------------------------------------------
# Simplified Temporal Fusion Transformer
# ---------------------------------------------------------------------------

class VariableSelectionNetwork(nn.Module):
    """Variable selection for TFT — learns which features matter."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.flattened_grn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.softmax = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1),
        )
        self.transform = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        weights = self.softmax(self.flattened_grn(x))  # (batch, seq_len, input_dim)
        weighted = x * weights  # Element-wise weighting
        return self.transform(weighted), weights


class GatedResidualNetwork(nn.Module):
    """GRN building block for TFT."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        h = self.elu(self.fc1(x))
        h2 = self.dropout(self.fc2(h))
        gate = self.gate(self.elu(self.fc1(x)))
        out = self.layer_norm(self.skip(x) + gate * h2)
        return out


class SimplifiedTFT(nn.Module):
    """Simplified Temporal Fusion Transformer with quantile output."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_heads: int = 4,
                 dropout: float = 0.3, quantiles: List[float] = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.quantiles = quantiles

        # Variable selection
        self.vsn = VariableSelectionNetwork(input_dim, hidden_dim)

        # Temporal processing (LSTM encoder)
        self.lstm_encoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=1,
            batch_first=True, dropout=dropout,
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Post-attention GRN
        self.grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

        # Quantile output heads
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in quantiles
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Variable selection
        selected, var_weights = self.vsn(x)

        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(selected)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(lstm_out + attn_out)

        # GRN
        grn_out = self.grn(attn_out[:, -1, :])  # Use last timestep
        grn_out = self.dropout(grn_out)

        # Quantile outputs
        quantile_preds = [head(grn_out) for head in self.quantile_heads]
        return torch.cat(quantile_preds, dim=-1)  # (batch, n_quantiles)


# ---------------------------------------------------------------------------
# Training & Inference
# ---------------------------------------------------------------------------

def quantile_loss(preds: torch.Tensor, targets: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    """Pinball loss for quantile regression."""
    losses = []
    for i, q in enumerate(quantiles):
        errors = targets - preds[:, i:i + 1]
        losses.append(torch.max(q * errors, (q - 1) * errors).mean())
    return sum(losses) / len(losses)


def _save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    input_dim: int,
    seq_len: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "seq_len": seq_len,
            "metadata": metadata or {},
        },
        checkpoint_path,
    )


def _try_load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    input_dim: int,
    seq_len: int,
    ticker: str,
    label: str,
) -> bool:
    if not checkpoint_path.exists():
        return False

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            if int(payload.get("input_dim", input_dim)) != input_dim or int(payload.get("seq_len", seq_len)) != seq_len:
                logger.info(f"    Existing {label} checkpoint incompatible for {ticker}; retraining")
                return False
            state_dict = payload["state_dict"]
        else:
            state_dict = payload

        model.load_state_dict(state_dict)
        logger.info(f"    Reusing cached {label} checkpoint for {ticker}")
        return True
    except Exception as exc:
        logger.warning(f"    Failed to load {label} checkpoint for {ticker}: {exc}")
        return False


def train_lstm(
    features: np.ndarray,
    targets: np.ndarray,
    ticker: str,
    epochs: int = 50,
    seq_len: int = 60,
    lr: float = 1e-3,
    batch_size: int = 32,
    reuse_models: bool = True,
) -> LSTMForecaster:
    """Train LSTM model for a single ticker."""
    # Train/val split (80/20)
    split_idx = int(len(features) * 0.8)
    train_ds = TimeSeriesDataset(features[:split_idx], targets[:split_idx], seq_len)
    val_ds = TimeSeriesDataset(features[split_idx:], targets[split_idx:], seq_len)

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.warning(f"  ✗ Not enough data to train LSTM for {ticker}")
        return None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    input_dim = features.shape[1]
    model = LSTMForecaster(input_dim)
    checkpoint_path = MODELS_DIR / f"lstm_{ticker}.pt"
    if reuse_models and _try_load_checkpoint(model, checkpoint_path, input_dim, seq_len, ticker, "LSTM"):
        return model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience, patience_counter = 10, 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            _save_checkpoint(model, checkpoint_path, input_dim, seq_len, {"ticker": ticker, "model": "lstm"})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"    Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch + 1}/{epochs} — train: {train_loss:.6f}, val: {val_loss:.6f}")

    # Load best model
    _try_load_checkpoint(model, checkpoint_path, input_dim, seq_len, ticker, "LSTM")
    return model


def train_tft(
    features: np.ndarray,
    targets: np.ndarray,
    ticker: str,
    epochs: int = 50,
    seq_len: int = 60,
    lr: float = 1e-3,
    batch_size: int = 32,
    quantiles: List[float] = None,
    reuse_models: bool = True,
) -> SimplifiedTFT:
    """Train simplified TFT model for a single ticker."""
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    split_idx = int(len(features) * 0.8)
    train_ds = TimeSeriesDataset(features[:split_idx], targets[:split_idx], seq_len)
    val_ds = TimeSeriesDataset(features[split_idx:], targets[split_idx:], seq_len)

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.warning(f"  ✗ Not enough data to train TFT for {ticker}")
        return None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    input_dim = features.shape[1]
    model = SimplifiedTFT(input_dim, quantiles=quantiles)
    checkpoint_path = MODELS_DIR / f"tft_{ticker}.pt"
    if reuse_models and _try_load_checkpoint(model, checkpoint_path, input_dim, seq_len, ticker, "TFT"):
        return model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience, patience_counter = 10, 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = quantile_loss(pred, y_batch.expand_as(pred), quantiles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch)
                val_loss += quantile_loss(pred, y_batch.expand_as(pred), quantiles).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            _save_checkpoint(model, checkpoint_path, input_dim, seq_len, {"ticker": ticker, "model": "tft", "quantiles": quantiles})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"    Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch + 1}/{epochs} — train: {train_loss:.6f}, val: {val_loss:.6f}")

    _try_load_checkpoint(model, checkpoint_path, input_dim, seq_len, ticker, "TFT")
    return model


def mc_dropout_predict(
    model: nn.Module,
    x_input: torch.Tensor,
    n_forward: int = 50,
    is_tft: bool = False,
) -> np.ndarray:
    """Monte Carlo dropout prediction — N forward passes with dropout enabled.

    Returns array of shape (n_forward,) for LSTM or (n_forward, n_quantiles) for TFT.
    """
    model.train()  # Enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(n_forward):
            pred = model(x_input.unsqueeze(0))
            preds.append(pred.numpy().flatten())
    return np.array(preds)


# ---------------------------------------------------------------------------
# Master forecasting function
# ---------------------------------------------------------------------------

def run_forecasting(
    ohlcv_data: Dict[str, pd.DataFrame],
    sentiment_scores: Dict[str, float],
    macro_data: Dict[str, pd.Series],
    rolling_sentiment: Optional[Dict[str, pd.DataFrame]] = None,
    gdelt_data: Optional[pd.DataFrame] = None,
    horizons: List[int] = None,
    epochs: int = 50,
    n_mc_passes: int = 50,
    dry_run: bool = False,
    reuse_models: bool = True,
) -> Dict[str, Any]:
    """Run full ML forecasting pipeline for all tickers.

    Returns per-ticker forecast distributions at each horizon.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    if dry_run:
        epochs = 5
        n_mc_passes = 10

    logger.info("=" * 60)
    logger.info("MODULE 3 — ML Forecasting Engine")
    logger.info("=" * 60)

    # Build cross-asset data dict
    cross_asset_data = {}
    for name, ca_ticker in CROSS_ASSETS.items():
        if ca_ticker in ohlcv_data:
            cross_asset_data[ca_ticker] = ohlcv_data[ca_ticker]

    results = {}
    scalers = {}

    for ticker in TICKERS:
        logger.info(f"\n  --- {ticker} ---")
        df = ohlcv_data.get(ticker, pd.DataFrame())
        if df.empty:
            logger.warning(f"  ✗ No OHLCV data for {ticker}, skipping")
            results[ticker] = None
            continue

        # Build features
        sentiment = sentiment_scores.get(ticker, 0.0)
        sentiment_series = None
        if rolling_sentiment:
            ticker_sentiment = rolling_sentiment.get(ticker)
            if ticker_sentiment is not None and not ticker_sentiment.empty and "composite" in ticker_sentiment.columns:
                sentiment_series = ticker_sentiment["composite"]
        feat_df, feat_cols = build_features(
            df,
            sentiment_score=sentiment,
            sentiment_series=sentiment_series,
            macro_data=macro_data,
            cross_asset_data=cross_asset_data,
            ticker=ticker,
            gdelt_data=gdelt_data,
        )

        if len(feat_df) < 120:
            logger.warning(f"  ✗ Only {len(feat_df)} rows after feature engineering, skipping")
            results[ticker] = None
            continue

        # Prepare arrays
        features = feat_df[feat_cols].values.astype(np.float32)
        targets = feat_df["Close"].values.astype(np.float32)
        hist_returns = df["Close"].pct_change().dropna()
        hist_daily_vol = float(hist_returns.std()) if len(hist_returns) > 20 else 0.02
        validation = {horizon: compute_validation_metrics(df, horizon) for horizon in horizons}

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        scalers[ticker] = scaler

        # Scale targets
        target_scaler = StandardScaler()
        targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()

        logger.info(f"  Features: {features.shape[1]} | Samples: {features.shape[0]}")

        # Train LSTM
        logger.info(f"  Training LSTM...")
        lstm_model = train_lstm(features_scaled, targets_scaled, ticker, epochs=epochs, reuse_models=reuse_models)

        # Train TFT
        logger.info(f"  Training TFT...")
        tft_model = train_tft(features_scaled, targets_scaled, ticker, epochs=epochs, reuse_models=reuse_models)

        # MC Dropout predictions at each horizon
        last_price = float(df["Close"].iloc[-1])
        seq_len = 60
        last_seq = torch.FloatTensor(features_scaled[-seq_len:])

        ticker_forecasts = {}
        for horizon in horizons:
            fc = {"horizon_days": horizon, "last_price": last_price}
            fc["baseline_percentiles"] = historical_baseline_percentiles(df, horizon)

            # LSTM MC dropout
            if lstm_model is not None:
                lstm_preds_scaled = mc_dropout_predict(lstm_model, last_seq, n_mc_passes)
                lstm_preds = target_scaler.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten()
                horizon_prices = project_horizon_prices(lstm_preds, last_price, horizon, hist_daily_vol)
                fc["lstm_percentiles"] = {
                    "p10": float(np.percentile(horizon_prices, 10)),
                    "p25": float(np.percentile(horizon_prices, 25)),
                    "p50": float(np.percentile(horizon_prices, 50)),
                    "p75": float(np.percentile(horizon_prices, 75)),
                    "p90": float(np.percentile(horizon_prices, 90)),
                }
                fc["lstm_mean"] = float(np.mean(horizon_prices))
                fc["lstm_std"] = float(np.std(horizon_prices))

            # TFT predictions (already gives quantiles)
            if tft_model is not None:
                tft_preds = mc_dropout_predict(tft_model, last_seq, n_mc_passes, is_tft=True)
                # Average across MC passes for each quantile
                tft_mean_quantiles = np.mean(tft_preds, axis=0)
                tft_prices = target_scaler.inverse_transform(
                    tft_mean_quantiles.reshape(-1, 1)
                ).flatten()
                if len(tft_prices) >= 5:
                    tft_horizon_prices = project_horizon_prices(tft_prices, last_price, horizon, hist_daily_vol)
                    fc["tft_percentiles"] = {
                        "p10": float(tft_horizon_prices[0]),
                        "p25": float(tft_horizon_prices[1]),
                        "p50": float(tft_horizon_prices[2]),
                        "p75": float(tft_horizon_prices[3]),
                        "p90": float(tft_horizon_prices[4]),
                    }

            # Ensemble (average LSTM and TFT percentiles)
            ensemble = {}
            for pkey in ["p10", "p25", "p50", "p75", "p90"]:
                weighted_sum = 0.0
                total_weight = 0.0
                if "lstm_percentiles" in fc:
                    weighted_sum += 0.8 * fc["lstm_percentiles"][pkey]
                    total_weight += 0.8
                if "tft_percentiles" in fc:
                    tft_weight = 0.2 if horizon <= 90 else 0.1
                    weighted_sum += tft_weight * fc["tft_percentiles"][pkey]
                    total_weight += tft_weight
                if total_weight > 0:
                    ensemble[pkey] = float(weighted_sum / total_weight)
                elif "baseline_percentiles" in fc:
                    ensemble[pkey] = fc["baseline_percentiles"][pkey]
            if ensemble and "baseline_percentiles" in fc:
                baseline_weight = 0.35 if horizon <= 90 else 0.5
                ensemble = {
                    pkey: float((1 - baseline_weight) * ensemble[pkey] + baseline_weight * fc["baseline_percentiles"][pkey])
                    for pkey in ensemble
                }
            if ensemble:
                ensemble = calibrate_forecast_band(ensemble, last_price, hist_daily_vol, horizon)
            fc["ensemble_percentiles"] = ensemble

            # Annualised volatility
            implied_annual_vol = hist_daily_vol * np.sqrt(252)
            if ensemble and last_price > 0:
                horizon_sigma = max(
                    abs(np.log(max(ensemble.get("p90", last_price), 0.01) / last_price)),
                    abs(np.log(max(ensemble.get("p10", last_price), 0.01) / last_price)),
                ) / 1.645
                implied_annual_vol = horizon_sigma / np.sqrt(max(horizon, 1) / 252)
            hist_annual_vol = hist_daily_vol * np.sqrt(252)
            fc["annualised_volatility"] = float(np.clip(0.65 * hist_annual_vol + 0.35 * implied_annual_vol, 0.12, 0.85))

            # Sharpe ratio (base case: using p50 return)
            if ensemble and last_price > 0:
                expected_return = np.log(max(ensemble.get("p50", last_price), 0.01) / last_price) * (252 / max(horizon, 1))
                expected_return = float(np.clip(expected_return, -0.35, 0.35))
                vol = fc.get("annualised_volatility", 0.3)
                risk_free = 0.04  # Approximate
                fc["sharpe_ratio"] = float((expected_return - risk_free) / vol) if vol > 0 else 0.0

            ticker_forecasts[horizon] = fc

        results[ticker] = {
            "forecasts": ticker_forecasts,
            "validation": validation,
            "feature_count": int(features.shape[1]),
            "sample_count": int(features.shape[0]),
        }
        logger.info(
            "  Validation: "
            + ", ".join(
                f"T+{h} dir={validation[h]['directional_accuracy']:.0%} mae={validation[h]['mae_pct']:.1%}"
                for h in horizons
            )
        )
        logger.info(f"  ✓ {ticker} forecasting complete")

    # Save scalers for later use
    with open(MODELS_DIR / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    logger.info("\n✓ ML Forecasting complete for all tickers.")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Module 3 — ML Forecasting Engine loaded. Run via main.py for full pipeline.")
