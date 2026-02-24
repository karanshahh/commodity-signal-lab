"""Technical indicators and price-derived features."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _returns(close: pd.Series, periods: int) -> pd.Series:
    """Forward-looking returns (no look-ahead: shift(1) applied by caller)."""
    return close.pct_change(periods)


def _momentum(close: pd.Series, periods: int) -> pd.Series:
    """Price momentum over horizon."""
    return close / close.shift(periods) - 1


def _rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualized rolling volatility."""
    return returns.rolling(window).std() * np.sqrt(252)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI indicator. Uses past data only."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def build_technical_features(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build technical features from OHLCV. No look-ahead: all features use only past data.

    Args:
        df: DataFrame with columns open, high, low, close, volume (lowercase).
        config: Optional config dict with technicals section.

    Returns:
        DataFrame with DatetimeIndex and feature columns.
    """
    cfg = config or {}
    tc = cfg.get("technicals", {})
    ret_horizons = tc.get("return_horizons", [1, 5, 10, 21])
    mom_horizons = tc.get("momentum_horizons", [5, 10, 21])
    vol_window = tc.get("volatility_window", 21)
    rsi_period = tc.get("rsi_period", 14)
    macd_cfg = tc.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    zscore_window = tc.get("zscore_window", 21)

    close = df["close"]
    out = pd.DataFrame(index=df.index)

    # Returns (1d, 5d, etc.) - shifted so we predict next-period
    for h in ret_horizons:
        out[f"returns_{h}d"] = _returns(close, h)

    # Momentum
    for h in mom_horizons:
        out[f"momentum_{h}d"] = _momentum(close, h)

    # Rolling volatility (annualized)
    ret_1d = _returns(close, 1)
    out["volatility"] = _rolling_vol(ret_1d, vol_window)
    out["volatility_21d"] = _rolling_vol(ret_1d, 21)

    # RSI
    out["rsi"] = _rsi(close, rsi_period)

    # MACD
    macd_line, signal_line, hist = _macd(
        close,
        fast=macd_cfg.get("fast", 12),
        slow=macd_cfg.get("slow", 26),
        signal=macd_cfg.get("signal", 9),
    )
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    # Carry proxy: (close - open) / close for intraday; for futures use front-back spread if available
    # Simplified: use 1d return as carry proxy
    out["carry_proxy"] = ret_1d

    # Z-scores of returns and volatility
    out["returns_zscore"] = _zscore(ret_1d, zscore_window)
    out["volatility_zscore"] = _zscore(out["volatility"], zscore_window)

    return out.dropna(how="all")
