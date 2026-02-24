"""Macro feature alignment and transformations."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _align_to_daily(
    macro_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    forward_fill_limit: int = 5,
) -> pd.Series:
    """Align macro series to daily frequency via reindex + forward fill."""
    if "value" in macro_df.columns:
        s = macro_df["value"]
    else:
        s = macro_df.iloc[:, 0]
    # Normalize timezones for reindex (yfinance uses tz-aware, FRED often naive)
    idx = target_index.tz_localize(None) if target_index.tz is not None else target_index
    s_idx = s.index.tz_localize(None) if s.index.tz is not None else s.index
    s = s.copy()
    s.index = s_idx
    s = s.reindex(idx, method="ffill", limit=forward_fill_limit)
    return s


def _changes(s: pd.Series, lags: list[int]) -> pd.DataFrame:
    """Period-over-period changes at given lags."""
    out = pd.DataFrame(index=s.index)
    for lag in lags:
        out[f"change_{lag}d"] = s.pct_change(lag)
    return out


def _surprises(s: pd.Series, window: int) -> pd.Series:
    """Surprise = actual - rolling mean (expectation proxy)."""
    return s - s.rolling(window).mean()


def build_macro_features(
    macro_dfs: dict[str, pd.DataFrame],
    target_index: pd.DatetimeIndex,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Align macro series to daily frequency and create changes/surprises.

    No look-ahead: forward-fill uses only past values; changes/surprises use past data.
    """
    cfg = config or {}
    mc = cfg.get("macro", {})
    ffill_limit = mc.get("forward_fill_limit", 5)
    change_lags = mc.get("change_lags", [1, 5, 21])
    surprise_window = mc.get("surprise_window", 5)

    out = pd.DataFrame(index=target_index)
    for name, df in macro_dfs.items():
        s = _align_to_daily(df, target_index, ffill_limit)
        prefix = name.replace(".", "_").lower()
        out[f"{prefix}"] = s
        for lag in change_lags:
            out[f"{prefix}_change_{lag}d"] = s.ffill().pct_change(lag)
        out[f"{prefix}_surprise"] = _surprises(s, surprise_window)
    return out
