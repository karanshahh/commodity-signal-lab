"""Walk-forward backtest engine with position sizing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from signal_lab.backtest.costs import apply_costs
from signal_lab.backtest.metrics import compute_metrics
from signal_lab.backtest.risk import compute_risk_metrics


def _position_from_signal(
    signal: pd.Series,
    threshold_long: float,
    threshold_short: float,
    sizing: str,
    volatility: pd.Series | None,
    target_vol: float,
    max_pct: float,
) -> pd.Series:
    """Convert signal to position (-max_pct to max_pct)."""
    pos = pd.Series(0.0, index=signal.index)
    long_mask = signal >= threshold_long
    short_mask = signal <= threshold_short
    pos[long_mask] = 1.0
    pos[short_mask] = -1.0

    if sizing == "volatility" and volatility is not None:
        vol_adj = target_vol / volatility.replace(0, np.nan)
        vol_adj = vol_adj.clip(0, 2)
        pos = pos * vol_adj
    else:
        pos = pos * max_pct
    return pos.clip(-max_pct, max_pct)


class BacktestEngine:
    """
    Walk-forward backtest: signal -> position -> strategy returns.
    Supports long/short rules, position sizing, costs.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        ec = cfg.get("engine", {})
        self.initial_capital = ec.get("initial_capital", 1_000_000)
        self.threshold_long = ec.get("signal_threshold_long", 0.55)
        self.threshold_short = ec.get("signal_threshold_short", 0.45)
        self.sizing = ec.get("position_sizing", "fixed")
        self.target_vol = ec.get("volatility_target", 0.15)
        self.max_position_pct = ec.get("max_position_pct", 0.25)
        self.costs_config = cfg.get("costs", {})
        self.risk_config = cfg.get("risk", {})

    def run(
        self,
        prices: pd.Series,
        signal: pd.Series,
        volatility: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Run backtest. prices: close series, signal: prob or return forecast.
        Returns dict with equity, returns, positions, metrics.
        """
        common = prices.index.intersection(signal.index)
        prices = prices.reindex(common).ffill().dropna()
        signal = signal.reindex(common).ffill().dropna()
        vol = volatility.reindex(common).ffill() if volatility is not None else None

        idx = prices.index.intersection(signal.index)
        prices = prices.loc[idx]
        signal = signal.loc[idx]
        vol = vol.loc[idx] if vol is not None else None

        pos = _position_from_signal(
            signal,
            self.threshold_long,
            self.threshold_short,
            self.sizing,
            vol,
            self.target_vol,
            self.max_position_pct,
        )

        returns = prices.pct_change()
        strategy_returns = pos.shift(1) * returns
        strategy_returns = strategy_returns.fillna(0)
        strategy_returns = apply_costs(strategy_returns, pos.shift(1), self.costs_config)

        cumret = (1 + strategy_returns).cumprod()
        equity = self.initial_capital * cumret

        metrics = compute_metrics(strategy_returns, pos.shift(1), self.costs_config)
        risk = compute_risk_metrics(strategy_returns, equity, self.risk_config)
        metrics.update(risk)

        return {
            "equity": equity,
            "returns": strategy_returns,
            "positions": pos,
            "metrics": metrics,
        }
