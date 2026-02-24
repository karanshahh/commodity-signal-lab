"""Performance metrics: CAGR, Sharpe, Sortino, hit rate, turnover."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def cagr(equity: pd.Series, periods_per_year: float = 252) -> float:
    """Compound annual growth rate."""
    n = len(equity)
    if n < 2:
        return 0.0
    e0, e1 = equity.iloc[0], equity.iloc[-1]
    if e0 <= 0:
        return 0.0
    total_return = e1 / e0
    years = n / periods_per_year
    if years <= 0 or total_return <= 0:
        return 0.0
    return float((total_return ** (1 / years) - 1) * 100)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: float = 252) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: float = 252) -> float:
    """Annualized Sortino ratio (downside deviation)."""
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def hit_rate(returns: pd.Series, positions: pd.Series) -> float:
    """Fraction of periods where position sign matches return sign (profitable direction)."""
    aligned = (returns > 0) == (positions > 0)
    valid = positions != 0
    if valid.sum() == 0:
        return 0.0
    return float(aligned[valid].mean() * 100)


def turnover(positions: pd.Series) -> float:
    """Average absolute position change (turnover)."""
    return float(positions.diff().abs().mean() * 100)


def compute_metrics(
    returns: pd.Series,
    positions: pd.Series,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute full metrics suite."""
    ret = returns.fillna(0)
    equity = (1 + ret).cumprod()
    return {
        "cagr_pct": cagr(equity),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "hit_rate_pct": hit_rate(returns, positions),
        "turnover_pct": turnover(positions),
        "total_return_pct": float((equity.iloc[-1] / equity.iloc[0] - 1) * 100)
        if len(equity) > 1 and equity.iloc[0] > 0
        else 0.0,
        "n_periods": len(returns),
    }
