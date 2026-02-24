"""Risk metrics: VaR, CVaR, max drawdown, rolling Sharpe."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR (Value at Risk)."""
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def cvar_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall)."""
    var = var_historical(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown from peak."""
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax.replace(0, np.nan)
    return float(dd.min())


def rolling_sharpe(returns: pd.Series, window: int = 252, risk_free: float = 0.0) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    excess = returns - risk_free / 252
    return excess.rolling(window).mean() / excess.rolling(window).std().replace(0, np.nan) * np.sqrt(252)


def compute_risk_metrics(
    returns: pd.Series,
    equity: pd.Series | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute VaR, CVaR, max drawdown, rolling Sharpe (final)."""
    cfg = config or {}
    conf = cfg.get("var_confidence", 0.95)
    equity = equity if equity is not None else (1 + returns).cumprod()

    return {
        "var_95": var_historical(returns, conf),
        "cvar_95": cvar_historical(returns, conf),
        "max_drawdown": max_drawdown(equity),
        "rolling_sharpe_252": float(rolling_sharpe(returns, 252).iloc[-1]) if len(returns) >= 252 else float("nan"),
    }
