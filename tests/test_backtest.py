"""Backtest engine tests."""

import numpy as np
import pandas as pd
import pytest

from signal_lab.backtest.engine import BacktestEngine
from signal_lab.backtest.metrics import cagr, sharpe_ratio


def test_backtest_engine() -> None:
    """Engine produces equity and metrics."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = np.random.randn(n) * 0.01
    prices = 100 * (1 + pd.Series(returns, index=dates)).cumprod()
    signal = pd.Series(np.random.rand(n), index=dates)
    engine = BacktestEngine({"engine": {"signal_threshold_long": 0.6, "signal_threshold_short": 0.4}})
    result = engine.run(prices, signal)
    assert "equity" in result
    assert "metrics" in result
    assert len(result["equity"]) == len(prices)
    assert "sharpe" in result["metrics"]


def test_cagr_positive() -> None:
    """CAGR for positive growth."""
    equity = pd.Series([1.0, 1.1, 1.21], index=pd.date_range("2020-01-01", periods=3, freq="B"))
    assert cagr(equity, periods_per_year=252) > 0
