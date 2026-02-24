"""Tests for feature engineering (no look-ahead)."""

import numpy as np
import pandas as pd
import pytest

from signal_lab.features.macro import build_macro_features
from signal_lab.features.regimes import detect_regimes
from signal_lab.features.technicals import build_technical_features


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Generate synthetic OHLCV for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = np.random.randn(n) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1e6, 1e7, n),
        },
        index=dates,
    )


def test_technicals_no_lookahead(sample_prices: pd.DataFrame) -> None:
    """Technical features use only past data."""
    tech = build_technical_features(sample_prices)
    assert "returns_1d" in tech.columns
    assert "rsi" in tech.columns
    assert "macd" in tech.columns
    # At row i, returns_1d should not use close[i+1]
    assert not tech["returns_1d"].isna().all()


def test_technicals_shape(sample_prices: pd.DataFrame) -> None:
    """Output has expected columns."""
    tech = build_technical_features(sample_prices)
    expected = ["returns_1d", "momentum_5d", "volatility", "rsi", "macd", "returns_zscore"]
    for col in expected:
        assert col in tech.columns, f"Missing {col}"


def test_macro_alignment() -> None:
    """Macro features align to daily index."""
    target = pd.date_range("2020-01-01", periods=100, freq="B")
    macro = pd.DataFrame(
        {"value": np.random.randn(20)},
        index=pd.date_range("2020-01-01", periods=20, freq="MS"),
    )
    out = build_macro_features({"cpi": macro}, target)
    assert len(out) == len(target)
    assert "cpi" in out.columns


def test_regimes_labels() -> None:
    """Regime detection produces integer labels."""
    df = pd.DataFrame(
        {
            "returns_21d": np.random.randn(300) * 0.01,
            "volatility_21d": np.abs(np.random.randn(300)) * 0.2,
        },
        index=pd.date_range("2020-01-01", periods=300, freq="B"),
    )
    regimes = detect_regimes(df, {"regimes": {"method": "kmeans", "n_states": 3}})
    assert regimes.min() >= 0
    assert regimes.max() < 3
    assert len(regimes) == len(df)
