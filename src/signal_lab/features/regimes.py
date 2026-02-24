"""Regime detection via HMM or k-means."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _hmm_regimes(
    X: np.ndarray,
    n_states: int = 3,
    min_samples: int = 252,
) -> np.ndarray:
    """Hidden Markov Model regime labels."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise ImportError("hmmlearn required for HMM regimes; install with pip install hmmlearn")

    if len(X) < min_samples:
        return np.zeros(len(X), dtype=int)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
    model.fit(X[:min_samples])
    labels, _ = model.predict(X)
    return labels


def _kmeans_regimes(
    X: np.ndarray,
    n_states: int = 3,
    min_samples: int = 252,
) -> np.ndarray:
    """K-means regime labels on volatility/returns."""
    from sklearn.cluster import KMeans

    if len(X) < min_samples:
        return np.zeros(len(X), dtype=int)
    km = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    km.fit(X[:min_samples])
    return km.predict(X)


def detect_regimes(
    features_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.Series:
    """
    Detect market regimes from volatility/returns. No look-ahead: fit on expanding window.

    Args:
        features_df: DataFrame with returns and volatility columns.
        config: Optional config with regimes section.

    Returns:
        Series of regime labels (0, 1, 2, ...) with same index.
    """
    cfg = config or {}
    rc = cfg.get("regimes", {})
    method = rc.get("method", "hmm")
    n_states = rc.get("n_states", 3)
    feat_names = rc.get("features_for_regime", ["returns_21d", "volatility_21d"])
    min_samples = rc.get("min_samples", 252)

    # Resolve column names (allow fuzzy match)
    avail = set(features_df.columns)
    cols = [c for c in feat_names if c in avail]
    if not cols:
        # Fallback: use any returns/vol columns
        cols = [c for c in features_df.columns if "return" in c.lower() or "vol" in c.lower()]
    if len(cols) < 2:
        cols = list(features_df.columns[:2]) if len(features_df.columns) >= 2 else list(features_df.columns)

    X = features_df[cols].fillna(0).values

    dispatch = {"hmm": _hmm_regimes, "kmeans": _kmeans_regimes}
    fn = dispatch.get(method, _kmeans_regimes)

    try:
        labels = fn(X, n_states=n_states, min_samples=min_samples)
    except Exception as e:
        logger.warning("Regime detection failed (%s), using zeros: %s", method, e)
        labels = np.zeros(len(X), dtype=int)

    return pd.Series(labels, index=features_df.index, name="regime")
