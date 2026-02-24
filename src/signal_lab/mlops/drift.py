"""Drift detection for feature distributions and signal stability."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ks_statistic(baseline: np.ndarray, current: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic between two samples."""
    from scipy import stats

    return float(stats.ks_2samp(baseline, current).statistic)


def compute_drift_summary(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute drift metrics for numeric features.
    Returns dict suitable for JSON serialization.
    """
    cols = feature_cols or list(baseline.select_dtypes(include=[np.number]).columns)
    cols = [c for c in cols if c in current.columns]
    summary = {"features": {}, "n_baseline": len(baseline), "n_current": len(current)}
    for col in cols:
        b = baseline[col].dropna().values
        c = current[col].dropna().values
        if len(b) < 10 or len(c) < 10:
            continue
        summary["features"][col] = {
            "ks_statistic": round(_ks_statistic(b, c), 4),
            "baseline_mean": float(np.mean(b)),
            "current_mean": float(np.mean(c)),
            "mean_shift_pct": float((np.mean(c) - np.mean(b)) / (np.mean(b) + 1e-10) * 100),
        }
    return summary


def save_daily_summary(summary: dict[str, Any], path: Path | str) -> None:
    """Save drift summary as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved drift summary to %s", p)
