#!/usr/bin/env python3
"""Generate daily drift summary JSON for feature distributions."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_lab.mlops.drift import compute_drift_summary, save_daily_summary
from signal_lab.utils.config import get_paths
from signal_lab.utils.io import load_parquet

logging.basicConfig(level=logging.INFO)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute drift summary")
    parser.add_argument("--commodity", default="CL_F")
    parser.add_argument("--baseline-days", type=int, default=252, help="Days for baseline window")
    parser.add_argument("--output", default="reports/drift_summary.json")
    args = parser.parse_args()

    paths = get_paths()
    feat_path = paths.processed / f"{args.commodity}_features.parquet"
    if not feat_path.exists():
        print(f"Features not found: {feat_path}", file=sys.stderr)
        return 1

    df = load_parquet(feat_path)
    feature_cols = [c for c in df.columns if c not in {"target_return", "target_direction"}]
    feature_cols = [c for c in feature_cols if df[c].dtype in ["float64", "int64"]]
    df = df[feature_cols].dropna(how="all")

    split = min(args.baseline_days, len(df) // 2)
    baseline = df.iloc[:split]
    current = df.iloc[-split:]

    summary = compute_drift_summary(baseline, current, feature_cols)
    summary["commodity"] = args.commodity
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_daily_summary(summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
