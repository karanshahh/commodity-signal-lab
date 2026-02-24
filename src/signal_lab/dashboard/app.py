"""Streamlit dashboard: signals, regime, equity curve, drawdown, rolling Sharpe, drift."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)


def _load_features(commodity: str, paths: Any) -> pd.DataFrame | None:
    from signal_lab.utils.io import load_parquet

    p = paths.processed / f"{commodity}_features.parquet"
    if not p.exists():
        return None
    return load_parquet(p)


def _load_prices(commodity: str, paths: Any) -> pd.Series | None:
    from signal_lab.utils.io import load_parquet

    p = paths.raw / f"{commodity}_prices.parquet"
    if not p.exists():
        return None
    df = load_parquet(p)
    return df["close"] if "close" in df.columns else df["Close"]


def _load_model(commodity: str, model_type: str, proj_root: Path) -> Any:
    from signal_lab.models.classifier import DirectionClassifier
    from signal_lab.models.regressor import ReturnRegressor

    p = proj_root / "models" / f"{commodity}_{model_type}.joblib"
    if not p.exists():
        return None
    return DirectionClassifier.load(p) if model_type == "classifier" else ReturnRegressor.load(p)


def _drift_summary_path(proj_root: Path) -> Path:
    return proj_root / "reports" / "drift_summary.json"


def main() -> None:
    st.set_page_config(page_title="Commodity Signal Lab", layout="wide")
    st.title("Commodity Trading Signal Lab")

    proj_root = Path(__file__).resolve().parents[3]
    src_path = str(proj_root / "src")
    if src_path not in __import__("sys").path:
        __import__("sys").path.insert(0, src_path)

    from signal_lab.utils.config import get_paths, load_yaml

    paths = get_paths()
    # proj_root used for models/, reports/ (project root)
    cfg = load_yaml("backtest.yaml")

    commodities = [f.stem.replace("_features", "") for f in paths.processed.glob("*_features.parquet")]
    if not commodities:
        commodities = ["CL_F", "NG_F", "GC_F"]
    commodity = st.sidebar.selectbox("Commodity", commodities, index=0)

    df = _load_features(commodity, paths)
    if df is None:
        st.error(f"No features for {commodity}. Run fetch_data and build_features first.")
        return

    prices = _load_prices(commodity, paths)
    if prices is None:
        st.warning("No price data. Showing features only.")

    model_type = st.sidebar.radio("Model", ["classifier", "regressor"], index=0)
    model = _load_model(commodity, model_type, proj_root)

    col1, col2, col3 = st.columns(3)
    with col1:
        latest = df.index.max()
        st.metric("Latest Date", str(latest.date()) if hasattr(latest, "date") else str(latest))
    with col2:
        regime = df["regime"].iloc[-1] if "regime" in df.columns else 0
        st.metric("Current Regime", int(regime))
    with col3:
        if model is not None:
            feature_cols = [c for c in df.columns if c not in {"target_return", "target_direction"}]
            feature_cols = [c for c in feature_cols if df[c].dtype in ["float64", "int64"]]
            X = df[feature_cols].iloc[-1:].ffill().bfill()
            if model_type == "classifier":
                proba = model.predict_proba(X).item()
                st.metric("Signal (P(up))", f"{proba:.2%}")
            else:
                pred = model.predict(X).item()
                st.metric("Predicted Return", f"{pred:.4%}")

    st.subheader("Equity Curve & Drawdown")
    if model is not None and prices is not None:
        from signal_lab.backtest.engine import BacktestEngine

        feature_cols = [c for c in df.columns if c not in {"target_return", "target_direction"}]
        feature_cols = [c for c in feature_cols if df[c].dtype in ["float64", "int64"]]
        X = df[feature_cols].dropna(how="all").ffill().bfill()
        common = X.index.intersection(prices.index)
        X, prices_aligned = X.loc[common], prices.reindex(common).ffill().dropna()
        X = X.loc[prices_aligned.index]
        if model_type == "classifier":
            signal = pd.Series(model.predict_proba(X), index=X.index)
        else:
            pred = model.predict(X)
            signal = pd.Series(pred, index=X.index)
            signal = (signal - signal.mean()) / (signal.std() + 1e-10) * 0.5 + 0.5
        vol = df["volatility"].reindex(X.index) if "volatility" in df.columns else None
        engine = BacktestEngine(cfg)
        result = engine.run(prices_aligned, signal, volatility=vol)
        equity = result["equity"]
        returns = result["returns"]
        st.line_chart(equity)
        cummax = equity.cummax()
        dd = (equity - cummax) / cummax.replace(0, 1)
        st.line_chart(dd)
        from signal_lab.backtest.risk import rolling_sharpe

        rs = rolling_sharpe(returns, 63)
        st.line_chart(rs)
        st.json(result["metrics"])
    else:
        st.info("Load model and prices to see backtest.")

    st.subheader("Drift Detection")
    drift_path = _drift_summary_path(proj_root)
    if drift_path.exists():
        summary = json.loads(drift_path.read_text())
        st.json(summary)
    else:
        st.info("Run drift detection to generate daily summary. Use: python -c \"from signal_lab.mlops.drift import compute_drift_summary, save_daily_summary; ...\"")

    st.subheader("Regime Over Time")
    if "regime" in df.columns:
        st.line_chart(df["regime"])


if __name__ == "__main__":
    main()
