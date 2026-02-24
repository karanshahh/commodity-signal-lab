#!/usr/bin/env python3
"""Run backtest and produce HTML/MD report with plots and metrics."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from signal_lab.backtest.engine import BacktestEngine
from signal_lab.models.classifier import DirectionClassifier
from signal_lab.models.regressor import ReturnRegressor
from signal_lab.utils.config import get_paths, load_yaml
from signal_lab.utils.io import load_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _drawdown_chart(equity: pd.Series) -> go.Figure:
    """Drawdown from peak."""
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax.replace(0, 1)
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values * 100, fill="tozeroy", name="Drawdown %"))
    fig.update_layout(title="Drawdown", yaxis_title="Drawdown %", template="plotly_white")
    return fig


def _equity_chart(equity: pd.Series) -> go.Figure:
    """Equity curve."""
    fig = go.Figure(go.Scatter(x=equity.index, y=equity.values, name="Equity"))
    fig.update_layout(title="Equity Curve", yaxis_title="Portfolio Value", template="plotly_white")
    return fig


def _rolling_sharpe_chart(returns: pd.Series, window: int = 63) -> go.Figure:
    """Rolling Sharpe."""
    from signal_lab.backtest.risk import rolling_sharpe

    rs = rolling_sharpe(returns, window)
    fig = go.Figure(go.Scatter(x=rs.index, y=rs.values, name=f"Rolling Sharpe ({window}d)"))
    fig.update_layout(title="Rolling Sharpe Ratio", template="plotly_white")
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run backtest and generate report")
    parser.add_argument("--commodity", default="CL_F", help="Commodity ticker")
    parser.add_argument("--model", choices=["classifier", "regressor"], default="classifier")
    parser.add_argument("--config", default="backtest.yaml")
    parser.add_argument("--output", default="reports/backtest_report.html")
    args = parser.parse_args()

    paths = get_paths()
    cfg = load_yaml(args.config)
    feat_path = paths.processed / f"{args.commodity}_features.parquet"
    if not feat_path.exists():
        logger.error("Features not found: %s", feat_path)
        return 1

    df = load_parquet(feat_path)
    prices = load_parquet(paths.raw / f"{args.commodity}_prices.parquet")
    prices = prices["close"] if "close" in prices.columns else prices["Close"]

    all_feature_cols = [c for c in df.columns if c not in {"target_return", "target_direction", "close"}]
    all_feature_cols = [c for c in all_feature_cols if df[c].dtype in ["float64", "int64"]]
    X_full = df[all_feature_cols].dropna(how="all").ffill().bfill().fillna(0)

    model_path = Path("models") / f"{args.commodity}_{args.model}.joblib"
    if not model_path.exists():
        logger.error("Model not found: %s. Run train first.", model_path)
        return 1

    if args.model == "classifier":
        clf = DirectionClassifier.load(model_path)
        model_feats = getattr(clf, "feature_names_", None) or all_feature_cols
        X = X_full.reindex(columns=model_feats, fill_value=0)[model_feats]
        signal = pd.Series(clf.predict_proba(X), index=X.index)
    else:
        reg = ReturnRegressor.load(model_path)
        model_feats = getattr(reg, "feature_names_", None) or all_feature_cols
        X = X_full.reindex(columns=model_feats, fill_value=0)[model_feats]
        pred = reg.predict(X)
        signal = pd.Series(pred, index=X.index)
        signal = (signal - signal.mean()) / (signal.std() + 1e-10) * 0.5 + 0.5

    vol = df["volatility"].reindex(X.index) if "volatility" in df.columns else None
    engine = BacktestEngine(cfg)
    result = engine.run(prices, signal, volatility=vol)

    equity = result["equity"]
    returns = result["returns"]
    metrics = result["metrics"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig_eq = _equity_chart(equity)
    fig_dd = _drawdown_chart(equity)
    fig_rs = _rolling_sharpe_chart(returns)

    metrics_html = "<table><tr><th>Metric</th><th>Value</th></tr>"
    for k, v in metrics.items():
        if isinstance(v, float):
            metrics_html += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        else:
            metrics_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
    metrics_html += "</table>"

    html = f"""
<!DOCTYPE html>
<html>
<head><title>Backtest Report - {args.commodity}</title></head>
<body>
<h1>Backtest Report: {args.commodity}</h1>
<h2>Metrics</h2>
{metrics_html}
<h2>Equity Curve</h2>
{fig_eq.to_html(full_html=False, include_plotlyjs='cdn')}
<h2>Drawdown</h2>
{fig_dd.to_html(full_html=False, include_plotlyjs='cdn')}
<h2>Rolling Sharpe</h2>
{fig_rs.to_html(full_html=False, include_plotlyjs='cdn')}
</body>
</html>
"""
    out_path = Path(args.output)
    out_path.write_text(html)
    logger.info("Report saved to %s", out_path)

    md_path = out_path.with_suffix(".md")
    md_lines = [f"# Backtest Report: {args.commodity}\n", "## Metrics\n", "| Metric | Value |", "|--------|-------|"]
    for k, v in metrics.items():
        md_lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    md_path.write_text("\n".join(md_lines))
    logger.info("Markdown report saved to %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
