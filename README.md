# Commodity Signal Lab

Production-grade commodity trading signal & risk modeling framework. Built for data scientists and quant researchers.

## Features

- **Data ingestion**: Futures/spot via yfinance, macro via FRED
- **Feature engineering**: Technicals (returns, momentum, RSI, MACD, volatility, z-scores), macro alignment, HMM/k-means regime detection
- **Signal modeling**: Direction classifier (LightGBM/XGBoost) and return regressor with MLflow tracking
- **Backtesting**: Walk-forward engine, position sizing, transaction costs, VaR/CVaR, drawdown
- **Dashboard**: Streamlit app with signals, regime state, equity curve, drift detection

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Mac + LightGBM: if you see libomp errors, run: brew install libomp
# Or the framework auto-falls back to sklearn GradientBoosting

# Optional: FRED API key for macro data
cp .env.example .env
# Edit .env and add FRED_API_KEY=

# Full reproducible flow
python scripts/fetch_data.py --all
python scripts/build_features.py --commodity CL_F
python scripts/train.py --commodity CL_F --model both
python scripts/backtest.py --commodity CL_F --model classifier
python scripts/drift_report.py --commodity CL_F
python scripts/run_dashboard.py
```

## Reproducible Flow

| Step | Command | Output |
|------|---------|--------|
| 1. Fetch | `fetch_data.py --all` | `data/raw/*.parquet` |
| 2. Features | `build_features.py --commodity CL_F` | `data/processed/CL_F_features.parquet` |
| 3. Train | `train.py --commodity CL_F` | `models/CL_F_*.joblib`, MLflow runs |
| 4. Backtest | `backtest.py --commodity CL_F` | `reports/backtest_report.html` |
| 5. Drift | `drift_report.py --commodity CL_F` | `reports/drift_summary.json` |
| 6. Dashboard | `run_dashboard.py` | http://localhost:8501 |

## Project Structure

```
commodity-signal-lab/
├── configs/          # YAML configs (universe, features, model, backtest)
├── data/raw/         # Raw parquet (prices, macro)
├── data/processed/   # Feature parquet
├── src/signal_lab/
│   ├── ingestion/    # prices.py, macro_fred.py
│   ├── features/     # technicals.py, macro.py, regimes.py
│   ├── models/       # classifier.py, regressor.py
│   ├── backtest/     # engine.py, costs.py, risk.py, metrics.py
│   ├── mlops/        # tracking.py, drift.py
│   └── dashboard/    # app.py
├── scripts/         # CLI entrypoints
└── tests/
```

## Configuration

- **universe.yaml**: Commodity tickers, date range
- **features.yaml**: Technical horizons, RSI/MACD params, regime method
- **model.yaml**: LightGBM params, training split
- **backtest.yaml**: Position sizing, thresholds, costs

## Tests

```bash
pytest tests/ -v
```

## Analysis

See [ANALYSIS.md](ANALYSIS.md) for:
- Data source verification (real yfinance + FRED data)
- End-to-end testing summary
- Backtest results and drift analysis
- Fixes applied during verification
