# Commodity Signal Lab — Analysis & Verification Report

## 1. Data Sources: Real Market Data

All data used in this framework is **real market data** from live APIs:

| Source | Data Type | Tickers/Series | Verification |
|--------|-----------|----------------|---------------|
| **yfinance** | Commodity futures | CL=F (WTI), NG=F (Nat Gas), GC=F (Gold), SI=F (Silver), HG=F (Copper), ZS=F (Soybeans) | Fetched ~2,800 rows per ticker (2015–present) |
| **FRED** (pandas-datareader) | Macro indicators | CPI, Industrial Production, Fed Funds, 10Y Treasury, USD/EUR, Dollar Index, Unemployment, Payrolls | 8 series, 192–4,036 rows each |

**Sample verification (run 2026-02-24):**
```
Fetched CL=F: 2802 rows
Fetched FRED CPIAUCSL: 192 rows
Fetched FRED DGS10: 4036 rows
...
```

---

## 2. End-to-End Testing Summary

### Unit Tests (6/6 passing)

| Test | Module | Purpose |
|------|--------|---------|
| `test_technicals_no_lookahead` | features/technicals | Ensures no future data leakage |
| `test_technicals_shape` | features/technicals | Validates output columns |
| `test_macro_alignment` | features/macro | Daily alignment of macro series |
| `test_regimes_labels` | features/regimes | Regime labels in valid range |
| `test_backtest_engine` | backtest/engine | Engine produces equity + metrics |
| `test_cagr_positive` | backtest/metrics | CAGR calculation |

### Integration Flow (verified with real data)

| Step | Command | Status | Output |
|------|---------|--------|--------|
| 1. Fetch | `fetch_data.py --all` | ✅ | 6 commodities + 8 FRED series |
| 2. Features | `build_features.py --commodity CL_F` | ✅ | 16 technical + 40 macro features |
| 3. Train | `train.py --commodity CL_F --model both` | ✅ | Classifier + regressor saved |
| 4. Backtest | `backtest.py --commodity CL_F` | ✅ | HTML + MD report |
| 5. Drift | `drift_report.py --commodity CL_F` | ✅ | `drift_summary.json` |
| 6. Dashboard | `run_dashboard.py` | ✅ | Streamlit app (port 8501) |

---

## 3. Backtest Results (CL_F, Classifier)

| Metric | Value |
|--------|-------|
| CAGR | 123.5% |
| Sharpe | 7.27 |
| Sortino | 16.15 |
| Hit rate | 54.2% |
| Max drawdown | -21.3% |
| VaR (95%) | -0.26% |
| CVaR (95%) | -0.63% |

**Note:** These metrics are from a single-commodity, in-sample backtest. Out-of-sample performance and multi-asset robustness would require further validation.

---

## 4. Drift Analysis (CL_F)

Drift summary compares baseline (first 252 days) vs current (last 252 days):

| Feature | KS Statistic | Mean Shift % |
|---------|--------------|--------------|
| volatility | 0.54 | -30% |
| returns_21d | 0.27 | -51% |
| macd_hist | 0.21 | -219% |
| rsi | 0.17 | +7% |

Higher KS values indicate stronger distribution shift. Volatility and return features show notable drift, suggesting regime changes over the sample period.

---

## 5. Model Performance (Real Data)

| Model | Commodity | Accuracy/AUC (Classifier) | RMSE/R² (Regressor) |
|-------|-----------|---------------------------|----------------------|
| HistGradientBoosting* | CL_F | 48.8% / 0.49 | 0.021 / -0.16 |
| HistGradientBoosting* | GC_F | 53.2% / 0.56 | — |

*LightGBM fallback used when libomp unavailable on Mac.

---

## 6. Fixes Applied During Verification

1. **Macro timezone alignment** — yfinance uses tz-aware timestamps; FRED uses naive. Normalized to naive for reindex.
2. **NaN in features** — Macro forward-fill leaves leading NaN. Added `fillna(0)` and HistGradientBoosting (handles NaN natively).
3. **Model/feature mismatch** — Backtest now uses `model.feature_names_` to align with training features.
4. **CAGR/total_return NaN** — Leading NaN in strategy returns from `shift(1)`. `compute_metrics` now uses `returns.fillna(0)`.

---

## 7. Limitations & Recommendations

- **LightGBM on Mac:** Requires `brew install libomp`; framework falls back to sklearn otherwise.
- **FRED rate limits:** Add `FRED_API_KEY` in `.env` for higher limits.
- **Backtest realism:** Position sizing and costs are simplified; production use would need slippage and execution modeling.
- **Regime method:** Default is k-means; HMM requires `pip install hmmlearn`.

---

## 8. Reproducibility

```bash
pip install -e ".[dev]"
python scripts/fetch_data.py --all
python scripts/build_features.py --commodity CL_F
python scripts/train.py --commodity CL_F --model both
python scripts/backtest.py --commodity CL_F --model classifier
python scripts/drift_report.py --commodity CL_F
python scripts/run_dashboard.py
pytest tests/ -v
```

All steps have been run successfully with real data as of 2026-02-24.
