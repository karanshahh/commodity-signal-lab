"""Futures/spot price ingestion via yfinance."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field

from signal_lab.utils.config import load_yaml
from signal_lab.utils.io import naming_convention, save_parquet

logger = logging.getLogger(__name__)


class PriceSchema(BaseModel):
    """Schema for price data validation."""

    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    ticker: str


def _validate_price_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate price DataFrame has required columns and types."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    df = df.rename(columns=str.lower).copy()
    df["ticker"] = ticker
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df


def fetch_commodity_prices(
    tickers: list[str],
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    config_path: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch commodity futures/spot prices from yfinance.

    Args:
        tickers: List of yfinance tickers (e.g. CL=F, NG=F).
        start: Start date (ISO or datetime).
        end: End date (ISO or datetime). None = today.
        config_path: Optional path to universe.yaml for date overrides.

    Returns:
        Dict mapping ticker -> OHLCV DataFrame with DatetimeIndex.
    """
    if config_path:
        cfg = load_yaml(config_path)
        dr = cfg.get("date_range", {})
        start = start or dr.get("start", "2015-01-01")
        end = end or dr.get("end")

    start_str = pd.Timestamp(start).strftime("%Y-%m-%d") if start else None
    end_str = pd.Timestamp(end).strftime("%Y-%m-%d") if end else None

    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start_str, end=end_str, auto_adjust=True)
            if df.empty:
                logger.warning("No data for %s", ticker)
                continue
            df = _validate_price_df(df, ticker)
            result[ticker] = df
            logger.info("Fetched %s: %d rows", ticker, len(df))
        except Exception as e:
            logger.exception("Failed to fetch %s: %s", ticker, e)
    return result


def fetch_and_save(
    tickers: list[str] | None = None,
    output_dir: str | Path | None = None,
    config_path: str = "universe.yaml",
) -> dict[str, pd.DataFrame]:
    """
    Fetch prices and save to parquet with naming convention.

    Uses universe.yaml for tickers and date range if tickers not provided.
    """
    from signal_lab.utils.config import get_paths

    cfg = load_yaml(config_path)
    tickers = tickers or [c["ticker"] for c in cfg.get("commodities", [])]
    paths = get_paths()
    out = Path(output_dir) if output_dir else paths.raw
    out.mkdir(parents=True, exist_ok=True)

    data = fetch_commodity_prices(tickers, config_path=config_path)
    for ticker, df in data.items():
        asset = ticker.replace("=", "_").replace(".", "_")
        fname = naming_convention(asset, "prices")
        save_parquet(df, out / fname)
    return data
