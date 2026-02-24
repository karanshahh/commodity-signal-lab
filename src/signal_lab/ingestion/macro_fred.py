"""Macro indicator ingestion from FRED API."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel

from signal_lab.utils.config import env, load_yaml
from signal_lab.utils.io import naming_convention, save_parquet

logger = logging.getLogger(__name__)

# FRED series IDs for common macro indicators
FRED_SERIES_MAP = {
    "CPIAUCSL": "cpi",
    "INDPRO": "industrial_production",
    "FEDFUNDS": "fed_funds_rate",
    "DGS10": "treasury_10y",
    "DEXUSEU": "usd_eur",
    "DTWEXBGS": "dollar_index",
    "UNRATE": "unemployment",
    "PAYEMS": "nonfarm_payrolls",
}


def fetch_fred_series(
    series_ids: list[str] | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    api_key: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch macro series from FRED via pandas-datareader or direct API.

    Args:
        series_ids: FRED series IDs. Defaults to FRED_SERIES_MAP keys.
        start: Start date.
        end: End date.
        api_key: FRED API key (or FRED_API_KEY env).

    Returns:
        Dict mapping series_id -> single-column DataFrame with DatetimeIndex.
    """
    api_key = api_key or env("FRED_API_KEY")
    if not api_key:
        logger.warning("No FRED_API_KEY; pandas-datareader may have rate limits.")

    series_ids = series_ids or list(FRED_SERIES_MAP.keys())
    start_str = pd.Timestamp(start or "2010-01-01").strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end or datetime.now()).strftime("%Y-%m-%d")

    result: dict[str, pd.DataFrame] = {}
    try:
        import pandas_datareader as pdr

        for sid in series_ids:
            try:
                df = pdr.get_data_fred(sid, start=start_str, end=end_str)
                if df is None or df.empty:
                    logger.warning("No data for FRED %s", sid)
                    continue
                df = df.rename(columns={sid: "value"})
                df = df.dropna()
                result[sid] = df
                logger.info("Fetched FRED %s: %d rows", sid, len(df))
            except Exception as e:
                logger.exception("Failed FRED %s: %s", sid, e)
    except ImportError:
        # Fallback: direct requests if pandas-datareader unavailable
        import requests

        base = "https://api.stlouisfed.org/fred/series/observations"
        for sid in series_ids:
            try:
                r = requests.get(
                    base,
                    params={
                        "series_id": sid,
                        "api_key": api_key or "",
                        "file_type": "json",
                        "observation_start": start_str,
                        "observation_end": end_str,
                    },
                )
                r.raise_for_status()
                data = r.json()
                obs = data.get("observations", [])
                if not obs:
                    continue
                df = pd.DataFrame(obs)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")[["value"]]
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna()
                result[sid] = df
                logger.info("Fetched FRED %s: %d rows", sid, len(df))
            except Exception as e:
                logger.exception("Failed FRED %s: %s", sid, e)
    return result


def fetch_and_save_macro(
    series_ids: list[str] | None = None,
    output_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch FRED series and save to parquet."""
    from pathlib import Path

    from signal_lab.utils.config import get_paths

    paths = get_paths()
    out = Path(output_dir) if output_dir else paths.raw
    out.mkdir(parents=True, exist_ok=True)

    data = fetch_fred_series(series_ids)
    for sid, df in data.items():
        name = FRED_SERIES_MAP.get(sid, sid.lower())
        fname = naming_convention(name, "macro")
        save_parquet(df, out / fname)
    return data
