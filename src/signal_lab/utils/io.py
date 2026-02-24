"""I/O utilities for parquet and structured data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def save_parquet(df: pd.DataFrame, path: Path | str, **kwargs: Any) -> None:
    """Save DataFrame to parquet with directory creation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=True, **kwargs)
    logger.info("Saved %s rows to %s", len(df), p)


def load_parquet(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    """Load parquet file to DataFrame."""
    df = pd.read_parquet(path, **kwargs)
    logger.info("Loaded %s rows from %s", len(df), path)
    return df


def naming_convention(asset: str, series_type: str, suffix: str = "") -> str:
    """Generate consistent parquet filename: {asset}_{series_type}{suffix}.parquet."""
    base = f"{asset}_{series_type}"
    return f"{base}{suffix}.parquet" if suffix else f"{base}.parquet"
