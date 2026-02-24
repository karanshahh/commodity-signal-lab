"""Data ingestion modules."""

from .macro_fred import fetch_fred_series
from .prices import fetch_commodity_prices

__all__ = ["fetch_commodity_prices", "fetch_fred_series"]
