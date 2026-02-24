"""Transaction costs and slippage."""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_costs(
    returns: pd.Series,
    positions: pd.Series,
    config: dict | None = None,
) -> pd.Series:
    """
    Apply transaction costs (commission + slippage + spread) to returns.

    Costs are proportional to |position change| and applied in bps.
    """
    cfg = config or {}
    commission_bps = cfg.get("commission_bps", 5)
    slippage_bps = cfg.get("slippage_bps", 3)
    spread_bps = cfg.get("spread_bps", 2)
    total_bps = commission_bps + slippage_bps + spread_bps

    pos_chg = positions.diff().abs()
    cost = pos_chg * (total_bps / 10_000)
    return returns - cost
