#!/usr/bin/env python3
"""Build features from raw data and save to processed."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_lab.features.macro import build_macro_features
from signal_lab.features.regimes import detect_regimes
from signal_lab.features.technicals import build_technical_features
from signal_lab.utils.config import get_paths, load_yaml
from signal_lab.utils.io import load_parquet, naming_convention, save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build features from raw data")
    parser.add_argument("--commodity", default="CL_F", help="Commodity ticker (e.g. CL_F)")
    parser.add_argument("--config", default="features.yaml", help="Features config")
    parser.add_argument("--universe", default="universe.yaml", help="Universe config")
    args = parser.parse_args()

    paths = get_paths()
    cfg = load_yaml(args.config)
    uni = load_yaml(args.universe)

    # Load price data
    price_file = paths.raw / naming_convention(args.commodity, "prices")
    if not price_file.exists():
        logger.error("Price file not found: %s. Run fetch_data first.", price_file)
        return 1

    prices = load_parquet(price_file)
    if "Open" in prices.columns:
        prices = prices.rename(columns={c: c.lower() for c in prices.columns})

    # Technicals
    tech = build_technical_features(prices, cfg)
    logger.info("Built %d technical features", tech.shape[1])

    # Macro (if available)
    macro_dfs = {}
    for f in paths.raw.glob("*_macro.parquet"):
        name = f.stem.replace("_macro", "")
        macro_dfs[name] = load_parquet(f)
    if macro_dfs:
        macro_feat = build_macro_features(macro_dfs, tech.index, cfg)
        tech = tech.join(macro_feat, how="left")
        logger.info("Added %d macro features", macro_feat.shape[1])

    # Regimes (use available returns/vol columns)
    regimes = detect_regimes(tech, cfg)
    tech["regime"] = regimes

    # Forward returns for target (next period)
    tech["target_return"] = prices["close"].pct_change().shift(-1)
    tech["target_direction"] = (tech["target_return"] > 0).astype(int)

    out_path = paths.processed / naming_convention(args.commodity, "features")
    save_parquet(tech, out_path)
    logger.info("Saved features to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
