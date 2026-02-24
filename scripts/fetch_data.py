#!/usr/bin/env python3
"""Fetch commodity prices and macro data; store as parquet."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_lab.ingestion.macro_fred import fetch_and_save_macro
from signal_lab.ingestion.prices import fetch_and_save
from signal_lab.utils.config import get_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch commodity and macro data")
    parser.add_argument("--prices", action="store_true", help="Fetch commodity prices")
    parser.add_argument("--macro", action="store_true", help="Fetch FRED macro data")
    parser.add_argument("--all", action="store_true", help="Fetch both (default)")
    parser.add_argument("--config", default="universe.yaml", help="Universe config path")
    parser.add_argument("--output", help="Override output directory")
    args = parser.parse_args()

    do_all = args.all or (not args.prices and not args.macro)

    paths = get_paths()
    out = Path(args.output) if args.output else paths.raw
    out.mkdir(parents=True, exist_ok=True)

    if do_all or args.prices:
        logger.info("Fetching commodity prices...")
        fetch_and_save(config_path=args.config, output_dir=str(out))

    if do_all or args.macro:
        logger.info("Fetching FRED macro data...")
        fetch_and_save_macro(output_dir=str(out))

    logger.info("Done. Data in %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
