#!/usr/bin/env python3
"""Train classifier and/or regressor; persist models and log to MLflow."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_lab.mlops.tracking import log_experiment
from signal_lab.models.classifier import DirectionClassifier
from signal_lab.models.regressor import ReturnRegressor
from signal_lab.utils.config import get_paths, load_yaml
from signal_lab.utils.io import load_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_RETURN = "target_return"
TARGET_DIRECTION = "target_direction"
EXCLUDE_COLS = {TARGET_RETURN, TARGET_DIRECTION}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train signal models")
    parser.add_argument("--commodity", default="CL_F", help="Commodity ticker")
    parser.add_argument("--model", choices=["classifier", "regressor", "both"], default="both")
    parser.add_argument("--config", default="model.yaml")
    parser.add_argument("--output", help="Model output directory")
    args = parser.parse_args()

    paths = get_paths()
    cfg = load_yaml(args.config)
    feat_path = paths.processed / f"{args.commodity}_features.parquet"
    if not feat_path.exists():
        logger.error("Features not found: %s. Run build_features first.", feat_path)
        return 1

    df = load_parquet(feat_path)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and df[c].dtype in ["float64", "int64"]]
    X = df[feature_cols].dropna(how="all").ffill().bfill().fillna(0)
    y_ret = df.loc[X.index, TARGET_RETURN]
    y_dir = df.loc[X.index, TARGET_DIRECTION]
    valid = y_ret.notna() & y_dir.notna()
    X, y_ret, y_dir = X[valid], y_ret[valid], y_dir[valid]

    train_size = int(len(X) * (1 - cfg.get("training", {}).get("test_size", 0.2)))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_ret_train, y_ret_test = y_ret.iloc[:train_size], y_ret.iloc[train_size:]
    y_dir_train, y_dir_test = y_dir.iloc[:train_size], y_dir.iloc[train_size:]

    out_dir = Path(args.output) if args.output else Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = None
    if args.model in ("classifier", "both"):
        clf = DirectionClassifier(cfg)
        clf.fit(X_train, y_dir_train, feature_names=feature_cols)
        proba = clf.predict_proba(X_test)
        from sklearn.metrics import accuracy_score, roc_auc_score

        acc = accuracy_score(y_dir_test, (proba >= 0.5).astype(int))
        auc = roc_auc_score(y_dir_test, proba) if len(set(y_dir_test)) > 1 else 0.0
        logger.info("Classifier: accuracy=%.4f, AUC=%.4f", acc, auc)
        clf.save(out_dir / f"{args.commodity}_classifier.joblib")
        imp = clf.feature_importance()
        if imp:
            top = sorted(imp.items(), key=lambda x: -x[1])[:5]
            logger.info("Top features: %s", top)
        run_id = log_experiment(
            f"classifier_{args.commodity}",
            {"commodity": args.commodity, "model": "classifier"},
            {"accuracy": acc, "roc_auc": auc},
            model=clf.model,
        )

    if args.model in ("regressor", "both"):
        reg = ReturnRegressor(cfg)
        reg.fit(X_train, y_ret_train, feature_names=feature_cols)
        pred = reg.predict(X_test)
        from sklearn.metrics import mean_squared_error, r2_score

        rmse = mean_squared_error(y_ret_test, pred) ** 0.5
        r2 = r2_score(y_ret_test, pred)
        logger.info("Regressor: RMSE=%.6f, R2=%.4f", rmse, r2)
        reg.save(out_dir / f"{args.commodity}_regressor.joblib")
        run_id = log_experiment(
            f"regressor_{args.commodity}",
            {"commodity": args.commodity, "model": "regressor"},
            {"rmse": rmse, "r2": r2},
            model=reg.model,
        )

    logger.info("Models saved to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
