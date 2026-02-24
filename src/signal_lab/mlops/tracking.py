"""MLflow experiment tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from signal_lab.utils.config import env

logger = logging.getLogger(__name__)


def log_experiment(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    model: Any = None,
    artifact_path: str = "model",
) -> str | None:
    """
    Log experiment to MLflow. Uses MLFLOW_TRACKING_URI from env or ./mlruns.
    """
    try:
        import mlflow

        uri = env("MLFLOW_TRACKING_URI") or str(Path.cwd() / "mlruns")
        mlflow.set_tracking_uri(uri)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if model is not None:
                mlflow.sklearn.log_model(model, artifact_path)
            return mlflow.active_run().info.run_id
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)
        return None
