"""Log a dummy run to the remote MLflow server to validate the DagsHub connection.

Checks params, metrics, artifacts and model logging, since MLflow 3 clients use
newer model APIs that a remote server may not support.

Usage:
    uv run python scripts/check_mlflow.py
"""

import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression

from fraud_detection.config import MLFLOW_EXPERIMENT_SETUP, SEED
from fraud_detection.tracking import setup_mlflow


def main() -> None:
    uri = setup_mlflow(MLFLOW_EXPERIMENT_SETUP)
    with mlflow.start_run(run_name="connection-check") as run:
        mlflow.log_param("seed", SEED)
        mlflow.log_metric("dummy_metric", 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "check.json"
            path.write_text(json.dumps({"status": "ok"}))
            mlflow.log_artifact(str(path))
        print("OK - params, metrics and artifacts")

        X = np.random.default_rng(SEED).normal(size=(50, 2))
        model = LogisticRegression().fit(X, (X[:, 0] > 0).astype(int))
        mlflow.sklearn.log_model(model, name="model", input_example=X[:2])
        print("OK - model logged")
    print(f"OK - run {run.info.run_id} logged to {uri}")


if __name__ == "__main__":
    main()
