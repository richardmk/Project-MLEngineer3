"""Log a dummy run to the remote MLflow server to validate the DagsHub connection.

Usage:
    uv run python scripts/check_mlflow.py
"""

import json
import tempfile
from pathlib import Path

import mlflow

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
    print(f"OK - run {run.info.run_id} logged to {uri}")


if __name__ == "__main__":
    main()
