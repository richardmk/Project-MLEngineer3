"""MLflow tracking setup against a remote DagsHub server."""

import os

import mlflow
from dotenv import load_dotenv

from fraud_detection.config import PROJECT_ROOT

_REQUIRED_VARS = ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")


def setup_mlflow(experiment_name: str) -> str:
    """Load credentials from `.env` and point MLflow at the remote tracking server.

    MLflow reads `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` from the
    environment for HTTP basic auth, so no interactive login is needed.

    Args:
        experiment_name: Experiment to create (if missing) and activate.

    Returns:
        The tracking URI in use.

    Raises:
        OSError: If any required environment variable is missing.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    missing = [var for var in _REQUIRED_VARS if not os.getenv(var)]
    if missing:
        raise OSError(f"Missing environment variables (see .env.example): {', '.join(missing)}")

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri
