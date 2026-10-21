"""MLflow tracking setup against a remote DagsHub server, plus logging helpers."""

import os
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator

from fraud_detection.config import PROJECT_ROOT

_URI_VAR = "MLFLOW_TRACKING_URI"
_AUTH_VARS = ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")


def setup_mlflow(experiment_name: str) -> str:
    """Load credentials from `.env` and point MLflow at the tracking server.

    MLflow reads `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` from the
    environment for HTTP basic auth, so no interactive login is needed. A non-HTTP
    URI (e.g. `sqlite:///...`) is accepted without credentials for local dry runs.

    Args:
        experiment_name: Experiment to create (if missing) and activate.

    Returns:
        The tracking URI in use.

    Raises:
        OSError: If any required environment variable is missing.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    is_remote = os.getenv(_URI_VAR, "").startswith("http")
    required = (_URI_VAR, *_AUTH_VARS) if is_remote else (_URI_VAR,)
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        raise OSError(f"Missing environment variables (see .env.example): {', '.join(missing)}")

    tracking_uri = os.environ[_URI_VAR]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def log_model_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    figures: dict[str, plt.Figure] | None = None,
    tables: dict[str, pd.DataFrame] | None = None,
    model: BaseEstimator | None = None,
    input_example: pd.DataFrame | None = None,
) -> tuple[str, str | None]:
    """Log one model (or baseline) run with params, metrics and artifacts.

    Args:
        run_name: MLflow run name.
        params: Parameters to log.
        metrics: Metrics to log (NaN values are skipped).
        tags: Optional run tags.
        figures: Optional figures, logged as PNG under `plots/`.
        tables: Optional DataFrames, logged as CSV under `tables/`.
        model: Optional fitted estimator, logged with the sklearn flavor as `model`.
        input_example: Small feature sample used to infer the model signature.

    Returns:
        (run_id, model_uri or None).
    """
    model_uri = None
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags or {})
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)})
        for name, fig in (figures or {}).items():
            mlflow.log_figure(fig, f"plots/{name}.png")
        for name, table in (tables or {}).items():
            mlflow.log_table(table, f"tables/{name}.json")
        if model is not None:
            # cloudpickle + bundled package code: the logreg pipeline uses a custom
            # transformer function that the default skops format refuses to load.
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=input_example,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                code_paths=[str(PROJECT_ROOT / "src" / "fraud_detection")],
            )
            model_uri = info.model_uri
    return run.info.run_id, model_uri


def register_champion(model_uri: str, model_name: str, alias: str = "champion") -> str:
    """Register a logged model and point the production alias at the new version.

    Args:
        model_uri: URI returned when the model was logged.
        model_name: Registered model name.
        alias: Alias marking the production version (MLflow 3 replaces stages with aliases).

    Returns:
        The registered version number.
    """
    version = mlflow.register_model(model_uri, model_name).version
    MlflowClient().set_registered_model_alias(model_name, alias, version)
    return str(version)
