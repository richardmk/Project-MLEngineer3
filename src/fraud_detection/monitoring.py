"""Production monitoring: scoring, periods, Evidently reports and per-period drift/performance."""

import gzip
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.core.report import Snapshot
from evidently.presets import ClassificationPreset, DataDriftPreset
from sklearn.base import BaseEstimator

from fraud_detection.config import PERIOD_STEPS
from fraud_detection.evaluate import compute_metrics
from fraud_detection.features import MODEL_FEATURES, add_features

# Low-cardinality features are tested as categorical (Jensen-Shannon) rather than
# numeric (Wasserstein), which suits binary flags and the 24-level hour.
CATEGORICAL_FEATURES: list[str] = [
    "is_orig_balance_zero",
    "is_full_drain",
    "is_dest_balance_zero",
    "is_transfer",
    "hour_of_day",
]
NUMERICAL_FEATURES: list[str] = [f for f in MODEL_FEATURES if f not in CATEGORICAL_FEATURES]
TARGET, PREDICTION, LABEL = "isFraud", "prediction", "prediction_label"
REPORT_COLUMNS: list[str] = MODEL_FEATURES + [PREDICTION, LABEL, TARGET]
RAW_COLUMNS: list[str] = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "is_merchant_dest",
    "isFlaggedFraud",
    "isFraud",
]


def score(raw: pd.DataFrame, model: BaseEstimator, threshold: float) -> pd.DataFrame:
    """Add features, fraud score, predicted label and monitoring period (simulated day).

    Args:
        raw: Transactions with the raw PaySim columns (no account names).
        model: Fitted champion model.
        threshold: Decision threshold.

    Returns:
        Feature frame with `prediction`, `prediction_label` and `period` columns.
    """
    df = add_features(raw)
    df[PREDICTION] = model.predict_proba(df[MODEL_FEATURES])[:, 1]
    return add_labels(df, threshold)


def add_labels(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Derive `prediction_label` and `period` from an already scored frame."""
    df = df.copy()
    df[LABEL] = (df[PREDICTION] >= threshold).astype(np.int8)
    df["period"] = ((df["step"] - 1) // PERIOD_STEPS + 1).astype(np.int16)
    return df


def period_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per monitoring period with its step range, volume and fraud rate."""
    return (
        df.groupby("period")
        .agg(
            step_min=("step", "min"),
            step_max=("step", "max"),
            transacciones=(TARGET, "size"),
            fraudes=(TARGET, "sum"),
            tasa_fraude=(TARGET, "mean"),
        )
        .reset_index()
    )


def _data_definition() -> DataDefinition:
    return DataDefinition(
        numerical_columns=NUMERICAL_FEATURES + [PREDICTION],
        categorical_columns=CATEGORICAL_FEATURES + [TARGET, LABEL],
        classification=[
            BinaryClassification(
                target=TARGET, prediction_labels=LABEL, prediction_probas=PREDICTION
            )
        ],
    )


def run_report(reference: pd.DataFrame, current: pd.DataFrame) -> Snapshot:
    """Evidently report for one period: feature drift, prediction drift, target drift, performance.

    Args:
        reference: Scored reference data (step <= 400).
        current: Scored production data for the period.

    Returns:
        The Evidently snapshot (renderable as HTML). The classification block is skipped
        when the period has a single class (e.g. the last simulated day only has frauds),
        since per-class quality metrics are undefined there.
    """
    definition = _data_definition()
    presets = [
        DataDriftPreset(columns=MODEL_FEATURES),
        DataDriftPreset(columns=[PREDICTION]),
        DataDriftPreset(columns=[TARGET]),
    ]
    if current[TARGET].nunique() == 2:
        presets.append(ClassificationPreset())
    report = Report(presets)
    return report.run(
        Dataset.from_pandas(current[REPORT_COLUMNS], data_definition=definition),
        Dataset.from_pandas(reference[REPORT_COLUMNS], data_definition=definition),
    )


def drift_summary(snapshot: Snapshot) -> dict[str, Any]:
    """Extract drift scores from a snapshot.

    Every method Evidently picks here is a distance (Wasserstein normed / Jensen-Shannon),
    so a column drifts when its score is >= its threshold.

    Args:
        snapshot: Output of `run_report`.

    Returns:
        Dict with feature drift share/count, per-column drift scores and drift flags.
    """
    out: dict[str, Any] = {}
    for metric in snapshot.dict()["metrics"]:
        name, config, value = metric["metric_name"], metric["config"], metric["value"]
        if name.startswith("ValueDrift"):
            col = config["column"]
            out[f"drift_score__{col}"] = float(value)
            out[f"drifted__{col}"] = bool(value >= config["threshold"])
        elif name.startswith("DriftedColumnsCount") and len(config["columns"]) > 1:
            out["features_drift_share"] = float(value["share"])
            out["features_drifted"] = int(value["count"])
    return out


def concept_summary(df: pd.DataFrame) -> dict[str, float]:
    """Relation features -> target for one period, centered on the dominant fraud pattern.

    Args:
        df: Scored transactions.

    Returns:
        Fraud rate, P(fraud | full drain), share of frauds that drain the account, and the
        precision/recall of the `is_full_drain` rule.
    """
    drain = df["is_full_drain"] == 1
    fraud = df[TARGET] == 1
    return {
        "fraud_rate": float(fraud.mean()),
        "p_fraud_given_drain": float(fraud[drain].mean()) if drain.any() else np.nan,
        "share_frauds_drain": float(drain[fraud].mean()) if fraud.any() else np.nan,
        "rule_recall": float((drain & fraud).sum() / fraud.sum()) if fraud.any() else np.nan,
        "rule_precision": float((drain & fraud).sum() / drain.sum()) if drain.any() else np.nan,
    }


def period_summary(
    reference: pd.DataFrame, current: pd.DataFrame, threshold: float, snapshot: Snapshot
) -> dict[str, Any]:
    """All monitoring indicators for one period (drift, concept, prediction, performance)."""
    perf = compute_metrics(current[TARGET].to_numpy(), current[PREDICTION].to_numpy(), threshold)
    return {
        "transacciones": len(current),
        "fraudes": int(current[TARGET].sum()),
        "predicted_positive_rate": float(current[LABEL].mean()),
        "mean_score": float(current[PREDICTION].mean()),
        **concept_summary(current),
        **drift_summary(snapshot),
        **perf,
    }


def save_snapshot(snapshot: Snapshot, path: Path) -> None:
    """Save the full snapshot (widgets included) gzip-compressed (~200 KB vs ~4.6 MB HTML)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(snapshot.dumps())


def load_snapshot(path: Path) -> Snapshot:
    """Load a snapshot saved with `save_snapshot`."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return Snapshot.loads(fh.read())
