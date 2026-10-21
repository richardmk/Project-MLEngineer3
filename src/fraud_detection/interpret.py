"""Interpretability helpers: champion loading, SHAP, LIME and local case selection."""

import json
from collections.abc import Callable
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.base import BaseEstimator

from fraud_detection.config import CHAMPION_META_PATH, CHAMPION_MODEL_PATH, SEED

# LIME cannot handle NaN; the ratio is NaN only when the origin balance is 0 and is
# otherwise >= 0, so -1 is an unambiguous sentinel that is mapped back before scoring.
_RATIO_COL = "amount_to_orig_balance_ratio"
_NAN_SENTINEL = -1.0


def load_champion() -> tuple[BaseEstimator, dict[str, Any]]:
    """Load the local copy of the champion model and its metadata (threshold, features).

    Returns:
        (fitted model, metadata dict written by notebook 02).
    """
    return joblib.load(CHAMPION_MODEL_PATH), json.loads(CHAMPION_META_PATH.read_text())


def explain_sample(df: pd.DataFrame, n_legit: int, seed: int = SEED) -> pd.DataFrame:
    """All frauds plus a random sample of legit rows, to keep explanations fast but rich.

    Args:
        df: Labelled transactions with an `isFraud` column.
        n_legit: Number of legit rows to sample.
        seed: Sampling seed.

    Returns:
        The sample, in original row order.
    """
    frauds = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0].sample(
        n=min(n_legit, (df["isFraud"] == 0).sum()), random_state=seed
    )
    return pd.concat([frauds, legit]).sort_index()


def shap_explanation(model: BaseEstimator, X: pd.DataFrame) -> shap.Explanation:
    """Exact TreeSHAP values (log-odds space) for a tree-based model.

    Args:
        model: Fitted tree model (XGBoost/LightGBM/RandomForest).
        X: Rows to explain.

    Returns:
        A shap.Explanation with one row per input row.
    """
    return shap.TreeExplainer(model)(X)


def shap_importance(explanation: shap.Explanation) -> pd.Series:
    """Global importance as mean |SHAP| per feature, sorted descending."""
    values = np.abs(explanation.values).mean(axis=0)
    return pd.Series(values, index=explanation.feature_names).sort_values(ascending=False)


def lime_explainer(background: pd.DataFrame) -> LimeTabularExplainer:
    """LIME explainer fitted on a NaN-free background sample (training data only).

    Args:
        background: Training feature sample.

    Returns:
        A LimeTabularExplainer for binary classification.
    """
    data = background.fillna({_RATIO_COL: _NAN_SENTINEL})
    return LimeTabularExplainer(
        data.to_numpy(dtype=float),
        feature_names=list(background.columns),
        class_names=["legítimo", "fraude"],
        mode="classification",
        random_state=SEED,
    )


def lime_predict_fn(model: BaseEstimator, dtypes: pd.Series) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap the model so LIME's numpy perturbations are scored like real rows.

    Args:
        model: Fitted model.
        dtypes: Feature dtypes of the training frame (restored before scoring).

    Returns:
        Function mapping an (n, d) array to (n, 2) class probabilities.
    """

    def predict(arr: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(arr, columns=dtypes.index)
        frame[_RATIO_COL] = frame[_RATIO_COL].where(frame[_RATIO_COL] >= 0)
        return model.predict_proba(frame.astype(dtypes.to_dict()))

    return predict


def lime_row(row: pd.Series) -> np.ndarray:
    """Encode one feature row the way the LIME background was encoded."""
    return row.fillna({_RATIO_COL: _NAN_SENTINEL}).to_numpy(dtype=float)


def select_local_cases(
    y: np.ndarray, scores: np.ndarray, threshold: float, rule: np.ndarray
) -> dict[str, int]:
    """Pick representative positions for local explanations.

    Cases: a typical detected fraud, a detected fraud the business rule misses, the
    most confident missed fraud, and the highest-scored legit row (a false positive
    if any exists, otherwise the closest call).

    Args:
        y: Labels.
        scores: Model scores.
        threshold: Decision threshold.
        rule: Binary rule flag (`is_full_drain`).

    Returns:
        Mapping case label -> positional index (missing cases are omitted).
    """
    flagged = scores >= threshold
    cases: dict[str, int] = {}
    tp = np.flatnonzero(flagged & (y == 1) & (rule == 1))
    if len(tp):
        cases["Fraude detectado (típico)"] = int(tp[np.argsort(scores[tp])[len(tp) // 2]])
    tp_no_rule = np.flatnonzero(flagged & (y == 1) & (rule == 0))
    if len(tp_no_rule):
        cases["Fraude detectado que la regla no ve"] = int(
            tp_no_rule[np.argmax(scores[tp_no_rule])]
        )
    fn = np.flatnonzero(~flagged & (y == 1))
    if len(fn):
        cases["Fraude no detectado (FN)"] = int(fn[np.argmax(scores[fn])])
    legit = np.flatnonzero(y == 0)
    top_legit = int(legit[np.argmax(scores[legit])])
    label = "Falso positivo" if flagged[top_legit] else "Legítimo más sospechoso (casi FP)"
    cases[label] = top_legit
    return cases


def partial_dependence_ice(
    model: BaseEstimator,
    X: pd.DataFrame,
    feature: str,
    grid_size: int = 30,
    n_rows: int = 1_000,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ICE curves (and hence PDP = their mean) for one feature.

    Unlike sklearn's helper this ignores NaN when building the grid, uses quantiles
    (1st-99th percentile) so heavy-tailed amounts get a meaningful grid, and keeps
    binary features as their two values.

    Args:
        model: Fitted classifier.
        X: Feature rows to perturb.
        feature: Feature to vary.
        grid_size: Number of quantile grid points for continuous features.
        n_rows: Rows sampled from `X` for the ICE curves.
        seed: Sampling seed.

    Returns:
        (grid, ice) where ice has shape (n_rows, len(grid)) with P(fraud).
    """
    values = X[feature].dropna()
    if values.nunique() <= 2:
        grid = np.sort(values.unique())
    else:
        grid = np.unique(np.quantile(values, np.linspace(0.01, 0.99, grid_size)))
    rows = X.sample(n=min(n_rows, len(X)), random_state=seed)
    ice = np.empty((len(rows), len(grid)))
    for j, value in enumerate(grid):
        perturbed = rows.copy()
        perturbed[feature] = np.asarray(value).astype(X[feature].dtype)
        ice[:, j] = model.predict_proba(perturbed)[:, 1]
    return grid, ice
