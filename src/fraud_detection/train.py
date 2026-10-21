"""Model factories, temporal cross-validation and Optuna objectives with MLflow logging."""

import time
from collections.abc import Callable
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from fraud_detection.config import PERIOD_STEPS, SEED

MODEL_NAMES: list[str] = ["logreg", "random_forest", "xgboost", "lightgbm"]
N_JOBS: int = 4  # leave CPU headroom in WSL

Folds = list[tuple[np.ndarray, np.ndarray]]


# --- Temporal folds & sampling ----------------------------------------------
def temporal_day_folds(steps: np.ndarray, n_folds: int = 4) -> Folds:
    """Expanding-window folds on whole simulated days.

    Fold k validates on one full day and trains on every earlier day, so no fold
    ever sees the future. The last `n_folds` days of `steps` are the validation days.

    Args:
        steps: `step` value of each training row.
        n_folds: Number of folds (validation days).

    Returns:
        List of (train_idx, val_idx) positional index arrays.
    """
    days = (steps - 1) // PERIOD_STEPS + 1
    val_days = np.sort(np.unique(days))[-n_folds:]
    return [(np.flatnonzero(days < d), np.flatnonzero(days == d)) for d in val_days]


def undersample_negatives(y: np.ndarray, negative_fraction: float, seed: int = SEED) -> np.ndarray:
    """Return sorted indices keeping every positive and a random share of negatives.

    Applied only to training folds; validation/test keep the real class balance.

    Args:
        y: Binary labels.
        negative_fraction: Share of negatives to keep (1.0 = no undersampling).
        seed: Random seed.

    Returns:
        Sorted positional indices of the kept rows.
    """
    if negative_fraction >= 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    kept_neg = rng.choice(neg, size=int(len(neg) * negative_fraction), replace=False)
    return np.sort(np.concatenate([pos, kept_neg]))


# --- Model factories --------------------------------------------------------
def _signed_log1p(x: np.ndarray) -> np.ndarray:
    """Compress heavy-tailed amounts/balances while keeping the sign of balance errors."""
    return np.sign(x) * np.log1p(np.abs(x))


def build_model(name: str, params: dict[str, Any]) -> BaseEstimator:
    """Instantiate a model (or pipeline) by name with the given hyperparameters.

    Args:
        name: One of `MODEL_NAMES`.
        params: Hyperparameters for that model.

    Returns:
        An unfitted scikit-learn compatible estimator.
    """
    if name == "logreg":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("log", FunctionTransformer(_signed_log1p)),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=SEED, **params)),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
                ("model", RandomForestClassifier(n_jobs=N_JOBS, random_state=SEED, **params)),
            ]
        )
    if name == "xgboost":
        return XGBClassifier(
            tree_method="hist", eval_metric="aucpr", n_jobs=N_JOBS, random_state=SEED, **params
        )
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            subsample_freq=1,
            importance_type="gain",
            n_jobs=N_JOBS,
            random_state=SEED,
            verbose=-1,
            **params,
        )
    raise ValueError(f"Unknown model: {name}")


def suggest_params(trial: optuna.Trial, name: str, prefix: str = "") -> dict[str, Any]:
    """Search space per model.

    Ranges are centered on values that work for large tabular, highly imbalanced data:
    learning rate on a log scale, tree complexity capped to limit overfitting to the
    ~3.5K frauds, and class weighting searched because undersampling already shifts
    the class ratio. `prefix` keeps names unique in the AutoML (multi-model) study.

    Args:
        trial: Optuna trial.
        name: Model name.
        prefix: Prefix for parameter names.

    Returns:
        Hyperparameters accepted by `build_model`.
    """
    p = prefix
    if name == "logreg":
        return {
            "C": trial.suggest_float(f"{p}C", 1e-3, 1e2, log=True),
            "class_weight": trial.suggest_categorical(f"{p}class_weight", [None, "balanced"]),
        }
    if name == "random_forest":
        return {
            # Kept small on purpose: one RF CV costs ~10x a boosting CV on this data.
            "n_estimators": trial.suggest_int(f"{p}n_estimators", 50, 200, step=50),
            "max_depth": trial.suggest_int(f"{p}max_depth", 6, 16),
            "min_samples_leaf": trial.suggest_int(f"{p}min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_float(f"{p}max_features", 0.3, 1.0),
            "class_weight": trial.suggest_categorical(
                f"{p}class_weight", [None, "balanced_subsample"]
            ),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int(f"{p}n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float(f"{p}learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int(f"{p}max_depth", 3, 10),
            "min_child_weight": trial.suggest_float(f"{p}min_child_weight", 1, 20, log=True),
            "subsample": trial.suggest_float(f"{p}subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(f"{p}colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float(f"{p}reg_lambda", 1e-3, 10, log=True),
            "scale_pos_weight": trial.suggest_float(f"{p}scale_pos_weight", 1, 50, log=True),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int(f"{p}n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float(f"{p}learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int(f"{p}num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int(f"{p}min_child_samples", 5, 200, log=True),
            "subsample": trial.suggest_float(f"{p}subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(f"{p}colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float(f"{p}reg_lambda", 1e-3, 10, log=True),
            "scale_pos_weight": trial.suggest_float(f"{p}scale_pos_weight", 1, 50, log=True),
        }
    raise ValueError(f"Unknown model: {name}")


# --- Cross-validation -------------------------------------------------------
def fit_model(
    name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    negative_fraction: float,
    seed: int = SEED,
) -> BaseEstimator:
    """Fit a model after undersampling negatives of the given training data.

    Args:
        name: Model name.
        params: Hyperparameters.
        X: Training features.
        y: Training labels.
        negative_fraction: Share of negatives kept for fitting.
        seed: Sampling seed.

    Returns:
        The fitted estimator.
    """
    idx = undersample_negatives(y, negative_fraction, seed)
    model = build_model(name, params)
    model.fit(X.iloc[idx], y[idx])
    return model


def cross_validate(
    name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    folds: Folds,
    negative_fraction: float,
    on_fold: Callable[[int, float], None] | None = None,
) -> tuple[list[float], np.ndarray]:
    """Temporal CV returning per-fold PR-AUC and out-of-fold scores.

    Args:
        name: Model name.
        params: Hyperparameters.
        X: Training features.
        y: Training labels.
        folds: Output of `temporal_day_folds`.
        negative_fraction: Share of negatives kept in each training fold.
        on_fold: Optional callback(fold_index, running_mean_pr_auc), used for pruning.

    Returns:
        (fold PR-AUCs, OOF scores; NaN for rows never used for validation).
    """
    oof = np.full(len(y), np.nan)
    scores: list[float] = []
    for k, (tr, va) in enumerate(folds):
        model = fit_model(name, params, X.iloc[tr], y[tr], negative_fraction)
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        scores.append(average_precision_score(y[va], oof[va]))
        if on_fold is not None:
            on_fold(k, float(np.mean(scores)))
    return scores, oof


# --- Optuna -----------------------------------------------------------------
def make_objective(
    models: list[str],
    X: pd.DataFrame,
    y: np.ndarray,
    folds: Folds,
    negative_fraction: float,
    log_to_mlflow: bool = True,
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective maximizing mean temporal-CV PR-AUC.

    With several `models` the algorithm itself becomes a categorical hyperparameter
    (AutoML). Each trial is logged as a nested MLflow run, pruned ones included.

    Args:
        models: Candidate model names.
        X: Training features.
        y: Training labels.
        folds: Temporal folds.
        negative_fraction: Share of negatives kept in training folds.
        log_to_mlflow: Log each trial as a nested run of the active run.

    Returns:
        The objective function.
    """

    def objective(trial: optuna.Trial) -> float:
        name = models[0] if len(models) == 1 else trial.suggest_categorical("model", models)
        params = suggest_params(trial, name, prefix=f"{name}__" if len(models) > 1 else "")
        start = time.perf_counter()

        def on_fold(k: int, running: float) -> None:
            trial.report(running, step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        state, scores = "COMPLETE", []
        try:
            scores, _ = cross_validate(name, params, X, y, folds, negative_fraction, on_fold)
            return float(np.mean(scores))
        except optuna.TrialPruned:
            state = "PRUNED"
            raise
        except Exception:
            state = "FAIL"
            raise
        finally:
            if log_to_mlflow:
                with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
                    mlflow.set_tags({"model": name, "trial_state": state})
                    mlflow.log_params({"model": name, **params})
                    metrics = {f"fold{i}_pr_auc": s for i, s in enumerate(scores)}
                    if state == "COMPLETE":
                        metrics["cv_pr_auc"] = float(np.mean(scores))
                    metrics["duration_s"] = time.perf_counter() - start
                    mlflow.log_metrics(metrics)

    return objective


def run_study(
    study_name: str,
    models: list[str],
    X: pd.DataFrame,
    y: np.ndarray,
    folds: Folds,
    n_trials: int,
    sampler: str = "tpe",
    negative_fraction: float = 0.1,
    log_to_mlflow: bool = True,
) -> optuna.Study:
    """Run an Optuna study (TPE + median pruning, or random search) under an MLflow parent run.

    Args:
        study_name: Name of the study and of the MLflow parent run.
        models: Candidate model names (one for single-model tuning, several for AutoML).
        X: Training features.
        y: Training labels.
        folds: Temporal folds.
        n_trials: Trial budget.
        sampler: "tpe" (Bayesian, with MedianPruner) or "random" (baseline, no pruning).
        negative_fraction: Share of negatives kept in training folds.
        log_to_mlflow: Log the parent run and nested trial runs.

    Returns:
        The finished study.
    """
    if sampler == "tpe":
        opt_sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True, group=True)
        pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=1
        )
    elif sampler == "random":
        opt_sampler, pruner = optuna.samplers.RandomSampler(seed=SEED), optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    study = optuna.create_study(
        study_name=study_name, direction="maximize", sampler=opt_sampler, pruner=pruner
    )
    if len(models) == 1:
        study.set_user_attr("model", models[0])
    objective = make_objective(models, X, y, folds, negative_fraction, log_to_mlflow)
    if not log_to_mlflow:
        study.optimize(objective, n_trials=n_trials)
        return study

    with mlflow.start_run(run_name=study_name):
        mlflow.set_tags({"study": study_name, "sampler": sampler, "models": ",".join(models)})
        mlflow.log_params(
            {
                "n_trials": n_trials,
                "sampler": sampler,
                "negative_fraction": negative_fraction,
                "n_folds": len(folds),
            }
        )
        study.optimize(objective, n_trials=n_trials)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        mlflow.log_metrics(
            {
                "best_cv_pr_auc": study.best_value,
                "n_completed": len(completed),
                "n_pruned": len(study.trials) - len(completed),
            }
        )
        mlflow.log_params({f"best__{k}": v for k, v in study.best_params.items()})
    return study


def best_model_params(study: optuna.Study) -> tuple[str, dict[str, Any]]:
    """Recover (model name, build_model params) from a study's best trial.

    Args:
        study: A finished study (single-model or AutoML).

    Returns:
        (model_name, params) with any AutoML prefixes removed.
    """
    params = dict(study.best_params)
    name = params.pop("model", None) or study.user_attrs.get("model")
    if name is None:
        raise ValueError("Single-model studies must set study.user_attrs['model']")
    prefix = f"{name}__"
    return name, {k.removeprefix(prefix): v for k, v in params.items()}


def study_history(study: optuna.Study) -> pd.DataFrame:
    """Per-trial history with best-so-far value and cumulative compute time.

    Args:
        study: A finished study.

    Returns:
        DataFrame with trial, state, value, duration_s, best_so_far, cumulative_s.
    """
    rows = [
        {
            "trial": t.number,
            "state": t.state.name,
            "model": t.params.get("model", study.user_attrs.get("model")),
            "value": t.value,
            "duration_s": t.duration.total_seconds() if t.duration else np.nan,
        }
        for t in study.trials
    ]
    hist = pd.DataFrame(rows)
    hist["best_so_far"] = hist["value"].cummax().ffill()
    hist["cumulative_s"] = hist["duration_s"].cumsum()
    return hist


def get_feature_importance(model: BaseEstimator, features: list[str]) -> pd.Series:
    """Native importance per model: |coef| for logreg, impurity/gain for tree models.

    Args:
        model: Fitted estimator or pipeline.
        features: Feature names in training order.

    Returns:
        Importance per feature, normalized to sum 1.
    """
    est = model.named_steps["model"] if isinstance(model, Pipeline) else model
    raw = np.abs(est.coef_[0]) if hasattr(est, "coef_") else est.feature_importances_
    imp = pd.Series(raw, index=features, dtype=float)
    return imp / imp.sum()
