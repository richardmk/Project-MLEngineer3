"""Temporal splits: reference vs production, and train vs test inside reference."""

import pandas as pd

from fraud_detection.config import MODEL_TYPES, PERIOD_STEPS, REFERENCE_MAX_STEP, TRAIN_MAX_STEP


def filter_model_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the transaction types the model scores (fraud only occurs there).

    Args:
        df: Transactions with a `type` column.

    Returns:
        Rows whose `type` is in `config.MODEL_TYPES`.
    """
    return df[df["type"].isin(MODEL_TYPES)].reset_index(drop=True)


def split_reference_production(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by the course rule: step <= 400 is reference, step >= 401 is production.

    Args:
        df: Transactions with a `step` column.

    Returns:
        (reference, production) DataFrames.
    """
    is_ref = df["step"] <= REFERENCE_MAX_STEP
    return df[is_ref].reset_index(drop=True), df[~is_ref].reset_index(drop=True)


def split_train_test(
    reference: pd.DataFrame, train_max_step: int = TRAIN_MAX_STEP
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the reference window in time: train up to `train_max_step`, test afterwards.

    Args:
        reference: Reference-window transactions.
        train_max_step: Last step included in train.

    Returns:
        (train, test) DataFrames.
    """
    is_train = reference["step"] <= train_max_step
    return reference[is_train].reset_index(drop=True), reference[~is_train].reset_index(drop=True)


def assign_period(steps: pd.Series, period_steps: int = PERIOD_STEPS) -> pd.Series:
    """Map production steps to monitoring periods of `period_steps` hours.

    Periods are aligned with simulated days (step 1 = hour 0 of day 1), so the
    first production period may be partial.

    Args:
        steps: Step values.
        period_steps: Period length in steps.

    Returns:
        Integer simulated-day index (1-based) for each step.
    """
    return ((steps - 1) // period_steps + 1).astype("int16")
