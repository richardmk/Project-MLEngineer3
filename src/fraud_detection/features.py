"""Stateless, row-wise feature engineering for PaySim transactions.

Every feature depends only on the current row, so nothing is fitted here and the
same function can be applied to train, test and production without leakage.
Scaling/imputation that does require fitting lives in the training pipeline.
"""

import numpy as np
import pandas as pd

from fraud_detection.data import TRANSACTION_TYPES

# Balances are recorded with cent precision; anything below this is float noise.
BALANCE_TOLERANCE: float = 0.01

RAW_NUMERIC_FEATURES: list[str] = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

DERIVED_FEATURES: list[str] = [
    "orig_balance_diff",
    "orig_balance_error",
    "dest_balance_diff",
    "dest_balance_error",
    "amount_to_orig_balance_ratio",
    "is_orig_balance_zero",
    "is_full_drain",
    "is_dest_balance_zero",
    "hour_of_day",
]

TYPE_DUMMIES: list[str] = [f"is_{t.lower()}" for t in TRANSACTION_TYPES]

# Inside the model scope (TRANSFER/CASH_OUT) `is_merchant_dest` is constant and
# `is_cash_out` is the complement of `is_transfer`, so both are left out.
MODEL_FEATURES: list[str] = RAW_NUMERIC_FEATURES + DERIVED_FEATURES + ["is_transfer"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with the derived fraud features appended.

    Features:
        - `orig_balance_diff`: oldbalanceOrg - newbalanceOrig (money that left the origin).
        - `orig_balance_error`: newbalanceOrig + amount - oldbalanceOrg (0 if consistent).
        - `dest_balance_diff`: newbalanceDest - oldbalanceDest (money that reached the dest).
        - `dest_balance_error`: oldbalanceDest + amount - newbalanceDest (0 if consistent).
        - `amount_to_orig_balance_ratio`: share of the origin balance moved (NaN if balance is 0).
        - `is_orig_balance_zero`: origin had no balance before the transaction.
        - `is_full_drain`: amount equals the whole origin balance (account emptied).
        - `is_dest_balance_zero`: non-merchant destination with 0 balance before and after
          despite a positive amount.
        - `hour_of_day`: (step - 1) % 24, a cyclic proxy that generalizes to new days,
          unlike the raw, non-stationary `step`.
        - `is_<type>`: one dummy per transaction type (fixed category list, nothing fitted).

    Args:
        df: Transactions with the raw PaySim columns plus `is_merchant_dest`.

    Returns:
        New DataFrame with the feature columns added (float32/int8 to save RAM).
    """
    out = df.copy()
    amount = out["amount"].to_numpy()
    old_orig = out["oldbalanceOrg"].to_numpy()
    new_orig = out["newbalanceOrig"].to_numpy()
    old_dest = out["oldbalanceDest"].to_numpy()
    new_dest = out["newbalanceDest"].to_numpy()

    # Differences are computed in float64 and only stored as float32.
    out["orig_balance_diff"] = (old_orig - new_orig).astype(np.float32)
    out["orig_balance_error"] = (new_orig + amount - old_orig).astype(np.float32)
    out["dest_balance_diff"] = (new_dest - old_dest).astype(np.float32)
    out["dest_balance_error"] = (old_dest + amount - new_dest).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(old_orig > 0, amount / old_orig, np.nan)
    out["amount_to_orig_balance_ratio"] = ratio.astype(np.float32)

    out["is_orig_balance_zero"] = (old_orig == 0).astype(np.int8)
    out["is_full_drain"] = (
        (old_orig > 0) & (np.abs(amount - old_orig) <= BALANCE_TOLERANCE)
    ).astype(np.int8)
    out["is_dest_balance_zero"] = (
        (old_dest == 0) & (new_dest == 0) & (amount > 0) & (out["is_merchant_dest"].to_numpy() == 0)
    ).astype(np.int8)
    out["hour_of_day"] = ((out["step"].to_numpy() - 1) % 24).astype(np.int8)

    tx_type = out["type"].astype(str).to_numpy()
    for t, col in zip(TRANSACTION_TYPES, TYPE_DUMMIES, strict=True):
        out[col] = (tx_type == t).astype(np.int8)
    return out
