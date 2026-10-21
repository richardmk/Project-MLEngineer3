"""Raw data ingestion: CSV -> typed Parquet without raw account identifiers."""

from pathlib import Path

import pandas as pd
import polars as pl

from fraud_detection.config import PROCESSED_DIR, RAW_CSV

TRANSACTION_TYPES: list[str] = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# float64 on purpose: amounts reach ~9e7 and float32 (~7 significant digits) would
# drop the cents, adding noise to the balance-consistency features.
RAW_SCHEMA: dict[str, pl.DataType] = {
    "step": pl.Int16,
    "type": pl.Enum(TRANSACTION_TYPES),
    "amount": pl.Float64,
    "nameOrig": pl.String,
    "oldbalanceOrg": pl.Float64,
    "newbalanceOrig": pl.Float64,
    "nameDest": pl.String,
    "oldbalanceDest": pl.Float64,
    "newbalanceDest": pl.Float64,
    "isFraud": pl.Int8,
    "isFlaggedFraud": pl.Int8,
}

TRANSACTIONS_PARQUET: Path = PROCESSED_DIR / "transactions.parquet"


def scan_raw_csv(csv_path: Path = RAW_CSV) -> pl.LazyFrame:
    """Lazily scan the raw PaySim CSV with an explicit, memory-efficient schema.

    Args:
        csv_path: Path to the raw CSV.

    Returns:
        A polars LazyFrame; nothing is loaded until it is collected or sunk.
    """
    return pl.scan_csv(csv_path, schema=RAW_SCHEMA)


def summarize_account_names(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    """Profile the account identifiers before they are dropped.

    Only aggregated patterns leave this function; raw names are never persisted.

    Args:
        csv_path: Path to the raw CSV.

    Returns:
        One-row-per-metric DataFrame with prefix counts and cardinalities.
    """
    stats = (
        scan_raw_csv(csv_path)
        .select(
            pl.len().alias("rows"),
            pl.col("nameOrig").n_unique().alias("unique_orig"),
            pl.col("nameDest").n_unique().alias("unique_dest"),
            pl.col("nameOrig").str.starts_with("C").sum().alias("orig_prefix_C"),
            pl.col("nameOrig").str.starts_with("M").sum().alias("orig_prefix_M"),
            pl.col("nameDest").str.starts_with("C").sum().alias("dest_prefix_C"),
            pl.col("nameDest").str.starts_with("M").sum().alias("dest_prefix_M"),
        )
        .collect()
    )
    return stats.to_pandas().T.rename(columns={0: "value"})


def build_transactions_parquet(
    csv_path: Path = RAW_CSV,
    out_path: Path = TRANSACTIONS_PARQUET,
    overwrite: bool = False,
) -> Path:
    """Convert the raw CSV into a typed Parquet file, replacing names by derived flags.

    `nameOrig`/`nameDest` are dropped to avoid identifier leakage and to save RAM;
    only `is_merchant_dest` (destination name starts with "M") is kept. Runs with
    polars' streaming engine so peak memory stays well below the CSV size.

    Args:
        csv_path: Path to the raw CSV.
        out_path: Destination Parquet path.
        overwrite: Rebuild even if `out_path` already exists.

    Returns:
        The Parquet path.
    """
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (
        scan_raw_csv(csv_path)
        .with_columns(
            pl.col("nameDest").str.starts_with("M").cast(pl.Int8).alias("is_merchant_dest")
        )
        .drop("nameOrig", "nameDest")
        .sink_parquet(out_path, compression="zstd")
    )
    return out_path


def load_transactions(
    path: Path = TRANSACTIONS_PARQUET, columns: list[str] | None = None
) -> pd.DataFrame:
    """Load the processed transactions as pandas, keeping compact dtypes.

    Args:
        path: Parquet path produced by `build_transactions_parquet`.
        columns: Optional subset of columns to read.

    Returns:
        DataFrame with `type` as a pandas categorical.
    """
    df = pl.read_parquet(path, columns=columns).to_pandas()
    if "type" in df.columns:
        df["type"] = df["type"].astype(pd.CategoricalDtype(TRANSACTION_TYPES))
    return df
