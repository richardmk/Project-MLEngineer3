"""Project-wide constants: paths, seeds and temporal split boundaries."""

from pathlib import Path

SEED: int = 42

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

RAW_CSV: Path = RAW_DIR / "PS_20174392719_1491204439457_log.csv"

# Processed splits (git-ignored) and light versioned samples.
TRAIN_PARQUET: Path = PROCESSED_DIR / "train.parquet"
TEST_PARQUET: Path = PROCESSED_DIR / "test.parquet"
PRODUCTION_PARQUET: Path = PROCESSED_DIR / "production.parquet"
REFERENCE_SAMPLE_CSV: Path = DATA_DIR / "reference.csv"
PRODUCTION_CSV: Path = DATA_DIR / "production.csv"
REFERENCE_SAMPLE_ROWS: int = 100_000

# --- Problem definition ------------------------------------------------------
TARGET: str = "isFraud"
# EDA: fraud only happens in these types; the rest are scored 0 by rule.
MODEL_TYPES: list[str] = ["TRANSFER", "CASH_OUT"]

# Course rule: step <= 400 is reference/training data, step >= 401 is production.
REFERENCE_MAX_STEP: int = 400
# Train = steps 1-312 (13 full simulated days), test = 313-400. See notebook 01.
TRAIN_MAX_STEP: int = 312
# Production monitoring window size in steps (24 steps = 1 simulated day).
PERIOD_STEPS: int = 24

# --- Experiment tracking -----------------------------------------------------
MLFLOW_EXPERIMENT_SETUP: str = "setup-check"
