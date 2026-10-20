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

# --- Problem definition ------------------------------------------------------
TARGET: str = "isFraud"

# Course rule: step <= 400 is reference/training data, step >= 401 is production.
REFERENCE_MAX_STEP: int = 400
# Tentative train/test boundary inside the reference window (revisited after EDA).
TRAIN_MAX_STEP: int = 320
# Production monitoring window size in steps (24 steps = 1 simulated day).
PERIOD_STEPS: int = 24

# --- Experiment tracking -----------------------------------------------------
MLFLOW_EXPERIMENT_SETUP: str = "setup-check"
