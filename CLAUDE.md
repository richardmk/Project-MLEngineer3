# CLAUDE.md

Fraud detection on PaySim (MLE specialization, Course III final project). Binary classification, target `isFraud`.

## Commands
- Install: `uv sync`
- Lint: `uv run ruff check . && uv run ruff format --check .`
- MLflow connection check: `uv run python scripts/check_mlflow.py` (needs `.env`, see `.env.example`)
- Run a notebook end-to-end: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<nb>.ipynb`
- Notebooks are generated from `scripts/build_notebook_XX.py`; edit the generator, regenerate, then execute.
- Monitoring app: `uv run streamlit run app/app.py`
- Final HTML report + README figures (after running notebooks): `uv run python scripts/build_report.py`
- Export requirements after changing deps: `uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt`

## Layout
- `src/fraud_detection/`: reusable code (config, data, features, split, train, evaluate, monitoring). Notebooks import from here; keep notebook cells thin.
- `notebooks/01..04`: preprocessing, training, interpretability, monitoring.
- `data/raw`, `data/processed`, `models/`: git-ignored. Only light samples `data/*.csv` are versioned.

## Non-negotiable rules
- Temporal split only. `step <= 400` = reference/training; `step >= 401` = production, used **only** for monitoring/online metrics (never training or tuning). Constants live in `config.py`.
- No leakage: never use raw `nameOrig`/`nameDest` as features (only derived flags such as `is_merchant_dest`). Fit scalers/encoders/samplers on train only. `isFlaggedFraud` excluded as a feature unless EDA justifies otherwise.
- Imbalanced target: never use accuracy as the main metric. Use PR-AUC, recall, precision, F1, F2, confusion matrix; ROC-AUC only as a complement. Tune the decision threshold explicitly.
- ~6.3M rows, limited WSL RAM: explicit dtypes, Parquet intermediates, polars lazy for ingestion. Warn before heavy operations.
- Fixed seeds (`config.SEED`). Notebooks must run top to bottom.
- Secrets only in `.env` (git-ignored), loaded with python-dotenv.
- Evidently is 0.7.x: use the new API (`from evidently import Report, Dataset, DataDefinition`, `evidently.presets`), not the legacy `evidently.report`/`metric_preset` API.
- Stop and flag suspiciously perfect results (e.g. PR-AUC ~ 1.0), leakage or RAM issues.

## Conventions
- Language: README/notebooks/docs in Spanish; code, identifiers and commits in English.
- Google-style docstrings, type hints, ruff.
- Git: GitHub Flow variant in `GIT_STRATEGY.md` (`feature/*` -> `development` -> `main`), Conventional Commits.
- Work proceeds by phases; the user approves each phase before the next.
