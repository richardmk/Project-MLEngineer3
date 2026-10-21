"""Evaluation for imbalanced fraud detection: metrics, threshold selection and plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_detection.plots import COLOR_FRAUD, COLOR_LEGIT, COLOR_NEUTRAL, INK_SECONDARY, finish


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute threshold-free and thresholded metrics.

    Args:
        y_true: Binary labels.
        y_score: Fraud scores (higher = more likely fraud).
        threshold: Decision threshold; `y_score >= threshold` is flagged as fraud.

    Returns:
        Dict with pr_auc, roc_auc, precision, recall, f1, f2 and confusion-matrix counts.
    """
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = len(np.unique(y_true)) == 2
    return {
        "pr_auc": average_precision_score(y_true, y_score) if has_both else np.nan,
        "roc_auc": roc_auc_score(y_true, y_score) if has_both else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": fbeta_score(y_true, y_pred, beta=1, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def find_best_threshold(
    y_true: np.ndarray, y_score: np.ndarray, beta: float = 2.0
) -> tuple[float, float]:
    """Find the threshold that maximizes F-beta on the given (out-of-fold) predictions.

    Args:
        y_true: Binary labels.
        y_score: Fraud scores.
        beta: F-beta weight; beta=2 values recall twice as much as precision.

    Returns:
        (best_threshold, best_fbeta).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    denom = beta**2 * precision + recall
    fbeta = np.divide(
        (1 + beta**2) * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0
    )
    best = int(np.argmax(fbeta))
    return float(thresholds[best]), float(fbeta[best])


def expected_cost_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    review_cost: float,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Business cost per threshold: missed fraud costs its amount, a false alarm costs a review.

    Args:
        y_true: Binary labels.
        y_score: Fraud scores.
        amounts: Transaction amounts (loss if a fraud is missed).
        review_cost: Cost of manually reviewing one flagged transaction (TP or FP).
        thresholds: Thresholds to evaluate; defaults to 200 log-spaced values.

    Returns:
        DataFrame with threshold, fraud_loss, review_cost, total_cost, flagged.
    """
    if thresholds is None:
        thresholds = np.unique(np.concatenate([np.logspace(-4, 0, 200), [0.5]]))
    is_fraud = y_true == 1
    rows = []
    for thr in thresholds:
        flagged = y_score >= thr
        fraud_loss = float(amounts[is_fraud & ~flagged].sum())
        reviews = float(flagged.sum()) * review_cost
        rows.append(
            {
                "threshold": thr,
                "fraud_loss": fraud_loss,
                "review_cost": reviews,
                "total_cost": fraud_loss + reviews,
                "flagged": int(flagged.sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_pr_curve(curves: dict[str, tuple[np.ndarray, np.ndarray]], base_rate: float) -> plt.Figure:
    """Plot precision-recall curves for one or more models.

    Args:
        curves: Mapping label -> (y_true, y_score).
        base_rate: Positive rate, drawn as the random-classifier baseline.

    Returns:
        The matplotlib figure.
    """
    colors = [COLOR_LEGIT, COLOR_FRAUD, "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for (label, (y_true, y_score)), color in zip(curves.items(), colors, strict=False):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color=color, label=f"{label} (PR-AUC {ap:.3f})")
    ax.axhline(
        base_rate, color=COLOR_NEUTRAL, linestyle=":", linewidth=1, label=f"Azar ({base_rate:.4f})"
    )
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="x")
    ax.legend(loc="lower left")
    finish(ax, "Curva Precision-Recall", "Recall", "Precision")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float, title: str
) -> plt.Figure:
    """Plot a labelled 2x2 confusion matrix at the given threshold.

    Args:
        y_true: Binary labels.
        y_score: Fraud scores.
        threshold: Decision threshold.
        title: Chart title.

    Returns:
        The matplotlib figure.
    """
    cm = confusion_matrix(y_true, (y_score >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(np.log1p(cm), cmap="Blues")
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if np.log1p(cm[i, j]) > np.log1p(cm).max() * 0.6 else "#0b0b0b"
            ax.text(
                j,
                i,
                f"{names[i][j]}\n{cm[i, j]:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
            )
    ax.set_xticks([0, 1], ["Legítimo", "Fraude"])
    ax.set_yticks([0, 1], ["Legítimo", "Fraude"])
    ax.grid(False)
    finish(ax, f"{title}\n(threshold = {threshold:.4f})", "Predicción", "Real")
    ax.title.set_fontsize(10)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance: pd.Series, title: str, top: int = 15) -> plt.Figure:
    """Horizontal bar chart of the top feature importances.

    Args:
        importance: Importance per feature (index = feature name).
        title: Chart title.
        top: Number of features to show.

    Returns:
        The matplotlib figure.
    """
    data = importance.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(data) + 1.2))
    ax.barh(data.index, data.values, color=COLOR_LEGIT, height=0.6)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK_SECONDARY)
    finish(ax, title, "Importancia", "")
    fig.tight_layout()
    return fig
