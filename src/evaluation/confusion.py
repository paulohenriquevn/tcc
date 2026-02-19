"""Confusion matrix computation and visualization.

Produces normalized confusion matrices for accent classification,
required by the validation protocol for every probe result.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for scripts
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def compute_normalized_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute row-normalized (recall per class) confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Class labels in display order. If None, inferred from data.

    Returns:
        Tuple of (normalized_matrix, labels).
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalize by row (recall per class)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero for classes with no samples
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_normalized = cm / row_sums

    return cm_normalized, labels


def plot_confusion_matrix(
    cm_normalized: np.ndarray,
    labels: list[str],
    title: str = "Confusion Matrix (Normalized by Class)",
    output_path: Path | None = None,
) -> None:
    """Plot and optionally save confusion matrix heatmap.

    Args:
        cm_normalized: Row-normalized confusion matrix.
        labels: Class labels.
        title: Plot title.
        output_path: If provided, saves figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {output_path}")

    plt.close(fig)


def check_degenerate_matrix(
    cm_normalized: np.ndarray,
    labels: list[str],
    min_recall: float = 0.20,
) -> list[str]:
    """Check if confusion matrix is degenerate.

    A matrix is degenerate if any class has recall below threshold,
    meaning the model essentially ignores that class.

    Args:
        cm_normalized: Row-normalized confusion matrix.
        labels: Class labels.
        min_recall: Minimum acceptable recall per class.

    Returns:
        List of warning messages (empty if non-degenerate).
    """
    warnings = []
    for i, label in enumerate(labels):
        recall = cm_normalized[i, i]
        if recall < min_recall:
            warnings.append(
                f"Class '{label}' has recall {recall:.3f} < {min_recall} — "
                f"matrix may be degenerate"
            )
    return warnings
