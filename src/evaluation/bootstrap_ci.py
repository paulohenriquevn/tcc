"""Bootstrap confidence intervals for evaluation metrics.

Implements non-parametric bootstrap for CI 95% estimation.
Used for all reported metrics (balanced accuracy, cosine similarity, etc.).
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score

logger = logging.getLogger(__name__)


@dataclass
class BootstrapCI:
    """Result of a bootstrap confidence interval estimation."""
    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence: float
    n_samples: int
    includes_chance: bool  # True if CI includes chance level


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    metric_name: str = "metric",
    chance_level: float | None = None,
) -> BootstrapCI:
    """Compute bootstrap CI for a classification metric.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        metric_fn: Function(y_true, y_pred) -> float.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (e.g., 0.95).
        seed: Random seed for reproducibility.
        metric_name: Name for reporting.
        chance_level: If provided, checks if CI includes this value.

    Returns:
        BootstrapCI with point estimate and interval.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    # Point estimate on full data
    point_estimate = metric_fn(y_true, y_pred)

    # Bootstrap resampling
    bootstrap_scores = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        bootstrap_scores[i] = metric_fn(y_true[indices], y_pred[indices])

    # Percentile method for CI
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))

    includes_chance = False
    if chance_level is not None:
        includes_chance = ci_lower <= chance_level <= ci_upper

    return BootstrapCI(
        metric_name=metric_name,
        point_estimate=float(point_estimate),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence=confidence,
        n_samples=n_bootstrap,
        includes_chance=includes_chance,
    )


def bootstrap_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    chance_level: float | None = None,
) -> BootstrapCI:
    """Convenience wrapper for balanced accuracy CI.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        seed: Random seed.
        chance_level: Chance level for the task (1/N_classes).

    Returns:
        BootstrapCI for balanced accuracy.
    """
    return bootstrap_metric(
        y_true=y_true,
        y_pred=y_pred,
        metric_fn=balanced_accuracy_score,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        metric_name="balanced_accuracy",
        chance_level=chance_level,
    )


def bootstrap_cosine_similarity(
    similarities: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap CI for a set of cosine similarity values.

    Args:
        similarities: Array of cosine similarity scores.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        BootstrapCI for mean cosine similarity.
    """
    rng = np.random.RandomState(seed)
    n = len(similarities)

    point_estimate = float(np.mean(similarities))

    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        bootstrap_means[i] = np.mean(similarities[indices])

    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        metric_name="cosine_similarity",
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence=confidence,
        n_samples=n_bootstrap,
        includes_chance=False,
    )
