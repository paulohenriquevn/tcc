"""Linear probes for accent, speaker, and leakage testing.

CRITICAL: Split assignments for leakage probes follow specific logic:

- accent probe: speaker-disjoint split (test speakers never seen in train)
- speaker probe: stratified split (same speakers in train/test, different utts)
- leakage A→speaker: STRATIFIED split (same speakers in train/test)
  Because we test if accent features contain speaker info — need known speakers
- leakage S→accent: SPEAKER-DISJOINT split (different speakers in test)
  Because we test if speaker features contain accent info — need unseen speakers

Getting these wrong was Achado 1 (CRITICAL) in the readiness audit.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

from src.evaluation.bootstrap_ci import bootstrap_balanced_accuracy, BootstrapCI
from src.evaluation.confusion import (
    compute_normalized_confusion_matrix,
    check_degenerate_matrix,
)

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of a single probe experiment."""
    probe_name: str
    feature_source: str       # e.g., "backbone_layer_12", "ecapa", "wavlm_layer_6"
    target: str               # "accent" or "speaker_id"
    split_type: str           # "speaker_disjoint" or "stratified"
    balanced_accuracy: float
    f1_macro: float
    ci: BootstrapCI | None
    chance_level: float
    delta_pp: float           # balanced_accuracy - chance_level, in percentage points
    n_train: int
    n_test: int
    n_classes: int
    confusion_matrix: np.ndarray | None
    confusion_labels: list[str] | None
    degenerate_warnings: list[str]
    regularization_C: float


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    probe_name: str,
    feature_source: str,
    target: str,
    split_type: str,
    C: float = 1.0,
    seed: int = 42,
    compute_ci: bool = True,
    n_bootstrap: int = 1000,
) -> ProbeResult:
    """Train a logistic regression probe and evaluate.

    Args:
        X_train: Training features, shape (n_train, n_features).
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        probe_name: Name for this probe (e.g., "accent_probe").
        feature_source: Where features came from.
        target: What we're predicting.
        split_type: How train/test were split.
        C: Regularization parameter (inverse strength).
        seed: Random seed.
        compute_ci: Whether to compute bootstrap CI.
        n_bootstrap: Number of bootstrap samples for CI.

    Returns:
        ProbeResult with all metrics and diagnostics.
    """
    # Determine number of classes
    classes = sorted(set(y_train) | set(y_test))
    n_classes = len(classes)
    chance_level = 1.0 / n_classes

    logger.info(
        f"Probe '{probe_name}': {len(X_train)} train, {len(X_test)} test, "
        f"{n_classes} classes, chance={chance_level:.4f}"
    )

    # Train logistic regression (linear probe)
    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Metrics
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    delta_pp = (bal_acc - chance_level) * 100  # percentage points

    # CI
    ci = None
    if compute_ci:
        ci = bootstrap_balanced_accuracy(
            y_true=np.array(y_test),
            y_pred=np.array(y_pred),
            n_bootstrap=n_bootstrap,
            seed=seed,
            chance_level=chance_level,
        )

    # Confusion matrix — use original label types for sklearn compatibility
    cm_norm, cm_labels = compute_normalized_confusion_matrix(
        y_test, y_pred, labels=classes
    )
    degenerate_warnings = check_degenerate_matrix(cm_norm, cm_labels)

    logger.info(
        f"  bal_acc={bal_acc:.4f}, f1_macro={f1_mac:.4f}, "
        f"chance={chance_level:.4f}, delta={delta_pp:+.1f}pp"
    )
    if ci:
        logger.info(
            f"  CI 95%: [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]"
            f"{' (includes chance!)' if ci.includes_chance else ''}"
        )

    return ProbeResult(
        probe_name=probe_name,
        feature_source=feature_source,
        target=target,
        split_type=split_type,
        balanced_accuracy=bal_acc,
        f1_macro=f1_mac,
        ci=ci,
        chance_level=chance_level,
        delta_pp=delta_pp,
        n_train=len(X_train),
        n_test=len(X_test),
        n_classes=n_classes,
        confusion_matrix=cm_norm,
        confusion_labels=cm_labels,
        degenerate_warnings=degenerate_warnings,
        regularization_C=C,
    )


def evaluate_probe_against_thresholds(
    result: ProbeResult,
    thresholds: dict,
) -> str:
    """Evaluate a probe result against Gate thresholds.

    Args:
        result: ProbeResult from a probe experiment.
        thresholds: Dict with 'go', 'go_conditional' balanced accuracy values,
                    or 'go_margin_pp', 'conditional_margin_pp' for leakage.

    Returns:
        "GO", "GO_CONDITIONAL", or "FAIL".
    """
    if "go_margin_pp" in thresholds:
        # Leakage probe — threshold is delta from chance
        if result.delta_pp <= thresholds["go_margin_pp"]:
            return "GO"
        elif result.delta_pp <= thresholds["conditional_margin_pp"]:
            return "GO_CONDITIONAL"
        else:
            return "FAIL"
    else:
        # Accent/speaker probe — threshold is absolute balanced accuracy
        if result.balanced_accuracy >= thresholds["go"]:
            return "GO"
        elif result.balanced_accuracy >= thresholds["go_conditional"]:
            return "GO_CONDITIONAL"
        else:
            return "FAIL"


def train_selectivity_control(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    real_result: ProbeResult,
    seed: int = 42,
    n_permutations: int = 5,
    C: float = 1.0,
) -> dict:
    """Run permutation baseline to measure probe selectivity.

    Selectivity = real_accuracy - permuted_accuracy.
    If selectivity is low, the probe may be memorizing rather than
    detecting a real signal.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        real_result: The ProbeResult from the real (unpermuted) probe.
        seed: Random seed.
        n_permutations: Number of permutation runs to average.
        C: Regularization parameter.

    Returns:
        Dict with selectivity metrics:
        {
            'real_bal_acc': float,
            'permuted_bal_acc_mean': float,
            'permuted_bal_acc_std': float,
            'selectivity_pp': float,  # difference in percentage points
            'n_permutations': int,
        }
    """
    rng = np.random.RandomState(seed)
    permuted_accs = []

    for i in range(n_permutations):
        # Permute training labels (break signal, keep structure)
        y_train_perm = rng.permutation(y_train)

        clf = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=seed + i,
        )
        clf.fit(X_train, y_train_perm)
        y_pred_perm = clf.predict(X_test)
        perm_acc = balanced_accuracy_score(y_test, y_pred_perm)
        permuted_accs.append(perm_acc)

    perm_mean = float(np.mean(permuted_accs))
    perm_std = float(np.std(permuted_accs))
    selectivity = (real_result.balanced_accuracy - perm_mean) * 100  # pp

    logger.info(
        f"Selectivity control for '{real_result.probe_name}': "
        f"real={real_result.balanced_accuracy:.4f}, "
        f"permuted={perm_mean:.4f}±{perm_std:.4f}, "
        f"selectivity={selectivity:+.1f}pp"
    )

    return {
        "real_bal_acc": real_result.balanced_accuracy,
        "permuted_bal_acc_mean": perm_mean,
        "permuted_bal_acc_std": perm_std,
        "selectivity_pp": selectivity,
        "n_permutations": n_permutations,
    }


def sweep_regularization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C_values: list[float] = (0.01, 0.1, 1.0, 10.0),
    **probe_kwargs,
) -> list[ProbeResult]:
    """Run probe with multiple C values to check robustness.

    Args:
        X_train, y_train, X_test, y_test: Data.
        C_values: Regularization values to try.
        **probe_kwargs: Passed to train_linear_probe.

    Returns:
        List of ProbeResult, one per C value.
    """
    results = []
    for c in C_values:
        result = train_linear_probe(
            X_train, y_train, X_test, y_test,
            C=c, compute_ci=False,  # Skip CI for sweep
            **probe_kwargs,
        )
        results.append(result)
        logger.info(f"  C={c}: bal_acc={result.balanced_accuracy:.4f}")

    return results
