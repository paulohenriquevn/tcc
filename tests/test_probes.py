"""Tests for linear probes and threshold evaluation."""

import numpy as np
import pytest

from src.evaluation.probes import (
    ProbeResult,
    evaluate_probe_against_thresholds,
    train_linear_probe,
    train_selectivity_control,
    sweep_regularization,
)


def _make_separable_data(
    n_samples: int = 200,
    n_features: int = 32,
    n_classes: int = 3,
    seed: int = 42,
    noise: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create linearly separable data for testing probes."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, n_classes, size=n_samples)

    # Add signal: shift mean by class
    for cls in range(n_classes):
        mask = y == cls
        X[mask, :3] += cls * 2  # Strong signal in first 3 features

    X += rng.randn(n_samples, n_features) * noise

    split = int(0.7 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def _make_random_data(
    n_samples: int = 200,
    n_features: int = 32,
    n_classes: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create random (non-separable) data for testing chance-level probes."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, n_classes, size=n_samples)

    split = int(0.7 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


class TestTrainLinearProbe:
    def test_separable_data_above_chance(self):
        X_train, y_train, X_test, y_test = _make_separable_data()

        result = train_linear_probe(
            X_train, y_train, X_test, y_test,
            probe_name="test_probe",
            feature_source="synthetic",
            target="accent",
            split_type="speaker_disjoint",
            compute_ci=False,
        )

        assert result.balanced_accuracy > result.chance_level
        assert result.delta_pp > 0

    def test_random_data_near_chance(self):
        X_train, y_train, X_test, y_test = _make_random_data()

        result = train_linear_probe(
            X_train, y_train, X_test, y_test,
            probe_name="test_probe",
            feature_source="synthetic",
            target="accent",
            split_type="speaker_disjoint",
            compute_ci=False,
        )

        # Should be near chance (1/3 ≈ 0.333) with some variance
        assert result.balanced_accuracy < 0.55
        assert result.chance_level == pytest.approx(1/3, abs=0.01)

    def test_returns_confusion_matrix(self):
        X_train, y_train, X_test, y_test = _make_separable_data()

        result = train_linear_probe(
            X_train, y_train, X_test, y_test,
            probe_name="test_probe",
            feature_source="synthetic",
            target="accent",
            split_type="speaker_disjoint",
            compute_ci=False,
        )

        assert result.confusion_matrix is not None
        assert result.confusion_matrix.shape == (3, 3)
        # Row sums should be ~1.0 (normalized)
        row_sums = result.confusion_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_with_ci_computation(self):
        X_train, y_train, X_test, y_test = _make_separable_data()

        result = train_linear_probe(
            X_train, y_train, X_test, y_test,
            probe_name="test_probe",
            feature_source="synthetic",
            target="accent",
            split_type="speaker_disjoint",
            compute_ci=True,
            n_bootstrap=100,  # Fewer for speed
        )

        assert result.ci is not None
        assert result.ci.ci_lower <= result.balanced_accuracy <= result.ci.ci_upper


class TestEvaluateThresholds:
    def _make_result(self, bal_acc: float, delta_pp: float) -> ProbeResult:
        return ProbeResult(
            probe_name="test",
            feature_source="test",
            target="accent",
            split_type="speaker_disjoint",
            balanced_accuracy=bal_acc,
            f1_macro=bal_acc,
            ci=None,
            chance_level=0.2,
            delta_pp=delta_pp,
            n_train=100,
            n_test=50,
            n_classes=5,
            confusion_matrix=None,
            confusion_labels=None,
            degenerate_warnings=[],
            regularization_C=1.0,
        )

    def test_accent_probe_go(self):
        result = self._make_result(0.60, 40.0)
        decision = evaluate_probe_against_thresholds(
            result, {"go": 0.55, "go_conditional": 0.50}
        )
        assert decision == "GO"

    def test_accent_probe_conditional(self):
        result = self._make_result(0.52, 32.0)
        decision = evaluate_probe_against_thresholds(
            result, {"go": 0.55, "go_conditional": 0.50}
        )
        assert decision == "GO_CONDITIONAL"

    def test_accent_probe_fail(self):
        result = self._make_result(0.45, 25.0)
        decision = evaluate_probe_against_thresholds(
            result, {"go": 0.55, "go_conditional": 0.50}
        )
        assert decision == "FAIL"

    def test_leakage_probe_go(self):
        result = self._make_result(0.22, 2.0)
        decision = evaluate_probe_against_thresholds(
            result, {"go_margin_pp": 5, "conditional_margin_pp": 12}
        )
        assert decision == "GO"

    def test_leakage_probe_conditional(self):
        result = self._make_result(0.28, 8.0)
        decision = evaluate_probe_against_thresholds(
            result, {"go_margin_pp": 5, "conditional_margin_pp": 12}
        )
        assert decision == "GO_CONDITIONAL"

    def test_leakage_probe_fail(self):
        result = self._make_result(0.35, 15.0)
        decision = evaluate_probe_against_thresholds(
            result, {"go_margin_pp": 5, "conditional_margin_pp": 12}
        )
        assert decision == "FAIL"


class TestSelectivityControl:
    def test_separable_data_has_high_selectivity(self):
        X_train, y_train, X_test, y_test = _make_separable_data()

        real_result = train_linear_probe(
            X_train, y_train, X_test, y_test,
            probe_name="test", feature_source="synthetic",
            target="accent", split_type="speaker_disjoint",
            compute_ci=False,
        )

        selectivity = train_selectivity_control(
            X_train, y_train, X_test, y_test,
            real_result=real_result,
            n_permutations=3,
        )

        assert selectivity["selectivity_pp"] > 10  # Strong signal
        assert selectivity["permuted_bal_acc_mean"] < real_result.balanced_accuracy

    def test_random_data_has_low_selectivity(self):
        X_train, y_train, X_test, y_test = _make_random_data()

        real_result = train_linear_probe(
            X_train, y_train, X_test, y_test,
            probe_name="test", feature_source="synthetic",
            target="accent", split_type="speaker_disjoint",
            compute_ci=False,
        )

        selectivity = train_selectivity_control(
            X_train, y_train, X_test, y_test,
            real_result=real_result,
            n_permutations=3,
        )

        # Selectivity should be near zero for random data
        assert abs(selectivity["selectivity_pp"]) < 20


class TestRegularizationSweep:
    def test_returns_results_for_each_c(self):
        X_train, y_train, X_test, y_test = _make_separable_data()

        results = sweep_regularization(
            X_train, y_train, X_test, y_test,
            C_values=[0.1, 1.0, 10.0],
            probe_name="sweep",
            feature_source="synthetic",
            target="accent",
            split_type="speaker_disjoint",
            seed=42,
        )

        assert len(results) == 3
        assert results[0].regularization_C == 0.1
        assert results[1].regularization_C == 1.0
        assert results[2].regularization_C == 10.0
