"""Tests for bootstrap confidence interval computation."""

import numpy as np
import pytest

from src.evaluation.bootstrap_ci import (
    bootstrap_balanced_accuracy,
    bootstrap_cosine_similarity,
    bootstrap_metric,
)


class TestBootstrapBalancedAccuracy:
    def test_perfect_predictions_give_narrow_ci(self):
        y_true = np.array([0, 1, 2, 0, 1, 2] * 50)
        y_pred = y_true.copy()

        ci = bootstrap_balanced_accuracy(y_true, y_pred, seed=42)

        assert ci.point_estimate == 1.0
        assert ci.ci_lower >= 0.95
        assert ci.ci_upper <= 1.0

    def test_random_predictions_include_chance(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 3, size=300)
        y_pred = rng.randint(0, 3, size=300)

        ci = bootstrap_balanced_accuracy(
            y_true, y_pred, seed=42, chance_level=1/3
        )

        # Random predictions should have CI that includes or is near chance
        assert ci.point_estimate < 0.5

    def test_deterministic_with_same_seed(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 3, size=100)
        y_pred = rng.randint(0, 3, size=100)

        ci1 = bootstrap_balanced_accuracy(y_true, y_pred, seed=42)
        ci2 = bootstrap_balanced_accuracy(y_true, y_pred, seed=42)

        assert ci1.ci_lower == ci2.ci_lower
        assert ci1.ci_upper == ci2.ci_upper

    def test_includes_chance_flag(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        y_pred = rng.randint(0, 2, size=100)

        ci = bootstrap_balanced_accuracy(
            y_true, y_pred, seed=42, chance_level=0.5
        )

        # includes_chance should be a boolean
        assert isinstance(ci.includes_chance, bool)


class TestBootstrapCosineSimilarity:
    def test_identical_embeddings_give_high_similarity(self):
        similarities = np.ones(100) * 0.95 + np.random.RandomState(42).normal(0, 0.01, 100)

        ci = bootstrap_cosine_similarity(similarities, seed=42)

        assert ci.point_estimate > 0.9
        assert ci.ci_lower > 0.9

    def test_ci_width_decreases_with_more_samples(self):
        rng = np.random.RandomState(42)
        sims_small = rng.normal(0.8, 0.1, size=20)
        sims_large = rng.normal(0.8, 0.1, size=500)

        ci_small = bootstrap_cosine_similarity(sims_small, seed=42)
        ci_large = bootstrap_cosine_similarity(sims_large, seed=42)

        width_small = ci_small.ci_upper - ci_small.ci_lower
        width_large = ci_large.ci_upper - ci_large.ci_lower

        assert width_large < width_small
