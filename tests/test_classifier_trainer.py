"""Tests for classifier training and evaluation."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.classifier.cnn_model import AccentCNN
from src.classifier.trainer import (
    TrainingConfig,
    TrainingResult,
    compute_class_weights,
    train_classifier,
    evaluate_classifier,
)


def _make_synthetic_loaders(n_classes=3, n_samples=30, batch_size=10):
    """Create tiny random DataLoaders for CNN smoke tests."""
    X = torch.randn(n_samples, 1, 80, 50)  # small mel spectrograms
    y = torch.randint(0, n_classes, (n_samples,))
    ds = TensorDataset(X, y)
    train_loader = DataLoader(ds, batch_size=batch_size)
    val_loader = DataLoader(ds, batch_size=batch_size)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------


class TestComputeClassWeights:
    def test_balanced_labels(self):
        """Equal class counts produce equal weights."""
        # Arrange
        labels = [0, 0, 1, 1]

        # Act
        weights = compute_class_weights(labels, n_classes=2)

        # Assert
        assert weights.shape == (2,)
        assert torch.allclose(weights[0], weights[1])

    def test_imbalanced_labels(self):
        """Minority class gets a higher weight than majority class."""
        # Arrange
        labels = [0, 0, 0, 1]

        # Act
        weights = compute_class_weights(labels, n_classes=2)

        # Assert
        assert weights[1] > weights[0]

    def test_empty_labels_raises(self):
        """Empty label list raises ValueError."""
        with pytest.raises(ValueError, match="labels must not be empty"):
            compute_class_weights([], n_classes=2)


# ---------------------------------------------------------------------------
# train_classifier
# ---------------------------------------------------------------------------


class TestTrainClassifier:
    def test_smoke_1_epoch(self, tmp_path):
        """Training for 1 epoch completes and returns a TrainingResult."""
        # Arrange
        n_classes = 3
        model = AccentCNN(n_classes=n_classes)
        train_loader, val_loader = _make_synthetic_loaders(n_classes=n_classes)
        config = TrainingConfig(
            n_epochs=1,
            patience=1,
            use_amp=False,
            device="cpu",
            checkpoint_dir=tmp_path / "ckpt",
            experiment_name="smoke_test",
        )

        # Act
        result = train_classifier(model, train_loader, val_loader, config)

        # Assert
        assert isinstance(result, TrainingResult)
        assert result.best_epoch >= 1
        assert result.total_epochs_run == 1

    def test_checkpoint_created(self, tmp_path):
        """After training, a checkpoint file exists at best_checkpoint_path."""
        # Arrange
        n_classes = 3
        model = AccentCNN(n_classes=n_classes)
        train_loader, val_loader = _make_synthetic_loaders(n_classes=n_classes)
        config = TrainingConfig(
            n_epochs=2,
            patience=5,
            use_amp=False,
            device="cpu",
            checkpoint_dir=tmp_path / "ckpt",
            experiment_name="ckpt_test",
        )

        # Act
        result = train_classifier(model, train_loader, val_loader, config)

        # Assert
        assert result.best_checkpoint_path.exists()

    def test_training_result_fields(self, tmp_path):
        """TrainingResult has consistent list lengths matching total_epochs_run."""
        # Arrange
        n_classes = 3
        model = AccentCNN(n_classes=n_classes)
        train_loader, val_loader = _make_synthetic_loaders(n_classes=n_classes)
        config = TrainingConfig(
            n_epochs=3,
            patience=10,
            use_amp=False,
            device="cpu",
            checkpoint_dir=tmp_path / "ckpt",
            experiment_name="fields_test",
        )

        # Act
        result = train_classifier(model, train_loader, val_loader, config)

        # Assert
        assert result.best_epoch >= 1
        assert len(result.train_losses) == result.total_epochs_run
        assert len(result.val_bal_accs) == result.total_epochs_run


# ---------------------------------------------------------------------------
# evaluate_classifier
# ---------------------------------------------------------------------------


class TestEvaluateClassifier:
    def test_evaluate_returns_metrics(self, tmp_path):
        """Evaluation returns dict with all required metric keys."""
        # Arrange
        n_classes = 3
        model = AccentCNN(n_classes=n_classes)
        train_loader, val_loader = _make_synthetic_loaders(n_classes=n_classes)
        config = TrainingConfig(
            n_epochs=2,
            patience=5,
            use_amp=False,
            device="cpu",
            checkpoint_dir=tmp_path / "ckpt",
            experiment_name="eval_test",
        )
        train_classifier(model, train_loader, val_loader, config)

        label_names = ["NE", "S", "SE"]
        test_loader = val_loader

        # Act
        metrics = evaluate_classifier(
            model, test_loader, label_names, device="cpu", n_bootstrap=50,
        )

        # Assert
        expected_keys = {
            "balanced_accuracy", "f1_macro", "confusion_matrix",
            "ci_95_lower", "ci_95_upper", "per_class_recall",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_evaluate_balanced_accuracy_range(self, tmp_path):
        """Balanced accuracy is in [0, 1]."""
        # Arrange
        n_classes = 3
        model = AccentCNN(n_classes=n_classes)
        train_loader, val_loader = _make_synthetic_loaders(n_classes=n_classes)
        config = TrainingConfig(
            n_epochs=1,
            patience=1,
            use_amp=False,
            device="cpu",
            checkpoint_dir=tmp_path / "ckpt",
            experiment_name="range_test",
        )
        train_classifier(model, train_loader, val_loader, config)

        label_names = ["NE", "S", "SE"]
        test_loader = val_loader

        # Act
        metrics = evaluate_classifier(
            model, test_loader, label_names, device="cpu", n_bootstrap=50,
        )

        # Assert
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
