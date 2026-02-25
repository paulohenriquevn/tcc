"""Training loop and evaluation for accent classifiers.

Provides a generic training function compatible with both AccentCNN
(mel-spectrogram input) and AccentWav2Vec2 (raw waveform input).
Uses early stopping on validation balanced accuracy, mixed precision,
and class-weighted loss for imbalanced accent distributions.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for classifier training.

    All hyperparameters are explicit and serializable for YAML logging.
    """

    learning_rate: float = 1e-3
    batch_size: int = 32
    n_epochs: int = 50
    patience: int = 10
    device: str = "cuda"
    seed: int = 42
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    experiment_name: str = "accent_classifier"
    use_amp: bool = True


@dataclass
class TrainingResult:
    """Summary of a completed training run."""

    best_epoch: int
    best_val_bal_acc: float
    best_checkpoint_path: Path
    train_losses: list[float]
    val_bal_accs: list[float]
    total_epochs_run: int


def compute_class_weights(labels: list[int], n_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Gives higher weight to underrepresented classes so the loss
    function penalizes misclassification of rare accents more.

    Args:
        labels: List of integer class labels from the training set.
        n_classes: Total number of classes.

    Returns:
        Float tensor of shape (n_classes,) with per-class weights.
    """
    if not labels:
        raise ValueError("labels must not be empty")

    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)

    # Avoid division by zero for classes with no samples
    counts = np.where(counts == 0, 1.0, counts)

    weights = len(labels) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    class_weights: torch.Tensor | None = None,
    resume_from: Path | None = None,
) -> TrainingResult:
    """Train an accent classifier with early stopping.

    Compatible with both AccentCNN and AccentWav2Vec2. The DataLoader
    must yield (input_tensor, label_idx) batches.

    Args:
        model: nn.Module to train (AccentCNN or AccentWav2Vec2).
        train_loader: DataLoader for training split.
        val_loader: DataLoader for validation split.
        config: Training hyperparameters.
        class_weights: Optional tensor of per-class weights for loss.
            Use compute_class_weights() to derive from label distribution.
        resume_from: Optional path to checkpoint for resuming training.

    Returns:
        TrainingResult with best checkpoint path and metric history.
    """
    set_global_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler(device.type, enabled=config.use_amp and device.type == "cuda")

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_losses: list[float] = []
    val_bal_accs: list[float] = []
    best_val_bal_acc = -1.0
    best_epoch = -1
    best_checkpoint_path = Path()
    epochs_without_improvement = 0
    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_from is not None and resume_from.exists():
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_bal_acc = checkpoint.get("val_bal_acc", -1.0)
        best_epoch = start_epoch
        best_checkpoint_path = resume_from
        logger.info(
            "Resumed from %s (epoch %d, val_bal_acc=%.4f)",
            resume_from, start_epoch, best_val_bal_acc,
        )

    logger.info(
        "Training %s: lr=%.1e, batch=%d, epochs=%d, patience=%d, amp=%s, device=%s",
        config.experiment_name,
        config.learning_rate,
        config.batch_size,
        config.n_epochs,
        config.patience,
        config.use_amp,
        device,
    )

    for epoch in range(start_epoch, config.n_epochs):
        # --- Training phase ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast(
                device_type=device.type,
                enabled=config.use_amp and device.type == "cuda",
            ):
                logits = model(inputs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # --- Validation phase ---
        val_bal_acc = _compute_val_balanced_accuracy(model, val_loader, device)
        val_bal_accs.append(val_bal_acc)

        vram_msg = ""
        if device.type == "cuda":
            alloc_gb = torch.cuda.memory_allocated(device) / 1e9
            max_gb = torch.cuda.max_memory_allocated(device) / 1e9
            vram_msg = f", vram_alloc={alloc_gb:.2f}GB, vram_peak={max_gb:.2f}GB"

        logger.info(
            "Epoch %d/%d — train_loss=%.4f, val_bal_acc=%.4f%s",
            epoch + 1,
            config.n_epochs,
            avg_loss,
            val_bal_acc,
            vram_msg,
        )

        # --- Checkpointing + early stopping ---
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            checkpoint_name = (
                f"{config.experiment_name}_epoch{best_epoch}_balacc{val_bal_acc:.4f}.pt"
            )
            best_checkpoint_path = config.checkpoint_dir / checkpoint_name

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": best_epoch,
                    "val_bal_acc": val_bal_acc,
                    "seed": config.seed,
                    "config": {
                        "learning_rate": config.learning_rate,
                        "batch_size": config.batch_size,
                        "n_epochs": config.n_epochs,
                        "patience": config.patience,
                        "seed": config.seed,
                        "experiment_name": config.experiment_name,
                        "use_amp": config.use_amp,
                    },
                },
                best_checkpoint_path,
            )
            logger.info("Saved best checkpoint: %s", best_checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d). "
                    "Best: epoch %d with val_bal_acc=%.4f",
                    epoch + 1,
                    config.patience,
                    best_epoch,
                    best_val_bal_acc,
                )
                break

    total_epochs_run = epoch + 1

    logger.info(
        "Training complete: %d epochs, best_epoch=%d, best_val_bal_acc=%.4f",
        total_epochs_run,
        best_epoch,
        best_val_bal_acc,
    )

    return TrainingResult(
        best_epoch=best_epoch,
        best_val_bal_acc=best_val_bal_acc,
        best_checkpoint_path=best_checkpoint_path,
        train_losses=train_losses,
        val_bal_accs=val_bal_accs,
        total_epochs_run=total_epochs_run,
    )


def _compute_val_balanced_accuracy(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute balanced accuracy on the validation set.

    Args:
        model: Trained model in eval mode.
        val_loader: Validation DataLoader.
        device: Computation device.

    Returns:
        Balanced accuracy as a float in [0, 1].
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    if not all_labels:
        logger.warning("Validation set is empty, returning bal_acc=0.0")
        return 0.0

    return balanced_accuracy_score(all_labels, all_preds)


def evaluate_classifier(
    model: nn.Module,
    test_loader: DataLoader,
    label_names: list[str],
    device: str = "cuda",
    n_bootstrap: int = 1000,
) -> dict:
    """Evaluate a trained classifier with bootstrap CI.

    Args:
        model: Trained nn.Module (must already be on device).
        test_loader: DataLoader for the test split.
        label_names: Ordered list of class label names matching indices.
        device: Device string ("cuda" or "cpu").
        n_bootstrap: Number of bootstrap samples for CI computation.

    Returns:
        Dict with keys: balanced_accuracy, f1_macro, per_class_recall,
        confusion_matrix, ci_95_lower, ci_95_upper.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(dev)
            logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    if not all_labels:
        raise ValueError("Test set is empty, cannot evaluate")

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Check for degenerate predictions
    unique_preds = set(y_pred.tolist())
    for idx, name in enumerate(label_names):
        if idx not in unique_preds:
            logger.warning(
                "Class '%s' (idx=%d) has 0 predictions — degenerate classifier",
                name,
                idx,
            )

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))

    # Per-class recall (row-normalized confusion matrix diagonal)
    per_class_recall = {}
    for idx, name in enumerate(label_names):
        row_sum = cm[idx].sum()
        recall = cm[idx, idx] / row_sum if row_sum > 0 else 0.0
        per_class_recall[name] = float(recall)

    # Bootstrap CI for balanced accuracy
    ci_lower, ci_upper = _bootstrap_ci(y_true, y_pred, n_bootstrap)

    logger.info(
        "Evaluation: bal_acc=%.4f (CI 95%%: [%.4f, %.4f]), f1_macro=%.4f",
        bal_acc,
        ci_lower,
        ci_upper,
        f1_mac,
    )

    return {
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_mac),
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
    }


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for balanced accuracy.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducible bootstrap sampling.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0

    rng = np.random.RandomState(seed)
    scores = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        scores[i] = balanced_accuracy_score(y_true[indices], y_pred[indices])

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(scores, 100 * alpha))
    upper = float(np.percentile(scores, 100 * (1.0 - alpha)))

    return lower, upper
