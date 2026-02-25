"""Inference utilities for trained accent classifiers.

Used in Stages 2-3 to evaluate accent controllability on generated audio.
Supports single-file and batch classification with both CNN and Wav2Vec2
model types.
"""

import logging
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_classifier(
    checkpoint_path: Path,
    model_class: type,
    n_classes: int,
    device: str = "cuda",
    **model_kwargs,
) -> nn.Module:
    """Load a trained classifier from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model_class: Model class (AccentCNN or AccentWav2Vec2).
        n_classes: Number of accent classes the model was trained on.
        device: Target device for the loaded model.
        **model_kwargs: Additional keyword arguments for the model constructor.

    Returns:
        Model in eval mode on the specified device.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    model = model_class(n_classes=n_classes, **model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(dev)
    model.eval()

    logger.info(
        "Loaded %s from %s (epoch=%s, val_bal_acc=%s)",
        model_class.__name__,
        checkpoint_path,
        checkpoint.get("epoch", "?"),
        checkpoint.get("val_bal_acc", "?"),
    )

    return model


def classify_audio(
    model: nn.Module,
    audio_path: Path,
    transform_fn: Callable[[Path], torch.Tensor],
    label_names: list[str],
    device: str = "cuda",
) -> dict:
    """Classify a single audio file.

    Args:
        model: Trained classifier in eval mode.
        audio_path: Path to the audio file.
        transform_fn: Function that takes an audio Path and returns a tensor
            ready for the model (with batch dimension added if needed).
            For AccentCNN: should return shape (1, 1, n_mels, n_frames).
            For AccentWav2Vec2: should return shape (1, n_samples).
        label_names: Ordered list of class label names matching model output.
        device: Device for inference.

    Returns:
        Dict with keys: predicted_label (str), probabilities (dict),
        logits (list of floats).

    Raises:
        FileNotFoundError: If audio_path does not exist.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    input_tensor = transform_fn(audio_path).to(dev)

    # Ensure batch dimension
    if input_tensor.dim() == len(input_tensor.shape) and input_tensor.shape[0] != 1:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)

    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = probs.argmax().item()

    probabilities = {
        name: float(probs[i]) for i, name in enumerate(label_names)
    }

    return {
        "predicted_label": label_names[pred_idx],
        "probabilities": probabilities,
        "logits": logits.squeeze(0).cpu().tolist(),
    }


def classify_batch(
    model: nn.Module,
    audio_paths: list[Path],
    transform_fn: Callable[[Path], torch.Tensor],
    label_names: list[str],
    device: str = "cuda",
    batch_size: int = 32,
) -> list[dict]:
    """Classify multiple audio files in batches.

    Args:
        model: Trained classifier in eval mode.
        audio_paths: List of paths to audio files.
        transform_fn: Function that takes an audio Path and returns a tensor
            (without batch dimension).
        label_names: Ordered list of class label names matching model output.
        device: Device for inference.
        batch_size: Number of files to process per batch.

    Returns:
        List of classification dicts (same format as classify_audio output),
        one per audio file. Failed files are included with predicted_label=None.
    """
    if not audio_paths:
        return []

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    results: list[dict] = []

    # Process in batches
    for batch_start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[batch_start:batch_start + batch_size]
        tensors: list[torch.Tensor] = []
        valid_indices: list[int] = []

        for i, path in enumerate(batch_paths):
            try:
                tensor = transform_fn(path)
                tensors.append(tensor)
                valid_indices.append(i)
            except Exception:
                logger.warning("Failed to load audio: %s", path, exc_info=True)

        # Add placeholder results for failed files
        batch_results: list[dict | None] = [None] * len(batch_paths)

        if tensors:
            batch_tensor = torch.stack(tensors).to(dev)

            with torch.no_grad():
                logits = model(batch_tensor)

            probs = torch.softmax(logits, dim=1)

            for tensor_idx, path_idx in enumerate(valid_indices):
                pred_idx = probs[tensor_idx].argmax().item()
                probabilities = {
                    name: float(probs[tensor_idx, i])
                    for i, name in enumerate(label_names)
                }
                batch_results[path_idx] = {
                    "predicted_label": label_names[pred_idx],
                    "probabilities": probabilities,
                    "logits": logits[tensor_idx].cpu().tolist(),
                }

        # Fill in failed entries
        for i in range(len(batch_paths)):
            if batch_results[i] is None:
                batch_results[i] = {
                    "predicted_label": None,
                    "probabilities": {},
                    "logits": [],
                }

        results.extend(batch_results)

    logger.info(
        "Batch classification: %d files, %d successful",
        len(audio_paths),
        sum(1 for r in results if r["predicted_label"] is not None),
    )

    return results
