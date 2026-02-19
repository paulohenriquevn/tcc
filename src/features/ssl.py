"""WavLM SSL feature extraction (layer-wise).

Extracts hidden states from specific layers of WavLM-Large.
Used for probing what information is encoded at different depths
of a self-supervised speech model.

Requires GPU for practical speed, but works on CPU.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

_ssl_model = None
_ssl_processor = None
_ssl_device = None


def _get_ssl_model(
    model_name: str = "microsoft/wavlm-large",
    device: str = "cpu",
) -> tuple:
    """Lazy-load WavLM model."""
    global _ssl_model, _ssl_processor, _ssl_device

    if _ssl_model is None or _ssl_device != device:
        _ssl_processor = AutoProcessor.from_pretrained(model_name)
        _ssl_model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        ).to(device).eval()
        _ssl_device = device
        logger.info(f"WavLM loaded on {device} ({model_name})")

    return _ssl_model, _ssl_processor, _ssl_device


def extract_ssl_features(
    audio_path: Path,
    layers: list[int] = (0, 6, 12, 18, 24),
    model_name: str = "microsoft/wavlm-large",
    device: str = "cpu",
    target_sr: int = 16000,
    pooling: str = "mean_temporal",
) -> dict[int, np.ndarray]:
    """Extract WavLM hidden states from specified layers.

    Args:
        audio_path: Path to audio file.
        layers: Which layers to extract (0=input embeddings, 24=last).
        model_name: HuggingFace model identifier.
        device: "cpu" or "cuda".
        target_sr: Target sampling rate.
        pooling: "mean_temporal" averages over time dimension.

    Returns:
        Dict mapping layer index to numpy array of shape (hidden_dim,).
    """
    model, processor, _ = _get_ssl_model(model_name, device)

    waveform, sr = torchaudio.load(str(audio_path))
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=target_sr,
        return_tensors="pt",
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)

    # hidden_states: tuple of (n_layers+1, batch, seq_len, hidden_dim)
    hidden_states = outputs.hidden_states

    result = {}
    for layer_idx in layers:
        if layer_idx >= len(hidden_states):
            logger.warning(
                f"Layer {layer_idx} out of range (max {len(hidden_states)-1}), skipping"
            )
            continue

        hs = hidden_states[layer_idx].squeeze(0)  # (seq_len, hidden_dim)

        if pooling == "mean_temporal":
            pooled = hs.mean(dim=0)  # (hidden_dim,)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        result[layer_idx] = pooled.cpu().numpy()

    return result
