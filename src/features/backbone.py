"""Qwen3-TTS backbone feature extraction (layer-wise).

Extracts hidden states from the talker component of Qwen3-TTS 1.7B-CustomVoice.
Probing these layers reveals where accent and speaker information is encoded.

REQUIRES GPU (model is ~3.4GB in fp16).
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

_backbone_model = None
_backbone_processor = None
_backbone_device = None


def _get_backbone_model(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device: str = "cuda",
) -> tuple:
    """Lazy-load Qwen3-TTS backbone.

    Note: This model requires significant VRAM (~4-6GB in fp16).
    """
    global _backbone_model, _backbone_processor, _backbone_device

    if _backbone_model is None or _backbone_device != device:
        _backbone_processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        _backbone_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            output_hidden_states=True,
        ).to(device).eval()
        _backbone_device = device

        # Log model architecture info
        if hasattr(_backbone_model, "config"):
            config = _backbone_model.config
            n_layers = getattr(config, "num_hidden_layers", "unknown")
            hidden_size = getattr(config, "hidden_size", "unknown")
            logger.info(
                f"Qwen3-TTS loaded on {device}: "
                f"{n_layers} layers, hidden_size={hidden_size}"
            )

    return _backbone_model, _backbone_processor, _backbone_device


def extract_backbone_features(
    audio_path: Path,
    text: str,
    layers: list[int] = (0, 4, 8, 12, 16, 20, 24, 27),
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device: str = "cuda",
    target_sr: int = 16000,
    pooling: str = "mean_temporal",
) -> dict[int, np.ndarray]:
    """Extract hidden states from Qwen3-TTS backbone layers.

    This extracts from the talker model, which processes both text and
    audio tokens through a decoder-only transformer.

    Args:
        audio_path: Path to audio file (used for speaker conditioning).
        text: Input text for the TTS model.
        layers: Which transformer layers to extract from.
        model_name: HuggingFace model identifier.
        device: "cpu" or "cuda" (cuda strongly recommended).
        target_sr: Target sampling rate.
        pooling: Temporal pooling method.

    Returns:
        Dict mapping layer index to numpy array of shape (hidden_dim,).
    """
    model, processor, _ = _get_backbone_model(model_name, device)

    waveform, sr = torchaudio.load(str(audio_path))
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Process through the model
    # Note: exact API depends on Qwen3-TTS implementation
    # This is a forward pass to get hidden states, NOT generation
    inputs = processor(
        text=text,
        audio=waveform.squeeze().numpy(),
        sampling_rate=target_sr,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

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
            pooled = hs.float().mean(dim=0)  # (hidden_dim,) - cast from fp16
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        result[layer_idx] = pooled.cpu().numpy()

    return result


def get_backbone_layer_count(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
) -> int | None:
    """Get number of transformer layers without loading the full model.

    Returns:
        Number of layers, or None if config not available.
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
        return getattr(config, "num_hidden_layers", None)
    except Exception as e:
        logger.warning(f"Could not read backbone config: {e}")
        return None
