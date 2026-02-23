"""Qwen3-TTS backbone feature extraction (layer-wise).

Extracts hidden states from the talker component of Qwen3-TTS 1.7B-CustomVoice.
Probing these layers reveals where accent and speaker information is encoded.

Architecture (from config.json):
  - Talker: 28 transformer layers, hidden_size=2048, 16 attention heads
  - Code predictor: 5 layers, hidden_size=1024
  - Model type: Qwen3TTSForConditionalGeneration
  - Package: qwen-tts (pip install qwen-tts)

REQUIRES GPU (~4-6GB in bf16). REQUIRES COLAB SMOKE TEST before first use.
The hook-based extraction needs validation against the real model.
"""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Architecture constants from config.json
TALKER_NUM_LAYERS = 28
TALKER_HIDDEN_SIZE = 2048

_backbone_model = None
_backbone_device = None
_hooks = []
_captured_hidden_states = {}


def _get_backbone_model(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device: str = "cuda",
) -> "Qwen3TTSModel":
    """Lazy-load Qwen3-TTS model via qwen_tts package.

    Note: Requires ~4-6GB VRAM in bf16.
    """
    global _backbone_model, _backbone_device

    if _backbone_model is not None and _backbone_device == device:
        return _backbone_model

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        raise ImportError(
            "qwen-tts package not installed. "
            "Install with: pip install qwen-tts"
        )

    _backbone_model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.bfloat16,
    )
    _backbone_device = device

    logger.info(
        f"Qwen3-TTS loaded on {device}: "
        f"talker={TALKER_NUM_LAYERS} layers, "
        f"hidden_size={TALKER_HIDDEN_SIZE}"
    )

    return _backbone_model


def _find_talker_layers(model) -> list:
    """Discover transformer layers inside the talker component.

    The Qwen3-TTS architecture nests the talker transformer inside
    the main model. This function searches for the layer list by
    inspecting common attribute patterns.

    Returns:
        List of nn.Module transformer layers, or empty list if not found.
    """
    # Search patterns for the talker's transformer layers
    candidates = [
        "model.talker.layers",
        "model.talker.model.layers",
        "talker.layers",
        "talker.model.layers",
        "model.model.layers",
        "model.layers",
    ]

    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, "__len__") and len(obj) > 0:
                logger.info(
                    f"Found talker layers at '{path}': "
                    f"{len(obj)} layers"
                )
                return list(obj)
        except AttributeError:
            continue

    # Fallback: search all named modules for a ModuleList with ~28 items
    for name, module in model.named_modules():
        if hasattr(module, "__len__"):
            try:
                if 20 <= len(module) <= 32:
                    logger.info(
                        f"Found candidate layer list at '{name}': "
                        f"{len(module)} items"
                    )
                    return list(module)
            except TypeError:
                continue

    logger.warning(
        "Could not find talker layers. "
        "Run smoke test on Colab to discover the correct path."
    )
    return []


def _register_hooks(layers: list, target_indices: list[int]) -> None:
    """Register forward hooks on target layers to capture hidden states."""
    global _hooks, _captured_hidden_states

    _remove_hooks()
    _captured_hidden_states = {}

    for layer_idx in target_indices:
        if layer_idx >= len(layers):
            logger.warning(
                f"Layer {layer_idx} out of range "
                f"(max {len(layers)-1}), skipping"
            )
            continue

        def make_hook(idx):
            def hook_fn(module, input, output):
                # Output is typically a tuple; first element is hidden states
                if isinstance(output, tuple):
                    hs = output[0]
                else:
                    hs = output
                _captured_hidden_states[idx] = hs.detach()
            return hook_fn

        h = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        _hooks.append(h)


def _remove_hooks() -> None:
    """Remove all registered hooks."""
    global _hooks
    for h in _hooks:
        h.remove()
    _hooks = []


def extract_backbone_features(
    audio_path: Path,
    text: str,
    layers: list[int] = (0, 4, 8, 12, 16, 20, 24, 27),
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device: str = "cuda",
    target_sr: int = 16000,
    pooling: str = "mean_temporal",
) -> dict[int, np.ndarray]:
    """Extract hidden states from Qwen3-TTS talker layers.

    Uses forward hooks on the talker transformer to capture hidden
    states at specified layers during voice clone processing.

    EXPERIMENTAL: This function needs a smoke test on Colab.
    The hook targets and processing pipeline may need adjustment
    based on the actual qwen_tts internal API.

    Args:
        audio_path: Path to audio file (reference for voice cloning).
        text: Input text (used as generation context).
        layers: Which talker layers to extract from (0-27).
        model_name: HuggingFace model identifier.
        device: "cuda" strongly recommended.
        target_sr: Target sampling rate.
        pooling: Temporal pooling method ("mean_temporal").

    Returns:
        Dict mapping layer index to numpy array of shape (hidden_dim,).
        hidden_dim is 2048 for the talker.
    """
    model = _get_backbone_model(model_name, device)

    # Find and hook into talker layers
    talker_layers = _find_talker_layers(model)
    if not talker_layers:
        logger.error(
            "Cannot extract features: talker layers not found. "
            "See BACKBONE_SMOKE_TEST instructions."
        )
        return {}

    _register_hooks(talker_layers, list(layers))

    try:
        # Use the Base model's voice clone path to process audio.
        # This feeds reference audio through the model's internal pipeline,
        # which lets us capture hidden states via hooks.
        #
        # NOTE: generate_voice_clone is on the Base model.
        # CustomVoice uses generate_custom_voice with predefined speakers.
        # For feature extraction, we attempt voice cloning to trigger
        # a forward pass that processes the reference audio.
        with torch.no_grad():
            model.generate_voice_clone(
                text=text,
                language="Portuguese",
                ref_audio=str(audio_path),
                ref_text=text,
            )
    except AttributeError:
        # CustomVoice model may not have generate_voice_clone.
        # Try generate_custom_voice with a short generation.
        try:
            with torch.no_grad():
                model.generate_custom_voice(
                    text=text,
                    language="Portuguese",
                    speaker="Vivian",
                )
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            _remove_hooks()
            return {}
    except Exception as e:
        logger.error(f"Voice clone forward pass failed: {e}")
        _remove_hooks()
        return {}

    _remove_hooks()

    # Pool captured hidden states
    result = {}
    for layer_idx, hs in _captured_hidden_states.items():
        if hs.dim() == 3:
            hs = hs.squeeze(0)  # (seq_len, hidden_dim)
        elif hs.dim() == 1:
            result[layer_idx] = hs.float().cpu().numpy()
            continue

        if pooling == "mean_temporal":
            pooled = hs.float().mean(dim=0)  # (hidden_dim,)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        result[layer_idx] = pooled.cpu().numpy()

    _captured_hidden_states.clear()

    if not result:
        logger.warning(
            "No hidden states captured. The hook targets may be wrong. "
            "Run the smoke test to verify layer discovery."
        )

    return result


def get_backbone_layer_count(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
) -> int:
    """Get number of talker transformer layers.

    Returns 28 (known from config.json) without loading the model.
    """
    return TALKER_NUM_LAYERS


def smoke_test_backbone(device: str = "cuda") -> dict:
    """Run diagnostic smoke test to validate backbone feature extraction.

    Call this on Colab BEFORE running the full pipeline.
    It loads the model, discovers layers, and reports what it finds.

    Returns:
        Dict with diagnostic information.
    """
    result = {
        "model_loaded": False,
        "talker_layers_found": 0,
        "layer_path": None,
        "sample_hidden_size": None,
        "errors": [],
    }

    try:
        model = _get_backbone_model(device=device)
        result["model_loaded"] = True
    except Exception as e:
        result["errors"].append(f"Model load failed: {e}")
        return result

    layers = _find_talker_layers(model)
    result["talker_layers_found"] = len(layers)

    if layers:
        # Try to get hidden size from first layer
        for name, param in layers[0].named_parameters():
            if "weight" in name:
                result["sample_hidden_size"] = param.shape[-1]
                break

    # Print model structure for debugging
    print("=== Qwen3-TTS Model Structure ===")
    for name, module in model.named_children():
        print(f"  {name}: {type(module).__name__}")
        for child_name, child in module.named_children():
            print(f"    {child_name}: {type(child).__name__}")

    return result
