"""ECAPA-TDNN speaker embedding extraction via SpeechBrain.

Used for:
1. Speaker probing (does the model encode speaker identity?)
2. Baseline intra/inter speaker similarity (reference for Stage 2)

Model: SpeechBrain spkrec-ecapa-voxceleb (192-dim embeddings)
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

# Lazy-loaded model to avoid import-time GPU allocation
_ecapa_model = None
_ecapa_device = None


def _get_ecapa_model(device: str = "cpu") -> tuple:
    """Lazy-load ECAPA-TDNN model.

    Returns:
        Tuple of (classifier, device_str).
    """
    global _ecapa_model, _ecapa_device

    if _ecapa_model is None or _ecapa_device != device:
        from speechbrain.inference.speaker import EncoderClassifier

        _ecapa_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        _ecapa_device = device
        logger.info(f"ECAPA-TDNN loaded on {device}")

    return _ecapa_model, _ecapa_device


def extract_ecapa_embedding(
    audio_path: Path,
    device: str = "cpu",
    target_sr: int = 16000,
) -> np.ndarray:
    """Extract ECAPA-TDNN speaker embedding from audio file.

    Args:
        audio_path: Path to audio file.
        device: "cpu" or "cuda".
        target_sr: Target sampling rate (ECAPA expects 16kHz).

    Returns:
        Numpy array of shape (192,) — speaker embedding.
    """
    model, _ = _get_ecapa_model(device)

    waveform, sr = torchaudio.load(str(audio_path))

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    with torch.no_grad():
        embedding = model.encode_batch(waveform.to(device))

    # Shape: (1, 1, 192) -> (192,)
    return embedding.squeeze().cpu().numpy()


def compute_cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings.

    Args:
        emb_a: First embedding vector.
        emb_b: Second embedding vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))


def compute_speaker_similarity_baseline(
    speaker_embeddings: dict[str, list[np.ndarray]],
    seed: int = 42,
) -> dict[str, dict]:
    """Compute intra-speaker and inter-speaker similarity baselines.

    Args:
        speaker_embeddings: Dict mapping speaker_id to list of embeddings.

    Returns:
        Dict with 'intra' and 'inter' similarity stats:
        {
            'intra': {'mean': float, 'std': float, 'values': list},
            'inter': {'mean': float, 'std': float, 'values': list},
        }
    """
    # Intra-speaker: similarity between different utterances of the same speaker
    intra_sims = []
    for speaker_id, embeddings in speaker_embeddings.items():
        if len(embeddings) < 2:
            continue
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = compute_cosine_similarity(embeddings[i], embeddings[j])
                intra_sims.append(sim)

    # Inter-speaker: similarity between different speakers
    # Sample pairs to keep computation manageable
    speakers = list(speaker_embeddings.keys())
    inter_sims = []
    rng = np.random.RandomState(seed)

    max_inter_pairs = min(5000, len(speakers) * (len(speakers) - 1) // 2)
    pair_count = 0

    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            if pair_count >= max_inter_pairs:
                break
            embs_i = speaker_embeddings[speakers[i]]
            embs_j = speaker_embeddings[speakers[j]]

            # Pick one random embedding from each speaker
            emb_a = embs_i[rng.randint(len(embs_i))]
            emb_b = embs_j[rng.randint(len(embs_j))]
            sim = compute_cosine_similarity(emb_a, emb_b)
            inter_sims.append(sim)
            pair_count += 1

    intra_arr = np.array(intra_sims) if intra_sims else np.array([0.0])
    inter_arr = np.array(inter_sims) if inter_sims else np.array([0.0])

    return {
        "intra": {
            "mean": float(np.mean(intra_arr)),
            "std": float(np.std(intra_arr)),
            "n_pairs": len(intra_sims),
            "values": intra_sims,
        },
        "inter": {
            "mean": float(np.mean(inter_arr)),
            "std": float(np.std(inter_arr)),
            "n_pairs": len(inter_sims),
            "values": inter_sims,
        },
    }
