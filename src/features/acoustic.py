"""Acoustic feature extraction: MFCC, pitch, energy, speech rate.

Low-level features used as a baseline for probing.
These do NOT require GPU — run on CPU.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AcousticFeatures:
    """Aggregated acoustic features for a single utterance."""
    utt_id: str
    mfcc_mean: np.ndarray       # (n_mfcc,)
    mfcc_std: np.ndarray        # (n_mfcc,)
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    energy_std: float
    speech_rate: float           # syllables/sec estimate
    duration_s: float


def extract_acoustic_features(
    audio_path: Path,
    utt_id: str,
    sr: int = 16000,
    n_mfcc: int = 13,
) -> AcousticFeatures:
    """Extract aggregated acoustic features from a single audio file.

    Args:
        audio_path: Path to WAV file.
        utt_id: Utterance identifier.
        sr: Target sampling rate (resamples if different).
        n_mfcc: Number of MFCC coefficients.

    Returns:
        AcousticFeatures with aggregated statistics.
    """
    y, actual_sr = librosa.load(audio_path, sr=sr)
    duration_s = len(y) / sr

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Pitch (F0) via yin (deterministic, ~10-50x faster than pyin)
    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C7")
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
    # Filter out-of-range estimates (yin returns fmin/fmax for unvoiced frames)
    f0_voiced = f0[(f0 > fmin) & (f0 < fmax)]
    pitch_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    pitch_std = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0

    # Energy (RMS)
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    # Speech rate estimate (onset-based)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    speech_rate = len(onsets) / duration_s if duration_s > 0 else 0.0

    return AcousticFeatures(
        utt_id=utt_id,
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        energy_mean=energy_mean,
        energy_std=energy_std,
        speech_rate=speech_rate,
        duration_s=duration_s,
    )


def features_to_vector(features: AcousticFeatures) -> np.ndarray:
    """Flatten acoustic features into a single vector for probing.

    Returns:
        1D numpy array: [mfcc_mean(13), mfcc_std(13), pitch_mean, pitch_std,
                         energy_mean, energy_std, speech_rate] = 31 dims.
    """
    return np.concatenate([
        features.mfcc_mean,
        features.mfcc_std,
        [features.pitch_mean, features.pitch_std],
        [features.energy_mean, features.energy_std],
        [features.speech_rate],
    ])
