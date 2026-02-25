"""Waveform dataset for Wav2Vec2-based accent classification.

Loads raw audio waveforms from ManifestEntry paths, resamples to the
target rate, and pads/truncates to a fixed length.
"""

import logging
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset

from src.data.manifest import ManifestEntry

logger = logging.getLogger(__name__)


class WaveformDataset(Dataset):
    """Dataset that yields (waveform_1d, label_idx) pairs.

    Waveforms are resampled, converted to mono, and padded/truncated
    to a fixed number of samples determined by max_length_s * sample_rate.

    Args:
        entries: Manifest entries with audio_path and accent fields.
        label_to_idx: Mapping from accent label string to integer index.
        max_length_s: Maximum audio length in seconds (pad/truncate target).
        sample_rate: Target sampling rate in Hz.
    """

    def __init__(
        self,
        entries: list[ManifestEntry],
        label_to_idx: dict[str, int],
        max_length_s: float = 15.0,
        sample_rate: int = 16000,
    ) -> None:
        if not entries:
            raise ValueError("entries must not be empty")
        if not label_to_idx:
            raise ValueError("label_to_idx must not be empty")

        self.entries = entries
        self.label_to_idx = label_to_idx
        self.max_length_s = max_length_s
        self.sample_rate = sample_rate
        self.max_samples = int(max_length_s * sample_rate)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        entry = self.entries[idx]
        audio_path = Path(entry.audio_path)

        waveform, sr = torchaudio.load(str(audio_path))

        # Resample if source rate differs from target
        if sr != self.sample_rate:
            waveform = F.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Squeeze to 1D: (n_samples,)
        waveform = waveform.squeeze(0)

        # Pad or truncate to max_samples
        n_samples = waveform.shape[0]
        if n_samples < self.max_samples:
            pad_size = self.max_samples - n_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        elif n_samples > self.max_samples:
            waveform = waveform[:self.max_samples]

        label_idx = self.label_to_idx[entry.accent]

        return waveform, label_idx
