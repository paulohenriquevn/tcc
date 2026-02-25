"""Mel-spectrogram dataset for CNN-based accent classification.

Loads audio from ManifestEntry paths, computes mel-spectrograms,
and returns fixed-length tensors suitable for Conv2d input.
"""

import logging
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset

from src.data.manifest import ManifestEntry

logger = logging.getLogger(__name__)


class MelSpectrogramDataset(Dataset):
    """Dataset that yields (mel_spectrogram, label_idx) pairs.

    Each mel-spectrogram is zero-padded or truncated to max_frames,
    producing tensors of shape (1, n_mels, max_frames).

    Args:
        entries: Manifest entries with audio_path and accent fields.
        label_to_idx: Mapping from accent label string to integer index.
        n_mels: Number of mel filterbank channels.
        max_frames: Maximum number of time frames (pad/truncate target).
        sample_rate: Target sampling rate in Hz.
    """

    def __init__(
        self,
        entries: list[ManifestEntry],
        label_to_idx: dict[str, int],
        n_mels: int = 80,
        max_frames: int = 300,
        sample_rate: int = 16000,
    ) -> None:
        if not entries:
            raise ValueError("entries must not be empty")
        if not label_to_idx:
            raise ValueError("label_to_idx must not be empty")

        self.entries = entries
        self.label_to_idx = label_to_idx
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.sample_rate = sample_rate

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=512,
        )

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

        # Compute mel-spectrogram: (1, n_mels, n_frames)
        mel = self.mel_transform(waveform)

        # Pad or truncate along the time axis to max_frames
        n_frames = mel.shape[-1]
        if n_frames < self.max_frames:
            pad_size = self.max_frames - n_frames
            mel = torch.nn.functional.pad(mel, (0, pad_size))
        elif n_frames > self.max_frames:
            mel = mel[..., :self.max_frames]

        label_idx = self.label_to_idx[entry.accent]

        return mel, label_idx

    @staticmethod
    def build_label_mapping(entries: list[ManifestEntry]) -> dict[str, int]:
        """Create a sorted label_to_idx mapping from manifest entries.

        Args:
            entries: Manifest entries with accent fields.

        Returns:
            Dict mapping accent label strings to integer indices.
        """
        labels = sorted({e.accent for e in entries})
        return {label: idx for idx, label in enumerate(labels)}
