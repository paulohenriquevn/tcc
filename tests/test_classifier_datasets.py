"""Tests for classifier dataset classes (MelSpectrogramDataset and WaveformDataset).

Note: torchaudio.load/save requires torchcodec which may not be installed.
We mock torchaudio.load to return synthetic tensors directly, so tests
validate dataset logic (padding, truncation, label encoding) without
depending on the audio I/O backend.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.data.manifest import ManifestEntry
from src.classifier.mel_dataset import MelSpectrogramDataset
from src.classifier.wav2vec2_dataset import WaveformDataset


def _make_audio_entry(audio_path: str = "/fake/test.wav", accent: str = "SE", **overrides) -> ManifestEntry:
    """Create a ManifestEntry pointing to a (fake) audio file."""
    defaults = {
        "utt_id": "test_001",
        "audio_path": audio_path,
        "speaker_id": "spk_001",
        "accent": accent,
        "gender": "M",
        "duration_s": 5.0,
        "sampling_rate": 16000,
        "text_id": None,
        "source": "test",
        "birth_state": "SP",
    }
    defaults.update(overrides)
    return ManifestEntry(**defaults)


def _fake_torchaudio_load(duration_s: float = 5.0, sr: int = 16000):
    """Return a mock function that mimics torchaudio.load output."""
    def _load(path, *args, **kwargs):
        n_samples = int(duration_s * sr)
        waveform = torch.sin(
            2 * torch.pi * 440 * torch.arange(n_samples, dtype=torch.float32) / sr
        ).unsqueeze(0)  # (1, n_samples)
        return waveform, sr
    return _load


# ---------------------------------------------------------------------------
# MelSpectrogramDataset
# ---------------------------------------------------------------------------


class TestMelSpectrogramDataset:
    @patch("src.classifier.mel_dataset.torchaudio.load", side_effect=_fake_torchaudio_load(5.0))
    def test_output_shape(self, mock_load):
        """Default params produce mel of shape (1, 80, 300)."""
        # Arrange
        entry = _make_audio_entry()
        label_to_idx = {"SE": 0, "NE": 1}
        ds = MelSpectrogramDataset([entry], label_to_idx, n_mels=80, max_frames=300)

        # Act
        mel, label = ds[0]

        # Assert
        assert mel.shape == (1, 80, 300)

    @patch("src.classifier.mel_dataset.torchaudio.load", side_effect=_fake_torchaudio_load(1.0))
    def test_padding_short_audio(self, mock_load):
        """Audio shorter than max_frames is zero-padded to max_frames."""
        # Arrange
        entry = _make_audio_entry(duration_s=1.0)
        label_to_idx = {"SE": 0}
        ds = MelSpectrogramDataset([entry], label_to_idx, n_mels=80, max_frames=300)

        # Act
        mel, _ = ds[0]

        # Assert
        assert mel.shape == (1, 80, 300)

    @patch("src.classifier.mel_dataset.torchaudio.load", side_effect=_fake_torchaudio_load(10.0))
    def test_truncation_long_audio(self, mock_load):
        """Audio longer than max_frames is truncated to max_frames."""
        # Arrange
        entry = _make_audio_entry(duration_s=10.0)
        label_to_idx = {"SE": 0}
        ds = MelSpectrogramDataset([entry], label_to_idx, n_mels=80, max_frames=300)

        # Act
        mel, _ = ds[0]

        # Assert
        assert mel.shape == (1, 80, 300)

    @patch("src.classifier.mel_dataset.torchaudio.load", side_effect=_fake_torchaudio_load(5.0))
    def test_label_encoding(self, mock_load):
        """Returned label index matches label_to_idx mapping."""
        # Arrange
        entry = _make_audio_entry(accent="NE", birth_state="BA")
        label_to_idx = {"NE": 0, "SE": 1}
        ds = MelSpectrogramDataset([entry], label_to_idx)

        # Act
        _, label = ds[0]

        # Assert
        assert label == 0

    def test_build_label_mapping(self):
        """Static method returns sorted label-to-index mapping."""
        # Arrange
        entries = [
            _make_audio_entry(audio_path="/fake/u1.wav", utt_id="u1", accent="SE"),
            _make_audio_entry(audio_path="/fake/u2.wav", utt_id="u2", accent="NE", birth_state="BA"),
            _make_audio_entry(audio_path="/fake/u3.wav", utt_id="u3", accent="S", birth_state="RS"),
        ]

        # Act
        mapping = MelSpectrogramDataset.build_label_mapping(entries)

        # Assert
        assert mapping == {"NE": 0, "S": 1, "SE": 2}

    def test_empty_entries_raises(self):
        """Empty entry list raises ValueError."""
        with pytest.raises(ValueError, match="entries must not be empty"):
            MelSpectrogramDataset([], {"SE": 0})


# ---------------------------------------------------------------------------
# WaveformDataset
# ---------------------------------------------------------------------------


class TestWaveformDataset:
    @patch("src.classifier.wav2vec2_dataset.torchaudio.load", side_effect=_fake_torchaudio_load(5.0))
    def test_output_shape(self, mock_load):
        """Waveform output shape is (max_samples,) for max_length_s=15.0."""
        # Arrange
        entry = _make_audio_entry()
        label_to_idx = {"SE": 0}
        ds = WaveformDataset([entry], label_to_idx, max_length_s=15.0, sample_rate=16000)

        # Act
        waveform, _ = ds[0]

        # Assert
        expected_samples = int(15.0 * 16000)
        assert waveform.shape == (expected_samples,)

    @patch("src.classifier.wav2vec2_dataset.torchaudio.load", side_effect=_fake_torchaudio_load(1.0))
    def test_padding_short_audio(self, mock_load):
        """Short audio is zero-padded to max_length_s * sample_rate samples."""
        # Arrange
        entry = _make_audio_entry(duration_s=1.0)
        label_to_idx = {"SE": 0}
        ds = WaveformDataset([entry], label_to_idx, max_length_s=15.0, sample_rate=16000)

        # Act
        waveform, _ = ds[0]

        # Assert
        expected_samples = int(15.0 * 16000)
        assert waveform.shape == (expected_samples,)

    @patch("src.classifier.wav2vec2_dataset.torchaudio.load", side_effect=_fake_torchaudio_load(5.0))
    def test_label_encoding(self, mock_load):
        """Returned label index matches label_to_idx mapping."""
        # Arrange
        entry = _make_audio_entry(accent="NE", birth_state="BA")
        label_to_idx = {"NE": 1, "SE": 0}
        ds = WaveformDataset([entry], label_to_idx)

        # Act
        _, label = ds[0]

        # Assert
        assert label == 1

    def test_empty_entries_raises(self):
        """Empty entry list raises ValueError."""
        with pytest.raises(ValueError, match="entries must not be empty"):
            WaveformDataset([], {"SE": 0})
