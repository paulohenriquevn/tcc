"""Tests for manifest_builder filters."""

import pytest

from src.data.manifest import ManifestEntry
from src.data.manifest_builder import _filter_speakers_by_utterance_count


def _make_entry(**overrides) -> ManifestEntry:
    """Factory for ManifestEntry with sensible defaults."""
    defaults = {
        "utt_id": "utt_001",
        "audio_path": "audio/utt_001.wav",
        "speaker_id": "spk_001",
        "accent": "SE",
        "gender": "M",
        "duration_s": 5.0,
        "sampling_rate": 16000,
        "text_id": "txt_001",
        "source": "CORAA-MUPE",
        "birth_state": "SP",
    }
    defaults.update(overrides)
    return ManifestEntry(**defaults)


class TestFilterSpeakersByUtteranceCount:
    def test_keeps_speakers_above_threshold(self):
        entries = [
            _make_entry(utt_id="u1", speaker_id="spk_A"),
            _make_entry(utt_id="u2", speaker_id="spk_A"),
            _make_entry(utt_id="u3", speaker_id="spk_A"),
            _make_entry(utt_id="u4", speaker_id="spk_B", accent="NE", birth_state="BA"),
            _make_entry(utt_id="u5", speaker_id="spk_B", accent="NE", birth_state="BA"),
            _make_entry(utt_id="u6", speaker_id="spk_B", accent="NE", birth_state="BA"),
        ]
        filtered, dropped = _filter_speakers_by_utterance_count(entries, min_utterances=3)
        assert len(filtered) == 6
        assert dropped == []

    def test_drops_speakers_below_threshold(self):
        entries = [
            _make_entry(utt_id="u1", speaker_id="spk_A"),
            _make_entry(utt_id="u2", speaker_id="spk_A"),
            _make_entry(utt_id="u3", speaker_id="spk_A"),
            _make_entry(utt_id="u4", speaker_id="spk_B", accent="NE", birth_state="BA"),
        ]
        filtered, dropped = _filter_speakers_by_utterance_count(entries, min_utterances=3)
        assert len(filtered) == 3
        assert "spk_B" in dropped
        assert all(e.speaker_id == "spk_A" for e in filtered)

    def test_drops_multiple_speakers(self):
        entries = [
            _make_entry(utt_id="u1", speaker_id="spk_A"),
            _make_entry(utt_id="u2", speaker_id="spk_A"),
            _make_entry(utt_id="u3", speaker_id="spk_A"),
            _make_entry(utt_id="u4", speaker_id="spk_B", accent="NE", birth_state="BA"),
            _make_entry(utt_id="u5", speaker_id="spk_C", accent="S", birth_state="RS"),
            _make_entry(utt_id="u6", speaker_id="spk_C", accent="S", birth_state="RS"),
        ]
        filtered, dropped = _filter_speakers_by_utterance_count(entries, min_utterances=3)
        assert len(filtered) == 3
        assert set(dropped) == {"spk_B", "spk_C"}

    def test_threshold_1_keeps_all(self):
        entries = [
            _make_entry(utt_id="u1", speaker_id="spk_A"),
            _make_entry(utt_id="u2", speaker_id="spk_B", accent="NE", birth_state="BA"),
        ]
        filtered, dropped = _filter_speakers_by_utterance_count(entries, min_utterances=1)
        assert len(filtered) == 2
        assert dropped == []

    def test_raises_when_all_dropped(self):
        entries = [
            _make_entry(utt_id="u1", speaker_id="spk_A"),
            _make_entry(utt_id="u2", speaker_id="spk_B", accent="NE", birth_state="BA"),
        ]
        with pytest.raises(ValueError, match="No entries remain"):
            _filter_speakers_by_utterance_count(entries, min_utterances=5)

    def test_exact_threshold_kept(self):
        """Speaker with exactly min_utterances should be kept."""
        entries = [
            _make_entry(utt_id="u1", speaker_id="spk_A"),
            _make_entry(utt_id="u2", speaker_id="spk_A"),
            _make_entry(utt_id="u3", speaker_id="spk_A"),
        ]
        filtered, dropped = _filter_speakers_by_utterance_count(entries, min_utterances=3)
        assert len(filtered) == 3
        assert dropped == []

    def test_returns_correct_dropped_ids(self):
        entries = [
            _make_entry(utt_id="u1", speaker_id="spk_A"),
            _make_entry(utt_id="u2", speaker_id="spk_A"),
            _make_entry(utt_id="u3", speaker_id="spk_A"),
            _make_entry(utt_id="u4", speaker_id="spk_B", accent="NE", birth_state="BA"),
            _make_entry(utt_id="u5", speaker_id="spk_B", accent="NE", birth_state="BA"),
        ]
        _, dropped = _filter_speakers_by_utterance_count(entries, min_utterances=3)
        assert dropped == ["spk_B"]
