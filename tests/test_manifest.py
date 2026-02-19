"""Tests for manifest schema validation and I/O."""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.manifest import (
    ManifestEntry,
    compute_file_hash,
    read_manifest,
    validate_manifest_consistency,
    write_manifest,
)


def _make_entry(**overrides) -> ManifestEntry:
    """Factory for ManifestEntry with sensible defaults."""
    defaults = {
        "utt_id": "utt_001",
        "audio_path": "audio/utt_001.wav",
        "speaker_id": "spk_001",
        "accent": "SE",
        "gender": "M",
        "duration_s": 5.0,
        "text_id": "txt_001",
        "source": "CORAA-MUPE",
        "birth_state": "SP",
    }
    defaults.update(overrides)
    return ManifestEntry(**defaults)


class TestManifestEntryValidation:
    def test_valid_entry_creates_successfully(self):
        entry = _make_entry()
        assert entry.accent == "SE"
        assert entry.gender == "M"

    def test_invalid_accent_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid accent"):
            _make_entry(accent="XX")

    def test_invalid_gender_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid gender"):
            _make_entry(gender="X")

    def test_negative_duration_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            _make_entry(duration_s=-1.0)

    def test_zero_duration_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            _make_entry(duration_s=0.0)

    def test_all_valid_accents_accepted(self):
        for accent, state in [("N", "AM"), ("NE", "BA"), ("CO", "GO"), ("SE", "SP"), ("S", "RS")]:
            entry = _make_entry(accent=accent, birth_state=state)
            assert entry.accent == accent

    def test_frozen_dataclass_prevents_mutation(self):
        entry = _make_entry()
        with pytest.raises(AttributeError):
            entry.accent = "NE"


class TestManifestIO:
    def test_write_and_read_roundtrip(self):
        entries = [
            _make_entry(utt_id="utt_001", speaker_id="spk_001"),
            _make_entry(utt_id="utt_002", speaker_id="spk_002", accent="NE", birth_state="BA"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            sha256 = write_manifest(entries, path)

            assert path.exists()
            assert len(sha256) == 64  # SHA-256 hex

            # Hash sidecar exists
            hash_path = path.with_suffix(".jsonl.sha256")
            assert hash_path.exists()

            # Roundtrip
            loaded = read_manifest(path)
            assert len(loaded) == 2
            assert loaded[0].utt_id == "utt_001"
            assert loaded[1].accent == "NE"

    def test_hash_is_deterministic(self):
        entries = [_make_entry()]

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "m1.jsonl"
            path2 = Path(tmpdir) / "m2.jsonl"

            h1 = write_manifest(entries, path1)
            h2 = write_manifest(entries, path2)

            assert h1 == h2

    def test_read_invalid_json_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            path.write_text("not json\n")

            with pytest.raises(ValueError, match="Invalid manifest entry"):
                read_manifest(path)


class TestManifestConsistency:
    def test_valid_manifest_returns_no_errors(self):
        entries = [
            _make_entry(utt_id="u1", speaker_id="s1", accent="SE"),
            _make_entry(utt_id="u2", speaker_id="s1", accent="SE"),
            _make_entry(utt_id="u3", speaker_id="s2", accent="NE", birth_state="BA"),
            _make_entry(utt_id="u4", speaker_id="s3", accent="S", birth_state="RS"),
            _make_entry(utt_id="u5", speaker_id="s4", accent="N", birth_state="AM"),
            _make_entry(utt_id="u6", speaker_id="s5", accent="CO", birth_state="GO"),
        ]
        errors = validate_manifest_consistency(entries)
        assert errors == []

    def test_duplicate_utt_id_detected(self):
        entries = [
            _make_entry(utt_id="u1", speaker_id="s1"),
            _make_entry(utt_id="u1", speaker_id="s2", accent="NE", birth_state="BA"),
        ]
        errors = validate_manifest_consistency(entries)
        assert any("Duplicate" in e for e in errors)

    def test_speaker_with_multiple_accents_detected(self):
        entries = [
            _make_entry(utt_id="u1", speaker_id="s1", accent="SE"),
            _make_entry(utt_id="u2", speaker_id="s1", accent="NE"),
        ]
        errors = validate_manifest_consistency(entries)
        assert any("multiple accents" in e for e in errors)
