"""Tests for manifest schema validation and I/O."""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.manifest import (
    BIRTH_STATE_TO_MACRO_REGION,
    STATE_FULL_NAME_TO_ABBREV,
    ManifestEntry,
    compute_file_hash,
    normalize_birth_state,
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
        "sampling_rate": 16000,
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


class TestNormalizeBirthState:
    """Tests for normalize_birth_state() — handles both abbreviations and full names."""

    def test_abbreviation_returned_as_is(self):
        """2-letter state abbreviation is returned unchanged."""
        assert normalize_birth_state("SP") == "SP"
        assert normalize_birth_state("RJ") == "RJ"
        assert normalize_birth_state("AM") == "AM"

    def test_full_name_accented(self):
        """Full Portuguese name with accents resolves correctly."""
        assert normalize_birth_state("São Paulo") == "SP"
        assert normalize_birth_state("Pará") == "PA"
        assert normalize_birth_state("Maranhão") == "MA"
        assert normalize_birth_state("Ceará") == "CE"

    def test_full_name_unaccented(self):
        """Full name without accents resolves (handles CORAA-MUPE variants)."""
        assert normalize_birth_state("SAO PAULO") == "SP"
        assert normalize_birth_state("PARA") == "PA"
        assert normalize_birth_state("MARANHAO") == "MA"

    def test_case_insensitive(self):
        """Lookup is case-insensitive."""
        assert normalize_birth_state("são paulo") == "SP"
        assert normalize_birth_state("minas gerais") == "MG"
        assert normalize_birth_state("rio de janeiro") == "RJ"

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is ignored."""
        assert normalize_birth_state("  SP  ") == "SP"
        assert normalize_birth_state(" São Paulo ") == "SP"

    def test_unknown_returns_none(self):
        """Unrecognized values return None."""
        assert normalize_birth_state("Unknown") is None
        assert normalize_birth_state("") is None
        assert normalize_birth_state("Brasil") is None

    def test_all_abbreviations_have_macro_region(self):
        """Every known abbreviation resolves and has a macro-region mapping."""
        for abbrev in BIRTH_STATE_TO_MACRO_REGION:
            result = normalize_birth_state(abbrev)
            assert result == abbrev, f"Failed for {abbrev}"

    def test_all_full_names_resolve(self):
        """Every full name in the mapping resolves to a valid abbreviation."""
        for full_name, expected_abbrev in STATE_FULL_NAME_TO_ABBREV.items():
            result = normalize_birth_state(full_name)
            assert result == expected_abbrev, f"Failed for {full_name}"

    def test_distrito_federal(self):
        """Multi-word state name: Distrito Federal."""
        assert normalize_birth_state("Distrito Federal") == "DF"

    def test_mato_grosso_do_sul(self):
        """Multi-word state name with 'do': Mato Grosso do Sul."""
        assert normalize_birth_state("Mato Grosso do Sul") == "MS"

    def test_rio_grande_do_norte_vs_sul(self):
        """Distinguish Rio Grande do Norte from Rio Grande do Sul."""
        assert normalize_birth_state("Rio Grande do Norte") == "RN"
        assert normalize_birth_state("Rio Grande do Sul") == "RS"
