"""Tests for CV accent normalization, combined manifest creation, and source distribution analysis."""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.manifest import (
    CV_ACCENT_TO_MACRO_REGION,
    ManifestEntry,
    normalize_cv_accent,
    read_manifest,
    write_manifest,
)
from src.data.combined_manifest import combine_manifests, analyze_source_distribution
from src.data.cv_manifest_builder import _CV_GENDER_MAP


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


# ---------------------------------------------------------------------------
# normalize_cv_accent
# ---------------------------------------------------------------------------


class TestNormalizeCvAccent:
    def test_demonym_to_region(self):
        """Regional demonyms map to the correct IBGE macro-region."""
        assert normalize_cv_accent("carioca") == "SE"
        assert normalize_cv_accent("gaúcho") == "S"
        assert normalize_cv_accent("baiano") == "NE"
        assert normalize_cv_accent("goiano") == "CO"
        assert normalize_cv_accent("paraense") == "N"

    def test_region_name(self):
        """Generic region name maps to the correct code."""
        assert normalize_cv_accent("nordeste") == "NE"
        assert normalize_cv_accent("sul") == "S"

    def test_state_abbreviation(self):
        """Lowercase state abbreviation maps to the correct macro-region."""
        assert normalize_cv_accent("sp") == "SE"
        assert normalize_cv_accent("rs") == "S"
        assert normalize_cv_accent("ba") == "NE"

    def test_unknown_returns_none(self):
        """Unrecognized or empty values return None."""
        assert normalize_cv_accent("unknown") is None
        assert normalize_cv_accent("") is None
        assert normalize_cv_accent("brasileiro") is None

    def test_case_insensitive(self):
        """Lookup is case-insensitive for demonyms."""
        assert normalize_cv_accent("CARIOCA") == "SE"
        assert normalize_cv_accent("Carioca") == "SE"
        assert normalize_cv_accent("carioca") == "SE"

    def test_fallback_to_birth_state(self):
        """Full state name falls back to normalize_birth_state."""
        assert normalize_cv_accent("São Paulo") == "SE"

    def test_all_mappings_valid(self):
        """Every value in CV_ACCENT_TO_MACRO_REGION is a valid IBGE code."""
        valid_regions = {"N", "NE", "CO", "SE", "S"}
        for key, region in CV_ACCENT_TO_MACRO_REGION.items():
            assert region in valid_regions, (
                f"Mapping '{key}' -> '{region}' is not a valid macro-region"
            )


# ---------------------------------------------------------------------------
# _CV_GENDER_MAP
# ---------------------------------------------------------------------------


class TestCvGenderMap:
    def test_standard_labels(self):
        """Standard Mozilla CV gender labels map correctly."""
        assert _CV_GENDER_MAP["male"] == "M"
        assert _CV_GENDER_MAP["female"] == "F"

    def test_extended_labels(self):
        """Community mirror labels (male_masculine/female_feminine) map correctly."""
        assert _CV_GENDER_MAP["male_masculine"] == "M"
        assert _CV_GENDER_MAP["female_feminine"] == "F"

    def test_all_values_valid(self):
        """All gender map values are M or F."""
        for key, val in _CV_GENDER_MAP.items():
            assert val in ("M", "F"), f"Invalid gender mapping: '{key}' -> '{val}'"


# ---------------------------------------------------------------------------
# combine_manifests
# ---------------------------------------------------------------------------


class TestCombineManifests:
    def _write_temp_manifest(self, entries: list[ManifestEntry], tmpdir: Path, name: str) -> Path:
        """Write entries to a temp manifest file and return its path."""
        path = tmpdir / name
        write_manifest(entries, path)
        return path

    def test_combine_two_sources(self, tmp_path):
        """Merging two source manifests produces the correct combined count."""
        # Arrange: two manifests with distinct IDs and enough speakers per region
        coraa_entries = [
            _make_entry(
                utt_id=f"coraa_{i:03d}",
                speaker_id=f"coraa_spk_{i:03d}",
                accent="SE",
                source="CORAA-MUPE",
            )
            for i in range(10)
        ] + [
            _make_entry(
                utt_id=f"coraa_ne_{i:03d}",
                speaker_id=f"coraa_ne_spk_{i:03d}",
                accent="NE",
                birth_state="BA",
                source="CORAA-MUPE",
            )
            for i in range(10)
        ]
        cv_entries = [
            _make_entry(
                utt_id=f"cv_{i:03d}",
                speaker_id=f"cv_spk_{i:03d}",
                accent="SE",
                source="CommonVoice-PT",
            )
            for i in range(5)
        ] + [
            _make_entry(
                utt_id=f"cv_ne_{i:03d}",
                speaker_id=f"cv_ne_spk_{i:03d}",
                accent="NE",
                birth_state="BA",
                source="CommonVoice-PT",
            )
            for i in range(5)
        ]

        coraa_path = self._write_temp_manifest(coraa_entries, tmp_path, "coraa.jsonl")
        cv_path = self._write_temp_manifest(cv_entries, tmp_path, "cv.jsonl")
        output_path = tmp_path / "combined.jsonl"

        # Act
        combined, stats = combine_manifests(
            [(coraa_path, "CORAA-MUPE"), (cv_path, "CommonVoice-PT")],
            output_path,
            min_speakers_per_region=1,
            min_utterances_per_speaker=0,
        )

        # Assert
        assert len(combined) == 30
        assert stats["total_utterances"] == 30

    def test_utt_id_collision_raises(self, tmp_path):
        """Same utt_id across sources raises ValueError."""
        entries_a = [
            _make_entry(utt_id="conflict_001", speaker_id="spk_a", source="A"),
            _make_entry(utt_id="a_002", speaker_id="spk_a2", accent="NE", birth_state="BA", source="A"),
        ]
        entries_b = [
            _make_entry(utt_id="conflict_001", speaker_id="spk_b", source="B"),
            _make_entry(utt_id="b_002", speaker_id="spk_b2", accent="NE", birth_state="BA", source="B"),
        ]

        path_a = self._write_temp_manifest(entries_a, tmp_path, "a.jsonl")
        path_b = self._write_temp_manifest(entries_b, tmp_path, "b.jsonl")
        output = tmp_path / "combined.jsonl"

        with pytest.raises(ValueError, match="utt_id collisions"):
            combine_manifests(
                [(path_a, "A"), (path_b, "B")],
                output,
                min_speakers_per_region=1,
                min_utterances_per_speaker=0,
            )

    def test_speaker_id_collision_raises(self, tmp_path):
        """Same speaker_id across different sources raises ValueError."""
        entries_a = [
            _make_entry(utt_id="a_001", speaker_id="shared_spk", source="A"),
            _make_entry(utt_id="a_002", speaker_id="spk_a_only", accent="NE", birth_state="BA", source="A"),
        ]
        entries_b = [
            _make_entry(utt_id="b_001", speaker_id="shared_spk", source="B"),
            _make_entry(utt_id="b_002", speaker_id="spk_b_only", accent="NE", birth_state="BA", source="B"),
        ]

        path_a = self._write_temp_manifest(entries_a, tmp_path, "a.jsonl")
        path_b = self._write_temp_manifest(entries_b, tmp_path, "b.jsonl")
        output = tmp_path / "combined.jsonl"

        with pytest.raises(ValueError, match="speaker_id collisions"):
            combine_manifests(
                [(path_a, "A"), (path_b, "B")],
                output,
                min_speakers_per_region=1,
                min_utterances_per_speaker=0,
            )

    def test_source_distribution_analysis(self):
        """analyze_source_distribution returns expected structure."""
        entries = [
            _make_entry(utt_id=f"u{i}", speaker_id=f"s{i}", source="A")
            for i in range(5)
        ] + [
            _make_entry(utt_id=f"v{i}", speaker_id=f"t{i}", source="B")
            for i in range(5)
        ]

        result = analyze_source_distribution(entries)

        assert "source_x_accent" in result
        assert "warnings" in result
        assert isinstance(result["warnings"], list)


# ---------------------------------------------------------------------------
# analyze_source_distribution — warning detection
# ---------------------------------------------------------------------------


class TestAnalyzeSourceDistribution:
    def test_warns_if_accent_dominated_by_one_source(self):
        """Warning emitted when >80% of one accent comes from a single source."""
        # 9 from source A, 1 from source B — 90% dominated
        entries = [
            _make_entry(utt_id=f"a_{i}", speaker_id=f"sa_{i}", accent="SE", source="A")
            for i in range(9)
        ] + [
            _make_entry(utt_id="b_0", speaker_id="sb_0", accent="SE", source="B"),
        ]

        result = analyze_source_distribution(entries)

        assert len(result["warnings"]) >= 1
        assert any("SE" in w and "A" in w for w in result["warnings"])

    def test_no_warning_if_balanced(self):
        """No warnings when sources are balanced within each accent."""
        entries = [
            _make_entry(utt_id=f"a_{i}", speaker_id=f"sa_{i}", accent="SE", source="A")
            for i in range(5)
        ] + [
            _make_entry(utt_id=f"b_{i}", speaker_id=f"sb_{i}", accent="SE", source="B")
            for i in range(5)
        ]

        result = analyze_source_distribution(entries)

        assert result["warnings"] == []
