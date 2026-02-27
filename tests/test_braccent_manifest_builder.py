"""Tests for BrAccent manifest builder.

Tests filename parsing, accent mapping (folder label vs CO re-mapping),
and integration with the manifest pipeline.
"""

import io
import struct
import zipfile
from pathlib import Path

import pytest

from src.data.braccent_manifest_builder import (
    BRACCENT_FOLDER_TO_IBGE,
    CO_STATES,
    GENDER_MAP,
    _BRACCENT_STATE_TO_ABBREV,
    build_manifest_from_braccent,
    parse_braccent_filename,
)
from src.data.manifest import ManifestEntry


# ---------------------------------------------------------------------------
# parse_braccent_filename
# ---------------------------------------------------------------------------


class TestParseBraccentFilename:
    def test_standard_filename(self):
        """Standard BrAccent filename parses correctly."""
        result = parse_braccent_filename(
            "converted_4rqv6rrro_frase_1_2018-08-29_60_Salvador_Bahia_Feminino_Mestrado.wav"
        )
        assert result is not None
        assert result["speaker_id"] == "4rqv6rrro"
        assert result["frase_num"] == "1"
        assert result["date"] == "2018-08-29"
        assert result["age"] == "60"
        assert result["city"] == "Salvador"
        assert result["state"] == "Bahia"
        assert result["gender"] == "Feminino"
        assert result["education"] == "Mestrado"

    def test_goiania_goias_filename(self):
        """CO speaker (Goiânia/Goiás) filename parses correctly."""
        result = parse_braccent_filename(
            "converted_s554xpzfx_frase_2_2018-02-05_24_Goiania_Goias_Masculino_EnsinoSuperiorIncompleto.wav"
        )
        assert result is not None
        assert result["speaker_id"] == "s554xpzfx"
        assert result["state"] == "Goias"
        assert result["gender"] == "Masculino"

    def test_brasilia_df_filename(self):
        """CO speaker (Brasília/DF) filename parses correctly."""
        result = parse_braccent_filename(
            "converted_r1o9odgok_frase_1_2018-06-11_27_Brasilia_DistritoFederal_Masculino_Mestrado.wav"
        )
        assert result is not None
        assert result["speaker_id"] == "r1o9odgok"
        assert result["state"] == "DistritoFederal"

    def test_filename_with_extra_number(self):
        """Filename with extra number before date (e.g. frase_1_36_2018-...)."""
        result = parse_braccent_filename(
            "converted_9er8q86jp_frase_1_36_2018-08-12_29_salvador_Bahia_Feminino_EnsinoSuperiorIncompleto.wav"
        )
        assert result is not None
        assert result["speaker_id"] == "9er8q86jp"
        assert result["frase_num"] == "1"

    def test_fake_file_returns_none(self):
        """FAKE files don't match the pattern (different prefix)."""
        result = parse_braccent_filename(
            "FAKE_converted_4rqv6rrro_frase_1_2018-08-29_60_Salvador_Bahia_Feminino_Mestrado.mp3"
        )
        assert result is None

    def test_invalid_filename_returns_none(self):
        """Non-matching filenames return None."""
        assert parse_braccent_filename("random_file.wav") is None
        assert parse_braccent_filename("") is None

    def test_mp3_returns_none(self):
        """MP3 files (FAKE) don't match the .wav pattern."""
        result = parse_braccent_filename(
            "converted_4rqv6rrro_frase_1_2018-08-29_60_Salvador_Bahia_Feminino_Mestrado.mp3"
        )
        assert result is None


# ---------------------------------------------------------------------------
# Mapping constants
# ---------------------------------------------------------------------------


class TestMappingConstants:
    def test_all_folder_labels_map_to_valid_regions(self):
        """Every BrAccent folder maps to a valid IBGE macro-region."""
        valid = {"N", "NE", "CO", "SE", "S"}
        for folder, region in BRACCENT_FOLDER_TO_IBGE.items():
            assert region in valid, f"Folder '{folder}' maps to invalid region '{region}'"

    def test_co_states_are_known(self):
        """CO states exist in the state abbreviation mapping."""
        for state in CO_STATES:
            assert state in _BRACCENT_STATE_TO_ABBREV, (
                f"CO state '{state}' not in _BRACCENT_STATE_TO_ABBREV"
            )
            abbrev = _BRACCENT_STATE_TO_ABBREV[state]
            from src.data.manifest import BIRTH_STATE_TO_MACRO_REGION
            assert BIRTH_STATE_TO_MACRO_REGION[abbrev] == "CO", (
                f"State '{state}' ({abbrev}) should map to CO"
            )

    def test_gender_map_values(self):
        """All gender map values are M or F."""
        for key, val in GENDER_MAP.items():
            assert val in ("M", "F"), f"Invalid gender mapping: '{key}' -> '{val}'"


# ---------------------------------------------------------------------------
# build_manifest_from_braccent (integration test with synthetic zip)
# ---------------------------------------------------------------------------


def _make_wav_bytes(duration_s: float = 5.0, sr: int = 16000) -> bytes:
    """Create minimal valid WAV file bytes (silence)."""
    n_samples = int(duration_s * sr)
    # WAV header (44 bytes) + PCM16 data
    data_size = n_samples * 2  # 16-bit = 2 bytes per sample
    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))   # PCM
    buf.write(struct.pack("<H", 1))   # mono
    buf.write(struct.pack("<I", sr))  # sample rate
    buf.write(struct.pack("<I", sr * 2))  # byte rate
    buf.write(struct.pack("<H", 2))   # block align
    buf.write(struct.pack("<H", 16))  # bits per sample
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(b"\x00" * data_size)  # silence
    return buf.getvalue()


def _create_test_zip(tmp_path: Path) -> Path:
    """Create a minimal BrAccent-like zip for testing."""
    zip_path = tmp_path / "archive.zip"
    wav_data = _make_wav_bytes(duration_s=5.0)

    with zipfile.ZipFile(zip_path, "w") as zf:
        # NE speaker (Baiano folder)
        zf.writestr(
            "FakeBrAccent 2/Baiano/Feminino/REAL/"
            "converted_speaker01_frase_1_2018-01-01_30_Salvador_Bahia_Feminino_Mestrado.wav",
            wav_data,
        )
        zf.writestr(
            "FakeBrAccent 2/Baiano/Feminino/REAL/"
            "converted_speaker01_frase_2_2018-01-01_30_Salvador_Bahia_Feminino_Mestrado.wav",
            wav_data,
        )
        # SE speaker (Fluminense folder)
        zf.writestr(
            "FakeBrAccent 2/Fluminense/Masculino/REAL/"
            "converted_speaker02_frase_1_2018-02-01_25_RiodeJaneiro_RiodeJaneiro_Masculino_Doutorado.wav",
            wav_data,
        )
        # CO speaker (Sulista folder, DF state -> re-mapped to CO)
        zf.writestr(
            "FakeBrAccent 2/Sulista/Masculino/REAL/"
            "converted_speaker03_frase_1_2018-03-01_27_Brasilia_DistritoFederal_Masculino_Mestrado.wav",
            wav_data,
        )
        # S speaker (Sulista folder, RS state -> stays S)
        zf.writestr(
            "FakeBrAccent 2/Sulista/Masculino/REAL/"
            "converted_speaker04_frase_1_2018-04-01_22_PortoAlegre_RioGrandedoSul_Masculino_Doutorado.wav",
            wav_data,
        )
        # FAKE file (should be ignored)
        zf.writestr(
            "FakeBrAccent 2/Baiano/Feminino/FAKE/"
            "FAKE_converted_speaker01_frase_1_2018-01-01_30_Salvador_Bahia_Feminino_Mestrado.mp3",
            b"fake audio data",
        )

    return zip_path


class TestBuildManifestFromBraccent:
    def test_builds_manifest_with_co_remapping(self, tmp_path):
        """CO re-mapping moves DF speaker from S to CO."""
        zip_path = _create_test_zip(tmp_path)

        entries, stats = build_manifest_from_braccent(
            zip_path=zip_path,
            audio_output_dir=tmp_path / "audio",
            manifest_output_path=tmp_path / "manifest.jsonl",
            min_duration_s=1.0,
            max_duration_s=30.0,
            min_speakers_per_region=0,
            include_co_remapping=True,
        )

        # Should have 5 entries (2 NE + 1 SE + 1 CO + 1 S)
        assert len(entries) == 5
        assert stats["filter_stats"]["co_remapped"] == 1

        # Check accent assignments
        accents = {e.speaker_id: e.accent for e in entries}
        assert accents["bra_speaker01"] == "NE"   # Baiano folder
        assert accents["bra_speaker02"] == "SE"   # Fluminense folder
        assert accents["bra_speaker03"] == "CO"   # DF -> CO re-mapped
        assert accents["bra_speaker04"] == "S"    # RS -> stays S

    def test_without_co_remapping(self, tmp_path):
        """Without CO re-mapping, DF speaker stays in S (Sulista folder)."""
        zip_path = _create_test_zip(tmp_path)

        entries, stats = build_manifest_from_braccent(
            zip_path=zip_path,
            audio_output_dir=tmp_path / "audio",
            manifest_output_path=tmp_path / "manifest.jsonl",
            min_duration_s=1.0,
            max_duration_s=30.0,
            min_speakers_per_region=0,
            include_co_remapping=False,
        )

        assert stats["filter_stats"]["co_remapped"] == 0

        accents = {e.speaker_id: e.accent for e in entries}
        assert accents["bra_speaker03"] == "S"  # Stays in Sulista

    def test_speaker_ids_prefixed(self, tmp_path):
        """All speaker IDs are prefixed with 'bra_'."""
        zip_path = _create_test_zip(tmp_path)

        entries, _ = build_manifest_from_braccent(
            zip_path=zip_path,
            audio_output_dir=tmp_path / "audio",
            manifest_output_path=tmp_path / "manifest.jsonl",
            min_duration_s=1.0,
            max_duration_s=30.0,
            min_speakers_per_region=0,
        )

        for entry in entries:
            assert entry.speaker_id.startswith("bra_"), (
                f"Speaker ID '{entry.speaker_id}' missing 'bra_' prefix"
            )

    def test_utt_ids_prefixed(self, tmp_path):
        """All utt IDs are prefixed with 'bra_'."""
        zip_path = _create_test_zip(tmp_path)

        entries, _ = build_manifest_from_braccent(
            zip_path=zip_path,
            audio_output_dir=tmp_path / "audio",
            manifest_output_path=tmp_path / "manifest.jsonl",
            min_duration_s=1.0,
            max_duration_s=30.0,
            min_speakers_per_region=0,
        )

        for entry in entries:
            assert entry.utt_id.startswith("bra_"), (
                f"Utt ID '{entry.utt_id}' missing 'bra_' prefix"
            )

    def test_source_is_braccent(self, tmp_path):
        """All entries have source='BrAccent'."""
        zip_path = _create_test_zip(tmp_path)

        entries, _ = build_manifest_from_braccent(
            zip_path=zip_path,
            audio_output_dir=tmp_path / "audio",
            manifest_output_path=tmp_path / "manifest.jsonl",
            min_duration_s=1.0,
            max_duration_s=30.0,
            min_speakers_per_region=0,
        )

        for entry in entries:
            assert entry.source == "BrAccent"

    def test_fake_files_ignored(self, tmp_path):
        """FAKE audio files are not included in the manifest."""
        zip_path = _create_test_zip(tmp_path)

        entries, stats = build_manifest_from_braccent(
            zip_path=zip_path,
            audio_output_dir=tmp_path / "audio",
            manifest_output_path=tmp_path / "manifest.jsonl",
            min_duration_s=1.0,
            max_duration_s=30.0,
            min_speakers_per_region=0,
        )

        # Only REAL WAVs counted
        assert stats["filter_stats"]["total_real_wavs"] == 5
        fake_utt_ids = [e.utt_id for e in entries if "FAKE" in e.utt_id]
        assert fake_utt_ids == []

    def test_duration_filter(self, tmp_path):
        """Entries outside duration range are rejected."""
        zip_path = _create_test_zip(tmp_path)

        entries, stats = build_manifest_from_braccent(
            zip_path=zip_path,
            audio_output_dir=tmp_path / "audio",
            manifest_output_path=tmp_path / "manifest.jsonl",
            min_duration_s=10.0,  # All test WAVs are 5s -> all rejected
            max_duration_s=30.0,
            min_speakers_per_region=0,
        )

        assert len(entries) == 0
        assert stats["filter_stats"]["rejected_duration"] == 5

    def test_missing_zip_raises(self, tmp_path):
        """FileNotFoundError raised for non-existent zip."""
        with pytest.raises(FileNotFoundError):
            build_manifest_from_braccent(
                zip_path=tmp_path / "nonexistent.zip",
                audio_output_dir=tmp_path / "audio",
                manifest_output_path=tmp_path / "manifest.jsonl",
            )

    def test_manifest_written_to_disk(self, tmp_path):
        """Manifest JSONL file is written and readable."""
        zip_path = _create_test_zip(tmp_path)
        manifest_path = tmp_path / "manifest.jsonl"

        entries, stats = build_manifest_from_braccent(
            zip_path=zip_path,
            audio_output_dir=tmp_path / "audio",
            manifest_output_path=manifest_path,
            min_duration_s=1.0,
            max_duration_s=30.0,
            min_speakers_per_region=0,
        )

        assert manifest_path.exists()
        assert stats["manifest_sha256"] is not None

        # Verify it's valid JSONL
        from src.data.manifest import read_manifest
        loaded = read_manifest(manifest_path)
        assert len(loaded) == len(entries)
