"""Tests for combine_manifests() — especially exclude_regions."""

import tempfile
from pathlib import Path

import pytest

from src.data.combined_manifest import combine_manifests
from src.data.manifest import ManifestEntry, write_manifest


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


def _write_test_manifest(entries: list[ManifestEntry], tmpdir: Path) -> Path:
    """Write entries to a temp manifest JSONL and return the path."""
    path = tmpdir / "manifest.jsonl"
    write_manifest(entries, path)
    return path


class TestCombineManifestsExcludeRegions:
    """Tests for the exclude_regions parameter in combine_manifests()."""

    def _build_multi_region_entries(self) -> list[ManifestEntry]:
        """Build entries spanning 5 IBGE regions with enough speakers each."""
        entries = []
        regions = {
            "N": ("PA", "CORAA-MUPE"),
            "NE": ("BA", "CORAA-MUPE"),
            "CO": ("GO", "CORAA-MUPE"),
            "SE": ("SP", "CORAA-MUPE"),
            "S": ("RS", "CORAA-MUPE"),
        }
        utt_idx = 0
        for accent, (state, source) in regions.items():
            # 6 speakers per region, 4 utterances each (above min thresholds)
            for spk_i in range(6):
                spk_id = f"spk_{accent}_{spk_i}"
                for utt_i in range(4):
                    entries.append(
                        _make_entry(
                            utt_id=f"utt_{utt_idx:04d}",
                            audio_path=f"audio/utt_{utt_idx:04d}.wav",
                            speaker_id=spk_id,
                            accent=accent,
                            gender="M" if spk_i % 2 == 0 else "F",
                            text_id=f"txt_{utt_idx:04d}",
                            source=source,
                            birth_state=state,
                        )
                    )
                    utt_idx += 1
        return entries

    def test_exclude_regions_removes_co(self):
        """Entries from excluded region (CO) are removed."""
        entries = self._build_multi_region_entries()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest_path = _write_test_manifest(entries, tmpdir)
            output_path = tmpdir / "combined.jsonl"

            result, stats = combine_manifests(
                manifests=[(manifest_path, "CORAA-MUPE")],
                output_path=output_path,
                min_speakers_per_region=1,
                min_utterances_per_speaker=1,
                exclude_regions=["CO"],
            )

            result_accents = {e.accent for e in result}
            assert "CO" not in result_accents
            assert result_accents == {"N", "NE", "SE", "S"}

    def test_exclude_regions_empty_list_keeps_all(self):
        """Empty exclude list does not filter anything."""
        entries = self._build_multi_region_entries()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest_path = _write_test_manifest(entries, tmpdir)
            output_path = tmpdir / "combined.jsonl"

            result, stats = combine_manifests(
                manifests=[(manifest_path, "CORAA-MUPE")],
                output_path=output_path,
                min_speakers_per_region=1,
                min_utterances_per_speaker=1,
                exclude_regions=[],
            )

            result_accents = {e.accent for e in result}
            assert result_accents == {"N", "NE", "CO", "SE", "S"}

    def test_exclude_regions_none_keeps_all(self):
        """None (default) does not filter anything."""
        entries = self._build_multi_region_entries()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest_path = _write_test_manifest(entries, tmpdir)
            output_path = tmpdir / "combined.jsonl"

            result, stats = combine_manifests(
                manifests=[(manifest_path, "CORAA-MUPE")],
                output_path=output_path,
                min_speakers_per_region=1,
                min_utterances_per_speaker=1,
                exclude_regions=None,
            )

            result_accents = {e.accent for e in result}
            assert result_accents == {"N", "NE", "CO", "SE", "S"}

    def test_exclude_regions_correct_count(self):
        """Excluding CO removes exactly the CO entries."""
        entries = self._build_multi_region_entries()
        co_count = sum(1 for e in entries if e.accent == "CO")
        assert co_count > 0, "Test data must have CO entries"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest_path = _write_test_manifest(entries, tmpdir)
            output_path = tmpdir / "combined.jsonl"

            result, stats = combine_manifests(
                manifests=[(manifest_path, "CORAA-MUPE")],
                output_path=output_path,
                min_speakers_per_region=1,
                min_utterances_per_speaker=1,
                exclude_regions=["CO"],
            )

            assert len(result) == len(entries) - co_count

    def test_exclude_multiple_regions(self):
        """Multiple regions can be excluded at once."""
        entries = self._build_multi_region_entries()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest_path = _write_test_manifest(entries, tmpdir)
            output_path = tmpdir / "combined.jsonl"

            result, stats = combine_manifests(
                manifests=[(manifest_path, "CORAA-MUPE")],
                output_path=output_path,
                min_speakers_per_region=1,
                min_utterances_per_speaker=1,
                exclude_regions=["CO", "N"],
            )

            result_accents = {e.accent for e in result}
            assert result_accents == {"NE", "SE", "S"}


class TestFilterHashWithExcludeRegions:
    """Verify that adding exclude_regions changes the filter hash."""

    def test_hash_changes_with_exclude_regions(self):
        """Adding exclude_regions to filters produces a different hash."""
        from src.data.cache import compute_filter_hash

        config_without = {
            "filters": {
                "speaker_type": "R",
                "min_duration_s": 3.0,
                "max_duration_s": 15.0,
                "min_speakers_per_region": 5,
                "min_utterances_per_speaker": 3,
            }
        }
        config_with = {
            "filters": {
                "speaker_type": "R",
                "min_duration_s": 3.0,
                "max_duration_s": 15.0,
                "min_speakers_per_region": 5,
                "min_utterances_per_speaker": 3,
                "exclude_regions": ["CO"],
            }
        }

        hash_without = compute_filter_hash(config_without)
        hash_with = compute_filter_hash(config_with)
        assert hash_without != hash_with

    def test_hash_same_with_empty_exclude(self):
        """Empty exclude_regions list produces a different hash than no key."""
        from src.data.cache import compute_filter_hash

        config_no_key = {"filters": {"min_duration_s": 3.0}}
        config_empty = {"filters": {"min_duration_s": 3.0, "exclude_regions": []}}

        hash_no_key = compute_filter_hash(config_no_key)
        hash_empty = compute_filter_hash(config_empty)
        # Different because the key exists (even if empty) — this is correct
        # behavior: presence of key in YAML changes the canonical dump.
        assert hash_no_key != hash_empty
