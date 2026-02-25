"""Tests for PipelineCache and compute_filter_hash."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.cache import PipelineCache, compute_filter_hash
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


def _make_config(**filter_overrides):
    """Build minimal config dict for cache tests."""
    filters = {
        "speaker_type": "R",
        "min_duration_s": 3.0,
        "max_duration_s": 15.0,
        "min_speakers_per_region": 8,
        "min_utterances_per_speaker": 3,
    }
    filters.update(filter_overrides)
    return {"dataset": {"filters": filters}}


class TestComputeFilterHash:
    def test_deterministic(self):
        """Same config produces same hash."""
        config = _make_config()
        h1 = compute_filter_hash(config["dataset"])
        h2 = compute_filter_hash(config["dataset"])
        assert h1 == h2

    def test_length_is_12(self):
        config = _make_config()
        h = compute_filter_hash(config["dataset"])
        assert len(h) == 12

    def test_hex_chars_only(self):
        config = _make_config()
        h = compute_filter_hash(config["dataset"])
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_config_different_hash(self):
        """Changing any filter value produces a different hash."""
        h1 = compute_filter_hash(_make_config()["dataset"])
        h2 = compute_filter_hash(_make_config(min_duration_s=5.0)["dataset"])
        assert h1 != h2

    def test_key_order_irrelevant(self):
        """YAML canonical dump sorts keys, so order doesn't matter."""
        config_a = {"dataset": {"filters": {"a": 1, "b": 2}}}
        config_b = {"dataset": {"filters": {"b": 2, "a": 1}}}
        assert compute_filter_hash(config_a["dataset"]) == compute_filter_hash(config_b["dataset"])

    def test_empty_filters(self):
        config = {"dataset": {"filters": {}}}
        h = compute_filter_hash(config["dataset"])
        assert len(h) == 12

    def test_missing_filters_key(self):
        config = {"dataset": {}}
        h = compute_filter_hash(config["dataset"])
        assert len(h) == 12


class TestPipelineCacheManifest:
    def test_has_manifest_false_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(_make_config(), drive_base=tmpdir)
            assert cache.has_manifest() is False

    def test_has_manifest_true_after_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(_make_config(), drive_base=tmpdir)
            entries = [
                _make_entry(utt_id="u1"),
                _make_entry(utt_id="u2", speaker_id="s2", accent="NE", birth_state="BA"),
            ]
            write_manifest(entries, cache.get_manifest_path())
            assert cache.has_manifest() is True

    def test_load_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(_make_config(), drive_base=tmpdir)
            entries = [
                _make_entry(utt_id="u1"),
                _make_entry(utt_id="u2", speaker_id="s2", accent="NE", birth_state="BA"),
            ]
            write_manifest(entries, cache.get_manifest_path())

            loaded = cache.load_manifest()
            assert len(loaded) == 2
            assert loaded[0].utt_id == "u1"

    def test_has_manifest_false_on_corrupted_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(_make_config(), drive_base=tmpdir)
            entries = [_make_entry(utt_id="u1")]
            manifest_path = cache.get_manifest_path()
            write_manifest(entries, manifest_path)

            # Corrupt the sidecar
            sha_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
            sha_path.write_text("0000000000000000  manifest.jsonl\n")

            assert cache.has_manifest() is False

    def test_has_manifest_false_without_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(_make_config(), drive_base=tmpdir)
            entries = [_make_entry(utt_id="u1")]
            manifest_path = cache.get_manifest_path()
            write_manifest(entries, manifest_path)

            # Delete sidecar
            sha_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
            sha_path.unlink()

            assert cache.has_manifest() is False


class TestPipelineCacheFeatures:
    def test_has_features_false_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(_make_config(), drive_base=tmpdir)
            assert cache.has_features("acoustic") is False

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(_make_config(), drive_base=tmpdir)
            features = {
                "utt_001": np.array([1.0, 2.0, 3.0]),
                "utt_002": np.array([4.0, 5.0, 6.0]),
            }
            cache.save_features("acoustic", features)
            assert cache.has_features("acoustic") is True

            loaded = cache.load_features("acoustic")
            assert set(loaded.keys()) == {"utt_001", "utt_002"}
            np.testing.assert_array_almost_equal(loaded["utt_001"], features["utt_001"])
            np.testing.assert_array_almost_equal(loaded["utt_002"], features["utt_002"])

    def test_save_multidim_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(_make_config(), drive_base=tmpdir)
            features = {
                "utt_001": np.random.randn(192),
                "utt_002": np.random.randn(192),
            }
            cache.save_features("ecapa", features)
            loaded = cache.load_features("ecapa")
            np.testing.assert_array_almost_equal(loaded["utt_001"], features["utt_001"])

    def test_different_feature_types_independent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(_make_config(), drive_base=tmpdir)
            cache.save_features("acoustic", {"u1": np.array([1.0])})
            cache.save_features("ecapa", {"u1": np.array([2.0])})

            assert cache.has_features("acoustic") is True
            assert cache.has_features("ecapa") is True
            assert cache.has_features("wavlm_layer_0") is False

            acoustic = cache.load_features("acoustic")
            ecapa = cache.load_features("ecapa")
            assert acoustic["u1"][0] == 1.0
            assert ecapa["u1"][0] == 2.0

    def test_load_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(_make_config(), drive_base=tmpdir)
            with pytest.raises(FileNotFoundError):
                cache.load_features("nonexistent")


class TestPipelineCachePaths:
    def test_filter_hash_in_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(_make_config(), drive_base=tmpdir)
            manifest_path = cache.get_manifest_path()
            assert cache.filter_hash in str(manifest_path)

    def test_audio_dir_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(_make_config(), drive_base=tmpdir)
            audio_dir = cache.get_audio_dir()
            assert audio_dir.exists()
            assert audio_dir.is_dir()

    def test_different_config_different_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_a = PipelineCache(_make_config(), drive_base=tmpdir)
            cache_b = PipelineCache(
                _make_config(min_duration_s=5.0), drive_base=tmpdir
            )
            assert cache_a.filter_hash != cache_b.filter_hash
            assert cache_a.get_manifest_path() != cache_b.get_manifest_path()


class TestPipelineCacheReport:
    def test_report_empty_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(_make_config(), drive_base=tmpdir)
            report = cache.report()
            assert "MISSING" in report
            assert cache.filter_hash in report

    def test_report_with_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(_make_config(), drive_base=tmpdir)
            write_manifest([_make_entry()], cache.get_manifest_path())
            report = cache.report()
            assert "CACHED" in report
