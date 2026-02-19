"""Tests for confound analysis (accent x gender, accent x duration)."""

import pytest

from src.analysis.confounds import (
    analyze_accent_x_duration,
    analyze_accent_x_gender,
    run_all_confound_checks,
)
from src.data.manifest import ManifestEntry


def _make_entry(speaker_id: str, accent: str, gender: str, duration_s: float, birth_state: str) -> ManifestEntry:
    return ManifestEntry(
        utt_id=f"utt_{speaker_id}_{duration_s}",
        audio_path=f"audio/{speaker_id}.wav",
        speaker_id=speaker_id,
        accent=accent,
        gender=gender,
        duration_s=duration_s,
        text_id=None,
        source="CORAA-MUPE",
        birth_state=birth_state,
    )


class TestAccentXGender:
    def test_balanced_distribution_not_blocking(self):
        """When gender is equally distributed across accents, no confound."""
        entries = []
        for i in range(50):
            entries.append(_make_entry(f"spk_se_{i}", "SE", "M" if i % 2 == 0 else "F", 5.0, "SP"))
            entries.append(_make_entry(f"spk_ne_{i}", "NE", "M" if i % 2 == 0 else "F", 5.0, "BA"))

        result = analyze_accent_x_gender(entries)

        assert not result.is_blocking
        assert result.effect_size < 0.3

    def test_perfectly_correlated_is_blocking(self):
        """When one accent is all-male and another all-female, strong confound."""
        entries = []
        for i in range(50):
            entries.append(_make_entry(f"spk_se_{i}", "SE", "M", 5.0, "SP"))
            entries.append(_make_entry(f"spk_ne_{i}", "NE", "F", 5.0, "BA"))

        result = analyze_accent_x_gender(entries, blocking_threshold=0.3)

        assert result.is_significant
        assert result.is_blocking
        assert result.effect_size > 0.3

    def test_returns_confound_result(self):
        entries = [
            _make_entry("s1", "SE", "M", 5.0, "SP"),
            _make_entry("s2", "NE", "F", 5.0, "BA"),
        ]
        result = analyze_accent_x_gender(entries)

        assert result.test_name == "chi_squared"
        assert result.variable_a == "accent"
        assert result.variable_b == "gender"
        assert result.effect_size_name == "cramers_v"


class TestAccentXDuration:
    def test_similar_durations_not_blocking(self):
        """When duration is similar across accents, no confound."""
        entries = []
        for i in range(50):
            entries.append(_make_entry(f"spk_se_{i}", "SE", "M", 5.0 + (i % 3), "SP"))
            entries.append(_make_entry(f"spk_ne_{i}", "NE", "F", 5.0 + (i % 3), "BA"))

        result = analyze_accent_x_duration(entries)

        assert not result.is_blocking

    def test_very_different_durations_is_blocking(self):
        """When one accent has much longer durations, confound detected."""
        entries = []
        for i in range(50):
            entries.append(_make_entry(f"spk_se_{i}", "SE", "M", 4.0, "SP"))
            entries.append(_make_entry(f"spk_ne_{i}", "NE", "F", 12.0, "BA"))

        result = analyze_accent_x_duration(entries, practical_diff_s=1.0)

        assert result.is_significant
        assert result.is_blocking


class TestRunAllChecks:
    def test_returns_two_results(self):
        entries = [
            _make_entry("s1", "SE", "M", 4.5, "SP"),
            _make_entry("s2", "SE", "F", 5.5, "SP"),
            _make_entry("s3", "NE", "F", 6.0, "BA"),
            _make_entry("s4", "NE", "M", 4.0, "BA"),
            _make_entry("s5", "S", "M", 5.0, "RS"),
            _make_entry("s6", "S", "F", 5.2, "RS"),
        ]
        results = run_all_confound_checks(entries)

        assert len(results) == 2
        assert results[0].variable_b == "gender"
        assert results[1].variable_b == "duration_s"
