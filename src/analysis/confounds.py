"""Confound analysis: accent x gender, accent x duration, accent x SNR.

These sanity checks are MANDATORY before any training.
If accent correlates strongly with gender, duration, or recording
conditions, any positive result may be attributable to the confound.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from src.data.manifest import ManifestEntry

logger = logging.getLogger(__name__)


@dataclass
class ConfoundResult:
    """Result of a single confound analysis."""
    test_name: str
    variable_a: str
    variable_b: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    is_significant: bool  # p < 0.05
    is_blocking: bool     # exceeds blocking threshold
    interpretation: str


def analyze_accent_x_gender(
    entries: list[ManifestEntry],
    blocking_threshold: float = 0.3,
) -> ConfoundResult:
    """Test association between accent and gender using chi-squared.

    Args:
        entries: Manifest entries with accent and gender fields.
        blocking_threshold: Cramer's V threshold for blocking confound.

    Returns:
        ConfoundResult with chi-squared test and Cramer's V.
    """
    # Build contingency table
    accents = sorted({e.accent for e in entries})
    genders = sorted({e.gender for e in entries})

    contingency = np.zeros((len(accents), len(genders)), dtype=int)
    accent_idx = {a: i for i, a in enumerate(accents)}
    gender_idx = {g: i for i, g in enumerate(genders)}

    for entry in entries:
        contingency[accent_idx[entry.accent], gender_idx[entry.gender]] += 1

    # Chi-squared test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Cramer's V (effect size)
    n = contingency.sum()
    k = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else 0.0

    is_significant = p_value < 0.05
    is_blocking = cramers_v >= blocking_threshold

    if is_blocking:
        interpretation = (
            f"BLOCKING: Cramer's V = {cramers_v:.3f} >= {blocking_threshold}. "
            f"Strong association accent x gender. Mitigation required."
        )
    elif is_significant:
        interpretation = (
            f"Significant but acceptable: Cramer's V = {cramers_v:.3f} < {blocking_threshold}. "
            f"Document as limitation."
        )
    else:
        interpretation = (
            f"No significant association (p={p_value:.4f}). "
            f"Cramer's V = {cramers_v:.3f}."
        )

    logger.info(f"Accent x Gender: chi2={chi2:.2f}, p={p_value:.4f}, V={cramers_v:.3f}")

    return ConfoundResult(
        test_name="chi_squared",
        variable_a="accent",
        variable_b="gender",
        statistic=chi2,
        p_value=p_value,
        effect_size=cramers_v,
        effect_size_name="cramers_v",
        is_significant=is_significant,
        is_blocking=is_blocking,
        interpretation=interpretation,
    )


def analyze_accent_x_duration(
    entries: list[ManifestEntry],
    practical_diff_s: float = 1.0,
) -> ConfoundResult:
    """Test association between accent and duration using Kruskal-Wallis.

    Non-parametric test: does duration distribution differ across accents?

    Args:
        entries: Manifest entries with accent and duration_s fields.
        practical_diff_s: Minimum practical difference in seconds.

    Returns:
        ConfoundResult with Kruskal-Wallis H and effect size.
    """
    # Group durations by accent
    accent_durations: dict[str, list[float]] = {}
    for entry in entries:
        accent_durations.setdefault(entry.accent, []).append(entry.duration_s)

    groups = [np.array(durs) for durs in accent_durations.values()]
    accents = list(accent_durations.keys())

    # Kruskal-Wallis H test
    h_stat, p_value = stats.kruskal(*groups)

    # Effect size: epsilon-squared (η² analog for Kruskal-Wallis)
    n = sum(len(g) for g in groups)
    k = len(groups)
    epsilon_sq = (h_stat - k + 1) / (n - k) if n > k else 0.0

    # Practical difference: max difference between group means
    means = {acc: np.mean(durs) for acc, durs in accent_durations.items()}
    max_diff = max(means.values()) - min(means.values())

    is_significant = p_value < 0.05
    is_blocking = is_significant and max_diff > practical_diff_s

    if is_blocking:
        interpretation = (
            f"Significant AND practical: max mean diff = {max_diff:.2f}s > {practical_diff_s}s. "
            f"Model might learn duration as accent proxy. Document as limitation."
        )
    elif is_significant:
        interpretation = (
            f"Statistically significant but small practical diff = {max_diff:.2f}s. "
            f"epsilon² = {epsilon_sq:.4f}."
        )
    else:
        interpretation = (
            f"No significant difference (p={p_value:.4f}). "
            f"Max mean diff = {max_diff:.2f}s."
        )

    # Log per-accent stats
    for acc in sorted(accents):
        durs = accent_durations[acc]
        logger.info(
            f"  {acc}: n={len(durs)}, mean={np.mean(durs):.2f}s, "
            f"std={np.std(durs):.2f}s, median={np.median(durs):.2f}s"
        )

    return ConfoundResult(
        test_name="kruskal_wallis",
        variable_a="accent",
        variable_b="duration_s",
        statistic=h_stat,
        p_value=p_value,
        effect_size=epsilon_sq,
        effect_size_name="epsilon_squared",
        is_significant=is_significant,
        is_blocking=is_blocking,
        interpretation=interpretation,
    )


def estimate_snr_rms(audio_path: Path, sr: int = 16000) -> float:
    """Estimate Signal-to-Noise Ratio using RMS energy and silence detection.

    Simple approach: compute ratio of RMS in speech segments vs silence.
    Uses a frame-level energy threshold to separate speech from silence.

    Args:
        audio_path: Path to audio file.
        sr: Sampling rate for loading.

    Returns:
        Estimated SNR in dB. Higher = cleaner audio.
    """
    import librosa

    y, _ = librosa.load(str(audio_path), sr=sr)

    # Frame-level RMS
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    if len(rms) == 0:
        return 0.0

    # Threshold: frames below 20th percentile are "silence"
    threshold = np.percentile(rms, 20)
    speech_rms = rms[rms > threshold]
    noise_rms = rms[rms <= threshold]

    if len(speech_rms) == 0 or len(noise_rms) == 0:
        return 0.0

    speech_power = np.mean(speech_rms ** 2)
    noise_power = np.mean(noise_rms ** 2)

    if noise_power == 0:
        return 60.0  # very clean

    snr_db = 10 * np.log10(speech_power / noise_power)
    return float(snr_db)


def analyze_accent_x_snr(
    entries: list[ManifestEntry],
    practical_diff_db: float = 5.0,
) -> ConfoundResult:
    """Test association between accent and recording conditions via SNR.

    Estimates SNR per utterance and tests whether it differs across accents.
    A significant difference suggests systematic recording quality variation
    by region — the model might learn channel instead of accent.

    Args:
        entries: Manifest entries with audio_path fields.
        practical_diff_db: Minimum practical SNR difference in dB.

    Returns:
        ConfoundResult with Kruskal-Wallis test on SNR.
    """
    # Estimate SNR per utterance
    accent_snrs: dict[str, list[float]] = {}
    for entry in entries:
        audio_path = Path(entry.audio_path)
        if not audio_path.exists():
            logger.warning(f"Audio not found for SNR: {audio_path}")
            continue
        snr = estimate_snr_rms(audio_path)
        accent_snrs.setdefault(entry.accent, []).append(snr)

    if len(accent_snrs) < 2:
        return ConfoundResult(
            test_name="kruskal_wallis",
            variable_a="accent",
            variable_b="snr_db",
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            effect_size_name="epsilon_squared",
            is_significant=False,
            is_blocking=False,
            interpretation="Insufficient data for SNR confound analysis.",
        )

    groups = [np.array(snrs) for snrs in accent_snrs.values()]
    accents = list(accent_snrs.keys())

    h_stat, p_value = stats.kruskal(*groups)

    n = sum(len(g) for g in groups)
    k = len(groups)
    epsilon_sq = (h_stat - k + 1) / (n - k) if n > k else 0.0

    means = {acc: np.mean(snrs) for acc, snrs in accent_snrs.items()}
    max_diff = max(means.values()) - min(means.values())

    is_significant = p_value < 0.05
    is_blocking = is_significant and max_diff > practical_diff_db

    if is_blocking:
        interpretation = (
            f"BLOCKING: Significant SNR difference across accents. "
            f"Max mean diff = {max_diff:.1f}dB > {practical_diff_db}dB. "
            f"Model might learn recording conditions as accent proxy."
        )
    elif is_significant:
        interpretation = (
            f"Statistically significant but small practical SNR diff = {max_diff:.1f}dB. "
            f"epsilon² = {epsilon_sq:.4f}. Document as limitation."
        )
    else:
        interpretation = (
            f"No significant SNR difference (p={p_value:.4f}). "
            f"Max mean diff = {max_diff:.1f}dB."
        )

    for acc in sorted(accents):
        snrs = accent_snrs[acc]
        logger.info(
            f"  {acc}: n={len(snrs)}, mean_snr={np.mean(snrs):.1f}dB, "
            f"std={np.std(snrs):.1f}dB"
        )

    return ConfoundResult(
        test_name="kruskal_wallis",
        variable_a="accent",
        variable_b="snr_db",
        statistic=h_stat,
        p_value=p_value,
        effect_size=epsilon_sq,
        effect_size_name="epsilon_squared",
        is_significant=is_significant,
        is_blocking=is_blocking,
        interpretation=interpretation,
    )


def run_all_confound_checks(
    entries: list[ManifestEntry],
    gender_blocking_threshold: float = 0.3,
    duration_practical_diff_s: float = 1.0,
    snr_practical_diff_db: float = 5.0,
    check_snr: bool = True,
) -> list[ConfoundResult]:
    """Run all mandatory confound analyses.

    Args:
        entries: Manifest entries.
        gender_blocking_threshold: Cramer's V threshold.
        duration_practical_diff_s: Practical difference threshold.
        snr_practical_diff_db: Practical SNR difference threshold in dB.
        check_snr: Whether to run SNR analysis (requires audio files).

    Returns:
        List of ConfoundResult objects.
    """
    results = [
        analyze_accent_x_gender(entries, gender_blocking_threshold),
        analyze_accent_x_duration(entries, duration_practical_diff_s),
    ]

    if check_snr:
        results.append(analyze_accent_x_snr(entries, snr_practical_diff_db))

    blocking = [r for r in results if r.is_blocking]
    if blocking:
        logger.warning(
            f"BLOCKING confounds detected: "
            f"{[r.variable_b for r in blocking]}"
        )

    return results
