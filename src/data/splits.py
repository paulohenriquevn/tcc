"""Speaker-disjoint split generation and persistence.

Splits are versioned artifacts — once generated, they never change
within an experiment. Each speaker appears in exactly one split.

The split is persisted as a JSON file containing speaker ID lists
and metadata for auditability.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.manifest import ManifestEntry

logger = logging.getLogger(__name__)


@dataclass
class SplitInfo:
    """Metadata about a generated split."""
    train_speakers: list[str]
    val_speakers: list[str]
    test_speakers: list[str]
    seed: int
    ratios: dict[str, float]
    total_speakers: int
    total_utterances: int
    utterances_per_split: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "train_speakers": self.train_speakers,
            "val_speakers": self.val_speakers,
            "test_speakers": self.test_speakers,
            "seed": self.seed,
            "ratios": self.ratios,
            "total_speakers": self.total_speakers,
            "total_utterances": self.total_utterances,
            "utterances_per_split": self.utterances_per_split,
        }


def generate_speaker_disjoint_splits(
    entries: list[ManifestEntry],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    min_speakers_per_region_per_split: int = 1,
) -> SplitInfo:
    """Generate speaker-disjoint splits stratified by accent.

    Each speaker appears in exactly one split. Stratification ensures
    each accent region is represented in all splits.

    Args:
        entries: List of manifest entries.
        train_ratio: Proportion of speakers for training.
        val_ratio: Proportion of speakers for validation.
        test_ratio: Proportion of speakers for test.
        seed: Random seed for reproducibility.
        min_speakers_per_region_per_split: Minimum speakers per region in each split.

    Returns:
        SplitInfo with speaker assignments and metadata.

    Raises:
        ValueError: If ratios don't sum to ~1.0 or insufficient speakers.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    rng = np.random.RandomState(seed)

    # Group speakers by accent
    speaker_accent: dict[str, str] = {}
    for entry in entries:
        if entry.speaker_id in speaker_accent:
            assert speaker_accent[entry.speaker_id] == entry.accent, (
                f"Speaker {entry.speaker_id} has inconsistent accent: "
                f"{speaker_accent[entry.speaker_id]} vs {entry.accent}"
            )
        else:
            speaker_accent[entry.speaker_id] = entry.accent

    accent_speakers: dict[str, list[str]] = defaultdict(list)
    for spk, acc in speaker_accent.items():
        accent_speakers[acc].append(spk)

    # Validate minimum speakers per region
    for acc, speakers in accent_speakers.items():
        min_needed = min_speakers_per_region_per_split * 3  # 3 splits
        if len(speakers) < min_needed:
            raise ValueError(
                f"Region '{acc}' has {len(speakers)} speakers, "
                f"need at least {min_needed} for 3 splits"
            )

    # Split speakers per accent (stratified)
    train_speakers: list[str] = []
    val_speakers: list[str] = []
    test_speakers: list[str] = []

    for acc in sorted(accent_speakers.keys()):
        speakers = sorted(accent_speakers[acc])
        rng.shuffle(speakers)

        n = len(speakers)
        n_train = max(min_speakers_per_region_per_split, round(n * train_ratio))
        n_val = max(min_speakers_per_region_per_split, round(n * val_ratio))
        n_test = n - n_train - n_val

        if n_test < min_speakers_per_region_per_split:
            n_test = min_speakers_per_region_per_split
            n_train = n - n_val - n_test

        train_speakers.extend(speakers[:n_train])
        val_speakers.extend(speakers[n_train:n_train + n_val])
        test_speakers.extend(speakers[n_train + n_val:])

        logger.info(
            f"Region {acc}: {n} speakers -> "
            f"train={n_train}, val={n_val}, test={n_test}"
        )

    # Critical assertions — NEVER remove these
    assert_speaker_disjoint(train_speakers, val_speakers, test_speakers)

    # Count utterances per split
    train_set = set(train_speakers)
    val_set = set(val_speakers)
    test_set = set(test_speakers)

    utt_per_split = {"train": 0, "val": 0, "test": 0}
    for entry in entries:
        if entry.speaker_id in train_set:
            utt_per_split["train"] += 1
        elif entry.speaker_id in val_set:
            utt_per_split["val"] += 1
        elif entry.speaker_id in test_set:
            utt_per_split["test"] += 1

    return SplitInfo(
        train_speakers=sorted(train_speakers),
        val_speakers=sorted(val_speakers),
        test_speakers=sorted(test_speakers),
        seed=seed,
        ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
        total_speakers=len(speaker_accent),
        total_utterances=len(entries),
        utterances_per_split=utt_per_split,
    )


def assert_speaker_disjoint(
    train: list[str], val: list[str], test: list[str]
) -> None:
    """Assert that no speaker appears in more than one split.

    This is a HARD FAIL condition. If this assertion fails,
    ALL downstream results are INVALID.
    """
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    assert len(overlap_tv) == 0, (
        f"Speaker leakage train->val: {overlap_tv}"
    )
    assert len(overlap_tt) == 0, (
        f"Speaker leakage train->test: {overlap_tt}"
    )
    assert len(overlap_vt) == 0, (
        f"Speaker leakage val->test: {overlap_vt}"
    )


def save_splits(split_info: SplitInfo, output_dir: Path) -> Path:
    """Persist splits to versioned JSON file.

    Args:
        split_info: Split metadata and speaker lists.
        output_dir: Directory to write split file.

    Returns:
        Path to the written split file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"splits_seed{split_info.seed}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_info.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Splits saved to {output_path}")
    return output_path


def load_splits(split_path: Path) -> SplitInfo:
    """Load splits from JSON file and re-validate disjointness.

    Args:
        split_path: Path to the split JSON file.

    Returns:
        SplitInfo with validated speaker assignments.
    """
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Re-validate on load — trust nothing
    assert_speaker_disjoint(
        data["train_speakers"],
        data["val_speakers"],
        data["test_speakers"],
    )

    return SplitInfo(
        train_speakers=data["train_speakers"],
        val_speakers=data["val_speakers"],
        test_speakers=data["test_speakers"],
        seed=data["seed"],
        ratios=data["ratios"],
        total_speakers=data["total_speakers"],
        total_utterances=data["total_utterances"],
        utterances_per_split=data["utterances_per_split"],
    )


def assign_entries_to_splits(
    entries: list[ManifestEntry],
    split_info: SplitInfo,
) -> dict[str, list[ManifestEntry]]:
    """Assign manifest entries to splits based on speaker membership.

    Args:
        entries: All manifest entries.
        split_info: Split with speaker assignments.

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to entry lists.
    """
    train_set = set(split_info.train_speakers)
    val_set = set(split_info.val_speakers)
    test_set = set(split_info.test_speakers)

    result: dict[str, list[ManifestEntry]] = {
        "train": [], "val": [], "test": []
    }

    for entry in entries:
        if entry.speaker_id in train_set:
            result["train"].append(entry)
        elif entry.speaker_id in val_set:
            result["val"].append(entry)
        elif entry.speaker_id in test_set:
            result["test"].append(entry)
        else:
            logger.warning(
                f"Speaker {entry.speaker_id} not in any split, skipping "
                f"utt_id={entry.utt_id}"
            )

    return result
