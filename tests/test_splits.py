"""Tests for speaker-disjoint and stratified split generation."""

import tempfile
from pathlib import Path

import pytest

from src.data.manifest import ManifestEntry
from src.data.splits import (
    assert_speaker_disjoint,
    assign_entries_to_splits,
    assign_entries_to_stratified_splits,
    generate_speaker_disjoint_splits,
    generate_stratified_splits,
    load_splits,
    load_stratified_splits,
    save_splits,
    save_stratified_splits,
)


def _make_entries(n_speakers_per_region: int = 4) -> list[ManifestEntry]:
    """Create test entries with speakers across 3 regions."""
    entries = []
    regions = [("SE", "SP"), ("NE", "BA"), ("S", "RS")]
    utt_counter = 0

    for region, state in regions:
        for spk_idx in range(n_speakers_per_region):
            speaker_id = f"spk_{region}_{spk_idx:03d}"
            for utt_idx in range(3):  # 3 utterances per speaker
                utt_counter += 1
                entries.append(ManifestEntry(
                    utt_id=f"utt_{utt_counter:04d}",
                    audio_path=f"audio/utt_{utt_counter:04d}.wav",
                    speaker_id=speaker_id,
                    accent=region,
                    gender="M" if spk_idx % 2 == 0 else "F",
                    duration_s=5.0 + utt_idx,
                    sampling_rate=16000,
                    text_id=f"txt_{utt_counter:04d}",
                    source="CORAA-MUPE",
                    birth_state=state,
                ))

    return entries


class TestSpeakerDisjointSplits:
    def test_no_speaker_in_multiple_splits(self):
        entries = _make_entries(n_speakers_per_region=10)
        split = generate_speaker_disjoint_splits(entries, seed=42)

        train_set = set(split.train_speakers)
        val_set = set(split.val_speakers)
        test_set = set(split.test_speakers)

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_all_speakers_assigned(self):
        entries = _make_entries(n_speakers_per_region=10)
        split = generate_speaker_disjoint_splits(entries, seed=42)

        all_assigned = (
            set(split.train_speakers)
            | set(split.val_speakers)
            | set(split.test_speakers)
        )
        all_speakers = {e.speaker_id for e in entries}

        assert all_assigned == all_speakers

    def test_deterministic_with_same_seed(self):
        entries = _make_entries(n_speakers_per_region=10)

        split1 = generate_speaker_disjoint_splits(entries, seed=42)
        split2 = generate_speaker_disjoint_splits(entries, seed=42)

        assert split1.train_speakers == split2.train_speakers
        assert split1.val_speakers == split2.val_speakers
        assert split1.test_speakers == split2.test_speakers

    def test_different_seeds_give_different_splits(self):
        entries = _make_entries(n_speakers_per_region=10)

        split1 = generate_speaker_disjoint_splits(entries, seed=42)
        split2 = generate_speaker_disjoint_splits(entries, seed=1337)

        # Very unlikely to be identical with different seeds
        assert split1.train_speakers != split2.train_speakers

    def test_each_region_in_each_split(self):
        entries = _make_entries(n_speakers_per_region=10)
        split = generate_speaker_disjoint_splits(entries, seed=42)

        # Build region lookup
        speaker_region = {}
        for e in entries:
            speaker_region[e.speaker_id] = e.accent

        for split_name, speakers in [
            ("train", split.train_speakers),
            ("val", split.val_speakers),
            ("test", split.test_speakers),
        ]:
            regions_in_split = {speaker_region[s] for s in speakers}
            assert len(regions_in_split) == 3, (
                f"Split '{split_name}' missing regions: "
                f"has {regions_in_split}, expected SE, NE, S"
            )

    def test_ratios_dont_sum_to_one_raises_error(self):
        entries = _make_entries()
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            generate_speaker_disjoint_splits(
                entries, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3
            )

    def test_too_few_speakers_raises_error(self):
        entries = _make_entries(n_speakers_per_region=1)
        with pytest.raises(ValueError, match="speakers"):
            generate_speaker_disjoint_splits(entries, seed=42)


class TestAssertSpeakerDisjoint:
    def test_disjoint_passes(self):
        assert_speaker_disjoint(["a", "b"], ["c", "d"], ["e", "f"])

    def test_overlap_train_val_fails(self):
        with pytest.raises(AssertionError, match="train->val"):
            assert_speaker_disjoint(["a", "b"], ["b", "c"], ["d", "e"])

    def test_overlap_train_test_fails(self):
        with pytest.raises(AssertionError, match="train->test"):
            assert_speaker_disjoint(["a", "b"], ["c", "d"], ["b", "e"])

    def test_overlap_val_test_fails(self):
        with pytest.raises(AssertionError, match="val->test"):
            assert_speaker_disjoint(["a", "b"], ["c", "d"], ["d", "e"])


class TestSplitPersistence:
    def test_save_and_load_roundtrip(self):
        entries = _make_entries(n_speakers_per_region=10)
        split = generate_speaker_disjoint_splits(entries, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_splits(split, Path(tmpdir))
            loaded = load_splits(path)

            assert loaded.train_speakers == split.train_speakers
            assert loaded.val_speakers == split.val_speakers
            assert loaded.test_speakers == split.test_speakers
            assert loaded.seed == split.seed

    def test_load_validates_disjointness(self):
        """If someone manually edits a split file to add overlap, load detects it."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad_split.json"
            bad_data = {
                "train_speakers": ["a", "b"],
                "val_speakers": ["b", "c"],  # overlap with train!
                "test_speakers": ["d", "e"],
                "seed": 42,
                "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                "total_speakers": 5,
                "total_utterances": 15,
                "utterances_per_split": {"train": 6, "val": 4, "test": 5},
            }
            path.write_text(json.dumps(bad_data))

            with pytest.raises(AssertionError, match="train->val"):
                load_splits(path)


class TestAssignEntries:
    def test_all_entries_assigned(self):
        entries = _make_entries(n_speakers_per_region=10)
        split = generate_speaker_disjoint_splits(entries, seed=42)

        assigned = assign_entries_to_splits(entries, split)

        total_assigned = (
            len(assigned["train"])
            + len(assigned["val"])
            + len(assigned["test"])
        )
        assert total_assigned == len(entries)

    def test_no_entry_in_wrong_split(self):
        entries = _make_entries(n_speakers_per_region=10)
        split = generate_speaker_disjoint_splits(entries, seed=42)

        assigned = assign_entries_to_splits(entries, split)
        val_speakers = set(split.val_speakers)

        for entry in assigned["val"]:
            assert entry.speaker_id in val_speakers


class TestStratifiedSplits:
    def test_same_speakers_in_train_and_test(self):
        entries = _make_entries(n_speakers_per_region=10)
        split = generate_stratified_splits(entries, seed=42)

        utt_map = {e.utt_id: e for e in entries}
        train_speakers = {utt_map[uid].speaker_id for uid in split.train_utt_ids}
        test_speakers = {utt_map[uid].speaker_id for uid in split.test_utt_ids}

        assert train_speakers == test_speakers

    def test_no_utterance_overlap(self):
        entries = _make_entries(n_speakers_per_region=10)
        split = generate_stratified_splits(entries, seed=42)

        overlap = set(split.train_utt_ids) & set(split.test_utt_ids)
        assert len(overlap) == 0

    def test_all_utterances_assigned(self):
        entries = _make_entries(n_speakers_per_region=10)
        split = generate_stratified_splits(entries, seed=42)

        total = len(split.train_utt_ids) + len(split.test_utt_ids)
        assert total == len(entries)

    def test_deterministic_with_same_seed(self):
        entries = _make_entries(n_speakers_per_region=10)

        s1 = generate_stratified_splits(entries, seed=42)
        s2 = generate_stratified_splits(entries, seed=42)

        assert s1.train_utt_ids == s2.train_utt_ids
        assert s1.test_utt_ids == s2.test_utt_ids

    def test_different_seeds_give_different_splits(self):
        entries = _make_entries(n_speakers_per_region=10)

        s1 = generate_stratified_splits(entries, seed=42)
        s2 = generate_stratified_splits(entries, seed=1337)

        assert s1.train_utt_ids != s2.train_utt_ids

    def test_invalid_train_ratio_raises_error(self):
        entries = _make_entries()
        with pytest.raises(ValueError, match="train_ratio"):
            generate_stratified_splits(entries, train_ratio=1.5)

    def test_save_and_load_roundtrip(self):
        entries = _make_entries(n_speakers_per_region=10)
        split = generate_stratified_splits(entries, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_stratified_splits(split, Path(tmpdir))
            loaded = load_stratified_splits(path)

            assert loaded.train_utt_ids == split.train_utt_ids
            assert loaded.test_utt_ids == split.test_utt_ids
            assert loaded.seed == split.seed

    def test_assign_entries(self):
        entries = _make_entries(n_speakers_per_region=10)
        split = generate_stratified_splits(entries, seed=42)

        assigned = assign_entries_to_stratified_splits(entries, split)
        total = len(assigned["train"]) + len(assigned["test"])
        assert total == len(entries)
