"""High-level dataset pipeline for Accents-PT-BR.

Orchestrates: manifest loading/building, combining, confound analysis,
and speaker-disjoint splits. Used by both the dataset and classifier
notebooks to avoid code duplication (DRY).

This module is the single entry point for the full data pipeline.
Individual steps remain in their respective modules (manifest_builder,
cv_manifest_builder, combined_manifest, splits, confounds).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from src.analysis.confounds import ConfoundResult, run_all_confound_checks
from src.data.combined_manifest import analyze_source_distribution, combine_manifests
from src.data.manifest import ManifestEntry, compute_file_hash, read_manifest
from src.data.splits import (
    SplitInfo,
    assign_entries_to_splits,
    generate_speaker_disjoint_splits,
    save_splits,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    """Result of the Accents-PT-BR dataset pipeline.

    Contains everything both notebooks need for downstream use:
    combined entries, splits, confound results, and provenance hashes.
    """

    combined_entries: list[ManifestEntry]
    split_info: SplitInfo
    split_entries: dict[str, list[ManifestEntry]]
    confound_results: list[ConfoundResult]
    combined_sha256: str
    source_distribution: dict


def load_or_build_accents_dataset(
    config: dict,
    drive_base: Path,
) -> DatasetBundle:
    """Load or build the Accents-PT-BR combined dataset.

    Handles the full pipeline: CORAA-MUPE + Common Voice manifest
    loading/building, combining, confound analysis, and speaker-disjoint
    splits. All intermediate results are cached on drive_base.

    Args:
        config: Loaded accent_classifier.yaml config dict.
        drive_base: Platform-aware persistent cache directory.

    Returns:
        DatasetBundle with all components needed for downstream use.

    Raises:
        AssertionError: If speaker-disjoint constraint is violated.
        ValueError: If manifest validation fails.
    """
    # --- 1. Load or build CORAA-MUPE manifest ---
    coraa_manifest_path = drive_base / "coraa_mupe" / "manifest.jsonl"
    coraa_audio_dir = drive_base / "coraa_mupe" / "audio"

    if coraa_manifest_path.exists():
        logger.info("Loading CORAA-MUPE from cache: %s", coraa_manifest_path)
        coraa_entries = read_manifest(coraa_manifest_path)
        coraa_sha = compute_file_hash(coraa_manifest_path)
        print(f"CORAA-MUPE: {len(coraa_entries):,} entries (cached, SHA: {coraa_sha[:16]}...)")
    else:
        from datasets import concatenate_datasets, load_dataset

        print("Downloading CORAA-MUPE-ASR from HuggingFace (~42 GB first time)...")
        ds = load_dataset("nilc-nlp/CORAA-MUPE-ASR", token=True)
        all_data = concatenate_datasets([ds[split] for split in ds.keys()])
        print(f"Total concatenado: {len(all_data):,} rows")

        from src.data.manifest_builder import build_manifest_from_hf_dataset

        coraa_entries, coraa_stats = build_manifest_from_hf_dataset(
            dataset=all_data,
            audio_output_dir=coraa_audio_dir,
            manifest_output_path=coraa_manifest_path,
            speaker_type_filter=config["dataset"]["filters"].get("speaker_type", "R"),
            min_duration_s=config["dataset"]["filters"]["min_duration_s"],
            max_duration_s=config["dataset"]["filters"]["max_duration_s"],
            min_speakers_per_region=config["dataset"]["filters"]["min_speakers_per_region"],
            min_utterances_per_speaker=config["dataset"]["filters"].get(
                "min_utterances_per_speaker", 3
            ),
        )
        print(
            f"CORAA-MUPE: {len(coraa_entries):,} entries, "
            f"SHA-256: {coraa_stats['manifest_sha256']}"
        )

    # --- 2. Load or build Common Voice manifest ---
    cv_manifest_path = drive_base / "common_voice_pt" / "manifest.jsonl"
    cv_audio_dir = drive_base / "common_voice_pt" / "audio"

    if cv_manifest_path.exists():
        logger.info("Loading Common Voice PT from cache: %s", cv_manifest_path)
        cv_entries = read_manifest(cv_manifest_path)
        cv_sha = compute_file_hash(cv_manifest_path)
        print(f"Common Voice PT: {len(cv_entries):,} entries (cached, SHA: {cv_sha[:16]}...)")
    else:
        from datasets import concatenate_datasets, load_dataset

        cv_hf_id = config["dataset"]["sources"][1]["hf_id"]
        cv_lang = config["dataset"]["sources"][1]["hf_lang"]

        print(f"Loading Common Voice PT from HuggingFace ({cv_hf_id})...")
        # Load validated splits and concatenate (mirrors may not have the
        # aggregate "validated" split — train+validation+test is equivalent).
        _cv_splits = []
        for _split_name in ("train", "validation", "test"):
            _s = load_dataset(
                cv_hf_id, cv_lang, split=_split_name,
                token=True, trust_remote_code=True,
            )
            print(f"  {_split_name}: {len(_s):,} rows")
            _cv_splits.append(_s)
        cv_dataset = concatenate_datasets(_cv_splits)
        print(f"Common Voice validated (concatenated): {len(cv_dataset):,} rows")

        from src.data.cv_manifest_builder import build_manifest_from_common_voice

        cv_entries, cv_stats = build_manifest_from_common_voice(
            dataset=cv_dataset,
            audio_output_dir=cv_audio_dir,
            manifest_output_path=cv_manifest_path,
            min_duration_s=config["dataset"]["filters"]["min_duration_s"],
            max_duration_s=config["dataset"]["filters"]["max_duration_s"],
            min_speakers_per_region=config["dataset"]["filters"]["min_speakers_per_region"],
            min_utterances_per_speaker=config["dataset"]["filters"].get(
                "min_utterances_per_speaker", 3
            ),
        )
        if cv_entries:
            print(
                f"Common Voice PT: {len(cv_entries):,} entries, "
                f"SHA-256: {cv_stats['manifest_sha256']}"
            )
        else:
            print(
                "Common Voice PT: 0 usable entries "
                f"(filter stats: {cv_stats['filter_stats']})"
            )
            print("  Accent metadata is sparse in CV-PT. Proceeding with CORAA-MUPE only.")

    # --- 3. Combine manifests ---
    combined_manifest_path = drive_base / "accents_pt_br" / "manifest.jsonl"

    # Only include manifests that exist and have entries
    manifests_to_combine = [(coraa_manifest_path, "CORAA-MUPE")]
    if cv_manifest_path.exists():
        manifests_to_combine.append((cv_manifest_path, "CommonVoice-PT"))

    if combined_manifest_path.exists():
        logger.info("Loading combined manifest from cache: %s", combined_manifest_path)
        combined_entries = read_manifest(combined_manifest_path)
        combined_sha256 = compute_file_hash(combined_manifest_path)
        print(f"Combined: {len(combined_entries):,} entries (cached, SHA: {combined_sha256[:16]}...)")
    else:
        combined_entries, combined_stats = combine_manifests(
            manifests=manifests_to_combine,
            output_path=combined_manifest_path,
            min_speakers_per_region=config["dataset"]["filters"]["min_speakers_per_region"],
            min_utterances_per_speaker=config["dataset"]["filters"].get(
                "min_utterances_per_speaker", 3
            ),
        )
        combined_sha256 = combined_stats["manifest_sha256"]
        print(f"Combined: {len(combined_entries):,} entries, SHA-256: {combined_sha256}")

    # --- 4. Source distribution analysis ---
    source_dist = analyze_source_distribution(combined_entries)

    if source_dist["warnings"]:
        for w in source_dist["warnings"]:
            print(f"  WARNING: {w}")

    # --- 5. Confound analysis ---
    confound_results = run_all_confound_checks(
        combined_entries,
        gender_blocking_threshold=config["confounds"]["accent_x_gender"]["threshold_blocker"],
        duration_practical_diff_s=config["confounds"]["accent_x_duration"]["practical_diff_s"],
        check_snr=False,
        source_blocking_threshold=config["confounds"]["accent_x_source"]["threshold_blocker"],
    )

    print("\n=== CONFOUND ANALYSIS ===")
    blocking = False
    for result in confound_results:
        if result.is_blocking:
            status = "BLOCKING"
            blocking = True
        elif result.is_significant:
            status = "SIGNIFICANT"
        else:
            status = "OK"
        print(
            f"  {result.variable_a} x {result.variable_b}: {status} "
            f"({result.effect_size_name}={result.effect_size:.4f})"
        )

    if blocking:
        print("*** BLOCKING CONFOUND DETECTED. Review before proceeding. ***")

    # --- 6. Speaker-disjoint splits ---
    split_info = generate_speaker_disjoint_splits(
        combined_entries,
        train_ratio=config["splits"]["ratios"]["train"],
        val_ratio=config["splits"]["ratios"]["val"],
        test_ratio=config["splits"]["ratios"]["test"],
        seed=config["splits"]["seed"],
    )

    split_path = save_splits(split_info, Path(config["splits"]["output_dir"]))
    split_entries = assign_entries_to_splits(combined_entries, split_info)

    # --- 7. Verify speaker-disjoint (HARD FAIL — KB_HARD_FAIL_RULES §1) ---
    train_spk = {e.speaker_id for e in split_entries["train"]}
    val_spk = {e.speaker_id for e in split_entries["val"]}
    test_spk = {e.speaker_id for e in split_entries["test"]}

    assert len(train_spk & val_spk) == 0, f"Speaker leakage train->val: {train_spk & val_spk}"
    assert len(train_spk & test_spk) == 0, f"Speaker leakage train->test: {train_spk & test_spk}"
    assert len(val_spk & test_spk) == 0, f"Speaker leakage val->test: {val_spk & test_spk}"

    print(f"\nSpeaker-disjoint verification: PASSED")
    print(f"Splits saved: {split_path}")
    print(
        f"Train: {len(split_info.train_speakers)} speakers, "
        f"{split_info.utterances_per_split['train']:,} utts"
    )
    print(
        f"Val:   {len(split_info.val_speakers)} speakers, "
        f"{split_info.utterances_per_split['val']:,} utts"
    )
    print(
        f"Test:  {len(split_info.test_speakers)} speakers, "
        f"{split_info.utterances_per_split['test']:,} utts"
    )

    return DatasetBundle(
        combined_entries=combined_entries,
        split_info=split_info,
        split_entries=split_entries,
        confound_results=confound_results,
        combined_sha256=combined_sha256,
        source_distribution=source_dist,
    )
