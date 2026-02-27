"""Build manifest from Common Voice Portuguese dataset.

Reads a HuggingFace Common Voice dataset object and produces a validated,
versioned JSONL manifest with all required fields. Speaker IDs and utt_ids
are prefixed with "cv_" to prevent collisions with CORAA-MUPE entries.

Two-phase approach for efficiency:
  Phase 1 — Filter on metadata columns (no audio decoding, fast).
  Phase 2 — Decode, resample to 16 kHz, and save audio only for rows
             that passed filters.
"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

from src.data.manifest import (
    ManifestEntry,
    normalize_cv_accent,
    validate_manifest_consistency,
    write_manifest,
)
from src.data.manifest_builder import (
    _filter_regions_by_speaker_count,
    _filter_speakers_by_utterance_count,
)

logger = logging.getLogger(__name__)

TARGET_SR = 16_000

_CV_GENDER_MAP: dict[str, str] = {
    "male": "M",
    "female": "F",
}


def build_manifest_from_common_voice(
    dataset,
    audio_output_dir: Path,
    manifest_output_path: Path,
    min_duration_s: float = 3.0,
    max_duration_s: float = 15.0,
    min_speakers_per_region: int = 5,
    min_utterances_per_speaker: int = 0,
) -> tuple[list[ManifestEntry], dict]:
    """Build manifest from a HuggingFace Common Voice Portuguese dataset.

    Two-phase approach for efficiency:
      Phase 1 — Filter on metadata columns (no audio decoding, fast).
      Phase 2 — Decode and save audio only for rows that passed filters.

    Expected dataset columns: client_id, path, sentence, age, gender,
    accent, audio. The 'duration' column is optional — if absent, duration
    is computed from the decoded audio array in Phase 2.

    Args:
        dataset: HuggingFace Dataset (Common Voice Portuguese split).
        audio_output_dir: Directory to save filtered audio WAV files.
        manifest_output_path: Where to write the manifest JSONL.
        min_duration_s: Minimum utterance duration in seconds.
        max_duration_s: Maximum utterance duration in seconds.
        min_speakers_per_region: Minimum speakers per macro-region.
        min_utterances_per_speaker: Minimum utterances per speaker to keep.

    Returns:
        Tuple of (entries, stats_dict).

    Raises:
        ValueError: If required columns are missing or validation fails.
    """
    audio_output_dir = Path(audio_output_dir)
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    # Validate required columns (duration is optional — computed from audio if absent)
    required_columns = {"client_id", "path", "gender", "accent", "audio"}
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Available: {dataset.column_names}"
        )

    has_duration_column = "duration" in dataset.column_names

    logger.info(f"Processing Common Voice dataset: {len(dataset)} rows")
    if not has_duration_column:
        logger.info(
            "No 'duration' column found — duration filter deferred to Phase 2 (audio decode)"
        )

    # ── Phase 1: Fast metadata filtering (no audio decoding) ──
    logger.info("Phase 1: Filtering by metadata (no audio decode)...")
    accents = dataset["accent"]
    durations = dataset["duration"] if has_duration_column else None
    genders = dataset["gender"]
    client_ids = dataset["client_id"]

    filter_stats = {
        "total_raw": len(dataset),
        "rejected_missing_accent": 0,
        "rejected_unknown_accent": 0,
        "rejected_duration": 0,
        "rejected_missing_gender": 0,
        "rejected_missing_client_id": 0,
        "rejected_audio_error": 0,
        "accepted": 0,
    }

    pass_indices: list[int] = []
    for idx in range(len(dataset)):
        # Filter: accent field exists and non-empty
        raw_accent = (accents[idx] or "").strip()
        if not raw_accent:
            filter_stats["rejected_missing_accent"] += 1
            continue

        # Filter: accent normalizes to a valid macro-region
        if normalize_cv_accent(raw_accent) is None:
            filter_stats["rejected_unknown_accent"] += 1
            continue

        # Filter: duration in range (skip if column absent — deferred to Phase 2)
        if durations is not None:
            dur = durations[idx]
            if dur is None:
                filter_stats["rejected_duration"] += 1
                continue
            dur = float(dur)
            if dur < min_duration_s or dur > max_duration_s:
                filter_stats["rejected_duration"] += 1
                continue

        # Filter: gender maps to M/F
        raw_gender = (genders[idx] or "").strip().lower()
        if raw_gender not in _CV_GENDER_MAP:
            filter_stats["rejected_missing_gender"] += 1
            continue

        # Filter: client_id exists
        cid = (client_ids[idx] or "").strip()
        if not cid:
            filter_stats["rejected_missing_client_id"] += 1
            continue

        pass_indices.append(idx)

    logger.info(
        f"Phase 1 complete: {len(pass_indices)} / {len(dataset)} rows "
        f"passed metadata filters"
    )

    # ── Phase 2: Decode audio and save (only filtered rows) ──
    logger.info("Phase 2: Saving audio for filtered rows...")
    filtered_ds = dataset.select(pass_indices)

    filtered: list[ManifestEntry] = []
    seen_utt_ids: set[str] = set()

    for i in range(len(filtered_ds)):
        row = filtered_ds[i]

        raw_accent = row["accent"].strip()
        accent = normalize_cv_accent(raw_accent)
        gender = _CV_GENDER_MAP[row["gender"].strip().lower()]
        client_id = row["client_id"].strip()
        speaker_id = f"cv_{client_id}"

        # Derive utt_id from audio path field
        audio_path_orig = row.get("path", "") or ""
        if audio_path_orig:
            utt_id = f"cv_{Path(audio_path_orig).stem}"
        else:
            utt_id = f"cv_{pass_indices[i]:06d}"

        # Guarantee uniqueness
        if utt_id in seen_utt_ids:
            utt_id = f"{utt_id}_{pass_indices[i]:06d}"
        seen_utt_ids.add(utt_id)

        # Decode and save audio as 16 kHz WAV
        audio_data = row["audio"]
        audio_array = np.array(audio_data["array"], dtype=np.float32)
        audio_sr = int(audio_data["sampling_rate"])

        # Compute duration from audio (always accurate, handles missing column)
        if has_duration_column:
            duration = float(row["duration"])
        else:
            duration = len(audio_array) / audio_sr
            # Deferred duration filter (Phase 1 skipped it when column was absent)
            if duration < min_duration_s or duration > max_duration_s:
                filter_stats["rejected_duration"] += 1
                continue

        if audio_sr != TARGET_SR:
            audio_tensor = torch.from_numpy(audio_array)
            audio_tensor = F.resample(audio_tensor, audio_sr, TARGET_SR)
            audio_array = audio_tensor.numpy()

        wav_path = audio_output_dir / f"{utt_id}.wav"
        try:
            sf.write(str(wav_path), audio_array, TARGET_SR)
        except Exception as e:
            filter_stats["rejected_audio_error"] += 1
            logger.warning(f"Failed to save audio for {utt_id}: {e}")
            continue

        entry = ManifestEntry(
            utt_id=utt_id,
            audio_path=str(wav_path),
            speaker_id=speaker_id,
            accent=accent,
            gender=gender,
            duration_s=duration,
            sampling_rate=TARGET_SR,
            text_id=None,
            source="CommonVoice-PT",
            birth_state=raw_accent,
        )
        filtered.append(entry)
        filter_stats["accepted"] += 1

        if filter_stats["accepted"] % 1000 == 0:
            logger.info(f"  Saved {filter_stats['accepted']} audio files...")

    logger.info(f"Phase 2 complete: {len(filtered)} entries with audio saved")

    # ── Early exit: no usable entries ──
    if not filtered:
        logger.warning(
            "No entries passed all filters. "
            f"Stats: {filter_stats}"
        )
        return [], {
            "filter_stats": filter_stats,
            "manifest_sha256": None,
            "regions": {},
            "total_speakers": 0,
            "total_utterances": 0,
        }

    # ── Post-filtering and validation ──
    filtered, region_speakers, dropped_regions = _filter_regions_by_speaker_count(
        filtered, min_speakers_per_region
    )
    filter_stats["dropped_regions"] = dropped_regions

    if min_utterances_per_speaker > 0:
        filtered, dropped_speakers = _filter_speakers_by_utterance_count(
            filtered, min_utterances_per_speaker
        )
        filter_stats["rejected_few_utterances"] = len(dropped_speakers)
        filter_stats["dropped_speakers"] = dropped_speakers
    else:
        filter_stats["rejected_few_utterances"] = 0
        filter_stats["dropped_speakers"] = []

    errors = validate_manifest_consistency(filtered)
    if errors:
        for error in errors:
            logger.error(f"Manifest validation error: {error}")
        raise ValueError(
            f"Manifest validation failed with {len(errors)} errors"
        )

    # Write manifest
    sha256 = write_manifest(filtered, manifest_output_path)
    logger.info(f"Manifest written to {manifest_output_path} (SHA-256: {sha256})")

    # Build stats
    region_speakers_after: dict[str, set] = {}
    for entry in filtered:
        region_speakers_after.setdefault(entry.accent, set()).add(entry.speaker_id)

    stats = {
        "filter_stats": filter_stats,
        "manifest_sha256": sha256,
        "regions": {
            region: {
                "n_speakers": len(speakers),
                "n_utterances": sum(
                    1 for e in filtered if e.accent == region
                ),
            }
            for region, speakers in sorted(region_speakers_after.items())
        },
        "total_speakers": sum(len(s) for s in region_speakers_after.values()),
        "total_utterances": len(filtered),
    }

    return filtered, stats
