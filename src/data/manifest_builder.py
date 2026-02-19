"""Build manifest from CORAA-MUPE dataset.

Reads raw CORAA-MUPE metadata and produces a validated, versioned
JSONL manifest with all required fields.

This script is the ONLY entry point for data ingestion.
All downstream code reads from the manifest, never from raw files.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.data.manifest import (
    BIRTH_STATE_TO_MACRO_REGION,
    ManifestEntry,
    validate_manifest_consistency,
    write_manifest,
)

logger = logging.getLogger(__name__)


def build_manifest_from_coraa(
    metadata_path: Path,
    audio_dir: Path,
    output_path: Path,
    speaker_type_filter: str = "R",
    min_duration_s: float = 3.0,
    max_duration_s: float = 15.0,
    min_speakers_per_region: int = 8,
) -> tuple[list[ManifestEntry], dict]:
    """Build manifest from CORAA-MUPE metadata.

    Args:
        metadata_path: Path to CORAA-MUPE metadata file (CSV or JSONL).
        audio_dir: Root directory containing audio files.
        output_path: Where to write the manifest JSONL.
        speaker_type_filter: Speaker type to keep ("R" = interviewee).
        min_duration_s: Minimum utterance duration.
        max_duration_s: Maximum utterance duration.
        min_speakers_per_region: Minimum speakers per macro-region.

    Returns:
        Tuple of (entries, stats_dict).

    Raises:
        ValueError: If validation fails or insufficient data.
    """
    raw_entries = _load_raw_metadata(metadata_path)

    logger.info(f"Raw entries loaded: {len(raw_entries)}")

    # Apply filters
    filtered = []
    filter_stats = {
        "total_raw": len(raw_entries),
        "rejected_speaker_type": 0,
        "rejected_duration": 0,
        "rejected_missing_birth_state": 0,
        "rejected_unknown_state": 0,
        "rejected_missing_gender": 0,
        "accepted": 0,
    }

    for raw in raw_entries:
        # Filter by speaker type
        if raw.get("speaker_type") != speaker_type_filter:
            filter_stats["rejected_speaker_type"] += 1
            continue

        # Filter by duration
        duration = raw.get("duration_s") or raw.get("duration")
        if duration is None:
            filter_stats["rejected_duration"] += 1
            continue
        duration = float(duration)
        if duration < min_duration_s or duration > max_duration_s:
            filter_stats["rejected_duration"] += 1
            continue

        # Validate birth_state
        birth_state = raw.get("birth_state", "").strip().upper()
        if not birth_state:
            filter_stats["rejected_missing_birth_state"] += 1
            continue
        if birth_state not in BIRTH_STATE_TO_MACRO_REGION:
            filter_stats["rejected_unknown_state"] += 1
            logger.warning(f"Unknown birth_state: '{birth_state}'")
            continue

        # Validate gender
        gender = raw.get("speaker_gender", raw.get("gender", "")).strip().upper()
        if gender not in ("M", "F"):
            filter_stats["rejected_missing_gender"] += 1
            continue

        # Build entry
        accent = BIRTH_STATE_TO_MACRO_REGION[birth_state]
        speaker_id = str(raw.get("speaker_id", raw.get("speaker", "")))

        entry = ManifestEntry(
            utt_id=str(raw.get("utt_id", raw.get("id", ""))),
            audio_path=str(raw.get("audio_path", raw.get("path", ""))),
            speaker_id=speaker_id,
            accent=accent,
            gender=gender,
            duration_s=duration,
            text_id=raw.get("text_id"),
            source="CORAA-MUPE",
            birth_state=birth_state,
        )
        filtered.append(entry)
        filter_stats["accepted"] += 1

    logger.info(f"After filters: {len(filtered)} entries")
    for key, count in filter_stats.items():
        logger.info(f"  {key}: {count}")

    # Validate speaker per region
    region_speakers: dict[str, set] = {}
    for entry in filtered:
        region_speakers.setdefault(entry.accent, set()).add(entry.speaker_id)

    for region, speakers in sorted(region_speakers.items()):
        n_speakers = len(speakers)
        if n_speakers < min_speakers_per_region:
            raise ValueError(
                f"Region '{region}' has {n_speakers} speakers, "
                f"minimum is {min_speakers_per_region}"
            )
        logger.info(
            f"Region {region}: {n_speakers} speakers, "
            f"{sum(1 for e in filtered if e.accent == region)} utterances"
        )

    # Validate manifest consistency
    errors = validate_manifest_consistency(filtered)
    if errors:
        for error in errors:
            logger.error(f"Manifest validation error: {error}")
        raise ValueError(
            f"Manifest validation failed with {len(errors)} errors"
        )

    # Write manifest
    sha256 = write_manifest(filtered, output_path)
    logger.info(f"Manifest written to {output_path} (SHA-256: {sha256})")

    # Build stats
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
            for region, speakers in sorted(region_speakers.items())
        },
        "total_speakers": sum(len(s) for s in region_speakers.values()),
        "total_utterances": len(filtered),
    }

    return filtered, stats


def build_manifest_from_hf_dataset(
    dataset,
    audio_output_dir: Path,
    manifest_output_path: Path,
    speaker_type_filter: str = "R",
    min_duration_s: float = 3.0,
    max_duration_s: float = 15.0,
    min_speakers_per_region: int = 8,
) -> tuple[list[ManifestEntry], dict]:
    """Build manifest from a HuggingFace CORAA-MUPE dataset object.

    Two-phase approach for efficiency:
      Phase 1 — Filter on metadata columns (no audio decoding, fast).
      Phase 2 — Decode and save audio only for rows that passed filters.

    Expected dataset columns: speaker_code, speaker_type, speaker_gender,
    birth_state, duration, audio.

    Args:
        dataset: HuggingFace Dataset (concatenate splits before calling).
        audio_output_dir: Directory to save filtered audio WAV files.
        manifest_output_path: Where to write the manifest JSONL.
        speaker_type_filter: Speaker type to keep ("R" = interviewee).
        min_duration_s: Minimum utterance duration.
        max_duration_s: Maximum utterance duration.
        min_speakers_per_region: Minimum speakers per macro-region.

    Returns:
        Tuple of (entries, stats_dict).

    Raises:
        ValueError: If required columns are missing or validation fails.
    """
    import soundfile as sf

    audio_output_dir = Path(audio_output_dir)
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    # Validate required columns
    required_columns = {
        "speaker_code", "speaker_type", "speaker_gender",
        "birth_state", "duration", "audio",
    }
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Available: {dataset.column_names}"
        )

    logger.info(f"Processing HuggingFace dataset: {len(dataset)} rows")

    # ── Phase 1: Fast metadata filtering (no audio decoding) ──
    logger.info("Phase 1: Filtering by metadata (no audio decode)...")
    speaker_types = dataset["speaker_type"]
    durations = dataset["duration"]
    birth_states = dataset["birth_state"]
    genders = dataset["speaker_gender"]

    filter_stats = {
        "total_raw": len(dataset),
        "rejected_speaker_type": 0,
        "rejected_duration": 0,
        "rejected_missing_birth_state": 0,
        "rejected_unknown_state": 0,
        "rejected_missing_gender": 0,
        "rejected_audio_error": 0,
        "accepted": 0,
    }

    pass_indices = []
    for idx in range(len(dataset)):
        if speaker_types[idx] != speaker_type_filter:
            filter_stats["rejected_speaker_type"] += 1
            continue

        dur = durations[idx]
        if dur is None:
            filter_stats["rejected_duration"] += 1
            continue
        dur = float(dur)
        if dur < min_duration_s or dur > max_duration_s:
            filter_stats["rejected_duration"] += 1
            continue

        bs = (birth_states[idx] or "").strip().upper()
        if not bs:
            filter_stats["rejected_missing_birth_state"] += 1
            continue
        if bs not in BIRTH_STATE_TO_MACRO_REGION:
            filter_stats["rejected_unknown_state"] += 1
            continue

        g = (genders[idx] or "").strip().upper()
        if g not in ("M", "F"):
            filter_stats["rejected_missing_gender"] += 1
            continue

        pass_indices.append(idx)

    logger.info(
        f"Phase 1 complete: {len(pass_indices)} / {len(dataset)} rows "
        f"passed metadata filters"
    )

    # ── Phase 2: Decode audio and save (only filtered rows) ──
    logger.info("Phase 2: Saving audio for filtered rows...")
    filtered_ds = dataset.select(pass_indices)

    filtered = []
    seen_utt_ids: set[str] = set()

    for i in range(len(filtered_ds)):
        row = filtered_ds[i]

        speaker_id = str(row["speaker_code"])
        duration = float(row["duration"])
        birth_state = row["birth_state"].strip().upper()
        gender = row["speaker_gender"].strip().upper()
        accent = BIRTH_STATE_TO_MACRO_REGION[birth_state]

        # Derive utt_id from audio path, fall back to global index
        audio_data = row["audio"]
        audio_path_orig = audio_data.get("path", "")
        if audio_path_orig:
            utt_id = Path(audio_path_orig).stem
        else:
            utt_id = f"coraa_{pass_indices[i]:06d}"

        # Guarantee uniqueness
        if utt_id in seen_utt_ids:
            utt_id = f"{utt_id}_{pass_indices[i]:06d}"
        seen_utt_ids.add(utt_id)

        # Save audio to WAV
        wav_path = audio_output_dir / f"{utt_id}.wav"
        try:
            sf.write(
                str(wav_path),
                audio_data["array"],
                audio_data["sampling_rate"],
            )
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
            text_id=None,
            source="CORAA-MUPE",
            birth_state=birth_state,
        )
        filtered.append(entry)
        filter_stats["accepted"] += 1

        if filter_stats["accepted"] % 1000 == 0:
            logger.info(f"  Saved {filter_stats['accepted']} audio files...")

    logger.info(f"Phase 2 complete: {len(filtered)} entries with audio saved")

    # ── Validation ──
    region_speakers: dict[str, set] = {}
    for entry in filtered:
        region_speakers.setdefault(entry.accent, set()).add(entry.speaker_id)

    for region, speakers in sorted(region_speakers.items()):
        n_speakers = len(speakers)
        if n_speakers < min_speakers_per_region:
            raise ValueError(
                f"Region '{region}' has {n_speakers} speakers, "
                f"minimum is {min_speakers_per_region}"
            )
        logger.info(
            f"Region {region}: {n_speakers} speakers, "
            f"{sum(1 for e in filtered if e.accent == region)} utterances"
        )

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
            for region, speakers in sorted(region_speakers.items())
        },
        "total_speakers": sum(len(s) for s in region_speakers.values()),
        "total_utterances": len(filtered),
    }

    return filtered, stats


def _load_raw_metadata(metadata_path: Path) -> list[dict]:
    """Load raw metadata from CSV or JSONL.

    Supports both formats to accommodate different CORAA-MUPE releases.
    """
    suffix = metadata_path.suffix.lower()

    if suffix == ".jsonl":
        entries = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    elif suffix == ".csv":
        import csv
        entries = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(dict(row))
        return entries

    else:
        raise ValueError(
            f"Unsupported metadata format: {suffix}. Use .jsonl or .csv"
        )
