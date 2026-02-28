"""Build manifest from CORAA-MUPE dataset.

Reads raw CORAA-MUPE metadata and produces a validated, versioned
JSONL manifest with all required fields.

This script is the ONLY entry point for data ingestion.
All downstream code reads from the manifest, never from raw files.
"""

import io
import json
import logging
import struct
from pathlib import Path
from typing import Optional

from src.data.manifest import (
    BIRTH_STATE_TO_MACRO_REGION,
    ManifestEntry,
    normalize_birth_state,
    validate_manifest_consistency,
    write_manifest,
)

logger = logging.getLogger(__name__)


def _filter_regions_by_speaker_count(
    entries: list["ManifestEntry"],
    min_speakers_per_region: int,
) -> tuple[list["ManifestEntry"], dict[str, set], list[str]]:
    """Drop regions with fewer than min_speakers_per_region speakers.

    Implements the fallback described in TECHNICAL_VALIDATION_PROTOCOL.md §4.3:
    if a region has < 8 speakers, drop it instead of hard-failing.

    Returns:
        Tuple of (filtered_entries, region_speakers_map, dropped_regions).

    Raises:
        ValueError: If no regions survive the filter.
    """
    region_speakers: dict[str, set] = {}
    for entry in entries:
        region_speakers.setdefault(entry.accent, set()).add(entry.speaker_id)

    dropped_regions = []
    kept_regions = []
    for region, speakers in sorted(region_speakers.items()):
        n_speakers = len(speakers)
        if n_speakers < min_speakers_per_region:
            dropped_regions.append(region)
            logger.warning(
                f"Region '{region}' has {n_speakers} speakers "
                f"(minimum {min_speakers_per_region}). "
                f"Dropping per protocol §4.3 fallback."
            )
        else:
            kept_regions.append(region)
            logger.info(
                f"Region {region}: {n_speakers} speakers, "
                f"{sum(1 for e in entries if e.accent == region)} utterances"
            )

    if not kept_regions:
        raise ValueError(
            "No regions have enough speakers after filtering. "
            f"min_speakers_per_region={min_speakers_per_region}, "
            f"all regions: {dict((r, len(s)) for r, s in region_speakers.items())}"
        )

    if dropped_regions:
        logger.warning(
            f"Dropped {len(dropped_regions)} regions: {dropped_regions}. "
            f"Keeping {len(kept_regions)} regions: {kept_regions}."
        )
        entries = [e for e in entries if e.accent not in set(dropped_regions)]
        # Rebuild region_speakers for kept regions only
        region_speakers = {
            r: s for r, s in region_speakers.items()
            if r not in set(dropped_regions)
        }

    return entries, region_speakers, dropped_regions


def _filter_speakers_by_utterance_count(
    entries: list["ManifestEntry"],
    min_utterances: int,
) -> tuple[list["ManifestEntry"], list[str]]:
    """Drop speakers with fewer than min_utterances utterances.

    Args:
        entries: List of manifest entries.
        min_utterances: Minimum utterances per speaker to keep.

    Returns:
        Tuple of (filtered_entries, dropped_speaker_ids).

    Raises:
        ValueError: If no entries survive the filter.
    """
    speaker_utts: dict[str, int] = {}
    for entry in entries:
        speaker_utts[entry.speaker_id] = speaker_utts.get(entry.speaker_id, 0) + 1

    dropped_speakers = [
        spk for spk, count in speaker_utts.items()
        if count < min_utterances
    ]

    if dropped_speakers:
        dropped_set = set(dropped_speakers)
        logger.warning(
            f"Dropping {len(dropped_speakers)} speakers with < {min_utterances} "
            f"utterances (total utterances removed: "
            f"{sum(speaker_utts[s] for s in dropped_speakers)})"
        )
        entries = [e for e in entries if e.speaker_id not in dropped_set]

    if not entries:
        raise ValueError(
            f"No entries remain after filtering speakers with < {min_utterances} "
            f"utterances. All speakers were below threshold."
        )

    return entries, dropped_speakers


def build_manifest_from_coraa(
    metadata_path: Path,
    audio_dir: Path,
    output_path: Path,
    speaker_type_filter: str = "R",
    min_duration_s: float = 3.0,
    max_duration_s: float = 15.0,
    min_speakers_per_region: int = 5,
    min_utterances_per_speaker: int = 0,
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

        # Validate birth_state (handles both "SP" and "São Paulo" formats)
        raw_birth_state = raw.get("birth_state", "").strip()
        if not raw_birth_state:
            filter_stats["rejected_missing_birth_state"] += 1
            continue
        birth_state = normalize_birth_state(raw_birth_state)
        if birth_state is None:
            filter_stats["rejected_unknown_state"] += 1
            logger.warning(f"Unknown birth_state: '{raw_birth_state}'")
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
            sampling_rate=int(raw.get("sampling_rate", 16000)),
            text_id=raw.get("text_id"),
            source="CORAA-MUPE",
            birth_state=birth_state,
        )
        filtered.append(entry)
        filter_stats["accepted"] += 1

    logger.info(f"After filters: {len(filtered)} entries")
    for key, count in filter_stats.items():
        logger.info(f"  {key}: {count}")

    # Validate speaker per region (with fallback for insufficient regions)
    filtered, region_speakers, dropped_regions = _filter_regions_by_speaker_count(
        filtered, min_speakers_per_region
    )
    filter_stats["dropped_regions"] = dropped_regions

    # Filter speakers with too few utterances
    if min_utterances_per_speaker > 0:
        filtered, dropped_speakers = _filter_speakers_by_utterance_count(
            filtered, min_utterances_per_speaker
        )
        filter_stats["rejected_few_utterances"] = len(dropped_speakers)
        filter_stats["dropped_speakers"] = dropped_speakers
    else:
        filter_stats["rejected_few_utterances"] = 0
        filter_stats["dropped_speakers"] = []

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
    region_speakers_after = {}
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


def build_manifest_from_hf_dataset(
    dataset,
    audio_output_dir: Path,
    manifest_output_path: Path,
    speaker_type_filter: str = "R",
    min_duration_s: float = 3.0,
    max_duration_s: float = 15.0,
    min_speakers_per_region: int = 5,
    min_utterances_per_speaker: int = 0,
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

        raw_bs = (birth_states[idx] or "").strip()
        if not raw_bs:
            filter_stats["rejected_missing_birth_state"] += 1
            continue
        bs = normalize_birth_state(raw_bs)
        if bs is None:
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

    # ── Phase 2: Save audio for filtered rows (raw bytes, no decode) ──
    #
    # Performance: HF datasets' __getitem__ with a single index does a
    # per-row Arrow seek + full audio decode (~1.6s/item on CORAA-MUPE).
    # Instead, we:
    #   1. cast_column("audio", Audio(decode=False)) → raw bytes, no decode
    #   2. enumerate() → sequential Arrow reads (batched internally)
    #   3. Write raw WAV bytes directly → skip decode→encode roundtrip
    # Result: ~0.01–0.05s/item instead of ~1.6s/item.
    logger.info("Phase 2: Saving audio for filtered rows (raw bytes)...")
    filtered_ds = dataset.select(pass_indices)

    from datasets import Audio as HFAudio

    no_decode_ds = filtered_ds.cast_column("audio", HFAudio(decode=False))

    filtered = []
    seen_utt_ids: set[str] = set()

    for i, row in enumerate(no_decode_ds):
        speaker_id = str(row["speaker_code"])
        duration = float(row["duration"])
        birth_state = normalize_birth_state(row["birth_state"])
        gender = row["speaker_gender"].strip().upper()
        accent = BIRTH_STATE_TO_MACRO_REGION[birth_state]

        # Derive utt_id from audio path, fall back to global index
        audio_raw = row["audio"]
        audio_path_orig = audio_raw.get("path", "")
        if audio_path_orig:
            utt_id = Path(audio_path_orig).stem
        else:
            utt_id = f"coraa_{pass_indices[i]:06d}"

        # Guarantee uniqueness
        if utt_id in seen_utt_ids:
            utt_id = f"{utt_id}_{pass_indices[i]:06d}"
        seen_utt_ids.add(utt_id)

        # Save audio (skip if already exists from a previous build)
        raw_bytes = audio_raw["bytes"]
        wav_path = audio_output_dir / f"{utt_id}.wav"

        if not wav_path.exists():
            try:
                if raw_bytes[:4] == b"RIFF":
                    # WAV format — direct byte copy (no decode/encode)
                    wav_path.write_bytes(raw_bytes)
                else:
                    # Non-WAV (FLAC/MP3/etc): decode and re-encode as WAV
                    data, sr = sf.read(io.BytesIO(raw_bytes))
                    sf.write(str(wav_path), data, sr)
            except Exception as e:
                filter_stats["rejected_audio_error"] += 1
                logger.warning(f"Failed to save audio for {utt_id}: {e}")
                continue

        # Get sampling rate without full decode
        if raw_bytes[:4] == b"RIFF":
            # Standard WAV header: sample rate at bytes 24-27 (uint32 LE)
            audio_sr = struct.unpack_from("<I", raw_bytes, 24)[0]
        else:
            # Fallback: read from the written file
            audio_sr = sf.info(str(wav_path)).samplerate

        entry = ManifestEntry(
            utt_id=utt_id,
            audio_path=str(wav_path),
            speaker_id=speaker_id,
            accent=accent,
            gender=gender,
            duration_s=duration,
            sampling_rate=audio_sr,
            text_id=None,
            source="CORAA-MUPE",
            birth_state=birth_state,
        )
        filtered.append(entry)
        filter_stats["accepted"] += 1

        if filter_stats["accepted"] % 5000 == 0:
            logger.info(f"  Saved {filter_stats['accepted']} audio files...")

    logger.info(f"Phase 2 complete: {len(filtered)} entries with audio saved")

    # ── Validation (with fallback for insufficient regions) ──
    filtered, region_speakers, dropped_regions = _filter_regions_by_speaker_count(
        filtered, min_speakers_per_region
    )
    filter_stats["dropped_regions"] = dropped_regions

    # Filter speakers with too few utterances
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

    # Rebuild region_speakers after all filtering
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
