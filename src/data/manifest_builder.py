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
