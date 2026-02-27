"""Combine multiple source manifests into the Accents-PT-BR dataset.

Merges CORAA-MUPE and Common Voice manifests while checking for
ID collisions, speaker consistency, and cross-source balance.
"""

import logging
from collections import Counter
from pathlib import Path

from src.data.manifest import (
    ManifestEntry,
    read_manifest,
    validate_manifest_consistency,
    write_manifest,
)
from src.data.manifest_builder import (
    _filter_regions_by_speaker_count,
    _filter_speakers_by_utterance_count,
)

logger = logging.getLogger(__name__)


def combine_manifests(
    manifests: list[tuple[Path, str]],
    output_path: Path,
    min_speakers_per_region: int = 5,
    min_utterances_per_speaker: int = 3,
    exclude_regions: list[str] | None = None,
) -> tuple[list[ManifestEntry], dict]:
    """Merge multiple source manifests into Accents-PT-BR.

    Reads each manifest JSONL, combines, validates no ID collisions,
    applies region exclusion, region and speaker filters, writes combined
    manifest with SHA-256.

    Args:
        manifests: List of (manifest_path, source_name) tuples.
        output_path: Where to write the combined manifest JSONL.
        min_speakers_per_region: Minimum speakers per IBGE region.
        min_utterances_per_speaker: Minimum utterances per speaker.
        exclude_regions: IBGE macro-regions to exclude (e.g. ["CO"]).
            Applied before region/speaker filters.

    Returns:
        Tuple of (combined_entries, stats_dict).

    Raises:
        ValueError: If ID collisions detected or validation fails.
    """
    all_entries: list[ManifestEntry] = []
    per_source: dict[str, int] = {}

    for manifest_path, source_name in manifests:
        entries = read_manifest(manifest_path)
        per_source[source_name] = len(entries)
        logger.info(
            f"Loaded {len(entries)} entries from '{source_name}' "
            f"({manifest_path})"
        )
        all_entries.extend(entries)

    logger.info(f"Total entries before validation: {len(all_entries)}")

    # Check utt_id collisions across sources
    utt_ids = [e.utt_id for e in all_entries]
    utt_id_counts = Counter(utt_ids)
    collisions = {uid: cnt for uid, cnt in utt_id_counts.items() if cnt > 1}
    if collisions:
        raise ValueError(
            f"utt_id collisions across sources: {dict(list(collisions.items())[:10])} "
            f"({len(collisions)} total duplicates)"
        )

    # Check speaker_id collisions across sources
    source_speakers: dict[str, set[str]] = {}
    for entry in all_entries:
        source_speakers.setdefault(entry.source, set()).add(entry.speaker_id)

    source_names = list(source_speakers.keys())
    for i, src_a in enumerate(source_names):
        for src_b in source_names[i + 1:]:
            overlap = source_speakers[src_a] & source_speakers[src_b]
            if overlap:
                raise ValueError(
                    f"speaker_id collisions between '{src_a}' and '{src_b}': "
                    f"{list(overlap)[:10]} ({len(overlap)} total). "
                    f"Use source-prefixed IDs to avoid this."
                )

    # Verify each speaker maps to exactly one accent within the combined set
    speaker_accents: dict[str, set[str]] = {}
    for entry in all_entries:
        speaker_accents.setdefault(entry.speaker_id, set()).add(entry.accent)
    inconsistent = {
        spk: accents for spk, accents in speaker_accents.items()
        if len(accents) > 1
    }
    if inconsistent:
        raise ValueError(
            f"Speakers with multiple accents: "
            f"{dict(list(inconsistent.items())[:10])} "
            f"({len(inconsistent)} total)"
        )

    # Exclude specified regions (applied BEFORE speaker/region filters)
    if exclude_regions:
        before = len(all_entries)
        excluded_set = set(exclude_regions)
        all_entries = [e for e in all_entries if e.accent not in excluded_set]
        logger.info(
            f"Excluded regions {exclude_regions}: {before} -> {len(all_entries)} entries"
        )

    # Apply region filter
    all_entries, _, dropped_regions = _filter_regions_by_speaker_count(
        all_entries, min_speakers_per_region
    )

    # Apply speaker utterance filter
    if min_utterances_per_speaker > 0:
        all_entries, dropped_speakers = _filter_speakers_by_utterance_count(
            all_entries, min_utterances_per_speaker
        )
    else:
        dropped_speakers = []

    # Full consistency validation on combined manifest
    errors = validate_manifest_consistency(all_entries)
    if errors:
        for error in errors:
            logger.error(f"Combined manifest validation error: {error}")
        raise ValueError(
            f"Combined manifest validation failed with {len(errors)} errors"
        )

    # Write combined manifest
    sha256 = write_manifest(all_entries, output_path)
    logger.info(
        f"Combined manifest written to {output_path} "
        f"({len(all_entries)} entries, SHA-256: {sha256})"
    )

    # Build stats
    region_speakers: dict[str, set[str]] = {}
    source_counts: dict[str, int] = Counter()
    for entry in all_entries:
        region_speakers.setdefault(entry.accent, set()).add(entry.speaker_id)
        source_counts[entry.source] += 1

    stats = {
        "manifest_sha256": sha256,
        "total_utterances": len(all_entries),
        "total_speakers": len({e.speaker_id for e in all_entries}),
        "per_source_input": per_source,
        "per_source_output": dict(source_counts),
        "regions": {
            region: {
                "n_speakers": len(speakers),
                "n_utterances": sum(
                    1 for e in all_entries if e.accent == region
                ),
            }
            for region, speakers in sorted(region_speakers.items())
        },
        "dropped_regions": dropped_regions,
        "dropped_speakers_count": len(dropped_speakers),
    }

    return all_entries, stats


def analyze_source_distribution(entries: list[ManifestEntry]) -> dict:
    """Report source x accent x gender cross-tabulation for confound analysis.

    Returns:
        Dict with keys: source_x_accent (contingency counts),
        source_x_gender (contingency counts), per_source_stats, warnings.
    """
    source_x_accent: dict[str, Counter] = {}
    source_x_gender: dict[str, Counter] = {}
    source_accent_gender: dict[tuple[str, str, str], int] = Counter()
    source_accent_speakers: dict[tuple[str, str], set[str]] = {}

    for entry in entries:
        src = entry.source
        source_x_accent.setdefault(src, Counter())[entry.accent] += 1
        source_x_gender.setdefault(src, Counter())[entry.gender] += 1
        source_accent_gender[(src, entry.accent, entry.gender)] += 1
        key = (src, entry.accent)
        source_accent_speakers.setdefault(key, set()).add(entry.speaker_id)

    # Total utterances per accent across all sources
    accent_totals: Counter = Counter()
    for entry in entries:
        accent_totals[entry.accent] += 1

    # Check for source dominance per accent (>80% from one source)
    warnings: list[str] = []
    for accent, total in accent_totals.items():
        for src, counts in source_x_accent.items():
            src_count = counts.get(accent, 0)
            if total > 0 and src_count / total > 0.80:
                pct = src_count / total * 100
                warnings.append(
                    f"Accent '{accent}': {pct:.1f}% of utterances from "
                    f"'{src}' ({src_count}/{total}). "
                    f"Potential source confound."
                )
                logger.warning(warnings[-1])

    # Per-source stats with speaker counts
    per_source_stats: dict[str, dict] = {}
    for src in source_x_accent:
        speakers_by_accent = {}
        for accent in source_x_accent[src]:
            key = (src, accent)
            speakers_by_accent[accent] = {
                "utterances": source_x_accent[src][accent],
                "speakers": len(source_accent_speakers.get(key, set())),
            }
        per_source_stats[src] = {
            "accents": speakers_by_accent,
            "genders": dict(source_x_gender[src]),
        }

    return {
        "source_x_accent": {
            src: dict(counts) for src, counts in source_x_accent.items()
        },
        "source_x_gender": {
            src: dict(counts) for src, counts in source_x_gender.items()
        },
        "source_accent_gender": {
            f"{src}|{acc}|{gen}": cnt
            for (src, acc, gen), cnt in sorted(source_accent_gender.items())
        },
        "per_source_stats": per_source_stats,
        "warnings": warnings,
    }
