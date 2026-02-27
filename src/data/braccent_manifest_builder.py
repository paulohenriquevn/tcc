"""Build manifest from Fake BrAccent dataset (REAL files only).

Reads archive.zip from the BrAccent corpus, extracts REAL wav files,
parses metadata from filenames, and maps to IBGE macro-regions.

Accent mapping strategy (decided 2026-02-27 meeting):
  - Non-CO speakers: use BrAccent folder label (perceived accent)
    Baiano -> NE, Nordestino -> NE, Carioca -> SE, Fluminense -> SE, Sulista -> S
  - CO speakers (DF, GO birth_state): re-mapped from "Sulista" to CO via birth_state

Speaker IDs are prefixed with "bra_" to prevent collisions with
CORAA-MUPE and Common Voice entries.

Filename pattern:
  converted_{speaker_id}_frase_{n}_{date}_{age}_{city}_{state}_{gender}_{education}.wav
"""

import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from src.data.manifest import (
    BIRTH_STATE_TO_MACRO_REGION,
    ManifestEntry,
    validate_manifest_consistency,
    write_manifest,
)
from src.data.manifest_builder import (
    _filter_regions_by_speaker_count,
    _filter_speakers_by_utterance_count,
)

logger = logging.getLogger(__name__)

TARGET_SR = 16_000

# BrAccent folder label -> IBGE macro-region (perceived accent mapping)
BRACCENT_FOLDER_TO_IBGE: dict[str, str] = {
    "Baiano": "NE",
    "Nordestino": "NE",
    "Carioca": "SE",
    "Fluminense": "SE",
    "Sulista": "S",
}

# States that trigger re-mapping to CO (regardless of folder label)
CO_STATES: set[str] = {"DistritoFederal", "Goias"}

# State names as they appear in filenames -> 2-letter abbreviations.
# BrAccent filenames use CamelCase without spaces (e.g. "RioGrandedoSul").
_BRACCENT_STATE_TO_ABBREV: dict[str, str] = {
    "Acre": "AC",
    "Alagoas": "AL",
    "Amapa": "AP",
    "Amazonas": "AM",
    "Bahia": "BA",
    "Ceara": "CE",
    "DistritoFederal": "DF",
    "EspiritoSanto": "ES",
    "Goias": "GO",
    "Maranhao": "MA",
    "MatoGrosso": "MT",
    "MatoGrossodoSul": "MS",
    "MinasGerais": "MG",
    "Para": "PA",
    "Paraiba": "PB",
    "Parana": "PR",
    "Pernambuco": "PE",
    "Piaui": "PI",
    "RiodeJaneiro": "RJ",
    "RioGrandedoNorte": "RN",
    "RioGrandedoSul": "RS",
    "Rondonia": "RO",
    "Roraima": "RR",
    "SantaCatarina": "SC",
    "SaoPaulo": "SP",
    "Sergipe": "SE",
    "Tocantins": "TO",
}

GENDER_MAP: dict[str, str] = {"Masculino": "M", "Feminino": "F"}

# Regex to parse BrAccent REAL filenames.
# Groups: speaker_id, frase_number, date, age, city, state, gender, education
_FILENAME_RE = re.compile(
    r"converted_"
    r"(?P<speaker_id>[a-z0-9]+)"
    r"_frase_"
    r"(?P<frase_num>\d+)"
    r"(?:_\d+)?"  # optional extra number before date (e.g. "frase_1_36_2018-...")
    r"_(?P<date>\d{4}-\d{2}-\d{2})"
    r"_(?P<age>\d+)"
    r"_(?P<city>[^_]+)"
    r"_(?P<state>[^_]+)"
    r"_(?P<gender>Masculino|Feminino)"
    r"_(?P<education>[^.]+)"
    r"\.wav$"
)


def parse_braccent_filename(filename: str) -> dict | None:
    """Parse metadata from a BrAccent REAL WAV filename.

    Args:
        filename: Just the filename (no directory path).

    Returns:
        Dict with keys: speaker_id, frase_num, date, age, city, state,
        gender, education. Returns None if parsing fails.
    """
    m = _FILENAME_RE.match(filename)
    if m is None:
        return None
    return m.groupdict()


def build_manifest_from_braccent(
    zip_path: Path,
    audio_output_dir: Path,
    manifest_output_path: Path,
    min_duration_s: float = 3.0,
    max_duration_s: float = 15.0,
    min_speakers_per_region: int = 0,
    min_utterances_per_speaker: int = 0,
    include_co_remapping: bool = True,
) -> tuple[list[ManifestEntry], dict]:
    """Build manifest from BrAccent archive.zip (REAL files only).

    Two-phase approach:
      Phase 1 — List zip contents, filter to REAL WAVs, parse metadata.
      Phase 2 — Extract, resample to 16 kHz, and save audio.

    Args:
        zip_path: Path to archive.zip containing BrAccent data.
        audio_output_dir: Directory to save extracted WAV files.
        manifest_output_path: Where to write the manifest JSONL.
        min_duration_s: Minimum utterance duration in seconds.
        max_duration_s: Maximum utterance duration in seconds.
        min_speakers_per_region: Minimum speakers per macro-region.
        min_utterances_per_speaker: Minimum utterances per speaker to keep.
        include_co_remapping: If True, re-map DF/GO speakers to CO accent.

    Returns:
        Tuple of (entries, stats_dict).

    Raises:
        FileNotFoundError: If zip_path doesn't exist.
        ValueError: If validation fails.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"BrAccent archive not found: {zip_path}")

    audio_output_dir = Path(audio_output_dir)
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    filter_stats = {
        "total_real_wavs": 0,
        "rejected_parse_error": 0,
        "rejected_unknown_gender": 0,
        "rejected_unknown_folder": 0,
        "rejected_unknown_state": 0,
        "rejected_duration": 0,
        "rejected_audio_error": 0,
        "co_remapped": 0,
        "accepted": 0,
    }

    entries: list[ManifestEntry] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Phase 1: Identify REAL WAV files and their folder labels
        real_wavs: list[tuple[str, str]] = []  # (zip_member_path, folder_label)
        for member in zf.namelist():
            if "/REAL/" in member and member.endswith(".wav"):
                # Extract folder label from path: FakeBrAccent 2/{label}/...
                parts = member.split("/")
                if len(parts) >= 2:
                    folder_label = parts[1]
                    if folder_label in BRACCENT_FOLDER_TO_IBGE:
                        real_wavs.append((member, folder_label))

        filter_stats["total_real_wavs"] = len(real_wavs)
        logger.info(f"Found {len(real_wavs)} REAL WAV files in BrAccent archive")

        # Phase 2: Parse metadata, extract audio, build entries
        for member_path, folder_label in real_wavs:
            filename = Path(member_path).name
            meta = parse_braccent_filename(filename)

            if meta is None:
                filter_stats["rejected_parse_error"] += 1
                logger.debug(f"Failed to parse filename: {filename}")
                continue

            # Validate gender
            gender = GENDER_MAP.get(meta["gender"])
            if gender is None:
                filter_stats["rejected_unknown_gender"] += 1
                continue

            # Determine accent: folder label or CO re-mapping
            state = meta["state"]
            state_abbrev = _BRACCENT_STATE_TO_ABBREV.get(state)

            if include_co_remapping and state in CO_STATES:
                accent = "CO"
                filter_stats["co_remapped"] += 1
            elif folder_label in BRACCENT_FOLDER_TO_IBGE:
                accent = BRACCENT_FOLDER_TO_IBGE[folder_label]
            else:
                filter_stats["rejected_unknown_folder"] += 1
                continue

            # Build IDs (prefixed to avoid collisions)
            raw_speaker_id = meta["speaker_id"]
            speaker_id = f"bra_{raw_speaker_id}"
            utt_id = f"bra_{raw_speaker_id}_frase_{meta['frase_num']}"

            # Extract and save audio
            wav_path = audio_output_dir / f"{utt_id}.wav"
            if not wav_path.exists():
                try:
                    audio_bytes = zf.read(member_path)
                    # Write raw first, then read back for resampling
                    tmp_path = wav_path.with_suffix(".tmp.wav")
                    tmp_path.write_bytes(audio_bytes)

                    audio_array, sr = sf.read(str(tmp_path), dtype="float32")
                    tmp_path.unlink()

                    # Resample if needed
                    if sr != TARGET_SR:
                        import torch
                        import torchaudio.functional as F

                        audio_tensor = torch.from_numpy(audio_array)
                        audio_tensor = F.resample(audio_tensor, sr, TARGET_SR)
                        audio_array = audio_tensor.numpy()

                    sf.write(str(wav_path), audio_array, TARGET_SR)
                    duration = len(audio_array) / TARGET_SR
                except Exception as e:
                    filter_stats["rejected_audio_error"] += 1
                    logger.warning(f"Audio error for {utt_id}: {e}")
                    if wav_path.exists():
                        wav_path.unlink()
                    continue
            else:
                # Already extracted — compute duration from file
                info = sf.info(str(wav_path))
                duration = info.duration

            # Duration filter
            if duration < min_duration_s or duration > max_duration_s:
                filter_stats["rejected_duration"] += 1
                continue

            birth_state = state_abbrev or state
            entry = ManifestEntry(
                utt_id=utt_id,
                audio_path=str(wav_path),
                speaker_id=speaker_id,
                accent=accent,
                gender=gender,
                duration_s=duration,
                sampling_rate=TARGET_SR,
                text_id=None,
                source="BrAccent",
                birth_state=birth_state,
            )
            entries.append(entry)
            filter_stats["accepted"] += 1

    logger.info(
        f"BrAccent: {filter_stats['accepted']} entries accepted "
        f"({filter_stats['co_remapped']} re-mapped to CO)"
    )

    # Early exit
    if not entries:
        logger.warning(f"No entries passed filters. Stats: {filter_stats}")
        return [], {
            "filter_stats": filter_stats,
            "manifest_sha256": None,
            "regions": {},
            "total_speakers": 0,
            "total_utterances": 0,
        }

    # Post-filtering: region and speaker count thresholds
    entries, _, dropped_regions = _filter_regions_by_speaker_count(
        entries, min_speakers_per_region
    )
    filter_stats["dropped_regions"] = dropped_regions

    if min_utterances_per_speaker > 0:
        entries, dropped_speakers = _filter_speakers_by_utterance_count(
            entries, min_utterances_per_speaker
        )
        filter_stats["rejected_few_utterances"] = len(dropped_speakers)
    else:
        filter_stats["rejected_few_utterances"] = 0

    # Validate consistency
    errors = validate_manifest_consistency(entries)
    if errors:
        for error in errors:
            logger.error(f"BrAccent manifest validation error: {error}")
        raise ValueError(
            f"BrAccent manifest validation failed with {len(errors)} errors"
        )

    # Write manifest
    sha256 = write_manifest(entries, manifest_output_path)
    logger.info(f"BrAccent manifest written to {manifest_output_path} (SHA-256: {sha256})")

    # Build stats
    region_speakers: dict[str, set[str]] = {}
    for entry in entries:
        region_speakers.setdefault(entry.accent, set()).add(entry.speaker_id)

    stats = {
        "filter_stats": filter_stats,
        "manifest_sha256": sha256,
        "regions": {
            region: {
                "n_speakers": len(speakers),
                "n_utterances": sum(1 for e in entries if e.accent == region),
            }
            for region, speakers in sorted(region_speakers.items())
        },
        "total_speakers": sum(len(s) for s in region_speakers.values()),
        "total_utterances": len(entries),
    }

    return entries, stats
