"""Manifest schema and I/O for CORAA-MUPE dataset.

The manifest is the single versioned artifact that describes all utterances
used in the experiment. Every downstream component (splits, features, probes)
depends on it.

Schema fields are defined as a frozen dataclass to prevent accidental mutation.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


# IBGE macro-region mapping: birth_state -> macro-region
BIRTH_STATE_TO_MACRO_REGION: dict[str, str] = {
    # Norte (N)
    "AC": "N", "AM": "N", "AP": "N", "PA": "N", "RO": "N", "RR": "N", "TO": "N",
    # Nordeste (NE)
    "AL": "NE", "BA": "NE", "CE": "NE", "MA": "NE", "PB": "NE",
    "PE": "NE", "PI": "NE", "RN": "NE", "SE": "NE",
    # Centro-Oeste (CO)
    "DF": "CO", "GO": "CO", "MS": "CO", "MT": "CO",
    # Sudeste (SE)
    "ES": "SE", "MG": "SE", "RJ": "SE", "SP": "SE",
    # Sul (S)
    "PR": "S", "RS": "S", "SC": "S",
}

VALID_MACRO_REGIONS = {"N", "NE", "CO", "SE", "S"}


@dataclass(frozen=True)
class ManifestEntry:
    """A single utterance in the manifest.

    All fields are required. Missing data must be investigated
    and resolved — never silently filled with defaults.
    """
    utt_id: str              # Unique utterance identifier
    audio_path: str          # Relative path to audio file
    speaker_id: str          # Speaker identifier (unique per person)
    accent: str              # IBGE macro-region: N, NE, CO, SE, S
    gender: str              # M or F
    duration_s: float        # Duration in seconds
    text_id: Optional[str]   # Text/transcription identifier (if available)
    source: str              # Source dataset (e.g., "CORAA-MUPE")
    birth_state: str         # Original birth_state from metadata

    def __post_init__(self) -> None:
        """Validate fields on creation."""
        if self.accent not in VALID_MACRO_REGIONS:
            raise ValueError(
                f"Invalid accent '{self.accent}' for utt_id={self.utt_id}. "
                f"Must be one of {VALID_MACRO_REGIONS}"
            )
        if self.gender not in ("M", "F"):
            raise ValueError(
                f"Invalid gender '{self.gender}' for utt_id={self.utt_id}. "
                f"Must be 'M' or 'F'"
            )
        if self.duration_s <= 0:
            raise ValueError(
                f"Invalid duration {self.duration_s}s for utt_id={self.utt_id}. "
                f"Must be positive"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ManifestEntry":
        return cls(**data)


def write_manifest(entries: list[ManifestEntry], output_path: Path) -> str:
    """Write manifest to JSONL and return SHA-256 hash.

    Args:
        entries: List of validated ManifestEntry objects.
        output_path: Path to write the .jsonl file.

    Returns:
        SHA-256 hex digest of the written file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    # Compute SHA-256
    sha256 = compute_file_hash(output_path)

    # Write hash sidecar
    hash_path = output_path.with_suffix(output_path.suffix + ".sha256")
    hash_path.write_text(f"{sha256}  {output_path.name}\n")

    return sha256


def read_manifest(manifest_path: Path) -> list[ManifestEntry]:
    """Read manifest from JSONL file.

    Args:
        manifest_path: Path to the .jsonl manifest file.

    Returns:
        List of ManifestEntry objects.

    Raises:
        FileNotFoundError: If manifest file doesn't exist.
        ValueError: If any entry fails validation.
    """
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(ManifestEntry.from_dict(data))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid manifest entry at line {line_num}: {e}"
                ) from e
    return entries


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        file_path: Path to the file.

    Returns:
        Hex digest string.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_manifest_hash(manifest_path: Path, expected_hash: str) -> bool:
    """Verify manifest integrity against expected hash.

    Args:
        manifest_path: Path to manifest file.
        expected_hash: Expected SHA-256 hex digest.

    Returns:
        True if hash matches.
    """
    actual = compute_file_hash(manifest_path)
    return actual == expected_hash


def validate_manifest_consistency(entries: list[ManifestEntry]) -> list[str]:
    """Validate manifest-level consistency constraints.

    Checks:
    - Each speaker_id maps to exactly one accent.
    - No duplicate utt_ids.
    - All accents have at least 1 speaker.

    Args:
        entries: List of manifest entries.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    # Check unique utt_ids
    utt_ids = [e.utt_id for e in entries]
    duplicates = {uid for uid in utt_ids if utt_ids.count(uid) > 1}
    if duplicates:
        errors.append(f"Duplicate utt_ids: {duplicates}")

    # Check speaker -> accent consistency
    speaker_accents: dict[str, set[str]] = {}
    for entry in entries:
        speaker_accents.setdefault(entry.speaker_id, set()).add(entry.accent)

    inconsistent = {
        spk: accents
        for spk, accents in speaker_accents.items()
        if len(accents) > 1
    }
    if inconsistent:
        errors.append(
            f"Speakers with multiple accents (must be 1): {inconsistent}"
        )

    # Check accent coverage
    accents_present = {e.accent for e in entries}
    missing = VALID_MACRO_REGIONS - accents_present
    if missing:
        errors.append(f"Missing macro-regions: {missing}")

    return errors
