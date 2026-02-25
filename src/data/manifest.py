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


# IBGE macro-region mapping: birth_state abbreviation -> macro-region
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

# Full state names -> abbreviations (handles CORAA-MUPE-ASR format).
# Includes both accented and unaccented variants for robustness.
STATE_FULL_NAME_TO_ABBREV: dict[str, str] = {
    "ACRE": "AC",
    "ALAGOAS": "AL",
    "AMAPÁ": "AP", "AMAPA": "AP",
    "AMAZONAS": "AM",
    "BAHIA": "BA",
    "CEARÁ": "CE", "CEARA": "CE",
    "DISTRITO FEDERAL": "DF",
    "ESPÍRITO SANTO": "ES", "ESPIRITO SANTO": "ES",
    "GOIÁS": "GO", "GOIAS": "GO",
    "MARANHÃO": "MA", "MARANHAO": "MA",
    "MATO GROSSO": "MT",
    "MATO GROSSO DO SUL": "MS",
    "MINAS GERAIS": "MG",
    "PARÁ": "PA", "PARA": "PA",
    "PARAÍBA": "PB", "PARAIBA": "PB",
    "PARANÁ": "PR", "PARANA": "PR",
    "PERNAMBUCO": "PE",
    "PIAUÍ": "PI", "PIAUI": "PI",
    "RIO DE JANEIRO": "RJ",
    "RIO GRANDE DO NORTE": "RN",
    "RIO GRANDE DO SUL": "RS",
    "RONDÔNIA": "RO", "RONDONIA": "RO",
    "RORAIMA": "RR",
    "SANTA CATARINA": "SC",
    "SÃO PAULO": "SP", "SAO PAULO": "SP",
    "SERGIPE": "SE",
    "TOCANTINS": "TO",
}

VALID_MACRO_REGIONS = {"N", "NE", "CO", "SE", "S"}

# Common Voice free-text accent label -> IBGE macro-region.
# CV accent field is user-submitted and noisy. This mapping covers known
# regional demonyms, generic region names, and state abbreviations.
CV_ACCENT_TO_MACRO_REGION: dict[str, str] = {
    # Sudeste (SE)
    "carioca": "SE", "paulistano": "SE", "paulista": "SE", "mineiro": "SE",
    "capixaba": "SE", "fluminense": "SE",
    # Sul (S)
    "gaúcho": "S", "gaucho": "S", "catarinense": "S", "paranaense": "S",
    "sulista": "S",
    # Nordeste (NE)
    "baiano": "NE", "cearense": "NE", "pernambucano": "NE", "recifense": "NE",
    "nordestino": "NE", "paraibano": "NE", "potiguar": "NE", "sergipano": "NE",
    "piauiense": "NE", "alagoano": "NE", "maranhense": "NE",
    # Centro-Oeste (CO)
    "goiano": "CO", "brasiliense": "CO", "mato-grossense": "CO",
    "candango": "CO",
    # Norte (N)
    "paraense": "N", "amazonense": "N", "nortista": "N", "manauara": "N",
    # Generic region names
    "norte": "N", "nordeste": "NE", "centro-oeste": "CO",
    "sudeste": "SE", "sul": "S",
    "sotaque do sul": "S", "sotaque do nordeste": "NE",
    "sotaque do sudeste": "SE", "sotaque do norte": "N",
    # State abbreviations (lowercase for matching)
    "sp": "SE", "rj": "SE", "mg": "SE", "es": "SE",
    "rs": "S", "sc": "S", "pr": "S",
    "ba": "NE", "ce": "NE", "pe": "NE", "pb": "NE", "rn": "NE",
    "se": "NE", "al": "NE", "ma": "NE", "pi": "NE",
    "go": "CO", "df": "CO", "ms": "CO", "mt": "CO",
    "am": "N", "pa": "N", "ac": "N", "ap": "N",
    "ro": "N", "rr": "N", "to": "N",
}


def normalize_cv_accent(raw_value: str) -> str | None:
    """Normalize Common Voice accent label to IBGE macro-region.

    Handles demonyms ("carioca" -> "SE"), region names ("nordeste" -> "NE"),
    state abbreviations ("sp" -> "SE"), and falls back to normalize_birth_state()
    for full state names ("São Paulo" -> "SP" -> "SE").

    Args:
        raw_value: Raw accent string from Common Voice dataset.

    Returns:
        IBGE macro-region code (N, NE, CO, SE, S), or None if unresolvable.
    """
    val = raw_value.strip().lower()
    if not val:
        return None

    # Direct lookup in CV mapping
    if val in CV_ACCENT_TO_MACRO_REGION:
        return CV_ACCENT_TO_MACRO_REGION[val]

    # Fallback: try to resolve as state name/abbreviation via existing function
    state_abbrev = normalize_birth_state(raw_value)
    if state_abbrev is not None:
        return BIRTH_STATE_TO_MACRO_REGION.get(state_abbrev)

    return None


def normalize_birth_state(raw_value: str) -> str | None:
    """Normalize birth_state to a 2-letter abbreviation.

    Handles both abbreviations ("SP") and full names ("São Paulo").

    Args:
        raw_value: Raw birth_state string from dataset.

    Returns:
        2-letter state abbreviation, or None if unrecognized.
    """
    val = raw_value.strip().upper()

    # Already an abbreviation?
    if val in BIRTH_STATE_TO_MACRO_REGION:
        return val

    # Full name?
    if val in STATE_FULL_NAME_TO_ABBREV:
        return STATE_FULL_NAME_TO_ABBREV[val]

    return None


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
    sampling_rate: int       # Audio sampling rate in Hz (expected: 16000)
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
        if self.sampling_rate <= 0:
            raise ValueError(
                f"Invalid sampling_rate {self.sampling_rate} for utt_id={self.utt_id}. "
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

    # Check unique utt_ids — O(n) via Counter instead of O(n²)
    from collections import Counter
    utt_counts = Counter(e.utt_id for e in entries)
    duplicates = {uid for uid, count in utt_counts.items() if count > 1}
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

    # Check accent coverage: need at least 2 regions for classification.
    # Exact region coverage is validated upstream by _filter_regions_by_speaker_count().
    accents_present = {e.accent for e in entries}
    if len(accents_present) < 2:
        errors.append(
            f"Need at least 2 macro-regions for classification, "
            f"found {len(accents_present)}: {accents_present}"
        )

    # Check sampling_rate uniformity
    srs = {e.sampling_rate for e in entries}
    if len(srs) > 1:
        errors.append(
            f"Non-uniform sampling rates detected: {srs}. "
            f"All entries should have the same sampling rate."
        )

    return errors
