#!/usr/bin/env python3
"""Streaming census of Common Voice Portuguese accent metadata.

Counts speakers, utterances, and gender distribution per IBGE macro-region
using HuggingFace streaming mode (no disk download). Processes all splits
(train, validation, test).

This is a CENSUS script — it counts everything, applies no duration/quality
filters, and reports all raw values to inform the accent mapping.

Usage:
    python3 scripts/cv_pt_census.py
"""

import re
import sys
import time
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Accent mapping (from src/data/manifest.py — duplicated here so the script
# is self-contained and doesn't require project imports).
# ---------------------------------------------------------------------------

BIRTH_STATE_TO_MACRO_REGION: dict[str, str] = {
    "AC": "N", "AM": "N", "AP": "N", "PA": "N", "RO": "N", "RR": "N", "TO": "N",
    "AL": "NE", "BA": "NE", "CE": "NE", "MA": "NE", "PB": "NE",
    "PE": "NE", "PI": "NE", "RN": "NE", "SE": "NE",
    "DF": "CO", "GO": "CO", "MS": "CO", "MT": "CO",
    "ES": "SE", "MG": "SE", "RJ": "SE", "SP": "SE",
    "PR": "S", "RS": "S", "SC": "S",
}

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
    # State abbreviations (lowercase)
    "sp": "SE", "rj": "SE", "mg": "SE", "es": "SE",
    "rs": "S", "sc": "S", "pr": "S",
    "ba": "NE", "ce": "NE", "pe": "NE", "pb": "NE", "rn": "NE",
    "se": "NE", "al": "NE", "ma": "NE", "pi": "NE",
    "go": "CO", "df": "CO", "ms": "CO", "mt": "CO",
    "am": "N", "pa": "N", "ac": "N", "ap": "N",
    "ro": "N", "rr": "N", "to": "N",
    # Additional labels found in CV-PT 17.0 census (unambiguous mappings)
    "sotaque sulista": "S",
    "interior paulista": "SE",
    "paulista do interior": "SE",
    "paulistano do interior": "SE",
    "sertanejo nordestino": "NE", "sertanejo nordestino.": "NE",
    "interiorano": "SE",
    # "brasileiro" intentionally excluded — it's a nationality, not an accent label.
    # When it appears in compound labels (e.g., "Brasileiro,Sulista"), we want
    # the parser to skip it and pick up the actual regional token.
    "caipira": "SE",
    "curitibano": "S",
    "sotaque carioca": "SE",
    "cearenses": "NE",
    "sotaque brasiliense": "CO",
    "manezinho": "S",  # Florianopolis native
    "gaúcho - interior": "S",
    "sotaque paraíbano": "NE",
    "sotaque paraibano": "NE",
}

# Gender values accepted — case-insensitive, handles both CV formats
# CV 17.0 (fsicoli mirror) uses "male_masculine"/"female_feminine"
# CV older versions use "male"/"female"
VALID_GENDERS = {"male", "female", "male_masculine", "female_feminine"}
GENDER_MAP = {
    "male": "M", "female": "F",
    "male_masculine": "M", "female_feminine": "F",
}


def normalize_birth_state(raw_value: str) -> str | None:
    """Normalize a birth state name/abbreviation to 2-letter code."""
    val = raw_value.strip().upper()
    if val in BIRTH_STATE_TO_MACRO_REGION:
        return val
    if val in STATE_FULL_NAME_TO_ABBREV:
        return STATE_FULL_NAME_TO_ABBREV[val]
    return None


def normalize_cv_accent_single(token: str) -> str | None:
    """Try to map a single accent token to a macro-region."""
    val = token.strip().lower()
    # Strip trailing punctuation (e.g., "Paulistano." -> "paulistano")
    val = re.sub(r'[.\s]+$', '', val)
    if not val:
        return None
    if val in CV_ACCENT_TO_MACRO_REGION:
        return CV_ACCENT_TO_MACRO_REGION[val]
    # Fallback: try as state name/abbreviation
    state_abbrev = normalize_birth_state(token)
    if state_abbrev is not None:
        return BIRTH_STATE_TO_MACRO_REGION.get(state_abbrev)
    return None


# Regex patterns that extract region info from free-text accent descriptions
_FREETEXT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "região sul/norte/nordeste/sudeste/centro-oeste do Brasil"
    (re.compile(r'regi[aã]o\s+(sul|norte|nordeste|sudeste|centro[- ]?oeste)', re.I), None),
    # "interior de são paulo" / "interior do estado de são paulo"
    (re.compile(r'interior\s+(?:de|do estado de)\s+s[aã]o\s+paulo', re.I), "SE"),
    # "Manezinho (Florianópolis - SC)"
    (re.compile(r'manezinho', re.I), "S"),
    # "baixada santista"
    (re.compile(r'baixada\s+santista', re.I), "SE"),
    # "Fortaleza" (CE)
    (re.compile(r'fortaleza', re.I), "NE"),
    # "southern brazil" / "south region"
    (re.compile(r'south(?:ern)?\s+(?:brazil|region)', re.I), "S"),
    # "sotaque do [region]" — sometimes with extra words
    (re.compile(r'sotaque\s+(?:da\s+)?regi[aã]o\s+(sul|norte|nordeste|sudeste|centro[- ]?oeste)', re.I), None),
]

# Map extracted region names to macro-region codes
_REGION_NAME_MAP = {
    "sul": "S", "norte": "N", "nordeste": "NE",
    "sudeste": "SE", "centro-oeste": "CO", "centro oeste": "CO",
    "centrooeste": "CO",
}


def normalize_cv_accent(raw_value: str) -> str | None:
    """Normalize Common Voice accent label to IBGE macro-region.

    Handles single labels, compound (comma-separated) labels,
    trailing punctuation, and common free-text patterns. For compound
    labels, returns the first resolvable token's region (if any).
    """
    val = raw_value.strip()
    if not val:
        return None

    # Try direct match first (exact, case-insensitive)
    result = normalize_cv_accent_single(val)
    if result is not None:
        return result

    # Try splitting on commas (Common Voice users often list multiple labels)
    if ',' in val:
        for part in val.split(','):
            part = part.strip()
            if part:
                result = normalize_cv_accent_single(part)
                if result is not None:
                    return result

    # Try free-text patterns
    for pattern, fixed_region in _FREETEXT_PATTERNS:
        m = pattern.search(val)
        if m:
            if fixed_region is not None:
                return fixed_region
            # Extract region name from capture group
            region_name = m.group(1).strip().lower()
            return _REGION_NAME_MAP.get(region_name)

    return None


def run_census() -> None:
    """Run the streaming census across all CV-PT splits."""
    from datasets import load_dataset

    # Dataset source configuration
    # NOTE: fsicoli mirror requires trust_remote_code=True (custom loading script).
    # The project pipeline (src/data/pipeline.py L125) also uses trust_remote_code=True.
    # Mozilla's official repo requires HF auth token — fallback only.
    HF_DATASETS = [
        ("fsicoli/common_voice_17_0", {"name": "pt", "trust_remote_code": True}),
        ("mozilla-foundation/common_voice_17_0", {"name": "pt", "trust_remote_code": False, "token": True}),
    ]
    SPLITS = ["train", "validation", "test"]

    # Counters
    total_rows = 0
    rows_missing_accent = 0
    rows_unmapped_accent = 0
    rows_invalid_gender = 0
    rows_missing_client_id = 0

    # Per-region counters (only for fully valid rows: accent+gender+client_id)
    region_speakers: dict[str, set[str]] = defaultdict(set)
    region_utterances: Counter = Counter()
    region_gender: dict[str, Counter] = defaultdict(Counter)

    # Raw value analysis (counted for ALL rows, regardless of other filters)
    raw_accent_counts: Counter = Counter()
    unmapped_accent_counts: Counter = Counter()
    raw_gender_counts: Counter = Counter()

    # Per-split counters
    split_row_counts: Counter = Counter()

    # Try dataset sources in order
    dataset_id = None
    dataset_kwargs = None
    for hf_id, kwargs in HF_DATASETS:
        print(f"Trying dataset: {hf_id} ...")
        try:
            test_ds = load_dataset(hf_id, streaming=True, split="train", **kwargs)
            _row = next(iter(test_ds))
            dataset_id = hf_id
            dataset_kwargs = kwargs
            print(f"  SUCCESS: Connected to {hf_id}")
            print(f"  Columns: {list(_row.keys())}")
            # Show sample row for debugging
            print(f"  Sample gender: {_row.get('gender')!r}")
            print(f"  Sample accent: {_row.get('accent')!r}")
            print(f"  Sample client_id: {str(_row.get('client_id', ''))[:20]!r}...")
            break
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    if dataset_id is None:
        print("\nERROR: Could not connect to any dataset source.")
        sys.exit(1)

    print(f"\nUsing dataset: {dataset_id}")
    print(f"Splits to process: {SPLITS}")
    print("=" * 70)

    t0 = time.time()

    for split_name in SPLITS:
        print(f"\n--- Processing split: {split_name} ---")
        try:
            ds = load_dataset(dataset_id, streaming=True, split=split_name, **dataset_kwargs)
        except Exception as e:
            print(f"  WARNING: Could not load split '{split_name}': {e}")
            continue

        split_count = 0
        for row in ds:
            total_rows += 1
            split_count += 1

            if total_rows % 50_000 == 0:
                elapsed = time.time() - t0
                print(f"  [{split_name}] {total_rows:>9,} rows processed ({elapsed:.0f}s)")

            # --- Extract raw fields ---
            raw_accent = (row.get("accent") or "").strip()
            raw_gender = (row.get("gender") or "").strip()
            client_id = (row.get("client_id") or "").strip()

            # --- Always count raw values (for census completeness) ---
            if raw_accent:
                raw_accent_counts[raw_accent] += 1
            raw_gender_counts[raw_gender if raw_gender else "(empty)"] += 1

            # --- Try accent mapping (independent of other filters) ---
            macro_region = None
            if not raw_accent:
                rows_missing_accent += 1
            else:
                macro_region = normalize_cv_accent(raw_accent)
                if macro_region is None:
                    rows_unmapped_accent += 1
                    unmapped_accent_counts[raw_accent] += 1

            # --- Filter: client_id ---
            if not client_id:
                rows_missing_client_id += 1
                continue

            # --- Filter: gender ---
            gender_lower = raw_gender.lower()
            if gender_lower not in VALID_GENDERS:
                rows_invalid_gender += 1
                continue

            # --- If accent didn't map, skip for per-region stats ---
            if macro_region is None:
                continue

            # --- Valid row: accumulate ---
            gender_code = GENDER_MAP[gender_lower]
            region_speakers[macro_region].add(client_id)
            region_utterances[macro_region] += 1
            region_gender[macro_region][gender_code] += 1

        split_row_counts[split_name] = split_count
        print(f"  [{split_name}] done: {split_count:,} rows")

    elapsed_total = time.time() - t0

    # ===================================================================
    # REPORT
    # ===================================================================
    print("\n" + "=" * 70)
    print("COMMON VOICE PT — STREAMING CENSUS REPORT")
    print(f"Dataset: {dataset_id}")
    print(f"Elapsed: {elapsed_total:.1f}s")
    print("=" * 70)

    # --- 1. Totals ---
    print(f"\n1. TOTAL ROWS")
    print(f"   Total processed:        {total_rows:>10,}")
    for sn in SPLITS:
        print(f"     {sn:>12s}:          {split_row_counts.get(sn, 0):>10,}")
    print(f"   Missing/empty accent:   {rows_missing_accent:>10,}")
    print(f"   Unmapped accent:        {rows_unmapped_accent:>10,}")
    print(f"   Invalid/empty gender:   {rows_invalid_gender:>10,}")
    print(f"   Missing client_id:      {rows_missing_client_id:>10,}")

    total_valid = sum(region_utterances.values())
    total_speakers = sum(len(s) for s in region_speakers.values())
    print(f"\n   VALID (mapped) rows:    {total_valid:>10,}")
    print(f"   VALID unique speakers:  {total_speakers:>10,}")

    # --- 1b. Gender distribution (raw) ---
    print(f"\n1b. RAW GENDER VALUES (all rows)")
    print(f"    {'Count':>10}  {'Value'}")
    for gval, cnt in raw_gender_counts.most_common():
        print(f"    {cnt:>10,}  {gval!r}")

    # --- 2. Per-region breakdown ---
    print(f"\n2. PER MACRO-REGION BREAKDOWN")
    print(f"   {'Region':<8} {'Speakers':>10} {'Utterances':>12} {'M':>8} {'F':>8} {'M%':>6} {'F%':>6}")
    print(f"   {'-'*6:<8} {'-'*10:>10} {'-'*12:>12} {'-'*8:>8} {'-'*8:>8} {'-'*6:>6} {'-'*6:>6}")

    for region in ["N", "NE", "CO", "SE", "S"]:
        n_spk = len(region_speakers.get(region, set()))
        n_utt = region_utterances.get(region, 0)
        n_m = region_gender.get(region, Counter()).get("M", 0)
        n_f = region_gender.get(region, Counter()).get("F", 0)
        total_g = n_m + n_f
        pct_m = (n_m / total_g * 100) if total_g > 0 else 0
        pct_f = (n_f / total_g * 100) if total_g > 0 else 0
        print(f"   {region:<8} {n_spk:>10,} {n_utt:>12,} {n_m:>8,} {n_f:>8,} {pct_m:>5.1f}% {pct_f:>5.1f}%")

    print(f"   {'TOTAL':<8} {total_speakers:>10,} {total_valid:>12,}")

    # --- 3. Top raw accent values ---
    print(f"\n3. TOP 20 RAW ACCENT VALUES (all rows, before filtering)")
    print(f"   {'#':>4} {'Count':>10}  {'Mapped':>6}  {'Accent Label'}")
    print(f"   {'--':>4} {'-----':>10}  {'------':>6}  {'------------'}")
    for rank, (accent, count) in enumerate(raw_accent_counts.most_common(20), 1):
        mapped = normalize_cv_accent(accent)
        tag = mapped if mapped else "--"
        print(f"   {rank:>4} {count:>10,}  {tag:>6}  {accent!r}")

    # --- 4. Unmapped accent values ---
    print(f"\n4. ALL UNMAPPED ACCENT VALUES ({len(unmapped_accent_counts)} unique values, {sum(unmapped_accent_counts.values()):,} total rows)")
    if unmapped_accent_counts:
        print(f"   {'Count':>10}  {'Accent Label'}")
        print(f"   {'-----':>10}  {'------------'}")
        for accent, count in unmapped_accent_counts.most_common():
            print(f"   {count:>10,}  {accent!r}")
    else:
        print("   (none — all accented rows mapped successfully)")

    # --- 5. Summary ---
    print(f"\n5. SUMMARY")
    print(f"   Total rows in dataset:       {total_rows:>10,}")
    rows_with_accent = total_rows - rows_missing_accent
    pct_accent = (rows_with_accent / total_rows * 100) if total_rows > 0 else 0
    print(f"   Rows with accent info:       {rows_with_accent:>10,} ({pct_accent:.1f}%)")
    pct_mapped = (total_valid / total_rows * 100) if total_rows > 0 else 0
    print(f"   Rows fully valid & mapped:   {total_valid:>10,} ({pct_mapped:.1f}%)")
    print(f"   Unique speakers (valid):     {total_speakers:>10,}")
    print(f"   Unique raw accent labels:    {len(raw_accent_counts):>10,}")
    print(f"   Unmappable accent labels:    {len(unmapped_accent_counts):>10,}")
    if total_rows > 0:
        pct_accent_missing = rows_missing_accent / total_rows * 100
        print(f"   Accent missing rate:         {pct_accent_missing:>9.1f}%")
    print()


if __name__ == "__main__":
    run_census()
