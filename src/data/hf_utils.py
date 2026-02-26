"""HuggingFace dataset utilities for Accents-PT-BR publication.

Provides conversion from ManifestEntry to HuggingFace format
and dataset card generation. Used by the dataset publication notebook.

Convention: the internal pipeline uses 'val' for the validation split,
but HuggingFace datasets use 'validation'. The INTERNAL_TO_HF_SPLITS
mapping handles this boundary. Callers should use to_hf_split_entries()
to convert before passing to build_dataset_card() or DatasetDict.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data.manifest import ManifestEntry

# Internal split name → HuggingFace split name.
# The pipeline uses 'val' everywhere; HF convention is 'validation'.
INTERNAL_TO_HF_SPLITS: dict[str, str] = {
    "train": "train",
    "val": "validation",
    "test": "test",
}


def to_hf_split_entries(
    split_entries: dict[str, list[ManifestEntry]],
) -> dict[str, list[ManifestEntry]]:
    """Rename internal split keys to HuggingFace convention.

    Converts 'val' → 'validation'. Passes through keys that are
    already in HF format (idempotent).

    Args:
        split_entries: Dict with internal keys ('train', 'val', 'test').

    Returns:
        Dict with HF keys ('train', 'validation', 'test').
    """
    return {
        INTERNAL_TO_HF_SPLITS.get(k, k): v
        for k, v in split_entries.items()
    }


def entries_to_hf_dict(entries: list[ManifestEntry]) -> dict:
    """Convert ManifestEntry list to dict-of-lists for HuggingFace Dataset.

    Args:
        entries: List of ManifestEntry objects.

    Returns:
        Dict with keys matching HuggingFace Dataset column names.
    """
    return {
        "audio": [e.audio_path for e in entries],
        "utt_id": [e.utt_id for e in entries],
        "speaker_id": [e.speaker_id for e in entries],
        "accent": [e.accent for e in entries],
        "gender": [e.gender for e in entries],
        "duration_s": [e.duration_s for e in entries],
        "source": [e.source for e in entries],
        "birth_state": [e.birth_state for e in entries],
        "text_id": [e.text_id or "" for e in entries],
    }


def build_dataset_card(
    combined_entries: list[ManifestEntry],
    split_entries: dict[str, list[ManifestEntry]],
    confound_summary: list[dict],
    accent_labels: list[str],
    gender_labels: list[str],
    source_labels: list[str],
    manifest_sha: str,
    commit_hash: str,
    seed: int,
) -> str:
    """Generate the HuggingFace dataset card (README.md) content.

    Args:
        combined_entries: All entries in the combined dataset.
        split_entries: Dict with HF split keys ('train', 'validation', 'test').
            Use to_hf_split_entries() to convert from internal naming.
        confound_summary: List of dicts with confound analysis results.
        accent_labels: Sorted list of accent class labels.
        gender_labels: Sorted list of gender class labels.
        source_labels: Sorted list of source class labels.
        manifest_sha: SHA-256 of the combined manifest.
        commit_hash: Git commit hash for provenance.
        seed: Global seed used.

    Returns:
        Complete dataset card as a string (Markdown with YAML frontmatter).
    """
    total_entries = len(combined_entries)
    total_speakers = len({e.speaker_id for e in combined_entries})
    total_duration_h = sum(e.duration_s for e in combined_entries) / 3600

    accent_stats = Counter(e.accent for e in combined_entries)
    source_stats = Counter(e.source for e in combined_entries)

    # Per-split stats
    split_stats = {}
    for name, entries in split_entries.items():
        split_stats[name] = {
            "utterances": len(entries),
            "speakers": len({e.speaker_id for e in entries}),
            "duration_h": sum(e.duration_s for e in entries) / 3600,
        }

    # Build accent distribution table
    accent_lines = []
    for acc in sorted(accent_stats.keys()):
        n = accent_stats[acc]
        pct = n / total_entries * 100
        spk_count = len({e.speaker_id for e in combined_entries if e.accent == acc})
        accent_lines.append(f"| {acc} | {n:,} | {pct:.1f}% | {spk_count} |")
    accent_table = "\n".join(accent_lines)

    # Build confound table
    confound_lines = []
    for cs in confound_summary:
        status = "BLOCKING" if cs["is_blocking"] else "OK"
        confound_lines.append(
            f"| {cs['variables']} | {cs['test']} | {cs['statistic']:.4f} | "
            f"{cs['p_value']:.6f} | {cs['effect_size_name']}={cs['effect_size']:.4f} | {status} |"
        )
    confound_table = "\n".join(confound_lines)

    return f"""---
language:
  - pt
license: cc-by-4.0
task_categories:
  - audio-classification
tags:
  - accent-classification
  - brazilian-portuguese
  - speech
  - regional-accent
  - ibge-macro-regions
  - tts-evaluation
size_categories:
  - 1K<n<10K
---

# Accents-PT-BR

A curated, multi-source dataset of Brazilian Portuguese speech annotated with IBGE macro-region
accent labels. Designed for training and evaluating accent classifiers used as external evaluators
in accent-controllable TTS research.

## Dataset Description

**Accents-PT-BR** combines two complementary sources of Brazilian Portuguese speech:

| Source | Type | Accent label origin |
|--------|------|--------------------|
| [CORAA-MUPE-ASR](https://huggingface.co/datasets/nilc-nlp/CORAA-MUPE-ASR) | Professional interviews | `birth_state` field (verified) |
| [Common Voice PT](https://commonvoice.mozilla.org/pt) (v17.0) | Crowd-sourced read speech | User-submitted `accent` field (noisy) |

Accent labels are normalized to **IBGE macro-regions**: N (Norte), NE (Nordeste), CO (Centro-Oeste),
SE (Sudeste), S (Sul).

### Key Properties

- **Speaker-disjoint splits**: No speaker appears in more than one split (train/validation/test).
  This is critical for fair evaluation of accent classifiers.
- **Source-prefixed IDs**: CORAA-MUPE and Common Voice entries use distinct ID namespaces
  (`cv_` prefix for Common Voice) to prevent collisions.
- **Multi-source**: Enables cross-source evaluation to detect source confounds
  (classifier learning recording conditions instead of accent).
- **All audio at 16 kHz mono WAV**.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total utterances | {total_entries:,} |
| Total speakers | {total_speakers} |
| Total duration | {total_duration_h:.1f} hours |
| Accent classes | {len(accent_stats)} (IBGE macro-regions) |
| Audio format | 16 kHz mono WAV |

### Accent Distribution

| Region | Utterances | % | Speakers |
|--------|-----------|---|----------|
{accent_table}

### Source Distribution

| Source | Utterances |
|--------|------------|
| CORAA-MUPE | {source_stats.get('CORAA-MUPE', 0):,} |
| CommonVoice-PT | {source_stats.get('CommonVoice-PT', 0):,} |

### Splits (Speaker-Disjoint)

| Split | Utterances | Speakers | Duration |
|-------|-----------|----------|----------|
| train | {split_stats['train']['utterances']:,} | {split_stats['train']['speakers']} | {split_stats['train']['duration_h']:.1f}h |
| validation | {split_stats['validation']['utterances']:,} | {split_stats['validation']['speakers']} | {split_stats['validation']['duration_h']:.1f}h |
| test | {split_stats['test']['utterances']:,} | {split_stats['test']['speakers']} | {split_stats['test']['duration_h']:.1f}h |

## Confound Analysis

Mandatory confound checks were run before publication:

| Variables | Test | Statistic | p-value | Effect size | Status |
|-----------|------|-----------|---------|-------------|--------|
{confound_table}

## Dataset Fields

| Field | Type | Description |
|-------|------|-------------|
| `audio` | Audio (16kHz) | Audio waveform |
| `utt_id` | string | Unique utterance identifier |
| `speaker_id` | string | Speaker identifier (unique per person, `cv_` prefix for Common Voice) |
| `accent` | ClassLabel | IBGE macro-region: {accent_labels} |
| `gender` | ClassLabel | Speaker gender: {gender_labels} |
| `duration_s` | float32 | Duration in seconds |
| `source` | ClassLabel | Source dataset: {source_labels} |
| `birth_state` | string | Original birth state / accent label from source |
| `text_id` | string | Transcription ID (if available) |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("paulohenriquevn/accents-pt-br")

# Access a sample
sample = ds['train'][0]
print(sample['accent'])      # e.g., 'SE'
print(sample['speaker_id'])  # e.g., 'coraa_spk123'
print(sample['audio'])       # {{'array': array([...]), 'sampling_rate': 16000}}

# Filter by accent
nordeste = ds['train'].filter(lambda x: x['accent'] == 'NE')
```

## Intended Use

This dataset is designed for:
1. **Training accent classifiers** for evaluating accent-controllable TTS systems.
2. **Cross-source generalization studies** (train on one source, test on another).
3. **Research on Brazilian Portuguese regional accent variation.**

It is NOT intended for:
- Speaker identification or re-identification.
- Commercial voice profiling.

## Limitations

- **Accent as proxy**: IBGE macro-regions are coarse. Intra-regional variation exists.
- **Common Voice labels are noisy**: User-submitted, not verified.
- **Source confound risk**: Different recording conditions between sources.
  Cross-source evaluation is recommended.
- **Class imbalance**: Some regions (N, CO) may have fewer speakers.

## Citation

If you use this dataset, please cite the underlying sources:

- **CORAA-MUPE**: Candido Jr. et al. (2023). CORAA: a large corpus of spontaneous and prepared speech.
- **Common Voice**: Ardila et al. (2020). Common Voice: A Massively-Multilingual Speech Corpus.

## Provenance

- **Manifest SHA-256**: `{manifest_sha}`
- **Pipeline commit**: `{commit_hash}`
- **Build date**: {datetime.now().strftime('%Y-%m-%d')}
- **Seed**: {seed}
- **Config**: `configs/accent_classifier.yaml`
"""
