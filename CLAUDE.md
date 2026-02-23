# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Controle Explícito de Sotaque Regional em pt-BR** — undergraduate thesis (TCC) investigating whether LoRA adapters on a frozen TTS backbone can achieve explicit accent control while preserving speaker identity, with disentangled embeddings S (speaker) and A (accent).

**Core question:** `Audio = TTS(text, S, A)` — can we vary A independently of S?

**Stack:** Qwen3-TTS 1.7B-CustomVoice (Apache-2.0) + LoRA + CORAA-MUPE dataset + 1x GPU 24GB (RTX 4090)

## Research Methodology: Stage-Gate

The project follows a **Stage-Gate** model. Each Stage has a formal Gate decision (PASS / ADJUST / FAIL) based on objective metrics, not subjective impressions.

| Stage | Focus | Key Artifacts |
|-------|-------|---------------|
| **Stage 1** | Baseline: zero-shot metrics, environment, dataset audit | `RESEARCH_ROADMAP.md` experiments 1.1–1.6 |
| **Stage 2** | Pilot: LoRA training, loss convergence, first results | Experiments 2.1–2.6 |
| **Stage 3** | Evaluation: full metrics with CI 95%, leakage probes, red team | Experiments 3.1–3.6 |

**Current status:** Stage 1 GO passed. Stage 1.5 (latent separability audit) — pipeline em construção. Nenhum resultado empírico ainda. Thresholds unificados adotados do template do mentor (ver `docs/protocol/TECHNICAL_VALIDATION_PROTOCOL.md`).

Gate decisions are taken by the `protocol-gate` Agent Team (all 5 personas).

## Agent Personas

Five research personas coordinate work via `.claude/agents/`. Each has domain expertise and veto power:

| Persona | Agent file | Domain | Veto criterion |
|---------|-----------|--------|----------------|
| **Rafael Monteiro** | `rafael-monteiro` | Backbone, LoRA, experiment design | Accent emerges as superficial prosody only |
| **Mariana Alves** | `mariana-alves` | Dataset, splits, confounds | Correlation accent↔speaker/mic/quality |
| **Lucas Andrade** | `lucas-andrade` | Pipeline, GPU, reproducibility | Pipeline not reexecutable |
| **Felipe Nakamura** | `felipe-nakamura` | Metrics, baselines, leakage probes | No measurable signal above baseline |
| **Ana K. Silva** | `ana-silva` | Red team (read-only, NO code) | Result doesn't survive hostile reading |

**Routing rule:** 1 persona → use Task tool (subagent). 2+ personas in parallel → use Agent Teams.

## Key Skills (slash commands)

| Skill | Persona | Purpose |
|-------|---------|---------|
| `/baseline-check` | Felipe | Measure zero-shot baselines |
| `/dataset-audit` | Mariana | Validate splits, metadata, confounds |
| `/experiment-design` | Rafael | Formulate hypothesis + ablations |
| `/leakage-test` | Felipe | Train probes, measure A→S and S→A leakage |
| `/metrics-evaluation` | Felipe | Compute all metrics with CI |
| `/model-validation` | Rafael | Verify backbone + LoRA config |
| `/pipeline-review` | Lucas | Check reproducibility, VRAM, seeds |
| `/red-team-review` | Ana | Adversarial audit (never implements) |
| `/deep-paper-search` | Rafael | RAG search on Arxiv + PubMed |
| `/meeting` | head-research | Multi-persona consolidated reporting |

## Validation Protocol (non-negotiable)

Defined in `docs/protocol/TECHNICAL_VALIDATION_PROTOCOL.md`. Three mandatory metrics:

1. **Accent controllability:** balanced accuracy of external classifier on generated audio (target: above baseline, non-degenerate confusion matrix)
2. **Identity preservation:** ECAPA/x-vector cosine similarity, same speaker across accent changes (target: < 10% drop)
3. **Leakage probes:** linear probes only (logistic regression). A→speaker and S→accent must be ≤ chance + 5 p.p.

**Baselines are measured BEFORE any LoRA training.** No result is interpretable without a baseline comparison.

## Critical Constraints

### Dataset hygiene
- **Speaker-disjoint splits are mandatory.** No speaker in more than one split. Violation invalidates all results.
- Accent labels use IBGE macro-regions (N, NE, CO, SE, S) from `birth_state` as proxy.
- Check confounds before training: accent vs gender, duration, recording conditions.
- CORAA-MUPE filters: interviewees only (`speaker_type='R'`), high audio quality, 3–15s duration.

### Reproducibility
- Every execution needs explicit seeds: `random`, `numpy`, `torch`, `torch.cuda`.
- All hyperparameters in YAML configs, never loose CLI args.
- Checkpoints: `{experiment_name}_epoch{N}_loss{L:.4f}.pt` with full state.
- Pin dependency versions with `==`. Log CUDA/cuDNN versions.

### Statistical rigor
- Always balanced accuracy, never simple accuracy on imbalanced data.
- CI 95% via bootstrap (1000 samples) or minimum 3 seeds.
- Overlapping CIs → do NOT claim superiority.
- Negative results are valid and must be documented.

## Directory Layout

```
CHANGELOG.md              # Project changelog (Keep a Changelog format)

docs/
├── protocol/             # Validation protocol and stage specifications
│   ├── TECHNICAL_VALIDATION_PROTOCOL.md
│   ├── STAGE_1.md
│   ├── STAGE_2.md
│   └── stage-gate-method.md
├── methodology/
│   └── PERSONAS.md       # Detailed persona profiles
└── meetings/             # Meeting notes with decision records

references/               # Literature: papers (PDFs) and dataset survey
├── dataset-survey.md     # 36KB survey of pt-BR speech datasets
├── emosteer-tts_xie-2025.pdf
├── qwen3-tts-technical_qwen-2026.pdf
└── segment-aware_liang-2026.pdf

notebooks/                # Jupyter notebooks (execution on Colab)
configs/                  # YAML experiment configs
src/                      # Python source code
tests/                    # Unit and integration tests

.claude/
├── agents/               # 6 persona definitions (head-research + 5 specialists)
├── rules/                # 10 protocol files (dataset, experiment, metrics, pytorch, etc.)
├── skills/               # 11 executable skills with SKILL.md each
├── hooks/                # task-completed.sh, teammate-idle.sh
├── AGENT_TEAMS.md        # Team composition patterns
└── RESEARCH_ROADMAP.md   # 3-stage experimental plan with DoD per stage
```

## Running Experiments

The primary execution environment is **Google Colab** (GPU). The notebook `notebooks/stage1_5_coraa_mupe.ipynb` references a companion repository (`accent-speaker-disentanglement`) that contains the `stage1_5` CLI tool and Python package.

Key CLI commands (run inside the companion repo):
```bash
stage1_5 features acoustic <manifest> <output_dir>     # Extract acoustic features
stage1_5 features ecapa <manifest> <output_dir>         # ECAPA speaker embeddings
stage1_5 features ssl <manifest> <output_dir>           # WavLM SSL features
stage1_5 features backbone <manifest> <texts> <output>  # Qwen3-TTS backbone features
stage1_5 run <config.yaml>                              # Full pipeline (probes + decision)
```

The `deep-paper-search` skill has its own venv at `.claude/.venv/` with LlamaIndex dependencies.

## Language

This project is bilingual. Documentation and agent communication are in **Brazilian Portuguese**. Code, variable names, and technical terms follow English conventions (Python/PyTorch standards).

## Experiment Registration

Every experiment must follow the template in `.claude/rules/experiment-protocol.md`:
```yaml
experiment:
  name: "exp_NNN_description"
  hypothesis: "If [intervention], then [expected], measured by [metric], because [justification]."
  independent_variable: "..."
  controlled_variables: { backbone, seed, data, ... }
  metrics: [balanced_accuracy, speaker_similarity_cosine, leakage_probe_accuracy]
  result: "pending | confirmed | refuted | inconclusive"
```

**Anti-patterns:** cherry-picking best config, comparing without CI, changing multiple variables at once, omitting failed experiments.
