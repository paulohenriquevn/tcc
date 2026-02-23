# Research Roadmap — TCC Accent Control

**Projeto:** Controle Explícito de Sotaque Regional em pt-BR
**Stack:** Qwen3-TTS 1.7B-CustomVoice + LoRA + CORAA-MUPE
**Data:** 2026-02-19

---

## Visão Geral

```
STAGE 1: BASELINE      STAGE 1.5: PROBING     STAGE 2: PILOTO        STAGE 3: AVALIAÇÃO
┌────────────────┐   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
│ Métricas zero  │   │ Separabilidade │   │ LoRA por       │   │ Classificador  │
│ Sem LoRA       │──▶│ Probes lineares│──▶│ sotaque        │──▶│ ECAPA, leakage │
│ Ambiente pronto│   │ Confounds      │   │ Loss curves    │   │ Gate: P/A/F    │
└────────────────┘   └────────────────┘   └────────────────┘   └────────────────┘
     GATE 1               GATE 1.5              GATE 2               GATE 3
```

---

## Stage 1: Baseline

> **Hipótese:** O modelo Qwen3-TTS 1.7B-CustomVoice pré-treinado produz fala em pt-BR sem controle explícito de sotaque regional. Métricas de classificação de sotaque no output zero-shot devem estar próximas de chance level.

### Experimentos

| # | Experimento | Persona Principal |
|---|-------------|-------------------|
| 1.1 | Setup de ambiente (GPU, seeds, wandb) | Lucas Andrade |
| 1.2 | Download e auditoria do CORAA-MUPE | Mariana Alves |
| 1.3 | Criar splits speaker-disjoint | Mariana Alves |
| 1.4 | Carregar Qwen3-TTS, verificar forward pass | Rafael Monteiro |
| 1.5 | Gerar áudio zero-shot (sem LoRA) | Rafael Monteiro |
| 1.6 | Medir baselines: todas as métricas | Felipe Nakamura |

### Métricas

- Balanced accuracy de classificação de sotaque (zero-shot)
- Speaker similarity intra/inter-speaker (referência)
- Random chance e majority class baselines
- VRAM usage do forward pass

### Definition of Done

- [ ] Ambiente configurado e reproduzível (Docker ou conda + requirements pinados)
- [ ] CORAA-MUPE baixado, auditado, splits criados e versionados
- [ ] Qwen3-TTS carregado, forward pass verificado
- [ ] Baselines medidos com CI 95% e tabelados
- [ ] Confounds analisados e documentados

### Gate 1

**Critérios para prosseguir:**
- Ambiente reproduzível confirmado (mesma seed = mesmo output)
- Dataset auditado sem issues bloqueantes
- Baselines medidos servem como referência válida
- VRAM dentro do budget para treinamento com LoRA

---

## Stage 1.5: Auditoria de Separabilidade Latente

> **Hipótese:** Representações internas do Qwen3-TTS 1.7B-CustomVoice codificam informação suficiente de sotaque regional (macro-região IBGE) para classificação acima de chance, E os subespaços speaker/accent são suficientemente disentangled para viabilizar controle explícito via LoRA.

### Experimentos

| # | Experimento | Persona Principal |
|---|-------------|-------------------|
| 1.5.1 | Construir manifest CORAA-MUPE (com gender, duration) | Mariana Alves |
| 1.5.2 | Gerar splits speaker-disjoint versionados | Mariana Alves |
| 1.5.3 | Análise de confounds (accent x gender, accent x duration) | Mariana Alves |
| 1.5.4 | Extrair features (acústicas, ECAPA, WavLM, backbone) | Rafael Monteiro |
| 1.5.5 | Probes lineares: accent, speaker, leakage A→S, S→A | Felipe Nakamura |
| 1.5.6 | Baseline ECAPA intra/inter speaker similarity | Felipe Nakamura |

### Métricas

- Accent probe: balanced accuracy com CI 95% (3 seeds)
- Speaker probe: balanced accuracy com CI 95%
- Leakage A→speaker: balanced accuracy vs chance level
- Leakage S→accent: balanced accuracy vs chance level
- ECAPA similarity: intra-speaker (média, desvio, CI) e inter-speaker
- Confounds: chi-quadrado accent x gender, Kruskal-Wallis accent x duration

### Definition of Done

- [ ] Manifest versionado (SHA-256) com todos os campos (gender, duration)
- [ ] Splits speaker-disjoint persistidos em arquivo com assertions
- [ ] Confounds analisados e documentados
- [ ] Features extraídas de 4 tipos (acústicas, ECAPA, WavLM, backbone)
- [ ] Probes executados em 8 camadas do backbone + WavLM + ECAPA
- [ ] Resultados com CI 95% (3 seeds) e confusion matrix
- [ ] Baseline ECAPA intra/inter speaker calculado
- [ ] Report final: `stage1_5_report.md` + `stage1_5_report.json`
- [ ] Pipeline reproduzível via `make stage15`

### Gate 1.5

**Critérios (ver `docs/protocol/TECHNICAL_VALIDATION_PROTOCOL.md` seções 9.1-9.5):**

| Decisão | Critério |
|---------|----------|
| **GO** | Accent probe bal_acc >= 0.55 E leakage <= chance+5pp E confounds analisados E baseline ECAPA medido |
| **GO_CONDITIONAL** | Accent probe bal_acc >= 0.50 OU leakage <= chance+12pp (ajustes necessários) |
| **FAIL** | Accent probe bal_acc < 0.50 OU leakage > chance+12pp OU hard fail (ver `KB_HARD_FAIL_RULES.md`) |

---

## Stage 2: Treinamento Piloto

> **Hipótese:** LoRA adapters aplicados ao Qwen3-TTS, treinados com dados de sotaque específico do CORAA-MUPE, conseguem alterar as características prosódicas do output na direção do sotaque alvo, medido por aumento na classificação de sotaque pelo classificador treinado em áudio real.

### Experimentos

| # | Experimento | Persona Principal |
|---|-------------|-------------------|
| 2.1 | Configurar LoRA (rank, alpha, target modules) | Rafael Monteiro |
| 2.2 | Treinamento piloto: 1 sotaque alvo | Rafael + Lucas |
| 2.3 | Monitorar loss curves, VRAM, gradient flow | Lucas Andrade |
| 2.4 | Gerar amostras com LoRA treinado | Rafael Monteiro |
| 2.5 | Avaliação rápida: classificação + speaker similarity | Felipe Nakamura |
| 2.6 | Ablation: rank do LoRA | Rafael + Felipe |

### Métricas

- Loss de treinamento e validação (convergência)
- Balanced accuracy de classificação de sotaque (pós-LoRA)
- Speaker similarity (preservação de identidade)
- VRAM peak durante treinamento
- Tempo de treinamento por epoch

### Definition of Done

- [ ] LoRA treinado em pelo menos 1 sotaque com loss convergida
- [ ] Amostras geradas auditáveis
- [ ] Classificação de sotaque melhorou vs baseline com CI
- [ ] Speaker similarity não degradou significativamente vs baseline
- [ ] Ablation de rank documentado
- [ ] Pipeline de treinamento reproduzível (checkpoint → resume funciona)

### Gate 2

**Critérios para prosseguir:**
- Loss convergiu (não divergiu, não platô alto)
- Classificação de sotaque melhorou vs Stage 1 (com CI 95%)
- Speaker similarity manteve-se acima do threshold
- Pipeline reproduzível confirmado (re-run com seed diferente, resultado consistente)

---

## Stage 3: Avaliação Técnica

> **Hipótese:** O LoRA adapter consegue transferir sotaque mantendo disentanglement: embeddings A codificam sotaque sem vazar speaker, embeddings S codificam speaker sem vazar sotaque.

### Experimentos

| # | Experimento | Persona Principal |
|---|-------------|-------------------|
| 3.1 | Treinar classificador de sotaque em áudio real | Felipe Nakamura |
| 3.2 | Extrair embeddings ECAPA-TDNN (speaker) | Felipe Nakamura |
| 3.3 | Leakage probes: A→speaker, S→accent | Felipe Nakamura |
| 3.4 | Confusion matrix por sotaque | Felipe Nakamura |
| 3.5 | Múltiplas seeds: variância e CI | Lucas + Felipe |
| 3.6 | Red team review completo | Ana Silva |

### Métricas

- Balanced accuracy de classificação de sotaque (full evaluation)
- Speaker similarity (cosine ECAPA-TDNN)
- Leakage probe A→speaker (vs chance level)
- Leakage probe S→accent (vs chance level)
- Confusion matrix normalizada
- CI 95% para todas as métricas (3+ seeds)

### Definition of Done

- [ ] Todas as métricas medidas com CI 95%
- [ ] Leakage probes executados em ambas as direções
- [ ] Confusion matrix analisada (quais sotaques confundem?)
- [ ] Red team review completo com veredicto
- [ ] Resultados comparados formalmente com baselines do Stage 1
- [ ] Limitações documentadas

### Gate 3 — Decisão Final

**Padrão: `protocol-gate` (Agent Teams, todas as 5 personas)**

| Decisão | Critério |
|---------|----------|
| **PASS** | Todas as métricas acima dos thresholds, leakage probes ≈ chance, red team sem BLOCKs |
| **ADJUST** | Métricas parcialmente atendidas, leakage parcial, ajustes identificáveis |
| **FAIL** | Métricas abaixo dos thresholds, leakage significativo, ou falha metodológica |

---

## Princípios do Roadmap

1. **Cada Stage tem Gate** — não prosseguir sem validação formal
2. **Baselines primeiro** — sem baseline, nenhum resultado é interpretável
3. **Uma variável por vez** — mudanças isoladas, impacto mensurável
4. **Resultados negativos documentados** — se não funcionou, registrar por quê
5. **Red team antes de claims** — Ana revisa antes de qualquer conclusão
