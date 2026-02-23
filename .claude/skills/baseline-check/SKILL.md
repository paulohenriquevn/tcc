---
name: baseline-check
description: |
  Medição de baselines antes de qualquer adaptação. Modelo pré-treinado sem LoRA,
  random chance, majority class. Delega para Dr. Felipe Nakamura (Applied Scientist).
  Exemplos: "/baseline-check all", "/baseline-check accent-classifier", "/baseline-check speaker-similarity"
context: fork
agent: felipe-nakamura
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Baseline Check - Medição de Baselines

Você é **Dr. Felipe Nakamura**, conforme descrito em agents/felipe-nakamura.md.

## Tarefa

Meça baselines: $ARGUMENTS

## Baselines Obrigatórios

### 1. Modelo Pré-Treinado (sem LoRA)

```
- [ ] Qwen3-TTS 1.7B-CustomVoice sem adaptação
- [ ] Gerar áudio com cada sotaque de referência
- [ ] Medir todas as métricas no output zero-shot
```

### 2. Random Chance

```
- [ ] Para classificação de sotaque: 1/N_classes
- [ ] Para leakage probes: 1/N_speakers, 1/N_accents
```

### 3. Majority Class

```
- [ ] Classificador que sempre prediz a classe mais frequente
- [ ] Reportar balanced accuracy deste baseline
```

### 4. Speaker Similarity Reference

```
- [ ] Similaridade intra-speaker (mesmo speaker, amostras diferentes)
- [ ] Similaridade inter-speaker (speakers diferentes)
- [ ] Estabelece o range esperado para a métrica
```

## Formato de Output

```markdown
## Baseline Measurements

### Dataset Info
- N classes (sotaques): [N]
- N speakers: [N]
- Distribuição: [resumo]

### Baselines

| Métrica | Random | Majority | Zero-Shot | CI 95% |
|---------|--------|----------|-----------|--------|

### Speaker Similarity Reference
- Intra-speaker mean: X.XX (CI: [Y, Z])
- Inter-speaker mean: X.XX (CI: [Y, Z])

### Interpretação
- [o que os baselines significam para os thresholds do protocolo]
```
