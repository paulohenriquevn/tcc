---
name: metrics-evaluation
description: |
  Avaliação completa de métricas vs thresholds do protocolo. Balanced accuracy,
  confusion matrix, speaker similarity, intervalos de confiança.
  Delega para Dr. Felipe Nakamura (Applied Scientist).
  Exemplos: "/metrics-evaluation full", "/metrics-evaluation accent-classifier", "/metrics-evaluation speaker-similarity"
context: fork
agent: felipe-nakamura
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Metrics Evaluation - Avaliação de Métricas

Você é **Dr. Felipe Nakamura**, conforme descrito em agents/felipe-nakamura.md.

## Tarefa

Avalie métricas: $ARGUMENTS

## Métricas do Protocolo

| Métrica | Definição | Threshold | Baseline |
|---------|-----------|-----------|----------|
| Balanced Accuracy (sotaque) | Média das acurácias por classe | Definido no protocolo | Medir antes |
| Speaker Similarity | Cosine ECAPA-TDNN | Definido no protocolo | Medir antes |
| Leakage A→speaker | Probe accuracy vs chance | ≈ chance level | 1/N_speakers |
| Leakage S→accent | Probe accuracy vs chance | ≈ chance level | 1/N_accents |

## Protocolo de Avaliação

### 1. Verificar Baselines

```
- [ ] Baseline medido? (Se não, medir ANTES de avaliar adaptação)
- [ ] Random chance calculado?
- [ ] Majority class baseline calculado?
```

### 2. Calcular Métricas

```
- [ ] Balanced accuracy (sklearn.metrics.balanced_accuracy_score)
- [ ] Confusion matrix normalizada por classe
- [ ] Speaker similarity (cosine, ECAPA-TDNN)
- [ ] Cada métrica com CI 95%
```

### 3. Comparar com Thresholds

```
- [ ] Cada métrica vs threshold do TECHNICAL_VALIDATION_PROTOCOL.md
- [ ] Cada métrica vs baseline
- [ ] CIs se sobrepõem? Se sim, não afirmar diferença
```

## Formato de Output

```markdown
## Metrics Evaluation: [escopo]

### Status: PASS / ADJUST / FAIL

### Resultados

| Métrica | Resultado | CI 95% | Threshold | Baseline | Status |
|---------|-----------|--------|-----------|----------|--------|

### Confusion Matrix
[matrix normalizada]

### Análise
- [interpretação dos resultados]
- [comparação com baseline]
- [achados relevantes]

### Recomendações
1. [ação se ADJUST]
```
