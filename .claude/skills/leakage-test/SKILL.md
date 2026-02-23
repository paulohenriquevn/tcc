---
name: leakage-test
description: |
  Probes de leakage para verificar disentanglement: embedding A vaza speaker?
  Embedding S vaza sotaque? Usa probes lineares com null hypothesis.
  Delega para Dr. Felipe Nakamura (Applied Scientist).
  Exemplos: "/leakage-test both", "/leakage-test A-to-speaker", "/leakage-test S-to-accent"
context: fork
agent: felipe-nakamura
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Leakage Test - Probes de Leakage

Você é **Dr. Felipe Nakamura**, conforme descrito em agents/felipe-nakamura.md.

## Tarefa

Execute probes de leakage: $ARGUMENTS

## Probes Definidos

### Probe A→Speaker (embedding de accent vaza identidade?)

```
H0: "Embedding A não contém informação de speaker identity"
Rejeitar H0 se: probe accuracy > chance level + margem (com CI 95%)
Chance level: 1/N_speakers

Implementação:
- Extrair embeddings A para todas as amostras
- Treinar logistic regression (sklearn) para predizer speaker_id
- Split: speaker-disjoint (CRÍTICO — usar speakers do val set)
- Seed fixo para reprodutibilidade
- Reportar com CI 95% (bootstrap)
```

### Probe S→Accent (embedding de speaker vaza sotaque?)

```
H0: "Embedding S não contém informação de sotaque"
Rejeitar H0 se: probe accuracy > chance level + margem (com CI 95%)
Chance level: 1/N_accents

Implementação:
- Extrair embeddings S (ECAPA-TDNN) para todas as amostras
- Treinar logistic regression para predizer accent_label
- Split: speaker-disjoint
- Seed fixo
- Reportar com CI 95%
```

## Regras do Probe

```
- Probe DEVE ser linear (logistic regression)
- Probe NÃO pode ser MLP (complexo demais, sempre acha algo)
- Split do probe DEVE ser speaker-disjoint
- Probe DEVE ser treinado com seed fixo
- Resultado DEVE incluir CI 95%
```

## Formato de Output

```markdown
## Leakage Test: [escopo]

### Probe A→Speaker
- Chance level: X.XX (1/N_speakers)
- Probe accuracy: X.XX (CI 95%: [Y, Z])
- H0 (sem leakage): ACEITA / REJEITADA
- Interpretação: [o que significa]

### Probe S→Accent
- Chance level: X.XX (1/N_accents)
- Probe accuracy: X.XX (CI 95%: [Y, Z])
- H0 (sem leakage): ACEITA / REJEITADA
- Interpretação: [o que significa]

### Diagnóstico de Disentanglement
| Embedding | Informação Esperada | Leakage Detectado | Severidade |
|-----------|--------------------|--------------------|-----------|

### Veredicto
[PASS: disentanglement funciona | ADJUST: leakage parcial | FAIL: leakage significativo]
```
