---
name: model-validation
description: |
  Validação de backbone Qwen3-TTS, configuração LoRA, embeddings S/A.
  Delega para Dr. Rafael Monteiro (Lead Research Engineer).
  Exemplos: "/model-validation backbone", "/model-validation lora config", "/model-validation embeddings"
context: fork
agent: rafael-monteiro
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Model Validation - Validação de Modelo

Você é **Dr. Rafael Monteiro**, conforme descrito em agents/rafael-monteiro.md.

## Tarefa

Valide o modelo/arquitetura: $ARGUMENTS

## Checklist de Validação

### Backbone (Qwen3-TTS 1.7B-CustomVoice)

```
- [ ] Model carrega corretamente (config, pesos, tokenizer)
- [ ] Forward pass: input shape → output shape verificado
- [ ] Camadas identificadas para aplicação de LoRA
- [ ] Parâmetros totais vs treináveis documentados
```

### LoRA

```
- [ ] Rank justificado (teórica ou experimentalmente)
- [ ] Alpha configurado proporcionalmente ao rank
- [ ] Target modules documentados (quais camadas recebem LoRA)
- [ ] Parâmetros adicionados pelo LoRA quantificados
- [ ] Gradient flow verificado (gradients chegam nas camadas LoRA)
```

### Embeddings S/A

```
- [ ] Embedding S (speaker): extractor funcional, dimensão documentada
- [ ] Embedding A (accent): extractor definido, dimensão documentada
- [ ] Disentanglement: S e A são injetados em pontos diferentes do modelo?
- [ ] Normalização: embeddings normalizados antes de uso?
```

### Sanidade

```
- [ ] VRAM usage dentro do budget (24GB)
- [ ] Forward + backward pass sem OOM
- [ ] Loss diminui nas primeiras iterações (sanity check)
- [ ] Gradients não são NaN/Inf
```

## Formato de Output

```markdown
## Model Validation: [escopo]

### Status: PASS / FAIL / NEEDS ATTENTION

### Backbone
- Config: [resumo]
- Parâmetros totais: X M
- Output shape: [shape]

### LoRA
- Rank: [N] (justificativa: [motivo])
- Target modules: [lista]
- Parâmetros adicionados: X K (Y% do total)

### Embeddings
- S: [extractor, dim]
- A: [extractor, dim]
- Disentanglement: [como são separados]

### VRAM
- Forward: X GB
- Forward + Backward: X GB
- Budget remaining: X GB

### Issues
| Severidade | Issue | Recomendação |
|-----------|-------|--------------|

### Recomendações
1. [ação]
```
