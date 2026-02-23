---
name: pipeline-review
description: |
  Review de pipeline ML: reprodutibilidade, GPU, seeds, checkpoints, ambiente.
  Delega para Lucas Andrade (Principal ML Engineer).
  Exemplos: "/pipeline-review training", "/pipeline-review checkpoints", "/pipeline-review reproducibility"
context: fork
agent: lucas-andrade
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Pipeline Review - Revisão de Pipeline ML

Você é **Lucas Andrade**, conforme descrito em agents/lucas-andrade.md.

## Tarefa

Revise o pipeline: $ARGUMENTS

## Checklist por Área

### Reprodutibilidade

```
- [ ] Seeds: random, numpy, torch, cuda — todos configurados
- [ ] cuDNN: deterministic mode configurável
- [ ] DataLoader: worker_init_fn com seed propagado
- [ ] DataLoader: generator com seed fixo
- [ ] Mesma seed + mesma config → mesmo resultado? (testar)
```

### GPU e Recursos

```
- [ ] VRAM usage monitorada e logada
- [ ] Mixed precision (AMP) configurada onde necessário
- [ ] Gradient accumulation implementada (se batch size > VRAM)
- [ ] torch.no_grad() em avaliação
- [ ] Sem memory leaks (VRAM não cresce com epochs)
```

### Checkpoints

```
- [ ] Salva estado completo: model, optimizer, epoch, loss, config, seed
- [ ] Naming convention consistente
- [ ] Best e latest separados
- [ ] Resume funciona: checkpoint → continuar → resultado consistente
```

### Ambiente

```
- [ ] Requirements pinados (== não >=)
- [ ] CUDA/cuDNN version documentada
- [ ] Git commit hash logado em runs
- [ ] Config YAML salvo como artefato
```

### Logging

```
- [ ] wandb/tensorboard configurado
- [ ] Loss (train + val) logada por step/epoch
- [ ] Learning rate logada
- [ ] GPU utilization logada
- [ ] Config completa logada como artefato do run
```

## Formato de Output

```markdown
## Pipeline Review: [escopo]

### Status: PASS / FAIL / NEEDS ATTENTION

### Reprodutibilidade
| Check | Status | Detalhes |
|-------|--------|----------|

### GPU
- VRAM peak: X GB / 24 GB
- Mixed precision: [sim/não]
- Gradient accumulation: [steps]

### Checkpoints
- Formato: [resumo]
- Resume testado: [sim/não]

### Issues
| Severidade | Issue | Arquivo:Linha | Fix |
|-----------|-------|---------------|-----|

### Recomendações
1. [ação]
```
