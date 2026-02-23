# Reprodutibilidade e Determinismo

Toda execução deve ser reproduzível por qualquer pessoa, em qualquer máquina, a qualquer momento. Se não reproduz, não existe.

---

## Seeds e Determinismo

- **TODA** execução de treinamento ou avaliação DEVE ter seed explícito e documentado.
- Seeds obrigatórios: `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, `torch.cuda.manual_seed_all()`.
- Configurar `torch.backends.cudnn.deterministic = True` e `torch.backends.cudnn.benchmark = False` durante avaliação.
- DataLoader com `worker_init_fn` que propaga seed para cada worker.
- Seeds devem estar em config YAML, não hardcoded no código.

## Checkpoints

- Salvar checkpoint a cada epoch (mínimo) com: model state_dict, optimizer state_dict, epoch, loss, seed, config completa.
- Nomear checkpoints com informação suficiente: `{experiment_name}_epoch{N}_loss{L:.4f}.pt`.
- Manter checkpoint do melhor modelo (best) e do último (latest) separados.
- Checkpoint deve permitir retomada exata do treinamento de onde parou.

## Versionamento de Ambiente

- `requirements.txt` ou `pyproject.toml` com versões pinadas (não `>=`, usar `==`).
- Registrar versão de CUDA, cuDNN e driver NVIDIA no log de cada experimento.
- Dockerfile para reprodução exata do ambiente quando necessário.
- `conda env export --from-history` para ambientes conda.

## Configuração

- **TODA** configuração de experimento em arquivos YAML, nunca em argumentos de linha de comando soltos.
- Config YAML deve conter: hiperparâmetros, paths de dados, seeds, arquitetura do modelo, parâmetros de LoRA.
- Config YAML é artefato versionado — commitar junto com o código.
- Nunca modificar config após início do experimento. Novo experimento = nova config.

## Logging e Tracking

- Usar wandb e/ou tensorboard para tracking de experimentos.
- Logar no mínimo: loss (train/val), learning rate, GPU utilization, epoch time.
- Logar a config YAML completa como artefato do run.
- Logar versão do código (git commit hash) em cada run.
- Estrutura de diretórios para outputs:

```
experiments/
├── {experiment_name}/
│   ├── config.yaml
│   ├── checkpoints/
│   ├── logs/
│   ├── metrics/
│   └── artifacts/
```

## Anti-Patterns

- Rodar experimento sem seed e dizer "deu X% de acurácia".
- Modificar código entre treinamento e avaliação sem re-treinar.
- Perder o config/seed de um resultado que está no paper.
- Usar `pip install` sem versão fixa e esperar mesmo resultado meses depois.
- Logar apenas a loss final em vez do histórico completo.
