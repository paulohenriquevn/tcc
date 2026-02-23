# Python e PyTorch — Padrões de Código

Código de pesquisa não é descartável. Código mal escrito gera resultados irreproduzíveis.

---

## Versões e Ambiente

- Python 3.10+ (type hints nativos, match/case disponível).
- PyTorch 2.x com CUDA support.
- Type hints em toda função pública. `def train_epoch(model: nn.Module, loader: DataLoader, optimizer: Optimizer) -> float:`.
- Docstrings para funções de interface (não para helpers internos triviais).

## Organização de Diretórios

```
src/
├── models/          # Definições de modelo (nn.Module)
│   ├── backbone.py  # Wrapper do Qwen3-TTS
│   ├── lora.py      # Configuração e aplicação do LoRA
│   └── probes.py    # Probes lineares para leakage test
├── data/            # Dataset, DataLoader, preprocessing
│   ├── dataset.py   # Dataset class (CORAA-MUPE)
│   ├── transforms.py # Audio transforms
│   └── splits.py    # Geração e validação de splits
├── training/        # Loop de treinamento
│   ├── trainer.py   # Training loop
│   └── callbacks.py # Checkpoint, early stopping, logging
├── evaluation/      # Métricas e avaliação
│   ├── metrics.py   # Implementação de métricas
│   ├── evaluate.py  # Script de avaliação
│   └── probing.py   # Leakage probes
├── utils/           # Utilitários gerais
│   ├── config.py    # Parsing de YAML config
│   ├── seed.py      # Setup de seeds
│   └── logging.py   # Setup de logging
configs/             # YAML configs por experimento
scripts/             # Entry points (train.py, evaluate.py, probe.py)
tests/               # Testes unitários e de integração
```

## PyTorch Patterns

### nn.Module

```python
class AccentAdapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        # Inicializar layers aqui, NUNCA no forward

    def forward(self, x: Tensor) -> Tensor:
        # Forward puro, sem side effects
        ...
```

- Toda camada no `__init__`, lógica no `forward`.
- Sem estado mutável fora de `register_buffer` e `register_parameter`.
- `model.eval()` antes de inferência, `model.train()` antes de treino.

### DataLoader Reproduzível

```python
generator = torch.Generator()
generator.manual_seed(seed)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=generator,
    pin_memory=True,  # GPU
)
```

### Gestão de GPU (24GB)

- Monitorar VRAM com `torch.cuda.memory_allocated()` e `torch.cuda.max_memory_allocated()`.
- Usar `torch.cuda.amp.autocast()` para mixed precision quando VRAM é limitante.
- Gradient accumulation para simular batch sizes maiores sem estourar VRAM.
- `torch.cuda.empty_cache()` apenas quando necessário (entre fases distintas, não entre batches).
- Logar uso de GPU em cada epoch.

## Logging Estruturado

- Usar `logging` stdlib com formato estruturado, não `print()`.
- Níveis: DEBUG para detalhes de batch, INFO para epoch/resultado, WARNING para anomalias, ERROR para falhas.
- Métricas numéricas via wandb/tensorboard, não em logs de texto.

## Testes para Código ML

- Testar preprocessing: input conhecido, output esperado.
- Testar shapes: verificar que tensors passam pelo modelo sem erro de dimensão.
- Testar reprodutibilidade: mesma seed + mesma config = mesmo output.
- Testar splits: assertions de disjointness.
- NÃO testar convergência em unitários (isso é teste de integração lento).

## Anti-Patterns

- `print(f"loss: {loss}")` em vez de logging estruturado + wandb.
- Código de treino em Jupyter notebook sem versionar (notebooks são para exploração, scripts para execução).
- `model.cuda()` hardcoded em vez de `model.to(device)` configurável.
- `torch.no_grad()` esquecido na avaliação (desperdiça VRAM).
- `import *` em qualquer lugar.
- Variáveis globais para hiperparâmetros.
