# TCC - Estrutura de Pesquisa

**Data:** 2026-02-19
**Versão:** 1.0
**Projeto:** Controle Explícito de Sotaque Regional em pt-BR (Qwen3-TTS 1.7B-CustomVoice + LoRA + CORAA-MUPE)

---

## Visão Geral

```
┌──────────────────────────────────────────────────────────┐
│              TCC - EQUIPE DE PESQUISA                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│              HEAD RESEARCH (coordenador)                  │
│                        │                                  │
│    ┌───────┬───────┬───┴───┬───────┬───────┐            │
│    │       │       │       │       │       │            │
│    ▼       ▼       ▼       ▼       ▼       │            │
│  Rafael  Mariana  Lucas  Felipe   Ana      │            │
│  MODEL    DATA    INFRA   EVAL   REDTEAM   │            │
│                                             │            │
│  Estrutura flat — sem camada de times       │            │
│  Coordenador roteia diretamente             │            │
│  5 personas + 1 coordenador = 6 agents      │            │
│                                             │            │
└──────────────────────────────────────────────────────────┘
```

**Decisão arquitetural:** Com apenas 5 personas, a camada de times é overhead puro. O coordenador (head-research) roteia diretamente para os indivíduos.

---

## Personas

### Dr. Rafael Monteiro — Lead Research Engineer

| Atributo | Valor |
|----------|-------|
| **Agent** | `rafael-monteiro` |
| **Foco** | Backbone TTS, LoRA adapters, Representation Learning |
| **Skills** | model-validation, experiment-design |
| **Tools** | Read, Grep, Glob, Bash, Edit, Write |

Responsável pela arquitetura do modelo, configuração de LoRA, design de embeddings S/A e liderança de design de experimentos.

### Mariana Alves — Senior Speech Data Engineer

| Atributo | Valor |
|----------|-------|
| **Agent** | `mariana-alves` |
| **Foco** | Dataset CORAA-MUPE, splits, confounds, metadata |
| **Skills** | dataset-audit |
| **Tools** | Read, Grep, Glob, Bash, Edit, Write |

Responsável pela integridade do dataset, splits speaker-disjoint, detecção de confounds e análise de distribuições.

### Lucas Andrade — Principal ML Engineer

| Atributo | Valor |
|----------|-------|
| **Agent** | `lucas-andrade` |
| **Foco** | Pipeline ML, GPU, seeds, checkpoints, reprodutibilidade |
| **Skills** | pipeline-review |
| **Tools** | Read, Grep, Glob, Bash, Edit, Write |

Responsável pelo pipeline end-to-end, gestão de GPU (24GB), determinismo, checkpoints e ambiente de execução.

### Dr. Felipe Nakamura — Applied Scientist

| Atributo | Valor |
|----------|-------|
| **Agent** | `felipe-nakamura` |
| **Foco** | Métricas, avaliação, probes de leakage, significância estatística |
| **Skills** | metrics-evaluation, leakage-test, baseline-check |
| **Tools** | Read, Grep, Glob, Bash, Edit, Write |

Responsável por todas as métricas de avaliação, baselines, leakage probes e validação estatística de resultados.

### Ana K. Silva — Independent Technical Reviewer

| Atributo | Valor |
|----------|-------|
| **Agent** | `ana-silva` |
| **Foco** | Revisão adversarial, red team, auditoria de claims |
| **Skills** | red-team-review |
| **Tools** | Read, Grep, Glob, Bash **(sem Edit/Write)** |

Revisora independente. **NÃO implementa código** — apenas audita, questiona e reporta. Separação deliberada entre quem constrói e quem audita.

---

## RACI por Área

| Área | Rafael | Mariana | Lucas | Felipe | Ana |
|------|--------|---------|-------|--------|-----|
| **Model Architecture** | **R** | I | C | C | C |
| **LoRA Configuration** | **R** | I | C | I | C |
| **Dataset Integrity** | I | **R** | C | C | C |
| **Splits & Confounds** | I | **R** | I | C | C |
| **Pipeline & Infra** | C | I | **R** | I | I |
| **Reproducibility** | C | C | **R** | C | C |
| **Metrics & Evaluation** | C | I | I | **R** | C |
| **Leakage Probes** | C | C | I | **R** | C |
| **Baselines** | C | I | I | **R** | I |
| **Red Team Review** | I | I | I | I | **R** |
| **Experiment Design** | **R** | C | C | C | C |
| **Gate Decision** | C | C | C | C | C |

> R = Responsible, A = Accountable (Lead/coordenador para Gate), C = Consulted, I = Informed

---

## Interfaces entre Personas

```
Rafael (Model) ←→ Felipe (Eval)
  Arquitetura do modelo ↔ Métricas que validam a arquitetura

Rafael (Model) ←→ Lucas (Infra)
  Config de LoRA/modelo ↔ VRAM budget e pipeline

Mariana (Data) ←→ Felipe (Eval)
  Splits e distribuições ↔ Baselines calculados nos splits

Mariana (Data) ←→ Lucas (Infra)
  Preprocessing pipeline ↔ Reprodutibilidade do preprocessing

Ana (Red Team) ←→ Todos
  Recebe resultados de todos ↔ Reporta findings adversariais
```

---

## Fluxo de Trabalho Típico

```
1. USUÁRIO pede tarefa
         │
2. HEAD-RESEARCH analisa
         │
3. DECISÃO: qual persona?
         │
    ┌─────┴──────┐
    │             │
  1 persona    2+ personas
    │             │
4a. Subagent   4b. Agent Teams
    │             │
    │         5. Escolher padrão (AGENT_TEAMS.md)
    │         6. Spawnar teammates
    │         7. Monitorar progresso
    │             │
    └──────┬──────┘
           │
8. RESULTADO consolidado para o usuário
```

---

*Documento mantido por: Head Research (coordenador)*
*Última atualização: 2026-02-19*
