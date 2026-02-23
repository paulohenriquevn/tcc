# Como Implementar Agent Teams no Claude Code

**Data:** 2026-02-19
**Versão:** 2.0
**Contexto:** Guia de referência para replicar a implementação de Agent Teams — adaptado para projeto de pesquisa com 5 personas

---

## Sumário

1. [O Que Fizemos](#1-o-que-fizemos)
2. [Arquitetura da Solução](#2-arquitetura-da-solução)
3. [Pré-requisitos](#3-pré-requisitos)
4. [Passo a Passo: Implementação](#4-passo-a-passo-implementação)
5. [Como Usar no Dia a Dia](#5-como-usar-no-dia-a-dia)
6. [Adaptando para Outro Projeto](#6-adaptando-para-outro-projeto)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. O Que Fizemos

### Problema

O projeto TCC usa um "time virtual de pesquisa" com 5 personas, 10 skills e 1 coordenador (`head-research`). O mecanismo de delegação via subagents (Task tool) funciona para tarefas de persona única. Mas precisávamos de:
- **Múltiplas personas avaliando em paralelo** (ex: protocol-gate com 5 perspectivas)
- **Comunicação direta entre personas** (ex: data engineer informa evaluator sobre confound)
- **Task list compartilhada** com dependências (ex: red team espera todos terminarem)

### Solução

Agent Teams como mecanismo complementar:

```
Tarefa para 1 persona? → Subagent (Task tool)
Tarefa para 2+ personas em paralelo? → Agent Teams
```

### O Que Foi Criado

| Arquivo | Propósito |
|---------|-----------|
| `settings.json` | Feature flag + hooks |
| `hooks/task-completed.sh` | Quality gate: valida DoD |
| `hooks/teammate-idle.sh` | Aviso informativo de idle |
| `AGENT_TEAMS.md` | 5 spawn prompts, 4 padrões de composição |
| `agents/head-research.md` | Coordenador com framework de decisão |

---

## 2. Arquitetura da Solução

```
┌─────────────────────────────────────────────────────────┐
│                   USUÁRIO                                │
│                     │                                    │
│                     ▼                                    │
│              ┌──────────────┐                            │
│              │ head-research │ ← Coordenador             │
│              └──────┬───────┘                            │
│                     │                                    │
│          ┌──────────┴──────────┐                         │
│          │                     │                         │
│    ┌─────▼──────┐     ┌───────▼────────┐                │
│    │  SUBAGENT   │     │  AGENT TEAMS   │                │
│    │ (Task tool) │     │  (Teammates)   │                │
│    ├────────────┤     ├────────────────┤                │
│    │ 1 persona  │     │ 2-5 personas   │                │
│    │ Sequencial │     │ Paralelo       │                │
│    │ Fork+volta │     │ Mailbox+Tasks  │                │
│    └────────────┘     └────────────────┘                │
│                                                          │
│    ┌─────────────────────────────────────┐              │
│    │           HOOKS (Quality Gates)      │              │
│    │  task-completed.sh → valida DoD      │              │
│    │  teammate-idle.sh  → avisa pendências │              │
│    └─────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Pré-requisitos

- Claude Code instalado e funcional
- Diretório `.claude/` configurado no projeto
- Pelo menos 2 agents definidos em `.claude/agents/`
- Um agent coordenador que decide o mecanismo

### Feature Flag

```
CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
```

Configurado no `settings.json` (versionado, compartilhado).

---

## 4. Passo a Passo: Implementação

### Passo 1: Criar `settings.json`

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "hooks": {
    "TaskCompleted": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash $CLAUDE_PROJECT_DIR/.claude/hooks/task-completed.sh",
            "timeout": 30
          }
        ]
      }
    ],
    "TeammateIdle": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash $CLAUDE_PROJECT_DIR/.claude/hooks/teammate-idle.sh",
            "timeout": 15
          }
        ]
      }
    ]
  }
}
```

### Passo 2: Criar hooks

Hooks em `.claude/hooks/`. Depois: `chmod +x hooks/*.sh`

**Códigos de saída:**

| Exit Code | Significado | Efeito |
|-----------|-------------|--------|
| `0` | Aprovado | Task prossegue |
| `2` | Rejeitado | Task devolvida com feedback (stderr) |
| Outro | Erro | Hook ignorado (fail-open) |

### Passo 3: Criar AGENT_TEAMS.md

4 seções obrigatórias:
1. **Framework de Decisão** — quando usar cada mecanismo
2. **Spawn Prompts** — um template por agent (referencia o agent file, NÃO duplica)
3. **Padrões de Composição** — combinações recorrentes com task lists
4. **Instruções de Coordenação** — Delegate Mode, cleanup, monitoramento

### Passo 4: Atualizar o coordenador

Adicionar ao agent coordenador:
- Framework de decisão (Teams vs Subagents)
- Tabela de padrões disponíveis
- Instruções de como usar

---

## 5. Como Usar no Dia a Dia

### Exemplo: Avaliação de Resultados (experiment-review)

O usuário pede: *"Avalie os resultados do experimento de LoRA rank 16"*

1. head-research decide: 3 perspectivas (model + eval + red team) → Agent Teams
2. Cria task list com dependências
3. Spawna Rafael, Felipe e Ana com prompts do AGENT_TEAMS.md
4. Rafael e Felipe trabalham em paralelo; Ana espera ambos
5. Lead consolida findings

### Exemplo: Protocol Gate

O usuário pede: *"Rode o gate do Stage 2"*

1. head-research escolhe padrão `protocol-gate` → Agent Teams com 5 personas
2. Tasks 1-4 em paralelo (Mariana, Lucas, Felipe, Rafael)
3. Task 5 (Ana) espera todas completarem
4. Task 6 (Lead) consolida decisão PASS/ADJUST/FAIL

### Quando NÃO usar Agent Teams

- *"Audite os splits do dataset"* → 1 persona (Mariana) → **Subagent**
- *"Qual a VRAM do forward pass?"* → 1 persona (Lucas) → **Subagent**
- *"Calcule os baselines"* → 1 persona (Felipe) → **Subagent**

---

## 6. Adaptando para Outro Projeto

### Checklist

1. **Inventário de agents** — liste seus agents e identifique cenários de paralelismo
2. **Se < 3 agents** — Agent Teams provavelmente não compensa
3. **Identifique padrões** — quais combinações de personas se repetem?
4. **Crie AGENT_TEAMS.md** — spawn prompts + padrões
5. **Atualize coordenador** — framework de decisão
6. **Copie hooks** — são genéricos, funcionam em qualquer projeto

### O que é genérico (copie direto)

- `settings.json` (estrutura é a mesma)
- `hooks/task-completed.sh` (validação de DoD é universal)
- `hooks/teammate-idle.sh` (aviso de pendências é universal)

### O que é específico (crie para seu projeto)

- `AGENT_TEAMS.md` (spawn prompts e padrões dependem dos seus agents)
- Seções de Agent Teams no coordenador

---

## 7. Troubleshooting

### Hook não executa

1. Hook não tem permissão → `chmod +x hooks/*.sh`
2. `settings.json` com JSON malformado → `python -m json.tool settings.json`
3. Path errado → verificar `$CLAUDE_PROJECT_DIR`
4. Timeout muito baixo → aumentar no `settings.json`

### Hook rejeita tudo

```bash
echo '{"task_subject":"Minha task de teste","task_status":"completed"}' | bash hooks/task-completed.sh
echo "Exit: $?"
```

### Teammates não se comunicam

Adicione ao spawn prompt:
```
Comunicação: Mande mensagem direta para outros teammates quando precisar de input.
```

### settings.json e settings.local.json conflitam

`settings.local.json` sobrescreve `settings.json` para campos iguais. Na prática: `settings.json` tem `env` e `hooks` (do projeto), `settings.local.json` tem `outputStyle` (da pessoa). Não há conflito.
