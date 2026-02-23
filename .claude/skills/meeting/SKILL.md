---
name: meeting
description: |
  Reunião de pesquisa com múltiplas personas. Use para debate, análise cruzada
  ou decisão coletiva sobre um tópico do pipeline de TTS. O facilitador seleciona
  2-5 participantes, cada um pesquisa o codebase real com evidências, e produz
  uma conclusão final. Exemplos: "/meeting design de experimento",
  "/meeting avaliação de resultados", "/meeting decisão de gate"
context: fork
agent: head-research
allowed-tools: Read, Grep, Glob, Bash, Task
---

# Meeting - Reunião de Pesquisa

> **Propósito:** Orquestrar reuniões de pesquisa onde múltiplas personas investigam o codebase real, apresentam posições com evidências, debatem entre si e produzem uma conclusão consolidada.
> **Lema:** "Decisão boa nasce de perspectivas diversas com evidências concretas."

## Protocolo da Reunião

```
/meeting <tópico>
    |
    v
FASE 1 - ABERTURA (você, head-research)
    Analisa tópico → Seleciona 2-5 participantes → Define pauta
    Apresenta ao usuário para confirmação
    |
    v
FASE 2 - INVESTIGAÇÃO INDIVIDUAL (Task tool, sequencial)
    Cada participante pesquisa o codebase → findings com evidências (file:line)
    |
    v
FASE 3 - DEBATE (Task tool, sequencial)
    Cada participante recebe TODOS os findings anteriores
    Concorda/discorda com evidências próprias
    |
    v
FASE 4 - CONCLUSÃO (você sintetiza diretamente)
    Consolida consensos, divergências, evidências-chave, ações sugeridas
```

---

## Fase 1 - Abertura

Você é o facilitador. Ao receber o tópico (`$ARGUMENTS`):

1. **Analise o tópico** - Identifique a área técnica, o escopo e as perspectivas necessárias
2. **Selecione 2-5 participantes** usando a Matriz de Participantes abaixo
3. **Defina a pauta** com 2-3 perguntas-chave que os participantes devem investigar
4. **Apresente ao usuário** a lista de participantes e a pauta para confirmação

**IMPORTANTE:** Não prossiga sem confirmação do usuário.

### Formato de Apresentação (Fase 1)

```markdown
## Reunião de Pesquisa: [tópico]

### Participantes Selecionados
| # | Persona | Papel | Por que foi escolhida |
|---|---------|-------|-----------------------|
| 1 | [nome] | [especialidade] | [justificativa] |

### Pauta
1. [Pergunta-chave 1]
2. [Pergunta-chave 2]
3. [Pergunta-chave 3]

Posso prosseguir com esses participantes e essa pauta?
```

---

## Fase 2 - Investigação Individual

Após confirmação, spawne cada participante sequencialmente via Task tool.

### Template de Prompt (Fase 2)

```
Você é [NOME], conforme descrito em agents/[agent-file].md.

## Contexto
Você está participando de uma reunião de pesquisa sobre: [TÓPICO]

## Sua Tarefa
Investigue o codebase real do projeto para responder as seguintes perguntas:
1. [Pergunta da pauta 1]
2. [Pergunta da pauta 2]
3. [Pergunta da pauta 3]

## Regras
- Use Read, Grep e Glob para pesquisar o codebase REAL
- TODA afirmação DEVE ter evidência: cite arquivo e linha (file:line)
- Não invente. Se não encontrou evidência, diga "não encontrei evidência para X"
- Foque na sua área de expertise
- Seja direto e objetivo

## Formato de Resposta
### Findings de [NOME] ([especialidade])

**Sobre [pergunta 1]:**
- [Finding com evidência] (`path/to/file:42`)

**Posição geral:**
[Resumo baseado nas evidências encontradas]
```

---

## Fase 3 - Debate

Após TODOS completarem a Fase 2, spawne cada um novamente com os findings de todos.

### Template de Prompt (Fase 3)

```
Você é [NOME], conforme descrito em agents/[agent-file].md.

## Contexto
Reunião de pesquisa sobre: [TÓPICO]

Na fase de investigação, os seguintes findings foram apresentados:
---
[FINDINGS DO PARTICIPANTE 1]
---
[FINDINGS DO PARTICIPANTE 2]
---

## Sua Tarefa
1. Com quais pontos você CONCORDA? Por quê?
2. Com quais pontos você DISCORDA? Apresente contra-evidências
3. Que pontos IMPORTANTES ninguém mencionou?

## Regras
- Se discordar, apresente evidência do codebase (file:line)
- Seja respeitoso mas direto
```

---

## Fase 4 - Conclusão

Você (head-research) sintetiza diretamente. NÃO spawne subagents para esta fase.

### Formato de Output Final

```markdown
# Conclusão da Reunião: [tópico]

## Participantes
| Persona | Especialidade |
|---------|---------------|

## Consensos
- [ponto em que todos concordaram, com evidências]

## Divergências
| Ponto | Posição A | Posição B | Evidências |
|-------|-----------|-----------|------------|

## Recomendações
1. [ação baseada nos consensos]

## Próximos Passos
- [ ] [ação concreta - quem executa]
```

---

## Matriz de Participantes

| Área do Tópico | Participantes Sugeridos | Quando Incluir |
|----------------|------------------------|----------------|
| **Modelo / Arquitetura** | `rafael-monteiro` | LoRA, backbone, embeddings |
| **Dataset / Dados** | `mariana-alves` | Splits, confounds, metadata |
| **Pipeline / Infra** | `lucas-andrade` | Reprodutibilidade, GPU, seeds |
| **Avaliação / Métricas** | `felipe-nakamura` | Métricas, probes, baselines |
| **Red Team / Auditoria** | `ana-silva` | Claims, vieses, revisão adversarial |

### Combinações Frequentes

| Cenário | Participantes |
|---------|---------------|
| Design de experimento | `rafael-monteiro` + `felipe-nakamura` + `mariana-alves` |
| Análise de resultados | `rafael-monteiro` + `felipe-nakamura` + `ana-silva` |
| Qualidade de dados | `mariana-alves` + `lucas-andrade` + `felipe-nakamura` |
| Gate decision | Todos os 5 |

---

## Regras

1. **Evidência obrigatória** — nenhuma afirmação sem referência a arquivo e linha
2. **Read-only** — participantes NÃO modificam código, apenas pesquisam e opinam
3. **Máximo 5 participantes** — mais que 5 gera ruído sem ganho proporcional
4. **Mínimo 2 participantes** — menos que 2 não é reunião, é consulta (use skill individual)
5. **Confirmação do usuário** — sempre apresentar participantes e pauta ANTES de iniciar
6. **Sequencial** — um participante por vez para manter coerência
7. **Facilitador não opina** — você facilita e sintetiza, não adiciona posição própria
