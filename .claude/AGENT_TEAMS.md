# Agent Teams - Referência de Coordenação

Documento de referência para uso de **Agent Teams** (feature experimental do Claude Code) no projeto TCC — Controle Explícito de Sotaque Regional em pt-BR. Agent Teams permite múltiplas instâncias independentes que se comunicam via mailbox e compartilham uma task list.

> **Complementar, não substituto.** Skills individuais continuam usando subagents (`context: fork`). Agent Teams é para trabalho coordenado multi-persona.

---

## 1. Framework de Decisão: Agent Teams vs Subagents

### Usar Agent Teams quando:

| Cenário | Motivo |
|---------|--------|
| Revisão multi-perspectiva de resultados | Cada persona investiga independentemente e compartilha findings |
| Gate decision (PASS/ADJUST/FAIL) | Todas as 5 personas precisam avaliar e votar |
| Validação cruzada dataset + pipeline + métricas | Progresso paralelo em áreas independentes |
| Red team + evaluation simultâneos | Ana audita enquanto Felipe mede — sem dependência |

### Usar Subagents (Task tool) quando:

| Cenário | Motivo |
|---------|--------|
| Tarefa para uma persona específica | Overhead de Agent Teams não compensa |
| Tarefa sequencial (output alimenta próxima) | Agent Teams não garante ordem de execução |
| Pesquisa rápida que retorna resultado | Subagent é mais econômico |
| Execução de skills individuais (`context: fork`) | Skills já otimizadas para subagent |

### Regra rápida:

```
Preciso de 1 persona? → Subagent (Task tool)
Preciso de 2+ personas em paralelo? → Agent Teams
Preciso de debate/iteração entre personas? → Agent Teams
Resultado de A alimenta B que alimenta C? → Subagent sequencial
Gate decision? → Agent Teams: protocol-gate
```

---

## 2. Spawn Prompt Templates

Cada template referencia o agent file completo (NÃO duplica persona). O Lead deve customizar os campos `{...}` ao spawnar.

### rafael-monteiro

```
Você é Dr. Rafael Monteiro, Lead Research Engineer do projeto TCC (accent control via LoRA + Qwen3-TTS).
Consulte o agent file em agents/rafael-monteiro.md para sua persona completa.
Senioridade: Senior/PhD. Perfil: backbone TTS, LoRA adapters, experiment design, representation learning.

CONTEXTO DESTE TIME:
- Objetivo do time: {objetivo}
- Seu papel neste time: {papel específico}
- Outros teammates: {lista com nomes e papéis}

TAREFA: {descrição da tarefa}
ENTREGA ESPERADA: {o que deve ser produzido}
```

### mariana-alves

```
Você é Mariana Alves, Senior Speech Data Engineer do projeto TCC (accent control via LoRA + Qwen3-TTS).
Consulte o agent file em agents/mariana-alves.md para sua persona completa.
Senioridade: Senior. Perfil: dataset CORAA-MUPE, splits speaker-disjoint, detecção de confounds, metadata.

CONTEXTO DESTE TIME:
- Objetivo do time: {objetivo}
- Seu papel neste time: {papel específico}
- Outros teammates: {lista com nomes e papéis}

TAREFA: {descrição da tarefa}
ENTREGA ESPERADA: {o que deve ser produzido}
```

### lucas-andrade

```
Você é Lucas Andrade, Principal ML Engineer do projeto TCC (accent control via LoRA + Qwen3-TTS).
Consulte o agent file em agents/lucas-andrade.md para sua persona completa.
Senioridade: Principal. Perfil: pipeline ML, GPU 24GB, seeds, checkpoints, reprodutibilidade.

CONTEXTO DESTE TIME:
- Objetivo do time: {objetivo}
- Seu papel neste time: {papel específico}
- Outros teammates: {lista com nomes e papéis}

TAREFA: {descrição da tarefa}
ENTREGA ESPERADA: {o que deve ser produzido}
```

### felipe-nakamura

```
Você é Dr. Felipe Nakamura, Applied Scientist do projeto TCC (accent control via LoRA + Qwen3-TTS).
Consulte o agent file em agents/felipe-nakamura.md para sua persona completa.
Senioridade: Senior/PhD. Perfil: métricas de avaliação, leakage probes, baselines, significância estatística.

CONTEXTO DESTE TIME:
- Objetivo do time: {objetivo}
- Seu papel neste time: {papel específico}
- Outros teammates: {lista com nomes e papéis}

TAREFA: {descrição da tarefa}
ENTREGA ESPERADA: {o que deve ser produzido}
```

### ana-silva

```
Você é Ana K. Silva, Independent Technical Reviewer (Red Team) do projeto TCC (accent control via LoRA + Qwen3-TTS).
Consulte o agent file em agents/ana-silva.md para sua persona completa.
Senioridade: Senior. Perfil: revisão adversarial, red team, auditoria de claims e metodologia.
IMPORTANTE: Você NÃO implementa código. Você audita, questiona e reporta.

CONTEXTO DESTE TIME:
- Objetivo do time: {objetivo}
- Seu papel neste time: {papel específico}
- Outros teammates: {lista com nomes e papéis}

TAREFA: {descrição da tarefa}
ENTREGA ESPERADA: {o que deve ser produzido}
```

---

## 3. Padrões de Composição de Times

Padrões pré-definidos para cenários recorrentes. O Lead (head-research) escolhe o padrão, customiza os campos `{...}` e spawna os teammates.

### 3.1 `experiment-review` — Revisão Multi-Perspectiva de Resultados

**Quando usar:** Após completar um experimento, antes de aceitar os resultados como válidos.

**Composição:**

| Teammate | Agent | Papel no time |
|----------|-------|---------------|
| Model Lead | `rafael-monteiro` | Validar arquitetura e config, interpretar resultados do ponto de vista do modelo |
| Evaluator | `felipe-nakamura` | Verificar métricas, CIs, comparação com baseline, leakage probes |
| Red Team | `ana-silva` | Revisão adversarial, questionar claims, buscar confounds |

**Task list sugerida:**

```
Task 1: [rafael] Revisar config do modelo, LoRA params, e interpretar loss curves
Task 2: [felipe] Verificar todas as métricas, CIs, e comparar com baselines
Task 3: [ana] Red team review dos resultados e claims (blocked by 1, 2)
Task 4: [Lead] Consolidar findings e decidir próximos passos (blocked by 3)
```

**Instruções de coordenação:**
- Rafael e Felipe trabalham em paralelo (perspectivas complementares).
- Ana recebe os findings de ambos antes de iniciar (garante que a auditoria é informada).
- Lead consolida depois que os 3 terminam.

**Critério de done:** Report consolidado com veredicto por claim (suportada / parcial / refutada).

---

### 3.2 `data-quality-gate` — Validação de Qualidade do Dataset

**Quando usar:** Antes de iniciar treinamento, para garantir que o dataset está pronto.

**Composição:**

| Teammate | Agent | Papel no time |
|----------|-------|---------------|
| Data Engineer | `mariana-alves` | Integridade, splits, confounds, metadata |
| Infra | `lucas-andrade` | Pipeline de dados, reprodutibilidade do preprocessing |
| Evaluator | `felipe-nakamura` | Distribuições, balanceamento, baselines |

**Task list sugerida:**

```
Task 1: [mariana] Auditar metadata, splits, e confounds
Task 2: [lucas] Verificar pipeline de preprocessing e reprodutibilidade
Task 3: [felipe] Analisar distribuições e calcular baselines (blocked by 1)
Task 4: [Lead] Consolidar e decidir se dataset está pronto (blocked by 2, 3)
```

**Instruções de coordenação:**
- Mariana e Lucas trabalham em paralelo (dados vs pipeline).
- Felipe depende dos splits validados de Mariana para calcular baselines corretos.
- Se Mariana encontra confound, comunica imediatamente para Lead.

**Critério de done:** Dataset certificado para uso em treinamento, ou lista de issues para resolver.

---

### 3.3 `protocol-gate` — Decisão PASS/ADJUST/FAIL

**Quando usar:** No final de cada Stage do protocolo de validação, para decisão formal.

**Composição:**

| Teammate | Agent | Papel no time |
|----------|-------|---------------|
| Data | `mariana-alves` | Validar integridade do dataset |
| Infra | `lucas-andrade` | Validar reprodutibilidade do pipeline |
| Evaluator | `felipe-nakamura` | Rodar métricas e produzir relatório |
| Model | `rafael-monteiro` | Validar arquitetura e config LoRA |
| Red Team | `ana-silva` | Red team review de todos os achados |

**Task list sugerida:**

```
Task 1: [mariana] Validar integridade do dataset para este stage
Task 2: [lucas] Validar reprodutibilidade do pipeline
Task 3: [felipe] Rodar todas as métricas e produzir relatório vs thresholds
Task 4: [rafael] Validar arquitetura do modelo e config LoRA
Task 5: [ana] Red team review de todos os achados (blocked by 1, 2, 3, 4)
Task 6: [Lead] Consolidar decisão PASS/ADJUST/FAIL (blocked by 5)
```

**Instruções de coordenação:**
- Tasks 1-4 rodam em paralelo (áreas independentes).
- Ana espera todas as 4 tasks para fazer red team informado.
- Lead toma decisão formal baseada nos 5 reports.
- Se qualquer persona reporta BLOCK, a decisão é FAIL ou ADJUST.

**Critério de done:** Decisão formal documentada: PASS (prosseguir), ADJUST (corrigir e re-testar), ou FAIL (reavaliar abordagem).

---

### 3.4 `hostile-replication` — Tentativa Adversarial de Replicação

**Quando usar:** Para validar que resultados são robustos e não artefatos.

**Composição:**

| Teammate | Agent | Papel no time |
|----------|-------|---------------|
| Red Team | `ana-silva` | Liderar a tentativa adversarial, questionar cada passo |
| Evaluator | `felipe-nakamura` | Re-rodar métricas com seeds diferentes, verificar robustez |
| Infra | `lucas-andrade` | Verificar reprodução exata do pipeline, testar em config diferente |

**Task list sugerida:**

```
Task 1: [lucas] Reproduzir treinamento com seed diferente, verificar variância
Task 2: [felipe] Re-calcular métricas nos checkpoints reproduzidos
Task 3: [ana] Comparar resultados original vs reprodução, identificar inconsistências (blocked by 1, 2)
Task 4: [Lead] Documentar resultado da replicação (blocked by 3)
```

**Instruções de coordenação:**
- Lucas reproduz, Felipe mede, Ana compara.
- Se resultados divergem significativamente, Ana investiga a causa.
- Lead documenta se replicação foi bem-sucedida ou não.

**Critério de done:** Relatório de replicação com status (replicou / parcial / falhou).

---

## 4. Instruções de Coordenação

### Delegate Mode (Shift+Tab)

Ative Delegate Mode quando quiser que o Lead apenas coordene sem executar:

- **Quando ativar:** tarefas com 3+ teammates, especialmente protocol-gate.
- **Comportamento:** Lead cria tasks, spawna teammates, monitora progresso e consolida resultados.
- **Quando desativar:** se precisar que o Lead execute algo diretamente (pesquisa rápida, consolidação final).

### Plan Approval

Exigir aprovação do plano quando:

- Gate decision (protocol-gate).
- Experimento que envolve treinamento (> 1 hora de GPU).
- Mudança em splits ou preprocessing do dataset.
- Qualquer alteração na arquitetura do modelo.

### Cleanup

Sempre via Lead:

1. Verificar que todas as tasks estão `completed`.
2. Consolidar resultados em resposta ao usuário.
3. Shutdown de teammates (eles encerram quando não há mais tasks).

**Nunca** deixar teammates rodando sem tasks atribuídas.

### Monitoramento

- **Check-in:** Lead verifica progresso a cada ciclo.
- **Redirecionamento:** se um teammate está stuck, Lead intervém com contexto adicional.
- **Conflito de edição:** se 2 teammates precisam editar o mesmo arquivo, Lead serializa.
- **Comunicação direta:** teammates podem e devem se comunicar diretamente via mailbox.
