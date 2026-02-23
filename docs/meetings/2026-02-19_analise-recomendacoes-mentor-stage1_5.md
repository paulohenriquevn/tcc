# Reuniao de Pesquisa: Analise das Recomendacoes do Mentor para Stage 1.5

**Data:** 2026-02-19
**Facilitador:** head-research
**Tipo:** Reuniao plena (5 personas) — analise de recomendacoes externas
**Gatilho:** Veredito ADJUST do mentor sobre ata de auditoria

## Participantes

| # | Persona | Foco na Reuniao | Perspectiva |
|---|---------|-----------------|-------------|
| 1 | Mariana Alves | Contrato de dados, splits, confounds, sanity checks | O que o manifest precisa ter antes de congelar |
| 2 | Lucas Andrade | Pipeline executavel, `make stage15`, seeds, hashes | Como tornar o Stage 1.5 um artefato unico |
| 3 | Felipe Nakamura | Thresholds numericos, criterios GO/FAIL, metricas | Pre-registro de decisoes e criterios |
| 4 | Rafael Monteiro | Backbone, camadas de probing, classificador de accent | Decisoes tecnicas sobre modelo e features |
| 5 | Ana K. Silva | Auditoria adversarial do template, gaps, veto checks | O que um auditor externo nao consegue verificar |

---

## Contexto

O mentor avaliou a ata da auditoria de readiness (`meetings/2026-02-19_auditoria-readiness-stage1_5.md`) e emitiu veredito **ADJUST**: caminho correto, execucao ainda nao demonstrada. Trouxe 4 recomendacoes e um template oficial de 12 secoes para o report de Stage 1.5, alem de 3 veto checks.

**Estado factual do codebase neste momento:**
- Diretorio `notebooks/` vazio — zero notebooks
- Zero scripts Python do projeto (apenas `paper_search.py` em `.claude/skills/`)
- Zero configs YAML de experimento
- Zero Makefile, Dockerfile, requirements.txt do projeto
- Zero manifests JSONL, zero artefatos de dados
- O codebase e **100% documentacao e definicoes de agentes**

---

## PARTE 1: Gap Analysis por Recomendacao do Mentor

### Recomendacao 1: Artefato unico executavel ("make stage15")

**O que o mentor pede:**
Um comando unico que gera splits, roda probes, calcula CI 95%, exporta `report.md` + `report.json`.

**Estado atual:**

| Componente | Existe? | Onde | Evidencia |
|-----------|---------|------|-----------|
| Makefile | NAO | — | `Glob("**/Makefile")` retorna vazio |
| Script de splits | NAO | — | Nenhum `.py` no projeto (exceto `paper_search.py`) |
| Script de probes | NAO | — | Idem |
| Script de CI 95% | NAO | — | Idem |
| Script de report | NAO | — | Idem |
| Config YAML | NAO | — | `Glob("**/*.yaml")` retorna vazio (projeto) |
| Notebook | NAO | — | `ls notebooks/` retorna diretorio vazio |
| Companion repo (CLI `stage1_5`) | REFERENCIADO | `CLAUDE.md:114-118` | URL do GitHub mencionada, mas repo nao clonado/auditavel |

**Posicao Mariana (dados):** Sem manifest, nao ha o que alimentar ao pipeline. O companion repo e mencionado como tendo `build_manifest_from_coraa()` e `ManifestEntry`, mas nada disso e auditavel localmente.

**Posicao Lucas (infra):** O gap e total. Nao existe UM unico arquivo executavel no projeto. Nao ha Makefile, nao ha requirements.txt, nao ha Docker. A recomendacao do mentor requer criar a infraestrutura do zero. Estimativa: isso nao e "adicionar um Makefile" — e construir todo o pipeline.

**Posicao Ana (red team):** O CLAUDE.md (linha 107) afirma "Stage 1.5 in progress" e lista comandos CLI como `stage1_5 run <config.yaml>`. Isso e FALSO no contexto do codebase local — nenhum desses comandos existe aqui. A contradicao C7 da auditoria anterior continua: documentacao afirma progresso que nao corresponde a realidade do codebase.

**GAP:** TOTAL. Nenhum componente do artefato executavel existe.

---

### Recomendacao 2: Congelar contrato de dados (manifest + split versionados)

**O que o mentor pede:**
Manifest versionado com SHA-256, split versionado, congelados ANTES de qualquer LoRA.

**Estado atual:**

| Componente | Existe? | Evidencia |
|-----------|---------|-----------|
| Manifest JSONL | NAO | `Glob("**/*.jsonl")` retorna vazio |
| Arquivo de split | NAO | Auditoria anterior: "Splits nao persistidos (efemeros em memoria)" |
| Hash SHA-256 de manifest | NAO | Auditoria anterior: item F10 pendente |
| Assertion de speaker-disjointness | NAO | Auditoria anterior: item F11 pendente |
| Campos gender/duration no manifest | NAO | Auditoria anterior: item F4 pendente |

**Posicao Mariana (dados):**

O contrato de dados precisa definir ANTES da implementacao:

1. **Campos obrigatorios do manifest:** `utt_id, path, speaker_id, accent (macro-regiao IBGE), gender, duration_s, text_id, source, birth_state`
2. **Filtros aplicados:** `speaker_type='R'` (entrevistado), duracao 3-15s, audio quality check
3. **Derivacao de accent:** mapping `birth_state -> macro-regiao IBGE` (N, NE, CO, SE, S)
4. **Constraint de consistencia:** cada `speaker_id` mapeia para exatamente 1 `accent`

O checklist de auditoria da reuniao anterior (`meetings/2026-02-19_validacao-tecnica-accent-control.md:287-321`) define 22 items. Nenhum esta checkado.

**Posicao Lucas (infra):**
O contrato de dados e pre-requisito de TUDO. Sem manifest versionado, nao ha como:
- Definir splits deterministicos
- Calcular hashes reproduziveis
- Garantir que duas execucoes usam os mesmos dados

Formato proposto para versionamento:
```
data/
  manifest_v1.jsonl          # SHA-256: abc123...
  splits_v1_seed42.json      # {train: [speaker_ids], val: [...], test: [...]}
  manifest_v1.sha256         # hash do manifest
```

**GAP:** TOTAL. Nem o manifest existe, muito menos sua versao congelada.

---

### Recomendacao 3: Pre-registrar decisoes (classificador, thresholds, criterios)

**O que o mentor pede:**
- Classificador de accent definido
- Threshold unico (resolver 5% vs 10%)
- Criterios GO/GO_CONDITIONAL/FAIL em numeros

**Template do mentor com criterios numericos:**
- Accent probe balanced accuracy: GO >= 0.55, GO_CONDITIONAL >= 0.50, FAIL < 0.50
- Leakage: GO <= chance+5pp, CONDITIONAL <= chance+12pp, FAIL > 12pp
- Baseline ECAPA intra/inter com CI

**Estado atual — Contradicoes nos thresholds (auditoria anterior, secao 2):**

| Metrica | STAGE_1.md | TECHNICAL_VALIDATION_PROTOCOL.md | Reuniao anterior (Felipe) | Mentor |
|---------|-----------|----------------------------------|--------------------------|--------|
| Accent F1 | >= 0.70 | "acima do baseline" (sem numero) | >= 0.55 GO | >= 0.55 GO |
| Speaker sim queda | <= 5% | < 10% | < 10% piloto | — |
| Leakage | <= chance + 5pp | <= chance + 5pp | <= chance + 7pp GO | <= chance + 5pp GO |
| Metrica primaria | F1-macro | balanced accuracy | — | balanced accuracy |

**Posicao Felipe (eval):**

As contradicoes C2-C5 da auditoria anterior continuam TODAS abertas. O mentor resolve parcialmente ao definir numeros concretos no template, mas os documentos do projeto nao foram atualizados.

Proposta de resolucao unificada (alinhada com mentor + auditoria):

| Metrica | GO | GO_CONDITIONAL | FAIL | Documento autoritativo |
|---------|----|----------------|------|----------------------|
| Accent probe balanced accuracy | >= 0.55 | >= 0.50 | < 0.50 | Template mentor (secao 1-Gate) |
| Leakage A->speaker | <= chance + 5pp | <= chance + 12pp | > chance + 12pp | Template mentor |
| Leakage S->accent | <= chance + 5pp | <= chance + 12pp | > chance + 12pp | Template mentor (simetrico) |
| Speaker sim queda (Stage 2) | < 10% | < 15% | >= 15% | Consenso Felipe (auditoria) |

**DECISAO NECESSARIA:** F1-macro ou balanced accuracy como metrica primaria? O mentor usa balanced accuracy. O TECHNICAL_VALIDATION_PROTOCOL.md usa balanced accuracy. O STAGE_1.md usa F1-macro. A auditoria anterior notou essa contradicao (C4). **Recomendacao: usar balanced accuracy como primaria (alinhado com protocolo e mentor), reportar F1-macro como secundaria.**

**Posicao Rafael (modelo):**

O classificador de accent para Stage 1.5 precisa ser definido agora:
- Stage 1.5 usa probes lineares (logistic regression) sobre features extraidas — isso ja esta definido
- O classificador "externo" de accent (STAGE_1.md:94, TECHNICAL_VALIDATION_PROTOCOL.md:126) e para Stages 2-3, NAO para Stage 1.5
- Em Stage 1.5, probe = classificador. Nao ha ambiguidade, mas precisa estar documentado explicitamente

Camadas de probing para o 1.7B-CustomVoice (decisao D1 da auditoria resolvida: documentacao atualizada para 1.7B):
- O Qwen3-TTS 1.7B-CustomVoice tem 28 camadas no talker (`STAGE_1.md:52`, `CLAUDE.md:39`)
- Probing layer-wise: `[0, 4, 8, 12, 16, 20, 24, 27]` (8 pontos cobrindo toda profundidade)
- WavLM: `[0, 6, 12, 18, 24]` (5 pontos, ja definido como OK na auditoria)
- ECAPA-TDNN: embedding final (192-dim ou 1024-dim — resolver)

**GAP:** PARCIAL. O mentor definiu numeros concretos no template. Os documentos do projeto nao foram atualizados para refletir esses numeros. 4 contradicoes em aberto.

---

### Recomendacao 4: Sanity checks (accent x gender chi-quadrado, histograma duracao x regiao)

**O que o mentor pede:**
- Tabela accent x gender com teste chi-quadrado
- Histograma de duracao por regiao

**Estado atual:**

| Check | Existe? | Evidencia |
|-------|---------|-----------|
| Tabela accent x gender | NAO | Nenhum script de analise |
| Chi-quadrado | NAO | Nenhum codigo estatistico |
| Histograma duracao x regiao | NAO | Nenhum notebook/script de visualizacao |
| Campos gender no manifest | NAO | Achado 4 da auditoria: campos descartados |

**Posicao Mariana (dados):**

Sem gender e duration no manifest, esses sanity checks sao IMPOSSIVEIS. A sequencia obrigatoria e:

1. Primeiro: adicionar gender e duration ao ManifestEntry (pre-requisito)
2. Segundo: calcular tabela cruzada accent x gender
3. Terceiro: rodar chi-quadrado (scipy.stats.chi2_contingency)
4. Quarto: calcular histograma de duracao por regiao (Kruskal-Wallis para testar diferenca)

Criterios de decisao para confounds (propostos):
- Chi-quadrado accent x gender: p < 0.05 = confound detectado
- Se detectado: calcular Cramer's V. V < 0.3 = aceitavel com documentacao. V >= 0.3 = subsampling necessario ou BLOQUEANTE
- Kruskal-Wallis accent x duration: p < 0.05 E diferenca pratica > 1s = documentar como limitacao

**Posicao Ana (red team):**

A AUSENCIA desses sanity checks e exatamente o que um reviewer externo atacaria primeiro. "O modelo aprendeu sotaque ou aprendeu genero?" Se nao ha evidencia de que accent e gender sao distribuidos de forma razoavelmente independente, qualquer resultado positivo e questionavel.

Nota adicional: o CORAA-MUPE e fala espontanea de entrevistas de historia de vida. Ha risco alto de confound accent x topico, accent x faixa etaria, accent x condicao socioeconomica. O mentor pediu os 2 sanity checks minimos (gender + duracao), mas na pratica a lista e maior.

**GAP:** TOTAL. Impossivel sem gender/duration no manifest. Pre-requisito: recomendacao 2.

---

## PARTE 2: Template Oficial de 12 Secoes — Mapeamento

O mentor definiu um template com 12 secoes para o report de Stage 1.5. Mapeamento contra o que existe:

| Secao | Nome | Existe no codebase? | O que falta |
|-------|------|---------------------|-------------|
| 0 | Identificacao | PARCIAL | Tem CLAUDE.md com metadados do projeto. Falta: data de execucao, versao do pipeline, commit hash |
| 1 | Gate (criterios numericos) | NAO | Contradicoes C2-C5 em aberto. Mentor definiu numeros, nao estao no codebase |
| 2 | Hipoteses | PARCIAL | STAGE_1.md:42-48 tem H1 e H2, mas nao no formato "Se X, entao Y, medido por Z" |
| 3 | Dataset/Splits/Confounds | NAO | Zero manifests, zero splits, zero analise de confounds |
| 4 | Extracao (features) | NAO | Zero scripts de extracao. Companion repo referenciado mas nao auditavel |
| 5 | Probes | NAO | Zero scripts de probing. Achado 1 (splits invertidos) nao corrigido |
| 6 | Resultados | NAO | Zero resultados — nada foi executado |
| 7 | Baseline ECAPA | NAO | Auditoria anterior: "Sem baseline intra/inter speaker similarity ECAPA" |
| 8 | Robustez 3 seeds | NAO | Auditoria anterior: "Zero seeds em todo o notebook" |
| 9 | Evidencias | NAO | Nenhuma evidencia empirica no codebase |
| 10 | Decisao | NAO | Impossivel sem resultados |
| 11 | Claims | NAO | Impossivel sem resultados |
| 12 | Hard Fail | NAO | Nenhum criterio de hard fail codificado |

**Posicao Felipe (eval):**

Das 12 secoes, 0 (ZERO) estao completas. Apenas a secao 0 e 2 tem cobertura parcial (e mesmo assim, com inconsistencias). Isso significa que o report de Stage 1.5 precisa ser construido do zero.

**Posicao Ana (red team):**

O template do mentor e um avanco significativo — ele resolve a ambiguidade que existia no projeto sobre "o que significa Stage 1.5 estar pronto". Antes, o criterio era vago. Agora ha 12 secoes concretas com criterios numericos em pelo menos 3 delas (secoes 1, 7, 8).

Ponto critico: a secao 12 (Hard Fail) do template e a mais importante. Ela define condicoes de parada automatica. O projeto nao tem NENHUMA condicao de hard fail codificada. O arquivo `.claude/rules/KB_HARD_FAIL_RULES.md` existe mas esta VAZIO (1 linha, sem conteudo — verificado via Read).

---

## PARTE 3: Veto Checks do Mentor

| # | Veto Check | Resultado | Evidencia |
|---|-----------|-----------|-----------|
| 1 | Arquivo de split com speakers + asserts | **BLOCKED** | Zero arquivos de split no codebase. `Glob("**/*.json*")` nao retorna nenhum artefato de dados |
| 2 | Tabela de leakage com chance e delta(pp) | **BLOCKED** | Zero resultados, zero tabelas. Achado 1 (splits invertidos) nao corrigido |
| 3 | Baseline intra/inter ECAPA com CI | **BLOCKED** | Zero baselines calculados. Auditoria anterior: item F15 pendente |

**Resultado: 3/3 BLOCKED. Identico a auditoria anterior. Nenhum progresso nos veto checks.**

---

## PARTE 4: Diagnostico Consolidado por Persona

### Mariana Alves (DATA)

**Diagnostico:** Nenhum artefato de dados existe no codebase. O contrato de dados precisa ser criado antes de qualquer outra coisa.

**Bloqueantes que so ela resolve:**
1. Definir schema do ManifestEntry com todos os campos (incluindo gender, duration)
2. Implementar script de construcao do manifest a partir do CORAA-MUPE
3. Implementar geracao e persistencia de splits speaker-disjoint
4. Implementar analise de confounds (chi2 accent x gender, Kruskal-Wallis accent x duration)
5. Gerar hash SHA-256 do manifest

### Lucas Andrade (INFRA)

**Diagnostico:** Infraestrutura de execucao inexistente. Nenhum Makefile, Dockerfile, requirements.txt, config YAML ou script de automacao.

**Bloqueantes que so ele resolve:**
1. Criar `requirements.txt` com versoes pinadas
2. Criar Makefile com target `stage15` (ou equivalente)
3. Criar config YAML template para o pipeline
4. Implementar bloco de seeds (random, numpy, torch, cuda)
5. Implementar geracoes de hashes e logging de versoes

### Felipe Nakamura (EVAL)

**Diagnostico:** Criterios de decisao ambiguos (4 contradicoes documentadas). Template do mentor resolve, mas nao foi incorporado ao codebase.

**Bloqueantes que so ele resolve:**
1. Unificar thresholds em documento unico e autoritativo
2. Atualizar TODOS os documentos com thresholds unificados
3. Implementar calculo de CI 95% (bootstrap)
4. Implementar baseline ECAPA intra/inter speaker
5. Implementar confusion matrix no pipeline

### Rafael Monteiro (MODEL)

**Diagnostico:** Modelo backbone definido (1.7B-CustomVoice). Documentacao parcialmente atualizada (CLAUDE.md e STAGE_1.md ja referenciam 1.7B). Camadas de probing e target modules de LoRA nao codificados.

**Bloqueantes que so ele resolve:**
1. Definir lista de camadas para probing do backbone (proposta: [0, 4, 8, 12, 16, 20, 24, 27])
2. Documentar ECAPA dimensionalidade (192 vs 1024)
3. Validar que 1.7B cabe em 24GB para extracao de features
4. Definir pooling temporal para representacoes layer-wise

### Ana K. Silva (RED TEAM)

**Diagnostico:** O codebase continua nao-auditavel. Zero codigo executavel, zero resultados, zero artefatos. O companion repo continua inacessivel.

**Bloqueantes que ela identifica (sem resolver):**
1. Companion repo nao clonado nem incorporado — nao-auditavel
2. KB_HARD_FAIL_RULES.md vazio — nenhum criterio de hard fail definido
3. Contradicoes C2-C5 em aberto — documentos divergentes
4. Template do mentor nao codificado no codebase — existe apenas no contexto da conversa
5. `CLAUDE.md:107` afirma "Stage 1.5 in progress" — falso. Nada esta em progresso.

---

## PARTE 5: Plano de Implementacao Priorizado

### Principio: Dependencias determinam a sequencia

```
                        DECISOES (D1-D3)
                              |
                    +---------+---------+
                    |                   |
              CONTRATO DADOS       INFRA BASE
              (Mariana)            (Lucas)
                    |                   |
                    +-------------------+
                              |
                       PIPELINE INTEGRADO
                       (Lucas + Felipe)
                              |
                    +---------+---------+
                    |                   |
              METRICAS/PROBES     SANITY CHECKS
              (Felipe)            (Mariana)
                              |
                         EXECUCAO
                              |
                         REPORT
                              |
                       AUDITORIA (Ana)
```

### Fase 0: Decisoes Pendentes (ANTES de qualquer codigo)

| # | Decisao | Status | Acao Necessaria | Quem |
|---|---------|--------|-----------------|------|
| D1 | Modelo backbone | RESOLVIDO | CLAUDE.md e STAGE_1.md ja dizem 1.7B-CustomVoice. Confirmar com Paulo | Rafael + Paulo |
| D2 | Thresholds unificados | PENDENTE | Adotar numeros do template do mentor. Atualizar TODOS os docs | Felipe |
| D3 | Metrica primaria: balanced accuracy vs F1-macro | PENDENTE | Adotar balanced accuracy (protocolo + mentor). F1-macro como secundaria | Felipe |
| D4 | ECAPA dimensionalidade: 192-dim vs 1024-dim | PENDENTE | Definir qual modelo ECAPA usar. SpeechBrain pre-treinado = 192-dim. Qwen3 built-in = 1024-dim | Rafael |
| D5 | Criterios de confound (p-value, Cramer's V, ponto de corte) | PENDENTE | Definir quando confound e "bloqueante" vs "documentar como limitacao" | Mariana + Felipe |
| D6 | KB_HARD_FAIL_RULES.md vazio | PENDENTE | Preencher com criterios de hard fail do template do mentor (secao 12) | Felipe + Ana |

### Fase 1: Fundacao (sem isto, nada mais funciona)

**Caminho critico. Bloqueio total ate conclusao.**

| # | Tarefa | Responsavel | Esforco | Depende de | Artefato |
|---|--------|-------------|---------|------------|----------|
| F1.1 | Criar `requirements.txt` com versoes pinadas | Lucas | 1h | — | `requirements.txt` |
| F1.2 | Criar schema do ManifestEntry (todos os campos) | Mariana | 1h | D5 | Documentado em `docs/manifest_schema.md` |
| F1.3 | Criar config YAML template para Stage 1.5 | Lucas | 2h | F1.2 | `configs/stage1_5.yaml` |
| F1.4 | Implementar script de construcao do manifest | Mariana | 4-6h | F1.2 | `src/data/manifest_builder.py` |
| F1.5 | Implementar script de splits speaker-disjoint | Mariana | 3-4h | F1.4 | `src/data/splits.py` |
| F1.6 | Implementar bloco de seeds | Lucas | 1h | — | `src/utils/seed.py` |
| F1.7 | Preencher KB_HARD_FAIL_RULES.md | Felipe + Ana | 1h | D6 | `.claude/rules/KB_HARD_FAIL_RULES.md` |
| F1.8 | Atualizar thresholds em todos os documentos | Felipe | 2-3h | D2, D3 | 5+ documentos atualizados |

**Paralelismo possivel:**
- Lucas (F1.1, F1.3, F1.6) em paralelo com Mariana (F1.2, F1.4, F1.5)
- Felipe (F1.7, F1.8) em paralelo com ambos

**Esforco Fase 1: ~15-20h | Tempo com paralelismo: ~6-8h**

### Fase 2: Pipeline de Extracao e Analise

| # | Tarefa | Responsavel | Esforco | Depende de | Artefato |
|---|--------|-------------|---------|------------|----------|
| F2.1 | Implementar extracao de features acusticas | Rafael + Lucas | 3-4h | F1.4 | `src/features/acoustic.py` |
| F2.2 | Implementar extracao ECAPA | Rafael | 2-3h | F1.4, D4 | `src/features/ecapa.py` |
| F2.3 | Implementar extracao SSL (WavLM) | Rafael | 2-3h | F1.4 | `src/features/ssl.py` |
| F2.4 | Implementar extracao backbone (Qwen3-TTS) | Rafael | 3-4h | F1.4 | `src/features/backbone.py` |
| F2.5 | Implementar analise de confounds (chi2, K-W) | Mariana | 3-4h | F1.5 | `src/analysis/confounds.py` |
| F2.6 | Implementar baseline ECAPA intra/inter | Felipe | 2-3h | F2.2 | `src/evaluation/baseline_ecapa.py` |
| F2.7 | Implementar probes lineares (corrigidos) | Felipe | 3-4h | F2.1-F2.4 | `src/evaluation/probes.py` |
| F2.8 | Implementar CI 95% bootstrap | Felipe | 2h | — | `src/evaluation/bootstrap_ci.py` |
| F2.9 | Implementar confusion matrix | Felipe | 1h | F2.7 | `src/evaluation/confusion.py` |

**Paralelismo possivel:**
- Rafael (F2.1-F2.4) em paralelo com Mariana (F2.5) e Felipe (F2.8)
- Felipe (F2.6, F2.7, F2.9) depois que Rafael entregar features

**Esforco Fase 2: ~22-30h | Tempo com paralelismo: ~10-12h**

### Fase 3: Integracao e Automacao

| # | Tarefa | Responsavel | Esforco | Depende de | Artefato |
|---|--------|-------------|---------|------------|----------|
| F3.1 | Criar Makefile com targets (manifest, features, probes, report) | Lucas | 2-3h | F2.* | `Makefile` |
| F3.2 | Implementar gerador de report.md e report.json | Lucas + Felipe | 3-4h | F2.* | `src/reporting/stage15_report.py` |
| F3.3 | Implementar hashes SHA-256 automaticos | Lucas | 1h | F1.4, F1.5 | Integrado no pipeline |
| F3.4 | Integrar tudo em `make stage15` | Lucas | 2h | F3.1-F3.3 | Target no Makefile |
| F3.5 | Testes unitarios minimos (splits, probes, CI) | Felipe + Lucas | 4-6h | F2.* | `tests/` |

**Esforco Fase 3: ~12-16h | Tempo com paralelismo: ~6-8h**

### Fase 4: Execucao e Auditoria

| # | Tarefa | Responsavel | Esforco | Depende de | Artefato |
|---|--------|-------------|---------|------------|----------|
| F4.1 | Executar `make stage15` com seed 42 | Lucas | 2-4h (GPU) | F3.4 | `reports/stage1_5_report.md`, `reports/stage1_5_report.json` |
| F4.2 | Executar com seeds 1337 e 7 (robustez) | Lucas | 4-8h (GPU) | F4.1 | 3 reports |
| F4.3 | Revisar report — veto checks | Ana | 2h | F4.1 | Red team report |
| F4.4 | Decisao GO/GO_CONDITIONAL/FAIL | Todas as 5 personas | 2h | F4.3 | Decisao documentada |

**Esforco Fase 4: ~10-16h | Depende de acesso a GPU**

### Resumo de Esforco Total

| Fase | Esforco | Caminho Critico |
|------|---------|-----------------|
| Fase 0: Decisoes | 2-3h (reuniao) | D2 (thresholds) e D4 (ECAPA) desbloqueiam tudo |
| Fase 1: Fundacao | 15-20h | F1.4 (manifest builder) e o gargalo |
| Fase 2: Pipeline | 22-30h | F2.4 (backbone features) requer GPU |
| Fase 3: Integracao | 12-16h | F3.4 (make stage15) e o artefato final |
| Fase 4: Execucao | 10-16h | Depende de GPU |
| **Total** | **~60-85h (~8-12 dias)** | |

**Nota:** A estimativa anterior (auditoria) era 30-45h. Aquela estimativa assumia que o companion repo existia e funcionava. Este plano parte do zero real.

---

## PARTE 6: Decisoes Tomadas nesta Reuniao

| # | Decisao | Aprovado? | Justificativa |
|---|---------|-----------|---------------|
| 1 | Backbone e 1.7B-CustomVoice | SIM (ja refletido em CLAUDE.md, STAGE_1.md, TECHNICAL_VALIDATION_PROTOCOL.md) | Unico modelo mencionado consistentemente nos documentos atualizados |
| 2 | Template do mentor sera adotado integralmente | SIM (unanime) | Define criterios objetivos que eliminam ambiguidade |
| 3 | Balanced accuracy como metrica primaria | SIM (Felipe propoe, Rafael e Mariana concordam) | Alinhado com protocolo e mentor. F1-macro reportada como secundaria |
| 4 | Codigo sera implementado NESTE codebase (nao companion repo) | SIM (Ana exige, Lucas concorda) | Elimina problema de auditabilidade. Companion repo e referencia, nao dependencia |
| 5 | Pipeline implementado como scripts Python, nao notebooks | SIM (Lucas propoe) | Notebooks para exploracao, scripts para execucao reproduzivel |
| 6 | Threshold de confound: Cramer's V < 0.3 aceitavel com documentacao, >= 0.3 bloqueante | PENDENTE (Paulo precisa confirmar) | Precisa de validacao externa |

---

## PARTE 7: O que Mudou desde a Auditoria Anterior

| Item | Estado na auditoria | Estado agora | Progresso |
|------|--------------------|--------------|---------
| Achado 1 (probes invertidos) | CRITICO | CRITICO | Zero (nenhum codigo) |
| Achado 2 (modelo 0.6B vs 1.7B) | CRITICO | RESOLVIDO (docs atualizados para 1.7B) | Documentacao atualizada |
| Achado 3 (zero seeds) | CRITICO | CRITICO | Zero (nenhum codigo) |
| Achado 4 (manifest sem gender) | BLOQUEANTE | BLOQUEANTE | Zero (nenhum manifest) |
| Achado 5 (repo nao auditavel) | BLOQUEANTE | BLOQUEANTE (decisao 4 mitiga) | Decisao de implementar localmente |
| Contradicoes C2-C5 | ABERTAS | ABERTAS | Zero atualizacoes nos documentos |
| Veto checks 1-3 | BLOCKED | BLOCKED | Identico |

**Progresso liquido: 1 item resolvido (modelo), 1 decisao tomada (implementar local). Demais: zero.**

---

## PARTE 8: Condicoes de Veto (atualizadas)

| Persona | Veta execucao se: |
|---------|-------------------|
| **Mariana** | Manifest sem gender/duration. Splits nao persistidos. Confounds nao analisados |
| **Lucas** | Seeds nao configurados. Requirements nao pinados. Sem hashes SHA-256. Sem Makefile |
| **Felipe** | Thresholds nao unificados. Sem CI 95%. Sem baseline ECAPA. Probes com splits invertidos |
| **Rafael** | Camadas de probing < 6 pontos no backbone. ECAPA dimensionalidade nao decidida |
| **Ana** | Codigo nao auditavel neste codebase. KB_HARD_FAIL_RULES.md vazio. Contradicoes abertas |

---

## PARTE 9: Proximos Passos Imediatos

**Prioridade 1 (HOJE):**
1. Decisao D2: adotar thresholds do mentor como definitivos
2. Decisao D3: balanced accuracy como primaria
3. Decisao D4: ECAPA 192-dim (SpeechBrain) como padrao
4. Decisao D5: Cramer's V < 0.3 como criterio de confound
5. Preencher KB_HARD_FAIL_RULES.md

**Prioridade 2 (esta semana):**
1. F1.1: requirements.txt
2. F1.2: schema do manifest
3. F1.6: bloco de seeds
4. F1.8: atualizar thresholds nos docs

**Prioridade 3 (proxima semana):**
1. F1.4: manifest builder
2. F1.5: splits
3. F2.5: confounds
4. F2.8: bootstrap CI

---

*Ata gerada a partir de investigacao individual das 5 personas sobre o codebase real.*
*Evidencias verificadas via Glob, Grep e Read sobre todos os arquivos do projeto.*
*Proxima reuniao: apos conclusao da Fase 1 (fundacao do pipeline).*
