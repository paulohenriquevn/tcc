# Reuniao de Pesquisa: Auditoria de Readiness do Stage 1.5

**Data:** 2026-02-19
**Facilitador:** head-research
**Tipo:** Auditoria operacional (5 personas, time completo)

## Participantes

| # | Persona | Foco | Veredicto Individual |
|---|---------|------|---------------------|
| 1 | Mariana Alves | Splits, confounds, manifest | NAO APROVADO (5 bloqueantes) |
| 2 | Lucas Andrade | Pipeline, seeds, reproducibilidade | NEEDS ATTENTION (2 hard fails) |
| 3 | Felipe Nakamura | Probes, metricas, thresholds | NAO VALIDO (leakage invertido) |
| 4 | Rafael Monteiro | Backbone, camadas, LoRA targets | NEEDS ATTENTION (modelo errado) |
| 5 | Ana K. Silva | Red team (auditoria adversarial) | **FAIL** (7 contradicoes, repo nao auditavel) |

## Veredicto Consolidado: FAIL — Stage 1.5 nao esta pronto para execucao

---

## 1. Os 5 Achados Mais Graves (em ordem de severidade)

### ACHADO 1 (CRITICO): Probes de leakage com splits invertidos

**Quem encontrou:** Felipe Nakamura
**Severidade:** Invalida conclusoes sobre disentanglement

O `leakage_a2s` (accent features vazam speaker?) usa split **speaker-disjoint**, onde os speakers do test NUNCA aparecem no train. O probe tenta predizer speaker IDs que nao existem no treinamento — resultado sera sempre proximo de chance, nao porque nao ha leakage, mas porque a tarefa e impossivel por design.

O `leakage_s2a` (speaker features vazam accent?) usa split **stratified** (NAO speaker-disjoint), onde os mesmos speakers aparecem em train e test. O probe pode memorizar a associacao speaker->accent, inflando o resultado.

**Os splits estao trocados.** A correcao e:
- `leakage_a2s`: usar split stratified (mesmos speakers em train/test)
- `leakage_s2a`: usar split speaker-disjoint (speakers diferentes)

**Arquivo:** `trainer.py:80-86` no companion repo
**Esforco:** 1-2 horas (trocar indices + atualizar testes)

### ACHADO 2 (CRITICO): Modelo errado — notebook usa 1.7B, projeto define 0.6B

**Quem encontrou:** Ana K. Silva (confirmado por Rafael Monteiro)
**Severidade:** Invalida transferibilidade dos resultados para Stage 2

Toda a documentacao (CLAUDE.md, STAGE_1.md, TECHNICAL_VALIDATION_PROTOCOL.md, RESEARCH_ROADMAP.md — 15+ arquivos) declara backbone **Qwen3-TTS 0.6B**. O notebook (cell 19) extrai features de `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`.

Implicacoes:
- O 1.7B tem ~28-36 camadas vs 20 do 0.6B. Probing em camadas 4 e 8 corresponde a posicoes relativas diferentes.
- O 1.7B tem hidden_size 2048 vs 1024 do 0.6B. Embeddings incompativeis.
- O 1.7B-CustomVoice NAO tem speaker_encoder built-in.
- O 1.7B pode nao caber em 24GB para treinamento LoRA (Stage 2).

**Decisao necessaria:** Qual e o modelo correto? Se 0.6B, corrigir notebook. Se 1.7B, atualizar TODOS os 15+ documentos e reavaliar VRAM para LoRA.

### ACHADO 3 (CRITICO): Zero seeds em todo o notebook

**Quem encontrou:** Lucas Andrade (confirmado por Ana K. Silva)
**Severidade:** Hard fail automatico — resultados nao reproduziveis

Nenhuma ocorrencia de `random.seed`, `np.random.seed`, `torch.manual_seed` ou `torch.cuda.manual_seed_all` no notebook. O companion repo tem `seed: 1337` no config YAML, mas:
- O notebook nao configura seeds antes de operacoes de preprocessing
- A propagacao de seed para o companion repo nao e verificavel
- Duas execucoes podem dar resultados diferentes

**Arquivo:** Todo o notebook `stage1_5_coraa_mupe.ipynb`
**Esforco:** 30 min (adicionar bloco de seeds) + auditoria do companion repo

### ACHADO 4 (BLOQUEANTE): Manifest descarta gender e duration

**Quem encontrou:** Mariana Alves
**Severidade:** Impossibilita toda analise de confounds

O `ManifestEntry` e `build_manifest_from_coraa()` descartam os campos `speaker_gender`, `duration`, `age` do CORAA-MUPE. O manifest final contem apenas: `utt_id, path, speaker, accent, text_id, source`.

Sem `gender`: impossivel verificar confound accent x gender (chi-squared)
Sem `duration`: impossivel verificar confound accent x duration (Kruskal-Wallis)

**Arquivo:** `dataset_builder.py` linhas 313-320, `manifest.py` classe `ManifestEntry`
**Esforco:** 2-3 horas (adicionar campos + atualizar testes)

### ACHADO 5 (BLOQUEANTE): Companion repo nao auditavel

**Quem encontrou:** Ana K. Silva
**Severidade:** Todo o codigo critico (splits, probes, decisao) esta fora do alcance de auditoria

O notebook clona `https://github.com/paulohenriquevn/accent-speaker-disentanglement.git` com comentario `# TODO: update`. O codigo de:
- Splits (speaker-disjoint ou nao?)
- Probes (realmente linear?)
- Decisao GO/NOGO (calculo correto?)
- Propagacao de seeds (determinismo?)

...nao pode ser verificado. Nenhum auditor pode validar resultados sem acesso ao codigo.

**Esforco:** Publicar repo OU incorporar modulos criticos no codebase principal

---

## 2. Tabela Completa de Contradicoes entre Documentos

| # | Documento A | Documento B | Contradicao |
|---|-------------|-------------|-------------|
| C1 | Todos (15+ docs): "Qwen3-TTS 0.6B" | Notebook cell 19: "1.7B-CustomVoice" | **Modelo backbone** |
| C2 | STAGE_1.md: "F1-macro >= 0.70" | Notebook: "F1 >= 0.55 para GO" | **Threshold accent** |
| C3 | STAGE_1.md: "queda <= 5%" | TECHNICAL_VALIDATION_PROTOCOL.md: "< 10%" | **Threshold speaker sim** |
| C4 | Protocol: "balanced accuracy" | Notebook: "F1-macro" | **Metrica primaria** |
| C5 | Protocol: "chance + 5 pp" | Notebook: "chance + 7 pp GO / 12 pp COND" | **Threshold leakage** |
| C6 | Protocol: probes AMBAS direcoes | Notebook decisao: so leakage_a2s | **Direcionalidade leakage** |
| C7 | CLAUDE.md: "Stage 1.5 in progress" | Codebase: zero resultados | **Status do Stage 1.5** |

---

## 3. Thresholds Consolidados (Proposta Felipe — Resolver Contradicoes)

| Metrica | GO | GO_CONDITIONAL | NOGO/FAIL | Notas |
|---------|----|----------------|-----------|-------|
| Accent F1-macro (probe linear) | >= 0.55 | >= 0.45 | < 0.40 (all backbone+SSL) | 0.70 e para Stage 2/3 com classificador treinado |
| Leakage A->speaker | <= chance + 7 pp | <= chance + 12 pp | > chance + 12 pp | 5 pp e para Stage 3 (audio gerado) |
| Leakage S->accent | <= chance + 7 pp | <= chance + 12 pp | > chance + 12 pp | Simetrico ao A->speaker |
| Text drop | <= 10 pp | <= 10 pp | > 10 pp | Consistente em todas as fontes |
| Speaker sim queda | < 10% (piloto Stage 2) | < 15% | >= 15% | 5% e target final Stage 3 |

**Deve ser aprovada pela equipe e atualizada em TODOS os documentos.**

---

## 4. Checklist Consolidado: O Que Funciona vs O Que Nao Funciona

### O que funciona (base solida)

| Item | Status | Evidencia |
|------|--------|-----------|
| Pipeline end-to-end (manifest -> features -> probes -> decisao) | OK | Notebook executa linearmente |
| Accent probe (F1-macro, speaker-disjoint) | OK | `SpeakerDisjointSplitter` via `GroupShuffleSplit` |
| Text-disjoint evaluation | OK | `GroupShuffleSplit` por `text_id` |
| `class_weight="balanced"` no LogisticRegression | OK | Mitiga desbalanceamento |
| Extracao de 4 tipos de features (acoustic, ECAPA, SSL, backbone) | OK | Cells 16-19 |
| WavLM com 5 camadas (0, 6, 12, 18, 24) | OK | Boa cobertura |
| Config YAML centralizado | OK | `stage1_5.yaml` |
| Criterios de decisao GO/COND/NOGO implementados | OK | `_decide()` em `run_all.py` |

### O que NAO funciona (deve ser corrigido)

| Item | Severidade | Persona |
|------|-----------|---------|
| Leakage probes com splits invertidos | CRITICO | Felipe |
| Modelo 1.7B vs 0.6B | CRITICO | Ana + Rafael |
| Zero seeds no notebook | CRITICO | Lucas |
| Manifest sem gender/duration | BLOQUEANTE | Mariana |
| Companion repo nao auditavel | BLOQUEANTE | Ana |
| Sem CI 95% para nenhuma metrica | BLOQUEANTE | Felipe |
| Sem confusion matrix | BLOQUEANTE | Felipe |
| Dependencias nao pinadas (`-U` no pip) | BLOQUEANTE | Lucas |
| Commit do companion repo nao pinado | BLOQUEANTE | Lucas |
| Sem hash SHA-256 do manifest/dataset | BLOQUEANTE | Lucas + Mariana |
| Sem assertions explicitos de speaker-disjointness | BLOQUEANTE | Mariana |
| Splits nao persistidos (efemeros em memoria) | BLOQUEANTE | Mariana |
| Sem analise de confound (accent x gender, accent x duration) | BLOQUEANTE | Mariana |
| Criterios numericos de confound nao definidos | BLOQUEANTE | Mariana |
| Cobertura de camadas do backbone insuficiente (2 de 20) | ALTA | Rafael |
| Pooling temporal nao documentado | ALTA | Rafael |
| Sem baseline intra/inter speaker similarity ECAPA | ALTA | Felipe |
| Threshold NOGO hardcoded (nao no YAML) | MEDIA | Felipe |
| ECAPA do notebook (192-dim) difere do built-in Qwen3 (1024-dim) | MEDIA | Rafael |
| Logging via print() em vez de logging estruturado | MEDIA | Lucas |

---

## 5. Plano de Correcao Priorizado

### Fase 0: Decisoes (antes de tocar codigo)

| # | Decisao | Quem Decide | Impacto |
|---|---------|-------------|---------|
| D1 | **Qual modelo backbone?** 0.6B ou 1.7B? | Rafael + Paulo | Define TUDO: VRAM, camadas, dimensoes, documentacao |
| D2 | **Thresholds unificados** (tabela da secao 3 acima) | Felipe + Rafael | Elimina 4 contradicoes de uma vez |
| D3 | **Criterios de confound** (p-value, Cramer's V, protocolo de mitigacao) | Mariana + Felipe | Define quando confound e "problema" |

### Fase 1: Correcoes CRITICAS (sem estas, resultado e invalido)

| # | Correcao | Responsavel | Esforco | Depende de |
|---|----------|-------------|---------|------------|
| F1 | Corrigir splits dos leakage probes (trocar A2S e S2A) | Felipe | 1-2h | — |
| F2 | Corrigir modelo no notebook (0.6B ou atualizar docs para 1.7B) | Rafael | 1h ou 4h+ | D1 |
| F3 | Adicionar bloco de seeds no notebook + verificar propagacao | Lucas | 2-4h | — |

### Fase 2: Correcoes BLOQUEANTES (sem estas, resultado nao e auditavel)

| # | Correcao | Responsavel | Esforco | Depende de |
|---|----------|-------------|---------|------------|
| F4 | Adicionar gender/duration ao ManifestEntry | Mariana | 2-3h | — |
| F5 | Implementar analise de confounds (chi2, Kruskal-Wallis) | Mariana | 4-6h | F4 |
| F6 | Adicionar CI 95% (bootstrap ou 3 seeds) | Felipe | 3-4h | F1 |
| F7 | Adicionar confusion matrix ao pipeline | Felipe | 1h | — |
| F8 | Pinar versoes de dependencias | Lucas | 1-2h | — |
| F9 | Pinar commit do companion repo | Lucas | 30min | — |
| F10 | Gerar hashes SHA-256 (manifest, config, splits) | Lucas | 1-2h | — |
| F11 | Persistir splits em arquivo + assertions de disjointness | Mariana | 3-4h | — |
| F12 | Publicar/disponibilizar companion repo para auditoria | Lucas | 1h | — |

### Fase 3: Melhorias ALTAS (antes de interpretar resultados)

| # | Correcao | Responsavel | Esforco | Depende de |
|---|----------|-------------|---------|------------|
| F13 | Expandir camadas do backbone (0, 4, 8, 12, 16, 19 para 0.6B) | Rafael | 1-2h | D1 |
| F14 | Documentar pooling temporal | Rafael | 1h | — |
| F15 | Calcular baseline intra/inter speaker similarity ECAPA | Felipe | 2-3h | — |
| F16 | Reportar balanced accuracy em paralelo com F1-macro | Felipe | 30min | — |
| F17 | Atualizar TODOS os documentos com thresholds unificados | Felipe | 2-3h | D2 |

### Fase 4: Melhorias MEDIAS (robustez)

| # | Correcao | Responsavel | Esforco |
|---|----------|-------------|---------|
| F18 | Variar C do LogisticRegression (0.01, 0.1, 1.0, 10.0) | Felipe | 1h |
| F19 | Monitoramento de VRAM durante extracao | Lucas | 30min |
| F20 | Nao modificar config YAML in-place (escrever novo arquivo) | Lucas | 30min |
| F21 | Loudness normalization (pyloudnorm, -23 LUFS) | Mariana | 2-3h |

---

## 6. Estimativa Total de Esforco

| Fase | Esforco | Caminho Critico |
|------|---------|-----------------|
| Fase 0: Decisoes | 1-2h (reuniao) | Decisao D1 (modelo) desbloqueia tudo |
| Fase 1: Criticas | 4-8h | F1 (leakage) + F2 (modelo) + F3 (seeds) em paralelo |
| Fase 2: Bloqueantes | 16-22h | F4->F5 (confounds) e caminho critico (~8h sequenciais) |
| Fase 3: Altas | 6-10h | Paralelizavel com Fase 2 |
| Fase 4: Medias | 4-5h | Apos Fase 2 |
| **Total** | **~30-45 horas (~4-6 dias)** | |

---

## 7. Veto Checks Simulados (Ana K. Silva)

| Check | Pergunta | Resultado | Evidencia |
|-------|----------|-----------|-----------|
| 1 | Arquivo de split com speakers + asserts? | **BLOCKED** | Nao existe no codebase |
| 2 | Tabela de leakage com chance e delta(pp)? | **BLOCKED** | Nao existe no codebase |
| 3 | Baseline intra/inter ECAPA com CI? | **BLOCKED** | Nao existe no codebase |

**Resultado: 3/3 BLOCKED. Stage 1.5 nao e auditavel.**

---

## 8. Mapeamento de Arquitetura Qwen3-TTS (Rafael)

### Componentes Principais

```
Qwen3TTSForConditionalGeneration
  ├── talker (Qwen3TTSTalkerForConditionalGeneration)
  │     ├── model (Qwen3TTSTalkerModel)
  │     │     ├── codec_embedding (nn.Embedding)
  │     │     ├── text_embedding (nn.Embedding)
  │     │     ├── layers[0..N-1] (Qwen3TTSTalkerDecoderLayer)
  │     │     │     ├── self_attn (q_proj, k_proj, v_proj, o_proj)  ← LoRA candidates
  │     │     │     └── mlp (gate_proj, up_proj, down_proj)         ← LoRA candidates
  │     │     └── norm (RMSNorm)
  │     ├── text_projection (ResizeMLP)
  │     ├── codec_head (nn.Linear)
  │     └── code_predictor (MTP module)
  └── speaker_encoder (ECAPA-TDNN, 1024-dim) [somente modelo "base"]
```

### Dimensoes por Variante

| Componente | 0.6B | 1.7B |
|---|---|---|
| Talker hidden_size | 1024 | 2048 |
| Talker num_hidden_layers | 20 | ~28-36 |
| Speaker encoder enc_dim | 1024 | 1024 |

### Candidatos Primarios para LoRA

| Modulo | Path | Justificativa |
|---|---|---|
| q_proj, v_proj | `talker.model.layers.*.self_attn.{q,v}_proj` | Padrao LoRA classico (Hu et al. 2022) |
| o_proj | `talker.model.layers.*.self_attn.o_proj` | Modula mistura de informacao |
| gate_proj, up_proj | `talker.model.layers.*.mlp.{gate,up}_proj` | Reweighting de features |

### Injecao de Speaker Embedding

Speaker embedding e um token na sequencia de input (NAO cross-attention):
```
[think_tokens] [speaker_embed(1,1,1024)] [codec_pad, codec_bos] [text + codec tokens...]
```

Implicacao: speaker identity permeia TODAS as camadas via attention. LoRA precisa modular accent SEM perturbar representacoes de speaker.

### Camadas Recomendadas para Probing

Para o 0.6B (20 layers): `[0, 4, 8, 12, 16, 19]`
Para o 1.7B: escalar proporcionalmente

---

## 9. Condicoes de Veto

| Persona | Veta execucao se: |
|---------|-------------------|
| **Felipe** | Leakage probes nao corrigidos, sem CI 95%, sem confusion matrix |
| **Lucas** | Seeds nao configurados, deps nao pinadas, sem hashes |
| **Mariana** | Manifest sem gender/duration, sem analise de confounds |
| **Rafael** | Modelo 0.6B/1.7B nao resolvido, cobertura de camadas < 6 pontos |
| **Ana** | Companion repo nao auditavel, contradicoes nao resolvidas |

---

## 10. Proximos Passos Imediatos

1. **DECISAO D1:** Qual modelo backbone? (Paulo + Rafael — hoje)
2. **CORRECAO F1:** Trocar splits dos leakage probes (Felipe — 1-2h)
3. **CORRECAO F3:** Seeds no notebook (Lucas — 2-4h)
4. **CORRECAO F4:** Gender/duration no manifest (Mariana — 2-3h)
5. **CORRECAO F8+F9:** Pinar deps + commit (Lucas — 1.5h)

Apos Fase 0+1+2, re-auditar com Ana antes de executar.

---

*Ata gerada a partir dos relatorios individuais de 5 personas.*
*Proxima reuniao: apos implementacao das Fases 0-2 (estimativa: 4-6 dias).*
