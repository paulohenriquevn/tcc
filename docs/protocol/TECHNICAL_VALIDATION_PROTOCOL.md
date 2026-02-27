# TECHNICAL_VALIDATION_PROTOCOL
## Validação Técnica de Controle de Sotaque com Preservação de Identidade Vocal
Projeto: Controle Explícito de Sotaque Regional em pt-BR

---

## 1. Objetivo

Este protocolo define o **mínimo necessário** para validar a viabilidade técnica do pipeline de TTS com:
- controle explícito de sotaque regional;
- preservação de identidade vocal;
- ausência de atalhos evidentes (leakage).

O protocolo **não avalia impacto social**, **não avalia percepção humana** e **não busca sotaque perfeito**.
Seu único objetivo é responder: **a técnica funciona?**

---

## 2. Escopo da Validação

Inclui:
- backbone TTS;
- adaptação via LoRA / adapters;
- dataset CORAA-MUPE (uso técnico);
- métricas objetivas automáticas.

Exclui:
- avaliação humana;
- análise sociolinguística profunda;
- generalização comercial;
- publicação científica.

---

## 3. Definições Técnicas

### 3.1 Modelo

- Backbone: **Qwen3-TTS 1.7B-CustomVoice**
- Licença: Apache-2.0
- Treinamento: apenas LoRA (sem fine-tuning completo)
- Embeddings:
  - `S`: identidade vocal (fixa)
  - `A`: sotaque (variável)

Geração formal:
Audio = TTS(texto, S, A)


---

## 4. Dataset e Higienização (Obrigatório)

### 4.1 Fonte
- Dataset: **CORAA-MUPE**

### 4.2 Filtros obrigatórios
- Usar **apenas um tipo de speaker** (preferencialmente entrevistado);
- Remover entrevistadores;
- Remover áudios com qualidade baixa (se metadado disponível);
- Normalizar sampling rate e loudness.

### 4.3 Rótulo de sotaque
- Usar **macro-regiões IBGE**: N (Norte), NE (Nordeste), CO (Centro-Oeste), SE (Sudeste), S (Sul);
- `birth_state` é tratado como **proxy**, não ground truth;
- **Decisão data-driven (census 2026-02-27):** censos completos de ambas as fontes (streaming, sem download):
  - **CORAA-MUPE-ASR** (317.743 rows; filtros: speaker_type=R, duração 3-15s, gênero válido, birth_state válido):
    - N: 30 speakers / 12.942 utt — **PASS**
    - NE: 39 speakers / 19.333 utt — **PASS**
    - CO: 3 speakers / 1.406 utt — **insuficiente isolado**
    - SE: 193 speakers / 104.652 utt — **PASS**
    - S: 7 speakers / 3.740 utt — **PASS**
  - **Common Voice PT** (fsicoli/common_voice_17_0, 162.111 rows; filtros: accent mapeável, gênero válido):
    - N: 3 speakers / 121 utt
    - NE: 20 speakers / 592 utt
    - CO: 4 speakers / 18.247 utt
    - SE: 48 speakers / 7.396 utt
    - S: 16 speakers / 14.020 utt
  - **Combinado (CORAA-MUPE + Common Voice):**
    - N: ~33 speakers — **PASS**
    - NE: ~59 speakers — **PASS**
    - CO: ~7 speakers — **PASS** (marginal, CI mais largo esperado)
    - SE: ~241 speakers — **PASS**
    - S: ~23 speakers — **PASS**
- **Estratégia multi-source:** Common Voice contribui speakers para todas as 5 regiões, não apenas CO. Isso dilui a correlação accent×source — o classificador não pode aprender "qual dataset" em vez de "qual sotaque". Cross-source evaluation obrigatória (§ confounds);
- **Configuração adotada:** 5 macro-regiões (N, NE, CO, SE, S) com `min_speakers_per_region=5`;
- **Caveats CO:** menor número de speakers (7 combinados), gender imbalance severo (~99% M no CV), CI mais largo que outras regiões. Documentado como região com menor robustez estatística;
- Confound accent×source monitorado via chi-quadrado + Cramer's V (threshold blocker: V ≥ 0.3);
- Decisão de manter ou reduzir classes é tomada após análise de distribuição no manifest combinado.

---

## 5. Split de Dados (Anti-Atalho)

- Split **speaker-disjoint** obrigatório;
- Nenhum speaker aparece em mais de um split;
- Splits:
  - treino (LoRA)
  - validação
  - teste técnico (somente speakers inéditos)

Falha neste item invalida toda a validação.

---

## 6. Baseline Inicial (Antes de Treinar)

Antes de qualquer adaptação, medir:

- saída do backbone sem LoRA;
- classificador de sotaque (esperado: fraco);
- ECAPA similarity intra-speaker;
- WER via Whisper-large-v3 no áudio gerado pelo backbone sem LoRA (referência para seção 9.4);
- métricas registradas como **baseline zero**.

---

## 7. Treinamento Piloto (LoRA)

- Uma LoRA por sotaque (macro-região);
- Dataset reduzido (ex.: 6–8 speakers por classe);
- ≥ 10 minutos por speaker já é suficiente;
- Treinamento curto (horas, não dias);
- Checkpoints intermediários obrigatórios.

Objetivo: **detectar sinal**, não atingir SOTA.

---

## 8. Conjunto Controlado de Avaliação

Criar um conjunto fixo de frases:
- 50–100 frases curtas;
- semanticamente neutras;
- mesmo texto usado em todas as condições.

Para cada frase:
- mesmo `S`;
- variar apenas `A`.

Este conjunto é usado em TODAS as métricas abaixo.

---

## 9. Métricas Técnicas Obrigatórias

### 9.1 Controlabilidade de sotaque

- Métrica primária: **balanced accuracy** (nunca accuracy simples em dados desbalanceados);
- F1-macro reportada como métrica secundária;
- Confusion matrix normalizada obrigatória.

#### Stage 1.5 (probes lineares em áudio real)

| Decisão | Critério (balanced accuracy, probe linear, test set) |
|---------|------------------------------------------------------|
| **GO** | >= 0.55 |
| **GO_CONDITIONAL** | >= 0.50 |
| **FAIL** | < 0.50 |

#### Stage 2-3 (classificador externo em áudio gerado)

- Classificador externo de sotaque treinado em áudio real;
- Avaliado nas amostras geradas;
- Comparar: sem LoRA vs com LoRA;
- Critério mínimo: desempenho acima do baseline com CI 95% não sobreposto;
- Matriz de confusão não degenerada (nenhuma classe com recall < 0.20).

---

### 9.2 Preservação de identidade vocal

- Modelo: ECAPA ou x-vector;
- Medir similaridade:
  - mesmo speaker, troca de `A`;
- Calcular média e desvio.

Critério piloto:
- queda moderada aceitável (< 10%);
- colapso grande indica falha técnica.

---

### 9.3 Leakage (teste anti-atalho)

Treinar probes **lineares** (logistic regression) para prever:
- speaker a partir de `A` (ou id da LoRA) — testa se A vaza identidade;
- sotaque a partir de `S` — testa se S vaza sotaque.

**Ambas** as direções são obrigatórias. Reportar balanced accuracy com CI 95%.

| Decisão | Critério (balanced accuracy vs chance level) |
|---------|----------------------------------------------|
| **GO** | <= chance + 5 p.p. |
| **GO_CONDITIONAL** | <= chance + 12 p.p. |
| **FAIL** | > chance + 12 p.p. |

Chance level: `1/N_classes` (1/N_speakers para A→speaker, 1/N_accents para S→accent).

---

### 9.4 Qualidade de Fala (sanity check — Stage 2-3)

Métricas automáticas de qualidade aplicadas ao áudio **gerado** pelo pipeline com LoRA. Não se aplicam ao Stage 1.5 (áudio real).

#### UTMOS (preditor neural de MOS)

- Modelo: **UTMOS** (SpeechMOS, treinado no VoiceMOS Challenge);
- Medir score médio das amostras geradas no conjunto controlado de avaliação (seção 8);
- Reportar: média, desvio-padrão, CI 95%.

| Decisão | Critério (UTMOS médio) |
|---------|------------------------|
| **GO** | >= 3.0 |
| **ADJUST** | >= 2.5 e < 3.0 |
| **FAIL** | < 2.5 |

#### WER via Whisper (inteligibilidade)

- Modelo: **Whisper-large-v3** (OpenAI);
- Medir WER entre texto de entrada e transcrição do áudio gerado;
- Comparar: WER do áudio gerado vs WER do áudio baseline (backbone sem LoRA);
- Reportar: WER médio, CI 95%.

| Decisão | Critério |
|---------|----------|
| **GO** | WER gerado <= WER baseline + 10 p.p. |
| **ADJUST** | WER gerado > WER baseline + 10 p.p. e <= WER baseline + 20 p.p. |
| **FAIL** | WER gerado > 50% (áudio ininteligível) |

Nota: UTMOS e WER são **sanity checks** — garantem que o LoRA não degradou a qualidade/inteligibilidade do áudio. Não são métricas de avaliação de sotaque.

---

### 9.5 Análise de Confounds (obrigatória antes de treinamento)

Verificar independência entre variável alvo (accent) e variáveis espúrias:

| Confound | Teste | Critério |
|----------|-------|----------|
| Accent x Gender | Chi-quadrado + Cramer's V | V < 0.3: aceitável (documentar). V >= 0.3: mitigação obrigatória ou BLOQUEANTE |
| Accent x Duration | Kruskal-Wallis | p < 0.05 E diferença prática > 1s: documentar como limitação |
| Accent x Recording conditions | Kruskal-Wallis (SNR estimado) | p < 0.05 E diferença prática > 5dB: documentar como limitação. O modelo pode aprender canal/ruído em vez de sotaque |

Se confound detectado e não mitigado, qualquer resultado positivo é questionável.

### 9.6 Baseline Speaker Similarity (obrigatório antes de adaptação)

- Modelo: **ECAPA-TDNN** (SpeechBrain pré-treinado, embedding 192-dim);
- Medir similaridade cosseno **intra-speaker** (mesmo speaker, utterances diferentes) no áudio real;
- Medir similaridade cosseno **inter-speaker** (speakers diferentes) no áudio real;
- Reportar: média, desvio-padrão, CI 95%;
- Este baseline é referência para Stage 2 (preservação de identidade com LoRA).

---

## 10. Critérios de Decisão

### PASS (técnica validada)
- Accent balanced accuracy >= threshold GO com CI 95%;
- Identidade preservada (queda < 10% na similaridade ECAPA);
- Leakage <= chance + 5 p.p. em ambas as direções;
- Confusion matrix não degenerada;
- UTMOS >= 3.0 (Stage 2-3);
- WER gerado <= WER baseline + 10 p.p. (Stage 2-3);
- Todos os hard fail checks passaram (ver `KB_HARD_FAIL_RULES.md`).

### ADJUST
- Accent balanced accuracy em zona GO_CONDITIONAL (>= 0.50, < 0.55);
- OU leakage em zona condicional (> chance+5pp, <= chance+12pp);
- OU identidade com queda entre 10-15%;
- OU UTMOS entre 2.5 e 3.0 (Stage 2-3);
- OU WER gerado > WER baseline + 10 p.p. mas <= WER baseline + 20 p.p. (Stage 2-3);
- Sinal presente, ajustes identificáveis.

### FAIL
- Accent balanced accuracy < 0.50 (abaixo de GO_CONDITIONAL);
- OU leakage > chance + 12 p.p.;
- OU identidade colapsa (queda >= 15%);
- OU UTMOS < 2.5 (Stage 2-3);
- OU WER > 50% — áudio ininteligível (Stage 2-3);
- OU falha metodológica (ver `KB_HARD_FAIL_RULES.md`).

---

## 11. Artefatos Obrigatórios

Para considerar o protocolo executado:
- notebook(s) com execução reproduzível;
- logs de métricas;
- scripts de split e filtros;
- README técnico com resultados.

---

## 12. Conclusão do Protocolo

Este protocolo é considerado **completo** quando:
- todos os testes foram executados;
- métricas registradas;
- decisão PASS / ADJUST / FAIL tomada explicitamente.

Somente após **PASS**, o projeto avança para:
- experimentação humana;
- refinamento linguístico;
- avaliação de impacto percebido.

---
