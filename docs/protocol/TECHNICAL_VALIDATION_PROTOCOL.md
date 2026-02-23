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
- Usar **macro-regiões** (ex.: Norte/Nordeste, Sudeste, Sul);
- `birth_state` é tratado como **proxy**, não ground truth;
- Não usar mais de 3 classes no piloto.

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

### 9.4 Análise de Confounds (obrigatória antes de treinamento)

Verificar independência entre variável alvo (accent) e variáveis espúrias:

| Confound | Teste | Critério |
|----------|-------|----------|
| Accent x Gender | Chi-quadrado + Cramer's V | V < 0.3: aceitável (documentar). V >= 0.3: mitigação obrigatória ou BLOQUEANTE |
| Accent x Duration | Kruskal-Wallis | p < 0.05 E diferença prática > 1s: documentar como limitação |

Se confound detectado e não mitigado, qualquer resultado positivo é questionável.

### 9.5 Baseline Speaker Similarity (obrigatório antes de adaptação)

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
- Todos os hard fail checks passaram (ver `KB_HARD_FAIL_RULES.md`).

### ADJUST
- Accent balanced accuracy em zona GO_CONDITIONAL (>= 0.50, < 0.55);
- OU leakage em zona condicional (> chance+5pp, <= chance+12pp);
- OU identidade com queda entre 10-15%;
- Sinal presente, ajustes identificáveis.

### FAIL
- Accent balanced accuracy < 0.50 (abaixo de GO_CONDITIONAL);
- OU leakage > chance + 12 p.p.;
- OU identidade colapsa (queda >= 15%);
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
