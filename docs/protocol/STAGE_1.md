# STAGE 1 — IDEAÇÃO E VALIDAÇÃO DE VIABILIDADE
## Projeto: Controle Explícito de Sotaque Regional em pt-BR com Disentanglement Sotaque × Identidade Vocal

---

## 1. Objetivo do Stage 1

O Stage 1 tem como objetivo validar a **viabilidade científica, técnica e operacional** do projeto antes de qualquer implementação pesada.

Ao final deste estágio, deve estar claro que:
- o problema é cientificamente bem definido;
- existe um backbone adequado e permissivo;
- há um dataset minimamente viável;
- as métricas são executáveis;
- o experimento humano é mensurável;
- a infraestrutura suporta o plano;
- os riscos são conhecidos e mitigáveis.

Este estágio termina com a aprovação formal do **Gate 1**.

---

## 2. Problema Científico

Modelos modernos de Text-to-Speech (TTS) aprendem representações latentes onde sotaque regional, identidade vocal, prosódia e estilo estão parcialmente entangled.

Essa mistura impede:
- controle explícito de sotaque;
- preservação garantida da identidade vocal ao variar sotaque;
- avaliação causal do impacto perceptivo do sotaque.

---

## 3. Pergunta de Pesquisa

É possível controlar sotaque regional em português brasileiro utilizando um modelo open-source com licença permissiva, mantendo identidade vocal estável, e demonstrar ganho perceptivo estatisticamente significativo?

---

## 4. Hipóteses

### H1 — Disentanglement Técnico
É possível variar o condicionamento de sotaque mantendo a identidade vocal estável dentro de limites mensuráveis, sem degradação significativa de naturalidade.

### H2 — Impacto Percebido
Sotaque alinhado ao usuário aumenta afinidade, confiança e naturalidade percebida em relação a sotaque neutro ou desalinhado.

---

## 5. Backbone Arquitetural (Decisão Travada)

- Modelo: **Qwen3-TTS 1.7B-CustomVoice**
- Licença: **Apache-2.0**
- Tipo: Speech Foundation Model com representação por tokens discretos
- Estratégia de treino: **LoRA / adapters apenas**
- Motivo da escolha:
  - open weights;
  - licença permissiva;
  - compatível com GPU 24GB;
  - atual e competitivo em 2026.

Baseline comparativo open-source permissivo será usado apenas para comparação de qualidade.

---

## 6. Formulação Técnica

A geração é modelada como:

Audio = TTS(texto, S, A)


Onde:
- `S` = embedding fixo de identidade vocal;
- `A` = LoRA específica de sotaque regional.

---

## 7. Dataset — Especificação Mínima

- 3 macro-regiões brasileiras;
- ≥ 8 falantes por região;
- ≥ 20 minutos de áudio por falante;
- texto comparável entre regiões;
- split **speaker-disjoint** obrigatório;
- licença de uso validada;
- dataset manifest versionado com hashes.

---

## 8. Métricas Técnicas

### Controlabilidade de sotaque
- Métrica primária: **balanced accuracy** (nunca accuracy simples);
- F1-macro reportada como métrica secundária;
- Thresholds definidos em `TECHNICAL_VALIDATION_PROTOCOL.md` (mesma pasta, seção 9.1):
  - Stage 1.5 (probes lineares): GO >= 0.55, GO_CONDITIONAL >= 0.50, FAIL < 0.50;
  - Stage 2-3 (classificador externo em áudio gerado): acima do baseline com CI 95%.

### Preservação de identidade
- ECAPA-TDNN (SpeechBrain, 192-dim) similarity;
- queda máxima aceitável < 10% (piloto Stage 2);
- target final Stage 3: < 5%.

### Leakage
- probes **lineares** (logistic regression):
  - prever speaker a partir de `A`;
  - prever sotaque a partir de `S`;
- **ambas** as direções obrigatórias;
- thresholds em `TECHNICAL_VALIDATION_PROTOCOL.md` (mesma pasta, seção 9.3):
  - GO: <= chance + 5 p.p.;
  - GO_CONDITIONAL: <= chance + 12 p.p.;
  - FAIL: > chance + 12 p.p.

---

## 9. Experimento com Usuários (Planejamento)

- Condições:
  1. neutro;
  2. alinhado;
  3. desalinhado.
- Métricas:
  - MOS;
  - Similarity MOS;
  - Afinidade (Likert 1–7);
  - Confiança (Likert 1–7);
  - Clareza.
- Desenho: within-subjects com contrabalanço.
- N mínimo: 30 participantes.
- Estatística: modelo misto, α = 0.05, IC 95%.

---

## 10. Infraestrutura

- GPU: 1× 24GB VRAM (RTX 4090 ou equivalente);
- CPU: 16 cores;
- RAM: 64 GB;
- Disco: 2 TB NVMe;
- SO: Ubuntu 22.04 LTS;
- Docker com CUDA fixado;
- armazenamento persistente.

---

## 11. Riscos e Mitigações

- Entanglement persistente → regularização adversarial;
- dataset insuficiente → reduzir granularidade regional;
- sotaque superficial → análise fonético-proxy;
- ausência de efeito perceptivo → ajuste de contexto experimental.

---

## 12. Gate 1 — Critério de Aprovação

O Stage 1 é aprovado se:
- backbone estiver congelado;
- dataset mínimo definido;
- métricas executáveis com thresholds;
- experimento humano formalizado;
- infraestrutura validada;
- riscos documentados.

**Status do Gate 1: GO**

---
