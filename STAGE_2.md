# STAGE 2 — IMPLEMENTAÇÃO EXPERIMENTAL E VALIDAÇÃO
## Projeto: Controle Explícito de Sotaque Regional em pt-BR

---

## 1. Objetivo do Stage 2

Implementar o pipeline experimental definido no Stage 1, executar os experimentos técnicos e perceptivos, e avaliar empiricamente as hipóteses propostas.

Este estágio termina com o **Gate 2**, que decide continuidade, publicação ou pivot.

---

## 2. Preparação do Ambiente

- Congelar versão do backbone e LoRAs;
- Configurar ambiente Docker reproduzível;
- Versionar datasets, splits e manifests;
- Ativar logging de métricas e artefatos.

---

## 3. Implementação Técnica

### 3.1 Pipeline de Treinamento

- Treinar uma LoRA por sotaque regional;
- Manter embedding de identidade fixo;
- Aplicar regularização adversarial se necessário;
- Monitorar perda, leakage e qualidade perceptiva.

---

### 3.2 Pipeline de Inferência

- Gerar amostras controladas:
  - mesmo speaker;
  - sotaques distintos;
- Medir latência (RTF) e uso de VRAM;
- Armazenar áudios versionados.

---

## 4. Avaliação Técnica

- Executar classificador de sotaque;
- Medir similaridade de identidade (ECAPA/x-vector);
- Rodar probes de leakage;
- Comparar com baseline open-source.

Critérios:
- atingir thresholds definidos no Stage 1;
- documentar falhas e desvios.

---

## 5. Avaliação com Usuários

- Gerar conjunto A/B/C de estímulos;
- Conduzir experimento within-subjects;
- Coletar respostas (Likert + MOS);
- Analisar com modelo estatístico misto;
- Reportar efeito, IC e significância.

---

## 6. Análise de Resultados

- Verificar validade das hipóteses H1 e H2;
- Identificar trade-offs entre controle e naturalidade;
- Avaliar estabilidade da identidade vocal;
- Analisar diferenças regionais.

---

## 7. Gate 2 — Critério de Decisão

### GO
- Disentanglement comprovado;
- leakage abaixo do threshold;
- efeito perceptivo significativo;
- qualidade competitiva com baseline.

### ADJUST
- controle parcial;
- efeito fraco;
- necessidade de mais dados ou regularização.

### KILL
- entanglement persistente;
- identidade colapsa;
- ausência total de efeito perceptivo.

---

## 8. Saídas do Stage 2

- Código reprodutível;
- Datasets e splits documentados;
- Resultados técnicos e estatísticos;
- Relatório experimental;
- Base para publicação ou produto.

---
