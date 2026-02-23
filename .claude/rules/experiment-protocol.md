# Protocolo Experimental

Todo experimento segue um protocolo rigoroso. Resultado sem protocolo é anedota, não evidência.

---

## Hipótese Obrigatória

- **ANTES** de rodar qualquer experimento, formular hipótese explícita e documentada.
- Formato: "Se [intervenção], então [resultado esperado], medido por [métrica], porque [justificativa teórica]."
- Exemplo: "Se aplicarmos LoRA rank 16 com dados de sotaque paulistano, então o classificador de sotaque deve atingir >70% balanced accuracy nos áudios gerados, medido por um classificador treinado em áudio real, porque o LoRA captura padrões prosódicos regionais no espaço de embeddings A."
- Sem hipótese = sem experimento. "Vamos ver o que acontece" não é pesquisa.

## Variáveis Controladas

- Documentar explicitamente para cada experimento:
  - **Variável independente:** o que estamos mudando (ex: rank do LoRA, sotaque alvo).
  - **Variáveis dependentes:** o que estamos medindo (ex: balanced accuracy, similaridade de speaker).
  - **Variáveis controladas:** o que mantemos fixo (ex: backbone, seed, dados de treino, hiperparâmetros).
- Mudar apenas UMA variável por vez entre experimentos comparativos.

## Ablation Studies

- Para cada componente significativo do sistema, realizar ablation: remover/substituir e medir impacto.
- Ablation mínimo para o TCC:
  - Com LoRA vs sem LoRA (baseline).
  - LoRA rank X vs rank Y.
  - Com fine-tuning de embeddings A vs frozen.
- Documentar cada ablation no mesmo formato de experimento (hipótese, resultado, análise).

## Significância Estatística

- Resultados com intervalo de confiança (CI 95%) quando possível.
- Para métricas de classificação: reportar média +/- desvio padrão de múltiplas seeds (mínimo 3 runs).
- Se diferença entre métodos for menor que o CI, **NÃO** afirmar que um é melhor que outro.
- Bootstrap para CI quando distribuição é desconhecida.

## Resultados Negativos

- Resultados negativos (hipótese refutada) são resultados válidos e DEVEM ser documentados.
- Formato: "Hipótese X foi testada. Resultado: Y (abaixo do threshold Z). Análise: [por que não funcionou]. Implicação: [o que isso significa para próximos passos]."
- Nunca omitir resultados negativos. Eles informam o espaço de soluções.

## Referência ao Protocolo de Validação

- Todo experimento deve referenciar os thresholds definidos em `TECHNICAL_VALIDATION_PROTOCOL.md`.
- Decisão de PASS/ADJUST/FAIL segue critérios objetivos do protocolo, não impressões subjetivas.
- Gate de validação é chamado ao final de cada Stage, nunca pulado.

## Registro de Experimentos

```yaml
# Template mínimo por experimento
experiment:
  name: "exp_001_lora_rank16_paulistano"
  hypothesis: "..."
  independent_variable: "LoRA rank (16)"
  controlled_variables:
    backbone: "Qwen3-TTS-1.7B-CustomVoice"
    seed: 42
    data: "CORAA-MUPE v1.0 split_v2"
  metrics:
    - balanced_accuracy
    - speaker_similarity_cosine
    - leakage_probe_accuracy
  result: "pending | confirmed | refuted | inconclusive"
  notes: ""
```

## Anti-Patterns

- Rodar 10 configs e reportar só a melhor ("cherry picking").
- Comparar resultados com seeds diferentes sem reportar variância.
- Afirmar que método A é melhor que B com diferença de 0.5% sem CI.
- Mudar múltiplas variáveis ao mesmo tempo e atribuir melhoria a uma delas.
- Omitir experimentos que "não funcionaram".
- Não ter baseline documentado antes de comparar.
