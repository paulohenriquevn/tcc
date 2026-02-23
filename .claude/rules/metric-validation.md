# Validação de Métricas

Métricas são o contrato entre hipótese e conclusão. Métrica errada ou mal aplicada invalida a conclusão.

---

## Definição Formal de Cada Métrica

Toda métrica usada DEVE ter definição formal documentada antes do primeiro uso:

| Métrica | Definição | Range | Melhor | Quando Usar |
|---------|-----------|-------|--------|-------------|
| **Balanced Accuracy** | Média das acurácias por classe | [0, 1] | 1.0 | Classificação com classes desbalanceadas |
| **Speaker Similarity (cosine)** | Similaridade cosseno entre embeddings ECAPA-TDNN do áudio gerado vs referência | [-1, 1] | 1.0 | Preservação de identidade vocal |
| **Accent Classification Accuracy** | Acurácia de classificador treinado em áudio real, avaliado em áudio gerado | [0, 1] | 1.0 | Transferência de sotaque |
| **Leakage Probe (A→speaker)** | Acurácia de probe linear que prediz speaker a partir de embedding A | [0, 1] | chance level | Disentanglement A/S |
| **Leakage Probe (S→accent)** | Acurácia de probe linear que prediz sotaque a partir de embedding S | [0, 1] | chance level | Disentanglement S/A |
| **UTMOS** | Preditor neural de MOS (SpeechMOS, VoiceMOS Challenge). Sanity check de qualidade no áudio gerado | [1, 5] | 5.0 | Qualidade de síntese (Stage 2-3) |
| **WER (Whisper-large-v3)** | Word Error Rate entre texto de entrada e transcrição Whisper do áudio gerado | [0, ∞) | 0.0 | Inteligibilidade de síntese (Stage 2-3) |
| **MOS (Mean Opinion Score)** | Avaliação subjetiva de qualidade (se aplicável) | [1, 5] | 5.0 | Qualidade perceptual |

## Baselines Obrigatórios

- **ANTES** de qualquer adaptação, medir todas as métricas no modelo baseline (Qwen3-TTS sem LoRA).
- Baseline é o piso. Qualquer resultado deve ser comparado contra ele.
- Baselines incluem:
  - Modelo pré-treinado sem fine-tuning.
  - Random chance para classificação (1/N_classes).
  - Majority class baseline.
- Documentar baselines em tabela padronizada com CI.

## Balanced Accuracy (Métrica Primária para Classificação)

- SEMPRE usar balanced accuracy, NUNCA accuracy simples, quando classes são desbalanceadas.
- `sklearn.metrics.balanced_accuracy_score(y_true, y_pred)`.
- Reportar junto: confusion matrix normalizada por classe.

## Confusion Matrix

- Reportar confusion matrix para toda tarefa de classificação.
- Normalizar por linha (recall por classe) para visualização.
- Usar para identificar: quais sotaques são confundidos entre si? Há viés sistemático?

## Intervalos de Confiança

- Toda métrica principal reportada com CI 95%.
- Método: bootstrap (1000 samples) ou múltiplas seeds (mínimo 3).
- Formato de reporte: `metric = X.XX (CI 95%: [Y.YY, Z.ZZ])`.
- Se dois métodos têm CIs sobrepostos, NÃO afirmar superioridade.

## Null Hypothesis para Leakage Probes

- **Leakage probe de A→speaker:** H0 = "embedding A não contém informação de speaker identity".
  - Rejeitar H0 se probe accuracy > chance level + margem significativa.
  - Chance level = 1/N_speakers.
  - Se H0 rejeitada: embeddings A vazam identidade → disentanglement falhou.
- **Leakage probe de S→accent:** H0 = "embedding S não contém informação de sotaque".
  - Rejeitar H0 se probe accuracy > chance level + margem significativa.
  - Chance level = 1/N_accents.
  - Se H0 rejeitada: embeddings S vazam sotaque → disentanglement falhou.
- Probe deve ser linear (logistic regression) para testar informação linearmente separável.
- Treinar probe com seed fixo, split speaker-disjoint, reportar com CI.

## Anti-Patterns

- Reportar accuracy simples em dataset desbalanceado (80% classe majoritária = 80% "accuracy" sem aprender nada).
- Comparar métricas sem intervalo de confiança.
- Ignorar confusion matrix e reportar apenas número agregado.
- Usar probe não-linear (MLP) para leakage test (probe complexo demais sempre acha algo).
- Não ter baseline — "nosso modelo atinge 75%" sem dizer 75% comparado a quê.
- Declarar "disentanglement funciona" sem testar leakage em ambas as direções.
