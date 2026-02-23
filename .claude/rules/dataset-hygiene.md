# Higiene de Dataset e Anti-Confound

O dataset é a fundação do experimento. Contaminação no dataset invalida todo resultado construído sobre ele.

---

## Speaker-Disjoint Splits (Obrigatório)

- Splits de treino/validação/teste DEVEM ser speaker-disjoint: nenhum speaker aparece em mais de um split.
- Validar com assertion automático antes de qualquer treinamento:

```python
assert len(train_speakers & val_speakers) == 0, "Speaker leakage train→val"
assert len(train_speakers & test_speakers) == 0, "Speaker leakage train→test"
assert len(val_speakers & test_speakers) == 0, "Speaker leakage val→test"
```

- Se o dataset original (CORAA-MUPE) não fornece splits speaker-disjoint, criar e documentar o procedimento.
- Splits são artefatos versionados — uma vez definidos, não mudam dentro de um experimento.

## Validação de Metadata

- Verificar completude de metadata antes de usar: speaker_id, accent_label, gender, duration, sampling_rate.
- Rejeitar amostras com metadata ausente ou inconsistente — não preencher com defaults silenciosamente.
- Verificar que labels de sotaque são consistentes (mesmo speaker, mesmo sotaque em todas as amostras).
- Documentar distribuição de amostras por: sotaque, speaker, gênero, duração.

## Checks de Correlação e Confounds

- **ANTES** de treinar, verificar correlação entre variáveis:
  - Sotaque vs gênero (ex: todas amostras de sotaque X são masculinas?)
  - Sotaque vs duração média (ex: sotaque Y tem amostras mais longas?)
  - Sotaque vs condições de gravação (ex: sotaque Z é só microfone de estúdio?)
- Se correlação espúria for detectada, documentar e mitigar (subsampling balanceado, data augmentation).
- Nunca ignorar confounds — se não tem como mitigar, documentar como limitação do estudo.

## Versionamento de Dados

- Dataset processado deve ter hash (SHA-256) registrado no config do experimento.
- Qualquer alteração no preprocessing gera novo hash e novo nome de versão.
- Manter script de preprocessing completo e reproduzível — da raw data ao dataset final.
- Nunca editar dataset manualmente sem registro.

## Distribuição e Balanceamento

- Reportar distribuição de classes (sotaques) em cada split: contagem e percentual.
- Se desbalanceado, documentar estratégia de mitigação: weighted sampling, balanced batches, ou nada (justificar).
- Usar balanced accuracy como métrica primária quando classes são desbalanceadas.

## Anti-Patterns

- Dividir por amostras sem verificar speakers (data leakage garantido).
- Ignorar correlação sotaque-gênero e depois celebrar "o modelo aprendeu sotaque" (aprendeu gênero).
- Preencher metadata faltante com valores default sem investigar.
- Usar dataset diferente para treino e avaliação sem documentar.
- Mudar preprocessing no meio do experimento e comparar com resultados anteriores.
