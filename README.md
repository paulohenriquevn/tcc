# Controle Explicito de Sotaque Regional em pt-BR

**Trabalho de Conclusao de Curso (TCC)** investigando se LoRA adapters em um backbone TTS congelado conseguem controlar sotaque regional de forma explicita, preservando a identidade do falante, com embeddings disentangled S (speaker) e A (accent).

**Pergunta central:** `Audio = TTS(texto, S, A)` — e possivel variar A independentemente de S?

**Stack:** Qwen3-TTS 1.7B-CustomVoice (Apache-2.0) + LoRA + CORAA-MUPE + 1x GPU 24GB

**Status atual:** Stage 1 GO aprovado. Stage 1.5 (auditoria de separabilidade latente) em construcao.

---

## Mapa do Repositorio

| Diretorio | Conteudo |
|-----------|----------|
| [`docs/protocol/`](docs/protocol/) | Protocolo de validacao, especificacoes dos Stages 1 e 2, metodo Stage-Gate |
| [`docs/methodology/`](docs/methodology/) | Perfis das personas tecnicas do projeto |
| [`docs/meetings/`](docs/meetings/) | Atas de reuniao com decisoes e achados tecnicos |
| [`references/`](references/) | Papers de referencia (PDFs) e survey de datasets pt-BR (36KB) |
| [`notebooks/`](notebooks/) | Notebooks Jupyter para execucao no Google Colab |
| [`configs/`](configs/) | Configs YAML dos experimentos (single source of truth) |
| [`src/`](src/) | Codigo Python: dados, features, probes, metricas, analise de confounds |
| [`tests/`](tests/) | Testes unitarios |
| [`.claude/`](.claude/) | Infraestrutura de agentes de pesquisa (agents, rules, skills, roadmap) |

## Documentos-Chave

- **Protocolo de Validacao:** [`docs/protocol/TECHNICAL_VALIDATION_PROTOCOL.md`](docs/protocol/TECHNICAL_VALIDATION_PROTOCOL.md) — thresholds, metricas e criterios PASS/ADJUST/FAIL
- **Roadmap de Pesquisa:** [`.claude/RESEARCH_ROADMAP.md`](.claude/RESEARCH_ROADMAP.md) — plano de 3 stages com experimentos e Definition of Done
- **Survey de Datasets:** [`references/dataset-survey.md`](references/dataset-survey.md) — avaliacao de 10+ datasets de fala pt-BR
- **Changelog:** [`CHANGELOG.md`](CHANGELOG.md) — registro de todas as mudancas do projeto

## Metodologia

O projeto segue um modelo **Stage-Gate** com decisoes formais baseadas em metricas objetivas:

| Stage | Foco | Status |
|-------|------|--------|
| Stage 1 | Baseline: metricas zero-shot, ambiente, auditoria de dataset | **GO** |
| Stage 1.5 | Probing: separabilidade latente, confounds, probes lineares | Em construcao |
| Stage 2 | Piloto: treinamento LoRA, convergencia, primeiros resultados | Pendente |
| Stage 3 | Avaliacao: metricas completas com CI 95%, leakage probes, red team | Pendente |

Detalhes do metodo: [`docs/protocol/stage-gate-method.md`](docs/protocol/stage-gate-method.md)

## Como Executar

O ambiente primario de execucao e **Google Colab** (GPU). O guia completo de execucao esta em [`notebooks/README.md`](notebooks/README.md).

### Notebooks Disponiveis

| Notebook | Objetivo | Tempo |
|----------|----------|-------|
| [`stage1_5_coraa_mupe.ipynb`](notebooks/stage1_5_coraa_mupe.ipynb) | Auditoria de separabilidade latente (probes lineares) | ~2-4h |
| [`accents_pt_br_dataset.ipynb`](notebooks/accents_pt_br_dataset.ipynb) | Pipeline Accents-PT-BR + publicacao HuggingFace | ~3-5h |
| [`accents_pt_br_classifier.ipynb`](notebooks/accents_pt_br_classifier.ipynb) | Ablation CNN vs wav2vec2 (classificador externo) | ~4-6h |

### Execucao Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Rodar testes (196 testes, sem GPU)
make test
```

## Metricas de Validacao

Tres metricas obrigatorias (nao-negociaveis):

1. **Controlabilidade de sotaque:** balanced accuracy de classificador externo em audio gerado
2. **Preservacao de identidade:** similaridade cosseno ECAPA-TDNN (queda < 10%)
3. **Leakage probes:** probes lineares A→speaker e S→accent devem ser <= chance + 5 p.p.

Baselines sao medidos **antes** de qualquer treinamento LoRA.
