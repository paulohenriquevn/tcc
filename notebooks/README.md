# Notebooks — Guia de Execucao

Os notebooks suportam **Google Colab**, **Lightning.ai**, **Paperspace Gradient** e **execucao local**. A deteccao de plataforma e automatica (`src/utils/platform.py`) — nao e necessario alterar nenhum path manualmente. A logica de negocio esta em `src/` (testavel, auditavel) — os notebooks sao camadas de orquestracao que instalam dependencias, configuram ambiente, chamam modulos e exibem resultados.

## Plataformas Suportadas

| Plataforma | GPU | Storage persistente | Setup |
|------------|-----|---------------------|-------|
| **Google Colab** | T4/A100 (depende do plano) | Google Drive (~50 GB) | Abrir notebook no Colab |
| **Lightning.ai** | L4 24GB (~21h/mes free) | 100 GB persistent disk | `bash scripts/lightning_setup.sh` |
| **Paperspace Gradient** | A5000 24GB+ (depende do plano) | `/storage/` persistente | Criar notebook, clonar repo |
| **Local** | NVIDIA com CUDA | Disco local | `pip install -r requirements.txt` |

## Pre-Requisitos

| Requisito | Colab | Lightning.ai | Paperspace | Local |
|-----------|-------|-------------|------------|-------|
| **GPU** | Runtime com GPU (T4 minimo) | L4 incluida no Studio | A5000+ (24 GB VRAM) | NVIDIA com CUDA |
| **Storage** | Google Drive montado automaticamente | Persistent disk built-in | `/storage/` persistente | ~50 GB de disco |
| **Conta HuggingFace** | Apenas para publicacao (`accents_pt_br_dataset.ipynb`) | Idem | Idem | Idem |
| **Espaco em disco** | ~50 GB no Drive | ~50 GB no persistent disk | ~50 GB no `/storage/` | ~50 GB local |

## Notebooks Disponiveis

```
notebooks/
├── stage1_5_coraa_mupe.ipynb        # Stage 1.5: auditoria de separabilidade latente
├── accents_pt_br_classifier.ipynb   # Ablation: CNN vs wav2vec2 accent classifier
└── accents_pt_br_dataset.ipynb      # Pipeline de dataset + publicacao HuggingFace
```

### Ordem de Execucao

Os notebooks sao **independentes** — cada um clona o repositorio e instala dependencias do zero. Porem, os que dependem de dados compartilham cache via storage persistente:

```
                   ┌─────────────────────────────────────┐
                   │ stage1_5_coraa_mupe.ipynb            │
                   │ (CORAA-MUPE only, probes latentes)   │
                   └─────────────────────────────────────┘

┌─────────────────────────────────────┐    ┌─────────────────────────────────────┐
│ accents_pt_br_dataset.ipynb         │───>│ accents_pt_br_classifier.ipynb      │
│ (Build dataset + publish to HF)     │    │ (Train/eval CNN vs wav2vec2)         │
└─────────────────────────────────────┘    └─────────────────────────────────────┘
         Produz manifests no cache                Consome manifests do cache
```

**Recomendacao:** executar `accents_pt_br_dataset.ipynb` antes de `accents_pt_br_classifier.ipynb`, pois o dataset notebook constroi e valida os manifests que o classifier consome via cache.

---

## Setup por Plataforma

### Google Colab

1. Abrir o notebook desejado: `File > Open notebook > GitHub > paulohenriquevn/tcc > notebooks/<notebook>.ipynb`
2. Selecionar runtime com GPU: `Runtime > Change runtime type > T4 GPU`
3. Executar a cell de setup. Se aparecer "Restarting runtime...", **re-executar a cell de setup** apos o restart (so acontece 1 vez por conta do NumPy ABI)
4. O Google Drive sera montado automaticamente (requer autorizacao na primeira vez)
5. Executar todas as cells sequencialmente (`Runtime > Run all`)

### Lightning.ai

1. Criar um Studio no Lightning.ai com GPU (L4 recomendado)
2. Abrir o terminal do Studio e executar o setup inicial (apenas 1 vez):

```bash
git clone https://github.com/paulohenriquevn/tcc.git /teamspace/studios/this_studio/TCC
cd /teamspace/studios/this_studio/TCC
bash scripts/lightning_setup.sh
```

3. Abrir o notebook desejado em `notebooks/` pelo file browser do Studio
4. Executar as cells sequencialmente — a deteccao de plataforma e automatica
5. O cache e armazenado em `/teamspace/studios/this_studio/cache` (persistente entre sessoes)

**Vantagens do Lightning.ai:**
- GPU L4 com 24 GB VRAM (suficiente para wav2vec2 sem reducao de batch)
- 100 GB de storage persistente (sem necessidade de Google Drive)
- ~21h/mes de GPU gratuita
- Sem restart de runtime por NumPy ABI (ambiente controlado)

### Paperspace Gradient

1. Criar um notebook no Paperspace Gradient com GPU A5000 (ou superior, 24 GB VRAM minimo)
2. Abrir o terminal e clonar o repositorio:

```bash
git clone https://github.com/paulohenriquevn/tcc.git /notebooks/TCC
cd /notebooks/TCC
pip install -r requirements.txt
```

3. Abrir o notebook desejado em `notebooks/` pelo file browser
4. Executar as cells sequencialmente — a deteccao de plataforma e automatica
5. O cache e armazenado em `/storage/tcc-cache` (persistente entre instancias de notebook)

**Vantagens do Paperspace Gradient:**
- GPU A5000 com 24 GB VRAM (suficiente para todo o pipeline)
- `/storage/` persiste entre instancias de notebook (equivalente ao Google Drive)
- Sem restart de runtime por NumPy ABI (ambiente controlado)
- Sem necessidade de montar Google Drive

### Execucao Local

```bash
# 1. Instalar dependencias
pip install -r requirements.txt
pip install huggingface_hub  # apenas para publicacao

# 2. Executar como notebook:
jupyter notebook notebooks/<notebook>.ipynb

# Ou converter para script:
jupyter nbconvert --to script notebooks/<notebook>.ipynb
python notebooks/<notebook>.py
```

**Requisitos locais:** Python 3.10+, GPU NVIDIA com CUDA (24 GB VRAM para wav2vec2), ~50 GB de disco.

A deteccao de plataforma e automatica — nao e necessario comentar cells nem ajustar paths.

---

## 1. `stage1_5_coraa_mupe.ipynb` — Auditoria de Separabilidade Latente

**Objetivo:** verificar se representacoes internas do Qwen3-TTS codificam informacao de sotaque regional acima de chance, com leakage controlado.

**Dataset:** CORAA-MUPE (somente)
**Config:** `configs/stage1_5.yaml`
**Tempo estimado:** ~2-4h (download + extracao de features + probes)

### Secoes do Notebook

| # | Secao | O que faz |
|---|-------|-----------|
| 0 | Setup | Clona repo, instala deps, detecta plataforma, configura seeds |
| 1 | Manifest | Download CORAA-MUPE (~42 GB), filtra, constroi manifest JSONL |
| 2 | Splits | Gera splits speaker-disjoint + stratified (para leakage A→speaker) |
| 3 | Confounds | Chi-quadrado accent×gender, Kruskal-Wallis accent×duration, accent×SNR |
| 4 | Features | Extrai acoustic, ECAPA, WavLM (layers), Qwen3-TTS backbone (layers) |
| 5 | Baseline ECAPA | Mede similaridade intra/inter speaker (referencia para Stage 2) |
| 6 | Probes | Probes lineares: accent, leakage A→speaker, leakage S→accent |
| 7 | Robustness | Repete melhor probe com 3 seeds |
| 8 | Gate Decision | Avaliacao automatica GO/ADJUST/FAIL |
| 9 | Report | Salva JSON com todas as metricas e proveniencia |

### Outputs

| Artefato | Caminho |
|----------|---------|
| Manifest JSONL | `<cache>/<filter_hash>/manifest.jsonl` |
| Splits JSON | `data/splits/splits_seed42.json` |
| Features (cache) | `<cache>/<filter_hash>/features/` |
| Report JSON | `reports/stage1_5_report.json` |
| Confusion matrix | `reports/figures/confusion_matrix_accent.png` |

---

## 2. `accents_pt_br_dataset.ipynb` — Pipeline de Dataset + HuggingFace

**Objetivo:** construir o dataset derivado Accents-PT-BR (CORAA-MUPE + Common Voice PT), executar validacao de confounds, criar splits speaker-disjoint e publicar no HuggingFace Hub.

**Dataset:** CORAA-MUPE + Common Voice PT (combinado)
**Config:** `configs/accent_classifier.yaml`
**Tempo estimado:** ~3-5h (download de ambos datasets + build de manifests + upload)

### Secoes do Notebook

| # | Secao | O que faz |
|---|-------|-----------|
| 1 | Setup | Clona repo, instala deps + `huggingface_hub`, detecta plataforma |
| 2 | CORAA-MUPE | Download e build de manifest (cache-aware) |
| 3 | Common Voice PT | Download CV v17.0, normaliza labels de sotaque, build manifest |
| 4 | Combinado | Merge com validacao de colisoes, filtros de regiao/speaker |
| 5 | Confounds | accent×gender, accent×duration, accent×source |
| 6 | Splits | Speaker-disjoint train/val/test com verificacao |
| 7 | HF Dataset | Constroi `DatasetDict` com `Audio()` feature e `ClassLabel` |
| 8 | Publicacao | Login HF, gera dataset card, `push_to_hub()`, verificacao round-trip |

### Outputs

| Artefato | Caminho |
|----------|---------|
| Manifest CORAA-MUPE | `<cache>/coraa_mupe/manifest.jsonl` |
| Manifest Common Voice | `<cache>/common_voice_pt/manifest.jsonl` |
| Manifest combinado | `<cache>/accents_pt_br/manifest.jsonl` |
| Splits JSON | `data/splits/splits_seed42.json` |
| Dataset no HuggingFace | `huggingface.co/datasets/paulohenriquevn/accents-pt-br` |

### Customizacao

Para publicar em outro repositorio HuggingFace, altere a variavel `HF_REPO_ID` na cell de publicacao:

```python
HF_REPO_ID = 'seu-usuario/accents-pt-br'  # trocar aqui
```

---

## 3. `accents_pt_br_classifier.ipynb` — Ablation CNN vs wav2vec2

**Objetivo:** treinar e avaliar classificadores de sotaque (CNN mel-spectrogram vs wav2vec2 fine-tuned) no dataset Accents-PT-BR. Estes classificadores servem como avaliadores externos para Stages 2-3.

**Dataset:** Accents-PT-BR (CORAA-MUPE + Common Voice PT combinado)
**Config:** `configs/accent_classifier.yaml`
**Tempo estimado:** ~4-6h (download + treino CNN ~1h + treino wav2vec2 ~2h + cross-source)

### Secoes do Notebook

| # | Secao | O que faz |
|---|-------|-----------|
| 1 | Setup | Clona repo, instala deps, detecta plataforma, configura seeds |
| 2 | CORAA-MUPE | Download/cache manifest |
| 3 | Common Voice PT | Download/cache manifest |
| 4 | Combinado | Merge Accents-PT-BR |
| 5 | Confounds | Analise accent×gender, duration, source |
| 6 | Splits | Speaker-disjoint train/val/test |
| 7 | CNN | Treino com early stopping, avaliacao com CI 95%, confusion matrix |
| 8 | wav2vec2 | Treino (batch menor por VRAM), avaliacao com CI 95% |
| 9 | Cross-source | Treina em fonte A, testa em fonte B (e vice-versa) |
| 10 | Ablation | Tabela comparativa, verifica overlap de CIs, salva report JSON |

### Outputs

| Artefato | Caminho |
|----------|---------|
| Checkpoint CNN | `experiments/accent_classifier/checkpoints/cnn/` |
| Checkpoint wav2vec2 | `experiments/accent_classifier/checkpoints/wav2vec2/` |
| Training curves | `experiments/accent_classifier/reports/figures/` |
| Confusion matrices | `experiments/accent_classifier/reports/figures/` |
| Report JSON | `experiments/accent_classifier/reports/ablation_report.json` |

---

## Problemas Comuns

### NumPy ABI Mismatch (Colab apenas)

```
NumPy ABI mismatch: loaded=2.x.x, installed=1.26.4
Restarting runtime...
```

**Causa:** Colab pre-carrega NumPy 2.x na memoria, mas `requirements.txt` pina 1.26.4.
**Solucao:** O runtime reinicia automaticamente. Basta **re-executar a cell de setup**. Acontece apenas 1 vez por sessao.
**Nota:** Este problema nao ocorre no Lightning.ai, Paperspace Gradient nem em execucao local.

### Google Drive nao monta (Colab apenas)

```
Drive already mounted at /content/drive
```

Se o Drive nao montar, verificar se a conta Google tem espaco disponivel (~50 GB necessarios).
**Nota:** No Lightning.ai e Paperspace Gradient, o storage e persistente e nao depende de Google Drive.

### CUDA Out of Memory (wav2vec2)

Se o treino do wav2vec2 estourar VRAM:
- Reduzir `batch_size` em `configs/accent_classifier.yaml` (de 8 para 4)
- Usar GPU com mais VRAM (A100 no Colab, L4 no Lightning.ai, A5000+ no Paperspace)
- O `freeze_feature_extractor: true` ja esta ativo por padrao

### Download lento do CORAA-MUPE

O primeiro download e ~42 GB. Em sessoes subsequentes, o manifest e features sao carregados do cache. Se o download falhar no meio, re-executar a cell — a biblioteca `datasets` tem resume automatico.

### Token HuggingFace sem permissao write

```
HTTPError: 403 Forbidden
```

O token precisa da permissao `write`. Gerar um novo em https://huggingface.co/settings/tokens com scope "Write".

---

## Testes

Os modulos `src/` usados pelos notebooks tem 196 testes unitarios:

```bash
python3 -m pytest tests/ -v --tb=short
```

Todos os testes passam sem GPU — usam mocks e dados sinteticos.
