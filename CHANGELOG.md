# Changelog

Todas as mudanças notáveis do projeto são documentadas neste arquivo.

Formato: [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) + [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Changed
- `acoustic.py`: pitch tracker alterado de `librosa.pyin` para `librosa.yin` (~10-50x mais rápido) — para estatísticas agregadas (média/std), YIN determinístico é adequado e reduz extração de 63h para ~1-3h (#25)
- `manifest_builder.py`: iteração sequencial com raw bytes writing — evita random access Arrow + roundtrip decode/encode de áudio (#25)
- Notebook `stage1_5`: células de extração (18-21) agora fazem checkpoint periódico a cada 10K items no Google Drive + suporte a resume de cache parcial — se sessão Colab morrer, progresso é preservado (#25)

### Fixed
- Notebook `stage1_5` cell-11: `TypeError` ao formatar `filter_stats` — chave `dropped_speakers` é lista, não número; agora pula valores não-numéricos (#26)
- Notebook `stage1_5` cell-8: token HF hardcoded removido — restaurado `userdata.get('HF_TOKEN')` (#26)
- Notebook `stage1_5` cells 20-21 (WavLM/Backbone): bug de cache per-layer — se qualquer camada faltava no cache, TODAS camadas eram descartadas e re-extraídas do zero (#25)

### Changed
- Estratégia "5 regiões, CO de CORAA-MUPE apenas": BrAccent removido do pipeline por não ser dataset HuggingFace; CO re-incluído via CORAA-MUPE (3 speakers) com `min_speakers_per_region` reduzido de 5 para 3 (mínimo científico para splits speaker-disjoint); CO excluído do Common Voice via `exclude_accents: ["CO"]` para evitar confound V=0.8791; CO documentado como análise secundária (não Gate-eligible, CI reportado separadamente); pipeline, config, notebook, dataset card e protocolo atualizados (#24)

### Added
- `braccent_manifest_builder.py`: builder para o dataset Fake BrAccent — mantido no repositório mas não integrado ao pipeline (não é fonte HuggingFace) (#21)
- Parâmetro `exclude_accents` em `build_manifest_from_common_voice()` — permite excluir regiões IBGE específicas de uma fonte (#21)
- 19 testes para BrAccent builder: parsing de filename, mapeamentos, CO re-mapping, integração com zip sintético (#21)

### Added
- Parâmetro `exclude_regions` em `combine_manifests()` e config YAML — permite excluir macro-regiões IBGE do pipeline com invalidação automática de cache (#20)
- `_CV_GENDER_MAP` expandido para aceitar labels `male_masculine`/`female_feminine` do mirror `fsicoli/common_voice_17_0` — sem isso, 100% dos rows do Common Voice PT seriam rejeitados silenciosamente pelo filtro de gênero (#18)

### Changed
- Estratégia "5 regiões com 3 fontes": CO re-incluído no pipeline via exclusão de CO do Common Voice + adição de BrAccent como 3ª fonte — CO agora tem 5 speakers (3 CORAA-MUPE + 2 BrAccent DF/GO), V(accent×source) estimado cai de 0.8791 para ~0.14 (abaixo do threshold 0.3); CO tratado como análise secundária (não Gate-eligible, CI reportado separadamente) (#21)
- Estratégia "4+1": CO excluído do pipeline principal por confound accent×source Cramer's V=0.8791 (threshold: 0.3) — 4 regiões restantes (N, NE, SE, S) são 96-100% CORAA-MUPE, eliminando confound; CO mantido como análise exploratória opcional (#20)
- Estratégia de cobertura regional expandida de 4 para 5 macro-regiões (N, NE, CO, SE, S) via combinação multi-source CORAA-MUPE + Common Voice PT — CO viabilizado com 7 speakers combinados (3 CORAA + 4 CV); protocolo, configs e documentação atualizados com censos de ambas as fontes e caveats de CO (#18)
- `min_speakers_per_region` reduzido de 8 para 5 em configs, defaults de funções e protocolo — decisão data-driven baseada em census completo do CORAA-MUPE-ASR (317k rows via streaming): Sul tem 7 speakers qualificados com 3.740 utterances (#17)

### Fixed
- `pipeline.py` reescrito: filtro de região era aplicado per-source (cada builder independentemente descartava CO por ter < 5 speakers), impedindo a agregação multi-source (CORAA-CO 3 + CV-CO 4 = 7) — agora builders recebem `min_speakers_per_region=0` e o threshold real é aplicado APENAS no `combine_manifests()` (#19)
- `pipeline.py`: cache usava bare `file.exists()` sem validação de hash — config de filtros podia mudar e o pipeline continuava servindo manifests stale. Agora usa sidecars `.filter_hash` com `compute_filter_hash()` de `src.data.cache`; cache combinado inclui SHAs dos manifests-fonte (#19)
- `manifest_builder.py` e `cv_manifest_builder.py`: audio WAV era reescrito incondicionalmente em rebuilds de manifest, decodificando do HuggingFace e salvando mesmo quando o arquivo já existia — adicionado guard `if not wav_path.exists()` para skip-existing-audio (#19)

### Fixed
- Notebook `stage1_5` cell 30: gate decision comparava com `'PASS'` mas `evaluate_probe_against_thresholds()` retorna `'GO'`/`'GO_CONDITIONAL'`/`'FAIL'` — gate nunca passava, decisão era sempre FAIL (#16)
- Notebook `stage1_5` cells 23-25: estado mutável `all_probe_results` compartilhado entre células causava duplicação de resultados ao re-executar — listas agora inicializadas/filtradas no início de cada cell para idempotência (#16)
- Notebook `stage1_5` cell 31: comprehension aninhada O(n*k) para cálculo de region stats substituída por iteração O(n) com `defaultdict` (#16)
- Notebook `accents_pt_br_dataset` cells 13-14: publicação no HuggingFace Hub executava automaticamente em Run All — adicionado gate `PUBLISH = False` que requer ativação manual (#16)
- Notebook `accents_pt_br_dataset` cell 14: escrita intermediária em `/tmp` (colisão em sistemas compartilhados) substituída por `tempfile.NamedTemporaryFile` (#16)
- Notebook `accents_pt_br_classifier` cells 7, 11: referências hardcoded a números de célula (`cell-2`, `cell-15`) já incorretas — removidas (#16)
- Notebook `accents_pt_br_classifier` cells 7, 11, 15, 17: `num_workers=2` hardcoded em 12 DataLoaders substituído por `cnn_cfg['training']['num_workers']` / `w2v_cfg['training']['num_workers']` do YAML config (#16)
- Notebook `accents_pt_br_dataset` cell 2: `pip install huggingface_hub` redundante removido — já é dependência transitiva de `datasets==3.2.0` instalado pelo bootstrap (#16)
- Notebook `accents_pt_br_dataset` cells 9, 12: mapeamento inline `'val' → 'validation'` substituído por `to_hf_split_entries()` de `src.data.hf_utils` — centraliza convenção de nomes (DRY) (#16)
- Notebook `accents_pt_br_dataset` cell 16: dict inline `hf_to_internal` substituído por reversão de `INTERNAL_TO_HF_SPLITS` de `src.data.hf_utils` (#16)

### Added
- `src/data/pipeline.py` — módulo `load_or_build_accents_dataset()` com `DatasetBundle` dataclass: pipeline unificada de dataset que elimina duplicação entre notebooks dataset e classifier (#15)
- `src/data/hf_utils.py` — `entries_to_hf_dict()` e `build_dataset_card()` extraídos do notebook dataset para código testável (#15)
- `src/utils/git.py` — `get_commit_hash()` centralizado, elimina `import subprocess` espalhado nos notebooks (#15)
- `build_probe_data()` em `src/evaluation/probes.py` — extraído do notebook stage1_5 para código auditável (#15)
- `src/utils/notebook_bootstrap.py` — módulo stdlib-only de bootstrap para notebooks: detecção de plataforma (Colab, Lightning.ai, Paperspace, local), clone de repo, pip install e verificação de NumPy ABI (#14)
- Notebook `accents_pt_br_classifier`: seção "Robustness Check" com retreino multi-seed [42, 1337, 7] para ambos classifiers e comparação de variância (#14)
- Notebook `stage1_5`: assertions explícitos de speaker-disjoint splits após geração de splits (#14)
- `train_classifier()` agora aceita `resume_from: Path` para retomar treinamento de checkpoint — restaura model, optimizer, scaler e epoch (#12)
- `label_to_idx` persistido em JSON no notebook do classificador para mapeamento reproduzível de labels (#12)
- `tqdm==4.67.1` pinado explicitamente no `requirements.txt` — era dependência transitiva, agora versionada (#12)
- Notebook `stage1_5`: seções "## 1. Download e Build Manifest" e "## 4. Feature Extraction", histograma de duração, e célula de gate decision com avaliação GO/ADJUST/FAIL estruturada (#12)
- Notebook `accents_pt_br_classifier`: verificação de chance level antes da avaliação cross-source (#12)
- Suporte a Paperspace Gradient em `src/utils/platform.py` — detecção automática via env `PAPERSPACE`, cache em `/storage/tcc-cache` (#13)

### Changed
- Notebooks dataset e classifier: pipeline duplicada (~10 cells cada) substituída por chamada única a `load_or_build_accents_dataset()` — DRY (#15)
- Notebook classifier: `seed_worker` inline (sem `random.seed`) substituído por import de `src.utils.seed.seed_worker` (completo) (#15)
- Notebook dataset: dataset card gerado via `build_dataset_card()` em vez de f-string inline de 200 linhas (#15)
- Notebook dataset: `entries_to_hf_dict()` importado de `src.data.hf_utils` em vez de definição inline (#15)
- Notebook classifier: cross-source threshold usa `config['cross_source']['above_chance_margin_pp']` em vez de hardcoded `0.05` (#15)
- Notebook stage1_5: `NEUTRAL_TEXT` vem de `config['features']['backbone']['neutral_text']` em vez de hardcoded (#15)
- Notebook stage1_5: VRAM monitorado após cada fase de extração de features (#15)
- Notebook classifier: robustness check markdown inclui estimativa de tempo (~2-4h GPU) (#15)
- Notebook classifier: report JSON inclui `robustness_results` e seções renumeradas (§3-§7) (#15)
- Todos os notebooks: `torch.load(..., weights_only=False)` documentado com comentário explicando motivo (#15)
- `configs/accent_classifier.yaml`: `above_chance_margin_pp: 5` adicionado à seção cross_source; comentário de `cache.drive_base` corrigido (#15)
- `configs/stage1_5.yaml`: `neutral_text` adicionado à seção `features.backbone`; comentário de `cache.drive_base` corrigido (#15)
- Notebooks (stage1_5, classifier, dataset): setup cells reescritos para usar `notebook_bootstrap.bootstrap()` — elimina triplicação de boilerplate de plataforma (#14)
- Notebook `accents_pt_br_classifier`: numeração de seções atualizada (Robustness Check = §9, Cross-source = §10, Ablation = §11) (#14)
- Notebook `accents_pt_br_dataset`: mapeamento de split names `validation→val` substituído por dict explícito `hf_to_internal` (#14)

### Fixed
- Notebook classifier: `seed_worker` inline estava sem `random.seed(worker_seed)` — causava não-determinismo em DataLoader workers que usam `random` (#15)
- Notebook `stage1_5` cell-15: `np.mean()`/`np.std()`/`np.median()` usados antes de `import numpy as np` — adicionado import no topo da célula (#14)
- Todos os 3 notebooks: montagem dupla de Google Drive (bootstrap + `setup_environment()`) — removida chamada redundante (#14)
- Notebook `stage1_5`: variável `overall` usada em gate decision sem inicialização — adicionado safe default `'NOT_EVALUATED'` (#14)
- Notebook `accents_pt_br_classifier`: variável `train_labels_cnn` nomeada incorretamente (compartilhada por CNN e wav2vec2) — renomeada para `train_labels` (#14)

### Changed
- Checkpoint do trainer agora inclui `scaler_state_dict` e `seed` para reprodutibilidade completa de estado (#12)
- Trainer loga VRAM (allocated + peak) por epoch quando em GPU (#12)
- `_bootstrap_ci()` agora aceita `seed` configurável — era hardcoded 42 (#12)
- `configs/stage1_5.yaml`: `splits.output_dir` → `"data/splits/stage1_5/"` para evitar colisão com splits do classificador (#12)
- `configs/accent_classifier.yaml`: `splits.output_dir` → `"data/splits/classifier/"` e adicionado filtro `speaker_type: "R"` ausente (#12)

### Fixed
- Notebook `stage1_5`: células corrompidas (markdown contendo código Python) removidas — resíduo de edição posicional com cell_id shifting (#12)
- Notebook `stage1_5`: célula de manifest build crashava com `NameError` quando cache existia — adicionado guard `if entries is None:` (#12)
- Notebook `accents_pt_br_classifier`: `torch.load()` sem `weights_only=False` falhava no PyTorch 2.6 (checkpoints com dados não-tensor) (#12)
- Notebook `accents_pt_br_classifier`: validação cross-source usava dados de treino como fallback (`coraa_train[:N]`) — corrigido para usar split de validação completo (speaker-disjoint) (#12)
- `validate_manifest_consistency()` usava O(n²) `list.count()` para detecção de duplicatas — substituído por O(n) `Counter` (#12)
- `n_permutations=5` em probes de seletividade era insuficiente para baseline confiável — aumentado para 50 (#12)

### Added
- `src/utils/platform.py` com `detect_platform()` e `setup_environment()` — detecção automática de Colab, Lightning.ai e local com paths e cache apropriados para cada plataforma (#11)
- `scripts/lightning_setup.sh` — script de setup one-time para Lightning.ai Studios (clone, deps, GPU check, testes) (#11)

### Changed
- Notebooks (`stage1_5_coraa_mupe`, `accents_pt_br_dataset`, `accents_pt_br_classifier`): setup cells reescritas para detecção automática de plataforma — sem necessidade de alterar paths manualmente entre Colab, Lightning.ai e local (#11)
- `notebooks/README.md`: guia de execução atualizado com instruções para Google Colab, Lightning.ai e execução local, incluindo tabela comparativa de plataformas (#11)

### Added
- Pacote `src/classifier/` com classificador externo de sotaque para avaliação de áudio gerado (Stages 2-3) — CNN 3-block e wav2vec2 fine-tuned, com trainer, inference e datasets (#9)
- `AccentCNN` (mel-spectrogram → logits) e `AccentWav2Vec2` (waveform → logits) em `src/classifier/` (#9)
- `train_classifier()` e `evaluate_classifier()` com early stopping, balanced accuracy, bootstrap CI 95%, class weighting (#9)
- `load_classifier()`, `classify_audio()`, `classify_batch()` em `src/classifier/inference.py` para avaliação de áudio gerado (#9)
- `MelSpectrogramDataset` e `WaveformDataset` para CNN e wav2vec2 respectivamente (#9)
- `CV_ACCENT_TO_MACRO_REGION` e `normalize_cv_accent()` em `manifest.py` — mapeia labels de sotaque do Common Voice para macro-regiões IBGE (#9)
- `build_manifest_from_common_voice()` em `src/data/cv_manifest_builder.py` — filtragem em duas fases para Common Voice PT, com prefixo `cv_` para IDs (#9)
- `combine_manifests()` e `analyze_source_distribution()` em `src/data/combined_manifest.py` — merge de manifests multi-source com detecção de colisões (#9)
- `analyze_accent_x_source()` em `confounds.py` — chi-squared test de confound accent × data source (Cramer's V), auto-executado quando múltiplas fontes detectadas (#9)
- `configs/accent_classifier.yaml` com configuração completa do experimento de ablação CNN vs wav2vec2 (#9)
- `notebooks/accents_pt_br_classifier.ipynb` — notebook Colab para construção do dataset Accents-PT-BR e treinamento dos classificadores (#9)
- Testes: `test_cv_manifest_builder.py`, `test_classifier_models.py`, `test_classifier_datasets.py`, `test_classifier_trainer.py` (#9)
- `notebooks/accents_pt_br_dataset.ipynb` — notebook Colab para pipeline completa de construção do Accents-PT-BR e publicação no HuggingFace Hub com dataset card, speaker-disjoint splits e verificação round-trip (#10)
- `notebooks/README.md` — guia de execução dos 3 notebooks com pre-requisitos, ordem, troubleshooting e instrucoes de execucao local (#10)

### Changed
- `README.md` (raiz): seção "Como Executar" atualizada com tabela dos 3 notebooks e link para `notebooks/README.md` (#10)

- `run_all_confound_checks()` agora inclui automaticamente análise accent × source quando múltiplas fontes são detectadas nos entries (#9)

- `PipelineCache` em `src/data/cache.py` para persistência de manifest e features em Google Drive entre sessões Colab — usa hash SHA-256 dos filtros como chave de diretório, invalidação automática ao mudar config (#8)
- `_filter_speakers_by_utterance_count()` em `manifest_builder.py` — descarta speakers com menos de N utterances antes de construir manifest (#8)
- `min_utterances_per_speaker: 3` no config YAML (`dataset.filters`) (#8)
- Seção `cache:` no config YAML com `enabled` e `drive_base` (#8)
- Cell de montagem do Google Drive no notebook Colab (#8)
- `tests/test_cache.py` com 15 testes para `PipelineCache` e `compute_filter_hash()` (#8)
- `tests/test_manifest_builder.py` com 7 testes para `_filter_speakers_by_utterance_count()` (#8)

### Changed
- `validate_manifest_consistency()` em `manifest.py`: check de cobertura de regiões agora exige mínimo 2 (era 5 fixo) — compatível com fallback do protocolo §4.3 que descarta regiões com < 8 speakers (#8)
- Notebook cells 7-8: manifest build agora é cache-aware — carrega do Drive se existir, senão faz download + build + salva (#8)
- Notebook cells 15-18: extração de features agora é cache-aware — carrega NPZ do Drive se existir, senão extrai + salva (#8)
- Notebook cell 31 (report): inclui `filter_hash` no JSON de saída para rastreabilidade (#8)
- `build_manifest_from_coraa()` e `build_manifest_from_hf_dataset()`: novo parâmetro `min_utterances_per_speaker` com tracking em `filter_stats` (#8)

### Fixed
- `.gitignore` usava `data/` sem leading `/`, ignorando `src/data/` (pacote Python) além de `/data/` (artifacts) — corrigido para `/data/`, `/reports/`, `/experiments/` (#8)
- `qwen-tts==1.0.1` no requirements.txt não existe no PyPI (versão especulativa) — corrigido para `qwen-tts==0.1.1` (latest real, Feb 2026). Sem esta correção `pip install -r requirements.txt` falha e nenhuma dependência posterior é instalada (#7)
- `transformers==4.48.3` conflita com `qwen-tts==0.1.1` que hard-pina `transformers==4.57.3` — atualizado pin para `4.57.3`. Compatível com WavLM `AutoModel` (suportado desde ~4.12) e SpeechBrain (que não depende de transformers) (#7)
- `torch.cuda.get_device_properties(0).total_mem` não existe em PyTorch — corrigido para `total_memory` no notebook cell 4 (#7)
- Regiões com < 8 speakers (N, CO) causavam hard-fail no `manifest_builder.py` — implementado fallback do protocolo §4.3: regiões insuficientes são descartadas com warning, mantendo NE/SE/S (#7)

### Changed
- `manifest_builder.py`: validação de regiões extraída para `_filter_regions_by_speaker_count()` compartilhada entre `build_manifest_from_coraa()` e `build_manifest_from_hf_dataset()` — implementa fallback gracioso em vez de hard-fail (#7)
- Notebook cell 8: exibe regiões descartadas e referência ao protocolo §4.3 (#7)

### Added
- `accelerate==1.12.0` pinado explicitamente — dependência transitiva de `qwen-tts==0.1.1`, agora versionada para reprodutibilidade (#7)

### Added
- Seção 9.4 (UTMOS + WER via Whisper) no TECHNICAL_VALIDATION_PROTOCOL.md como sanity checks de qualidade de fala para Stage 2-3, com thresholds GO/ADJUST/FAIL e integração nos critérios de decisão da seção 10 (#6)
- UTMOS e WER na tabela formal de métricas em `metric-validation.md` (#6)
- WER baseline (backbone sem LoRA) adicionado à seção 6 como referência obrigatória (#6)
- `normalize_birth_state()` para converter nomes completos de estados (e.g. "São Paulo") em siglas UF, com variantes acentuadas e sem acento (#5)
- `STATE_FULL_NAME_TO_ABBREV` mapeamento de 27 estados brasileiros (54 variantes) em `manifest.py` (#5)
- 11 testes unitários para `normalize_birth_state()` cobrindo siglas, nomes completos, case-insensitivity e edge cases (#5)
- `smoke_test_backbone()` função diagnóstica para validar extração de features no Colab antes do pipeline (#5)
- `qwen-tts` adicionado ao `requirements.txt` como dependência do backbone (#5)
- `generate_stratified_splits()` para splits onde mesmos speakers aparecem em train e test com utterances diferentes — necessário para leakage probe A→speaker (#4)
- `StratifiedSplitInfo` dataclass com persistência JSON e roundtrip (save/load) (#4)
- `train_selectivity_control()` permutation baseline no pipeline de probes — mede sinal real vs memorização (#4)
- `analyze_accent_x_snr()` análise de confound accent × condições de gravação via Kruskal-Wallis em SNR estimado (#4)
- `estimate_snr_rms()` estimativa de SNR baseada em energia RMS com threshold de silêncio no percentil 20 (#4)
- Campo `sampling_rate: int` no `ManifestEntry` com validação e verificação de uniformidade no manifest (#4)
- `sweep_regularization()` para sweep de regularização C em probes com múltiplos valores (#4)
- 48 testes unitários para módulos `src/features/` (acoustic, ECAPA, WavLM, backbone) em `tests/test_features.py` (#4)
- 19 testes para stratified splits, selectivity control e confound SNR nos test files existentes (#4)
- Implementação completa do leakage probe A→speaker no notebook Colab usando stratified splits (#4)
- README.md na raiz como entry point do repositório para orientador, banca e pesquisadores (#3)
- `docs/meetings/README.md` com índice das atas de reunião (#3)
- Skill `deep-paper-search` para busca profunda de papers científicos com LlamaIndex RAG — Arxiv + PubMed, indexação local com embeddings HuggingFace, retrieval semântico sem LLM (#1)
- Roteamento PAPER/LITERATURE/ARXIV/PUBMED para rafael-monteiro no head-research (#1)
- Stage 1.5 no RESEARCH_ROADMAP.md com 6 experimentos, métricas e Definition of Done (#2)
- `KB_HARD_FAIL_RULES.md` com 5 categorias de critérios de parada automática (#2)
- Seção 9.4 (confounds) e 9.5 (baseline ECAPA) no TECHNICAL_VALIDATION_PROTOCOL.md (#2)
- Fundação do pipeline Stage 1.5: `requirements.txt`, schema de manifest, config YAML, seeds, splits (#2)
- Pipeline completo Stage 1.5: features (acoustic, ECAPA, WavLM, backbone), probes com CI, confound analysis (#2)
- Integração HuggingFace `datasets` para download direto do CORAA-MUPE-ASR no notebook (#2)
- `build_manifest_from_hf_dataset()` com filtragem em duas fases: metadata rápida + decode de áudio filtrado (#2)
- Notebook Colab `stage1_5_coraa_mupe.ipynb` com pipeline end-to-end de 9 seções (#2)
- Makefile com targets `stage15`, `manifest`, `splits`, `confounds`, `features`, `probes`, `report` (#2)

### Changed
- `backbone.py` reescrito: usa `qwen_tts.Qwen3TTSModel` com extração via forward hooks nos layers do talker, substituindo API incorreta de `AutoModel`/`AutoProcessor` (#5)
- `manifest_builder.py`: 3 call-sites atualizados para usar `normalize_birth_state()` em vez de lookup direto por sigla — corrige rejeição silenciosa de 100% dos entries do HuggingFace dataset (#5)
- Notebook cell 12: `snr_practical_diff_db` agora é passado do YAML config em vez de usar valor default (#5)
- Notebook cell 29: gate decision mostra `effect_size_name` e `effect_size` específicos de cada confound bloqueante (#5)
- Testes de backbone reescritos para API hook-based com `nn.Module` reais em vez de mocks de output (#5)
- TECHNICAL_VALIDATION_PROTOCOL.md seção 4.3: "≤3 classes no piloto" → 5 macro-regiões IBGE com fallback documentado para 3 se dados insuficientes (#4)
- TECHNICAL_VALIDATION_PROTOCOL.md seção 9.4: adicionado accent×SNR (condições de gravação) à tabela de confounds obrigatórios (#4)
- `run_all_confound_checks()` agora inclui análise SNR por padrão (`check_snr=True`) (#4)
- `compute_speaker_similarity_baseline()` em ecapa.py: parâmetro `seed` agora é configurável (era hardcoded) (#4)
- Notebook: gate decision agora verifica status de confounds bloqueantes antes de decidir GO/ADJUST/FAIL (#4)
- Notebook: relatório JSON inclui metadata de ambiente (CUDA, cuDNN, torch versions) para reprodutibilidade (#4)
- Notebook: probe pipeline usa sweep de regularização C em vez de valor único (#4)
- Estrutura de documentação reorganizada: `docs/protocol/`, `docs/methodology/`, `docs/meetings/` (#3)
- `pesquisas/` renomeado para `references/` com PDFs padronizados (autor-ano) (#3)
- `.gitignore` seletivo para `.claude/`: infraestrutura de pesquisa agora versionada (agents, rules, skills, roadmap, changelog) (#3)
- CHANGELOG.md movido de `.claude/` para raiz (convenção Keep a Changelog) (#3)
- Makefile simplificado: targets fantasma removidos, aponta para notebook Colab (#3)
- Thresholds unificados em TECHNICAL_VALIDATION_PROTOCOL.md como fonte autoritativa única (#2)
- Métrica primária: balanced accuracy (F1-macro como secundária) — resolve contradição C4 (#2)
- Leakage com thresholds tiered: GO ≤ chance+5pp, CONDITIONAL ≤ chance+12pp, FAIL > 12pp (#2)
- Speaker similarity threshold: < 10% piloto (era ≤ 5%), target final < 5% no Stage 3 (#2)
- ECAPA-TDNN padronizado: SpeechBrain pré-treinado, 192-dim (#2)
- STAGE_1.md seção 8 agora referencia TECHNICAL_VALIDATION_PROTOCOL.md em vez de hardcodar thresholds (#2)
- CLAUDE.md: status corrigido de "in progress" para "pipeline em construção" (#2)

### Fixed
- `birth_state` do CORAA-MUPE-ASR contém nomes completos ("São Paulo"), não siglas ("SP") — sem `normalize_birth_state()`, 100% dos entries seriam silenciosamente rejeitados (#5)
- `backbone.py` usava `AutoModel.from_pretrained()` que não funciona com Qwen3-TTS (requer `qwen_tts.Qwen3TTSModel`) — reescrito com API correta (#5)
- Notebook cell 29: gate decision hardcodava threshold de gênero para todos os confounds bloqueantes (#5)
- Notebook cell 3: output residual de erro NumPy ABI removido (#5)
- `requirements.txt` não listava `qwen-tts` como dependência do backbone (#5)
- Contradição protocolo vs config: "≤3 classes" no protocolo vs "5 macro-regiões" no config — unificado para 5 regiões IBGE (#4)
- Leakage probe A→speaker era impossível sem stratified splits — implementado split stratified + probe completo (#4)
- Removido `hashlib-additional==1.0.1` do requirements.txt — desnecessário, stdlib `hashlib` é suficiente (#4)
- `wandb==0.19.6` marcado como opcional no requirements.txt — não utilizado no codebase, reservado para Stage 2+ (#4)
- Referências cruzadas atualizadas após reorganização de diretórios (CLAUDE.md, stage1_5.yaml, STAGE_1.md, RESEARCH_ROADMAP.md) (#3)
- Nome de arquivo com espaço corrigido: `EMOSTEER -TTS.pdf` → `emosteer-tts_xie-2025.pdf` (#3)
- Typo corrigido: `Qwen_3_tecnical.pdf` → `qwen3-tts-technical_qwen-2026.pdf` (#3)
- Contradições C2-C5 entre STAGE_1.md, TECHNICAL_VALIDATION_PROTOCOL.md e notebook resolvidas (#2)
- Status falso em CLAUDE.md ("Stage 1.5 in progress") corrigido para refletir realidade (#2)
