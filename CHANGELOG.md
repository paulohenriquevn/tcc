# Changelog

Todas as mudanças notáveis do projeto são documentadas neste arquivo.

Formato: [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) + [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Added
- `train_classifier()` agora aceita `resume_from: Path` para retomar treinamento de checkpoint — restaura model, optimizer, scaler e epoch (#12)
- `label_to_idx` persistido em JSON no notebook do classificador para mapeamento reproduzível de labels (#12)
- `tqdm==4.67.1` pinado explicitamente no `requirements.txt` — era dependência transitiva, agora versionada (#12)
- Notebook `stage1_5`: seções "## 1. Download e Build Manifest" e "## 4. Feature Extraction", histograma de duração, e célula de gate decision com avaliação GO/ADJUST/FAIL estruturada (#12)
- Notebook `accents_pt_br_classifier`: verificação de chance level antes da avaliação cross-source (#12)

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
