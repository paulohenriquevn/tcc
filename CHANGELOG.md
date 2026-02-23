# Changelog

Todas as mudanças notáveis do projeto são documentadas neste arquivo.

Formato: [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) + [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Added
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
- Referências cruzadas atualizadas após reorganização de diretórios (CLAUDE.md, stage1_5.yaml, STAGE_1.md, RESEARCH_ROADMAP.md) (#3)
- Nome de arquivo com espaço corrigido: `EMOSTEER -TTS.pdf` → `emosteer-tts_xie-2025.pdf` (#3)
- Typo corrigido: `Qwen_3_tecnical.pdf` → `qwen3-tts-technical_qwen-2026.pdf` (#3)
- Contradições C2-C5 entre STAGE_1.md, TECHNICAL_VALIDATION_PROTOCOL.md e notebook resolvidas (#2)
- Status falso em CLAUDE.md ("Stage 1.5 in progress") corrigido para refletir realidade (#2)
