# Reuniao de Pesquisa: Validacao Tecnica de Controle de Sotaque com Preservacao de Identidade Vocal

**Data:** 2026-02-19
**Facilitador:** head-research
**Participantes:**

| # | Persona | Papel | Foco na Reuniao |
|---|---------|-------|-----------------|
| 1 | Dr. Rafael Monteiro | Lead Research Engineer (Backbone/Literature) | Papers sobre accent-controllable TTS, LoRA em TTS, disentanglement S/A |
| 2 | Dr. Felipe Nakamura | Applied Scientist (Evaluation/Probing) | Metricas e protocolos de validacao na literatura, comparacao com nosso protocolo |
| 3 | Mariana Alves | Senior Speech Data Engineer (Dataset/Confounds) | Datasets multi-accent, estrategias de split, confound mitigation |

**Contexto:** TCC "Controle Explicito de Sotaque Regional em pt-BR". Stage 1 GO passado, Stage 1.5 em andamento. Objetivo da reuniao: levantar papers, exemplos e montar plano claro para validacao tecnica.

---

## Pauta

1. Estado da arte em accent/dialect control para TTS
2. Metricas e protocolos de validacao em trabalhos relacionados
3. Lacunas e oportunidades para o nosso plano

---

## 1. Convergencias — Onde os Tres Concordam

### O projeto esta em lacuna real da literatura

Nenhum paper publicado usa **LoRA especificamente para accent control em TTS**. Isso e simultaneamente oportunidade (contribuicao original) e risco (sem precedentes para guiar decisoes). O CORAA-MUPE para TTS accent control em pt-BR tambem e inedito.

### Nosso protocolo de validacao e mais rigoroso que a media

Balanced accuracy (vs accuracy simples), CI 95%, leakage probes bidirecionais, speaker-disjoint splits — poucos papers fazem tudo isso. Os tres concordam que nosso rigor metodologico e um diferencial.

### Stage 1.5 e essencial e nao pode ser pulado

Os tres condicionam avanco ao Stage 2 aos resultados do Stage 1.5 (probing de separabilidade latente nas representacoes do Qwen3-TTS).

---

## 2. Relatorio Rafael Monteiro — Estado da Arte em Accent-Controllable TTS

### 2.1 Tabela de Papers Relevantes

| # | Titulo | Autores / Ano | Abordagem Tecnica | Metricas Reportadas | Dataset | Relevancia |
|---|--------|---------------|--------------------|--------------------|---------|------------|
| 1 | **DART: Disentanglement of Accent and Speaker Representation in Multispeaker TTS** | ~2024-2025 | Disentanglement explicito S vs A com mecanismo adversarial | Accent classification, speaker similarity | Multi-accent English | **CRITICA** |
| 2 | **Multi-Scale Accent Modeling and Disentangling for Multi-Speaker Multi-Accent TTS** | ~2024-2025 | Modelagem multi-escala (fonetico + prosodico) com disentanglement S/A | Accent classification accuracy, speaker similarity, MOS | Multi-accent datasets | **CRITICA** |
| 3 | **Controllable Accented Text-to-Speech Synthesis** | ~2023-2024 | TTS com accent variavel (L2 como variante de L1) | Accent controllability, naturalness MOS | English L1/L2 | **ALTA** |
| 4 | **Explicit Intensity Control for Accented TTS** | ~2024 | Speaker-adversarial loss para disentanglement + controle de intensidade | Accent intensity, speaker preservation | English accented | **ALTA** |
| 5 | **MacST: Multi-Accent Speech Synthesis via Text Transliteration** | ~2024 | Transliteracao de texto para criar pares multi-accent do mesmo speaker | Accent accuracy, speaker similarity | Multi-accent pairs | **ALTA** |
| 6 | **Accent and Speaker Disentanglement in Many-to-many Voice Conversion** | ~2023-2024 | Voice conversion com disentanglement accent/speaker | Accent accuracy, speaker similarity, MOS | Multi-accent English | **ALTA** |
| 7 | **Voice-preserving Zero-shot Multiple Accent Conversion** | ~2024 | Accent conversion zero-shot preservando timbre e pitch | Speaker preservation, accent transfer | Multi-accent | **MEDIA-ALTA** |
| 8 | **Investigating Speaker Embedding Disentanglement on Natural Read Speech** | ~2024 | Analise empirica de disentanglement em embeddings de speaker | Probing accuracy para atributos | Read speech | **ALTA** |
| 9 | **Mixture of LoRA Experts with Multi-Modal LLM for Accented Speech Recognition** | ~2024-2025 | Mixture of LoRA experts por accent para ASR | WER por accent | Multi-accent ASR | **MEDIA** |
| 10 | **EmoSteer-TTS: Fine-Grained Emotion-Controllable TTS via Activation Steering** | Xie et al., 2025 | Activation steering em DiT-TTS sem treinamento | Emotion classification, speaker similarity, MOS | Emotional speech | **MEDIA** |
| 11 | **Segment-Aware Conditioning for Training-Free Intra-Utterance Emotion and Duration Control** | Liang et al., 2026 | Conditioning segment-aware sem retreinamento | Emotion consistency, duration control, MOS | Multi-emotion TTS | **MEDIA** |
| 12 | **Qwen3-TTS Technical Report** | Qwen Team, 2026 | Dual-track LM architecture, speaker embedding + codec, DPO + GSPO | WER, speaker similarity, MOS | 5M horas multi-lingue | **CRITICA** (nosso backbone) |
| 13 | **Meta-TTS: Meta-Learning for Few-Shot Speaker Adaptive TTS** | ~2022-2023 | Meta-learning (MAML) para adaptacao few-shot de speaker | Speaker similarity, MOS | Multi-speaker TTS | **MEDIA** |
| 14 | **EmoShift: Lightweight Activation Steering for Enhanced Emotion-Aware Speech Synthesis** | ~2025 | Activation steering lightweight para emocao | Emotion control, speaker preservation | Emotional TTS | **MEDIA** |

### 2.2 Taxonomia de Metodos de Disentanglement S/A

A literatura revela **quatro familias** principais:

**Familia 1: Adversarial Disentanglement**
- Papers: DART, Explicit Intensity Control, Multi-Scale Accent Modeling
- Mecanismo: Classificador adversarial tenta prever speaker a partir de A (ou accent a partir de S). Loss adversarial penaliza quando o classificador consegue.
- Formulacao tipica: `L_total = L_reconstruction + lambda * L_adversarial`
- Implicacao: Abordagem mais fundamentada. DART faz exatamente `Audio = TTS(text, S, A)` com adversarial loss.

**Familia 2: Information Bottleneck / Embedding Space Design**
- Papers: Multi-Scale Accent Modeling, Investigating Speaker Embedding Disentanglement
- Mecanismo: Projeta espacos de embeddings para S e A ocuparem subespacos ortogonais ou minimamente sobrepostos.
- Implicacao: Reforca necessidade do Stage 1.5 e sugere possivel loss auxiliar alem do LoRA puro.

**Familia 3: Data Augmentation para Desacoplamento**
- Papers: MacST, accent conversion many-to-many
- Mecanismo: Cria pares sinteticos (mesmo speaker, accents diferentes) via transliteracao.
- Implicacao: CORAA-MUPE nao tem pares (speaker, accent) — limitacao fundamental.

**Familia 4: Activation Steering (Training-Free)**
- Papers: EmoSteer-TTS, EmoShift, Segment-Aware Conditioning
- Mecanismo: Extrai vetores de direcao das ativacoes internas (difference-in-means) e aplica steering na inferencia sem retreinar.
- Implicacao: Alternativa ao LoRA. Se Qwen3-TTS codifica accent nas ativacoes (Stage 1.5 deve verificar), activation steering pode funcionar sem treino.

### 2.3 LoRA em TTS — Estado Atual

**Lacuna confirmada:** nao ha papers publicados usando LoRA especificamente para accent control em TTS.

O que existe:
- LoRA em ASR para accents (Mixture of LoRA Experts)
- Speaker fine-tuning lightweight no Qwen3-TTS (secao 4.2.5 do tech report)
- LoRA em LLMs de texto (vasta literatura, mas nao TTS)

**Candidatos a LoRA target modules no Qwen3-TTS:**
- `q_proj, k_proj, v_proj, o_proj` das self-attention layers (classicos)
- `gate_proj, up_proj, down_proj` dos MLP layers
- NAO no tokenizer/detokenizer (frozen)
- NAO no speaker encoder (frozen — preservar identidade)
- Decisao deve ser data-driven via probing layer-wise do Stage 1.5

### 2.4 Gaps e Recomendacoes (Rafael)

| Gap | O que a literatura faz | O que nos planejamos | Risco |
|-----|------------------------|---------------------|-------|
| Adversarial loss | DART, Explicit Intensity Control | LoRA sem adversarial | **ALTO** |
| Pares sinteticos | MacST cria pares (speaker, accent-diferente) | CORAA-MUPE: 1 accent/speaker | **MEDIO** |
| Multi-scale accent | Fonetico + prosodico | Embedding global (1 LoRA/regiao) | **MEDIO** |
| Controle de intensidade | Ajuste de forca do accent | Binario (on/off) | **BAIXO** |
| Activation steering | EmoSteer-TTS (training-free) | Nao planejado | **MEDIO** |

**Recomendacoes:**
1. Adicionar ablation de activation steering ao Stage 2
2. Considerar adversarial loss no treinamento de LoRA como ablation
3. Definir LoRA target modules data-driven baseado em probing do Stage 1.5
4. Documentar limitacao "um accent por speaker" formalmente

---

## 3. Relatorio Felipe Nakamura — Metricas e Protocolos de Validacao

### 3.1 Como a Comunidade Mede Accent Controllability

**Classificador externo de sotaque:** padrao dominante. Treina em audio real, aplica em audio gerado.
- Features tipicas: log-mel 80-dim, MFCCs, ou embeddings SSL (WavLM, HuBERT)
- Modelos: CNN (ResNet, TDNN), Conformer, ou fine-tuned SSL
- Metrica: accuracy ou F1-macro
- Observacao critica: maioria usa accuracy simples (nao balanced). Nosso uso de balanced accuracy e mais rigoroso.

**Accent Identification Rate (AIR):** percentual de amostras geradas identificadas no accent correto (essencialmente accuracy do classificador).

**Accent Similarity MOS (AMOS):** avaliacao subjetiva 1-5 por painel humano. Fora do nosso escopo tecnico.

### 3.2 Como a Comunidade Mede Speaker Preservation

**Cosine similarity de embeddings (metodo dominante):**

| Modelo | Dimensao | Adocao |
|--------|----------|--------|
| ECAPA-TDNN (SpeechBrain) | 192 | Alta, crescente (padrao 2023+) |
| d-vector / GE2E | 256 | Media, declinante |
| x-vector | 512 | Media (baseline classico) |
| WavLM-TDNN | 768+ | Crescente |

Nosso protocolo especifica ECAPA-TDNN ou x-vector — alinhado com estado da arte.

**Thresholds na literatura:**
- Intra-speaker baseline: 0.70-0.85 (ECAPA-TDNN)
- Criterio "queda aceitavel": 5-10% informalmente aceito
- Nao ha threshold universal

### 3.3 Protocolos de Disentanglement/Leakage na Literatura

| Paper | Probing Method | Chance Comparison |
|-------|---------------|-------------------|
| Pasad et al. 2021 | Linear probe (logistic regression) | Chance level + margem |
| SUPERB (Yang et al. 2021) | Linear probe sobre frozen SSL | Baselines por task |
| ContentVec (Qian et al. 2022) | Linear probe para speaker em content embedding | Chance = 1/N_speakers |
| Kreuk et al. 2020 | MINE (Mutual Information) + linear probes | Random MI baseline |

Nosso uso de logistic regression como probe linear esta alinhado. Threshold chance + 5 p.p. e conservador (bom).

### 3.4 Gap Analysis: Nosso Protocolo vs Estado da Arte

**O que fazemos bem (melhor que a media):**
- Balanced accuracy obrigatorio
- CI 95% para todas as metricas
- Speaker-disjoint splits obrigatorios
- Leakage probes em ambas as direcoes
- Stage 1.5 (probing pre-treinamento) — inovador

**10 Lacunas identificadas:**

| # | Lacuna | Severidade | Descricao |
|---|--------|-----------|-----------|
| L1 | Sem metrica automatica de qualidade de fala | ALTA | Protocolo nao tem proxy para MOS (UTMOS, WER) |
| L2 | Classificador de accent nao especificado | ALTA | Nao define arquitetura, features, validacao |
| L3 | Inconsistencia threshold speaker sim | MEDIA | STAGE_1.md: 5% vs TECHNICAL_VALIDATION_PROTOCOL.md: 10% |
| L4 | Sem baseline intra/inter speaker sim formalizado | MEDIA | Nao mede referencia antes de avaliar audio gerado |
| L5 | Sem swap test formalizado | MEDIA | Teste direto S fixo, A variavel nao esta no protocolo |
| L6 | Sem metrica de colapso/diversidade | MEDIA | Se modelo gera sempre o mesmo audio, metricas nao detectam |
| L7 | Sem WER/inteligibilidade | MEDIA | Audio gerado pode ser ininteligivel |
| L8 | Probe com multiplos C nao formalizado | BAIXA | Nao varia regularizacao do probe |
| L9 | Sem Mutual Information | BAIXA | MI e mais geral que probe linear |
| L10 | Confusion matrix sem hipoteses linguisticas | BAIXA | Nao documenta confusoes esperadas |

### 3.5 Recomendacoes (Felipe)

**OBRIGATORIAS (antes do Stage 2):**

**R1. Adicionar metrica de qualidade de fala automatica:**
```
- UTMOS >= 3.0 (predictor neural de MOS) como sanity check
- WER via Whisper-large-v3: WER gerado <= WER baseline + 10 p.p.
- Se WER > 50%: audio ininteligivel (FAIL)
```

**R2. Especificar classificador de accent:**
```
- Features: embeddings WavLM-large (layer 12, pooled) OU log-mel 80-dim
- Modelo: Logistic Regression ou TDNN leve
- Treinamento: audio REAL do CORAA-MUPE, split speaker-disjoint
- Validacao: balanced accuracy no test set real (se < 50%, classificador fraco demais)
```

**RECOMENDADAS:**

**R3.** Unificar threshold speaker similarity: 10% piloto (Stage 2), 5% target (Stage 3)

**R4.** Formalizar baseline de speaker similarity (intra/inter no dataset real)

**R5.** Formalizar swap test:
```
Para cada speaker S e par de accents (A1, A2):
1. Gerar: audio_1 = TTS(texto, S, A1) e audio_2 = TTS(texto, S, A2)
2. Accent: classifier(audio_1) == A1; classifier(audio_2) == A2
3. Speaker: cosine_sim(audio_1, audio_2) >= threshold_intra_speaker
```

**R6.** Adicionar check de colapso/diversidade

**OPCIONAIS:** R7 (probe com multiplos C), R8 (FAD), R9 (SV-EER)

### 3.6 Avaliacao de Thresholds

| Metrica | Nosso Threshold | Literatura | Avaliacao |
|---------|----------------|-----------|-----------|
| Accent balanced accuracy | F1 >= 0.55 GO (Stage 1.5) | 60-85% para dialects bem separados (ingles). 50-65% para macro-regioes pt-BR | 0.55 razoavel. 0.70 (STAGE_1.md) agressivo. |
| Speaker sim queda | < 10% (protocolo) / < 5% (STAGE_1) | 5-10% informalmente aceito | 10% piloto razoavel. 5% target ambicioso mas defensavel. |
| Leakage probe | <= chance + 5 p.p. | Chance + 5-10 p.p. na literatura | 5 p.p. conservador (bom). Stage 1.5 usa 7 p.p. GO / 12 p.p. GO_CONDITIONAL. |
| UTMOS (proposto) | >= 3.0 | 3.5-4.0 para TTS alta qualidade | 3.0 como sanity check adequado. |

---

## 4. Relatorio Mariana Alves — Datasets e Confounds

### 4.1 Tabela de Datasets Multi-Accent

| Dataset | Lingua | #Speakers | #Accents | #Horas | Tipo de Label |
|---------|--------|-----------|----------|--------|---------------|
| L2-ARCTIC | Ingles (L2) | 24 | 6 L1 backgrounds | ~26h | L1 do falante |
| VCTK | Ingles | 110 | ~12 accents | ~44h | Curado/auto-declarado |
| Common Voice | 100+ linguas | 100k+ | Auto-declarado (opcional) | 30k+ | Auto-declarado (ruidoso) |
| AESRC2020 | Ingles (L2) | 160 | 8 L1 accents | ~160h | L1 nationality |
| CORAA-MUPE | pt-BR | ~289 entrevistados | 5 macro-regioes (via birth_state) | ~365h | Geografico |
| CMU ARCTIC | Ingles | 7 | 4 accents | ~7h | Declarado |
| GLOBE | Ingles | 23k+ | 164 accents | ~450h | Auto-declarado |
| CETUC | pt-BR | 101 | Nenhum | ~144h | Nenhum |

**Observacao critica:** A grande maioria dos datasets multi-accent sao em ingles. Para pt-BR, o cenario e drasticamente mais escasso.

### 4.2 Proxy birth_state -> Regiao IBGE — Veredicto

**Aceitavel com ressalvas documentadas:**
- Melhor proxy disponivel dado o CORAA-MUPE
- Agregacao em 5 macro-regioes reduz ruido (suaviza variacoes intra-regionais)
- Problemas documentados: migracao, classe social, exposicao mediatica
- Recomendacao: documentar como proxy (nao ground truth), fazer analise de sensibilidade

### 4.3 Confounds Criticos

| Confound | Severidade | Mitigacao |
|----------|-----------|-----------|
| Accent x Gender | **ALTA** | Tabela cruzada obrigatoria. Se >70% de um genero numa regiao, subsampling ou documentar. |
| Accent x Topic/Content | **ALTA** | Text-disjoint evaluation (ja implementado). Textos controlados na geracao. |
| Accent x Duration | **MEDIA** | Filtro 3-15s mitiga parcialmente. Verificar com histograma por regiao. |
| Accent x Recording conditions | **MEDIA** | Corpus unico (CORAA-MUPE) reduz risco. Loudness normalization. |
| Speaker x Accent (entanglement) | **ALTA** | Speaker-disjoint splits + leakage probes. Min 8 speakers/regiao. |

### 4.4 CORAA-MUPE vs Datasets da Literatura

**Pontos fortes:**
- Speaker ID disponivel (speaker_code)
- Licenca favoravel (CC BY-SA 4.0)
- Tamanho adequado (~365h vs L2-ARCTIC 26h, VCTK 44h)
- Corpus unico (reduz confound de canal)
- Gender disponivel (permite analise de confound)

**Limitacoes:**
- Fala espontanea (sem controle de texto) → confound accent x topic
- Proxy geografico (impreciso para migrantes)
- Macro-regioes amplas (SE inclui SP e MG, que sao diferentes)
- Desbalanceamento regional provavel (SE maioria)
- Sem texto controlado (impossivel "mesmo texto, speakers diferentes")

### 4.5 Checklist de Auditoria Pre-Treinamento (Mariana)

```
Metadata
- [ ] speaker_code presente e unico para cada entrevistado
- [ ] birth_state presente para todos os speakers usados
- [ ] gender presente para todos os speakers usados
- [ ] accent (macro-regiao) derivado corretamente de birth_state
- [ ] Cada speaker_code mapeia para exatamente 1 accent label
- [ ] duration de cada segmento dentro de [3, 15] segundos

Distribuicoes
- [ ] Tabela: amostras por regiao (absoluto e %)
- [ ] Tabela: speakers por regiao
- [ ] Tabela cruzada: accent x gender (proporcoes)
- [ ] Histograma: duracao por regiao
- [ ] Identificar regiao com menor representacao

Splits
- [ ] Speaker-disjoint verificado com assertion
- [ ] Distribuicao de regioes proporcional em cada split
- [ ] Distribuicao de genero proporcional em cada split
- [ ] Seed do split documentada no config YAML
- [ ] Split salvo como artefato versionado

Confounds
- [ ] Correlacao accent x gender: Chi-squared test + proporcoes
- [ ] Correlacao accent x duration: ANOVA ou Kruskal-Wallis
- [ ] Confounds mitigados OU documentados como limitacao

Versionamento
- [ ] Hash SHA-256 do manifest.jsonl registrado
- [ ] Hash SHA-256 do conjunto de audio files registrado
- [ ] Script de preprocessing reproduzivel (seed + config -> mesmo output)
```

### 4.6 Recomendacoes (Mariana)

1. **NAO misturar** CORAA-MUPE com outros corpora para treinamento (risco de confound corpus-origin x accent)
2. **Loudness normalization** se nao implementada (pyloudnorm, -23 LUFS)
3. **Datasets complementares** apenas para validacao: CETUC (textos controlados), Common Voice pt (validacao cruzada — Stage 3)
4. Documentar 5 limitacoes formalmente no TCC

---

## 5. Riscos Convergentes — Priorizados

| # | Risco | Quem Levantou | Severidade | Mitigacao |
|---|-------|---------------|------------|-----------|
| 1 | **Sem adversarial loss -> LoRA codifica speaker junto com accent** | Rafael | ALTA | Adversarial loss como ablation no Stage 2; leakage probes detectam post-hoc |
| 2 | **Confound accent x gender no CORAA-MUPE** | Mariana | ALTA | Tabela cruzada accent x gender ANTES de treinar |
| 3 | **Sem metrica de qualidade de fala** | Felipe | ALTA | Adicionar UTMOS >= 3.0 e WER via Whisper ao protocolo |
| 4 | **Um accent por speaker -> confound estrutural** | Rafael + Mariana | MEDIA-ALTA | Aceitar como limitacao, rigor nos probes, confusion matrix |
| 5 | **Classificador de accent indefinido** | Felipe | MEDIA-ALTA | Especificar antes do Stage 2 |
| 6 | **Qwen3-TTS sem precedente de LoRA** | Rafael | MEDIA | Stage 1.5 informa viabilidade; activation steering como fallback |

---

## 6. Plano de Acao

### FASE IMEDIATA (Antes de qualquer coisa)

| # | Acao | Responsavel | Bloqueante? |
|---|------|-------------|-------------|
| A1 | Verificar confound accent x gender no CORAA-MUPE filtrado | Mariana | SIM |
| A2 | Contar speakers por regiao apos filtros (minimo 8/regiao) | Mariana | SIM |
| A3 | Resolver inconsistencia threshold speaker sim (5% vs 10%) | Felipe | SIM |
| A4 | Documentar limitacao "um accent por speaker" formalmente | Rafael | NAO |

### ANTES DO STAGE 2 — Atualizar Protocolo

| # | Acao | Responsavel | Prioridade |
|---|------|-------------|------------|
| B1 | Adicionar UTMOS + WER via Whisper ao TECHNICAL_VALIDATION_PROTOCOL.md | Felipe | ALTA |
| B2 | Especificar classificador de accent (arquitetura, features, validacao) | Felipe | ALTA |
| B3 | Formalizar baseline intra/inter speaker similarity | Felipe | MEDIA |
| B4 | Formalizar swap test (S fixo, A variavel) | Felipe + Rafael | MEDIA |
| B5 | Adicionar check de colapso/diversidade ao protocolo | Felipe | MEDIA |
| B6 | Definir LoRA target modules data-driven baseado em probing do Stage 1.5 | Rafael | ALTA |

### STAGE 2 — Experimentos Adicionais ao Roadmap

| # | Experimento | Responsavel | Justificativa |
|---|-------------|-------------|---------------|
| C1 | Ablation: LoRA com vs sem adversarial loss | Rafael | Literatura unanime: adversarial melhora disentanglement |
| C2 | Ablation: activation steering (EmoSteer-style, training-free) | Rafael | Alternativa de custo zero; fallback se LoRA falhar |
| C3 | Ablation: LoRA rank (8, 16, 32) | Rafael | Necessario sem precedente na literatura |

---

## 7. Papers para Citacao Obrigatoria no TCC

| Paper | Relevancia |
|-------|------------|
| **DART** (Disentanglement of Accent and Speaker in TTS) | Formulacao S/A mais proxima da nossa |
| **Multi-Scale Accent Modeling and Disentangling** | Multi-escala accent + disentanglement |
| **Explicit Intensity Control for Accented TTS** | Adversarial loss para accent |
| **Qwen3-TTS Technical Report** (Qwen Team, 2026) | Nosso backbone |
| **LoRA** (Hu et al., 2022) | Fundamentacao do metodo de adaptacao |
| **EmoSteer-TTS** (Xie et al., 2025) | Activation steering como paradigma alternativo |
| **Investigating Speaker Embedding Disentanglement** | Fundamenta nosso probing protocol |
| **Mixture of LoRA Experts for Accented ASR** | LoRA por accent (ASR, mas relevante) |

---

## 8. Condicoes de Veto (Gates)

| Persona | Veta avanco ao Stage 2 se: |
|---------|---------------------------|
| **Rafael** | (1) Stage 1.5 nao concluido, (2) sem decisao sobre adversarial loss, (3) LoRA target modules nao definidos por probing |
| **Felipe** | (1) Sem UTMOS no protocolo, (2) classificador de accent nao especificado, (3) baseline speaker sim nao formalizado |
| **Mariana** | (1) Confound accent x gender nao verificado, (2) speakers/regiao < 8 sem mitigacao |

---

## 9. Conclusao

Os tres participantes convergem: o projeto esta numa **lacuna real e publicavel** da literatura. O plano e solido mas tem gaps concretos que devem ser resolvidos antes do Stage 2.

**Tres achados que mudam o jogo:**

1. **A lacuna e real** — nenhum paper usa LoRA para accent control em TTS. Qualquer resultado (positivo ou negativo) e contribuicao.
2. **Adversarial loss e o elefante na sala** — literatura converge que sem adversarial, disentanglement S/A e fragil. Adicionar como ablation e o caminho pragmatico.
3. **Activation steering e um fallback elegante** — se LoRA falhar, EmoSteer-style steering e barato e pode demonstrar controle sem treinamento. Dois metodos no arsenal protegem o TCC.

---

*Ata gerada automaticamente a partir dos relatorios individuais dos participantes.*
*Proxima reuniao: apos conclusao do Stage 1.5 (probing de separabilidade latente).*
