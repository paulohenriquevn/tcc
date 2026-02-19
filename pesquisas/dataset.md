# Datasets de fala em português brasileiro para construir um corpus científico controlado por sotaque

## Resumo executivo

Este relatório mapeia e avalia datasets públicos (ou publicamente acessíveis) de fala em português que contêm (ou podem ser filtrados para) português brasileiro, com foco na viabilidade de montar um **dataset controlado por sotaque** para experimento científico. O alvo mínimo que você definiu é: **3 sotaques × 8 falantes × 30 textos** (24 falantes, 720 gravações).  

Os achados centrais são:

1) **Há poucos datasets que simultaneamente**: (i) tenham **múltiplos falantes**, (ii) tragam **metadados regionais confiáveis** (estado/cidade/dialeto) e (iii) ofereçam **cobertura “mesmo texto” por falante** (todos os 24 falantes lendo os mesmos 30 textos). Em geral, datasets “grandes” e fáceis de baixar (audiolivros, fala espontânea) falham em “mesmo texto”; datasets “de benchmark” repetem textos, mas quase sempre não têm região. citeturn14view0turn10view0turn16search16turn19view0turn32search0  

2) Entre os recursos investigados, o melhor candidato **para adaptação direta** (tentando chegar ao mínimo científico sem gravar nada novo) é o dataset **Sidney (SID)**: ele foi descrito como contendo **72 falantes** e campos como **local de nascimento** (forte proxy geográfico), além de mais de **5.700 enunciados** (provável folga para selecionar os 30 textos comuns). O ponto fraco é que ele é “for research purposes” e o licenciamento/termos não aparecem claros nas fontes públicas resumidas; isso afeta a possibilidade de redistribuição do seu “dataset pronto”. citeturn14view0turn18search1  

3) O melhor recurso **para metadados regionais explícitos** em escala é o **MuPe Life Stories (CORAA-MUPE-ASR)**: ele inclui **birth_state** (estado de nascimento) e identificadores de falante (speaker_code). Porém, o conteúdo é de **fala espontânea (entrevistas)**; logo, “mesmo texto” entre falantes não é garantido (alto risco de confundir sotaque com conteúdo/tópico). A licença, por outro lado, é mais amigável para pesquisa aberta (**CC BY-SA 4.0**). citeturn16search16turn16search0turn16search6  

4) O **CORAA ASR v1.1** é excelente por já trazer um campo **accent** com categorias regionais (p.ex., Recife, Minas Gerais, São Paulo etc.), mas sua licença divulgada é **CC BY-NC-ND 4.0** (mais restritiva; “ND” complica redistribuição de versões adaptadas) e o README público não explicita um campo de *speaker_id*, o que dificulta garantir “8 falantes por sotaque” sem engenharia extra (heurísticas via file_path). citeturn10view0turn9search7turn9search3  

5) **Mozilla Common Voice (scripted)** é enorme e tem um campo **accent** (auto-declarado), mas, para o seu desenho, ele é mais útil como **complemento** (pré-treino/robustez/checagens) do que como fonte primária de “mesmo texto” por falante. Além disso, a distribuição atual via **Mozilla Data Collective** impõe regras contratuais relevantes: é **proibido tentar identificar falantes** e também é **proibido re-hospedar/recompartilhar** o dataset. Isso impacta diretamente workflows que publicam um “dataset derivado” pronto para download. citeturn24view0turn23search4turn23search18  

6) Recomendação prática: adotar um plano em dois trilhos.  
- **Trilho A (adaptação com dados públicos existentes forçando o mínimo científico)**: tentar montar a matriz 3×8×30 a partir do **SID** (primeira tentativa) e/ou uma combinação CORAA/VoxForge, validando por script se existe interseção de 30 textos comuns.  
- **Trilho B (mínima coleta nova para garantir rigor)**: usar datasets públicos (CETUC/LapsBM/Common Voice) apenas como **fonte de prompts e baseline**, e gravar **24 falantes** seguindo um protocolo uniforme (mesmo microfone/ambiente/prompt), porque isso elimina o maior risco de “leakage” (confusão por canal/dataset). citeturn14view0turn24view0turn18search13  

## Metodologia e critérios de adequação

A pesquisa priorizou fontes primárias (páginas oficiais, repositórios e “dataset cards”) do entity["organization","Mozilla Foundation","nonprofit"], do entity["organization","Mozilla Data Collective","dataset platform"], do entity["organization","OpenSLR","speech resources portal"], além de repositórios e distribuição via entity["company","Hugging Face","ml platform"] e entity["organization","GitHub","code hosting platform"]. citeturn24view0turn19view0turn20view0turn10view0turn16search16  

O critério central foi “**suporta um experimento controlado por sotaque**”, decompondo em:

- **Identidade de falante**: existe *speaker_id* confiável? sem isso, não dá para garantir “8 falantes por sotaque”. citeturn16search2turn16search16turn21view3  
- **Metadado regional/varietal**: há campo de sotaque/estado/cidade/dialeto? é auto-declarado ou curado? citeturn10view0turn24view0turn16search16turn32search8  
- **Cobertura de “mesmo texto”**:  
  - ideal: os mesmos 30 textos existem para todos os 24 falantes;  
  - aceitável (com ressalvas): existe um conjunto grande de textos repetidos e você consegue selecionar por interseção (completude) via script. citeturn14view0turn32search1turn24view0  
- **Risco de leakage** (vazamento de sinal): sotaques correlacionados com microfone, ruído, domínio (TEDx vs entrevista), compressão, demonstrate. Exemplos típicos: misturar corpora diferentes para sotaques diferentes (você mede “dataset” e não “sotaque”). citeturn10view0turn18search13turn14view0  
- **Licença e “re-hosting”**: CC BY-NC-ND e GPL têm implicações fortes; no caso do Data Collective, há regras explícitas contra re-hospedagem e contra tentativa de identificação. citeturn9search7turn24view0turn32search0turn32search2turn19view0  

## Catálogo analítico de datasets candidatos

A tabela abaixo cataloga os principais candidatos que apareceram na investigação (incluindo os obrigatórios do seu pedido), destacando o que é **explicitamente documentado** e o que fica **não especificado**.

| Dataset (PT-BR ou filtrável) | Mantenedor / onde baixar | Licença (dataset) | Tamanho (horas / clips) | Falantes | Metadados regionais | Áudio (formato / taxa) | Alinhamento / segmentação |
|---|---|---|---|---|---|---|---|
| Common Voice (scripted; “pt” inclui PT-BR/pt-PT via *accent*) | Mozilla Data Collective (MCV Scripted Speech 24.0) | CC0-1.0, com regras: proibido identificar falantes e proibido re-hospedar | Estatística específica de “pt” não obtida aqui; a página de cada idioma traz “clips/hours/speakers” | idem | campo `accent` (auto-declarado; pode estar ausente) | MP3; TSV com `client_id`, `path`, `text`, votos e demografia (opt-in) | Segmentado por clip; TSV por clip (sem timestamps obrigatórios) |
| CORAA ASR v1.1 | GitHub + links para GDrive/HF | CC BY-NC-ND 4.0 (padronizado no projeto) | 290,77 h; 400k+ segmentos | não especificado no README público | `accent` em 4 categorias + “miscellaneous” (curado por origem do subcorpus) | não especificado no README público (downloads separados de áudio/metadados) | Segmentos já definidos; metadados por arquivo |
| MuPe Life Stories (CORAA-MUPE-ASR) | Hugging Face (nilc-nlp) | CC BY-SA 4.0 | 365 h; 289 entrevistas; ~317k segmentos | não especificado no resumo; há `speaker_code` | `birth_state` (estado de nascimento) para entrevistado (`speaker_type=R`) | WAV 16 kHz indicado no card de exemplo (estrutura típica HF áudio); tamanho ~41,8 GB | Segmentação com `start_time`; transcrição por segmento |
| NURC‑SP Audio Corpus | GitHub + Hugging Face | CC BY-NC-ND 4.0 (conforme descrição do projeto) | 239,30 h; 170k+ segmentos | 401 falantes (reportado no paper/descrições) | `accent` fixo (“sp-city”); metadados etários/sexo | não especificado no README, mas pipeline HF típ. em WAV/16k | tem `start_time`, `end_time`, `duration`; `speaker_id` (auto, pode ter erro) |
| CETUC | Distribuição via repositórios/espelhos do grupo e/ou registro FalaBrasil | uso acadêmico em algumas descrições; licença explícita não consolidada publicamente | 144h39m (aprox.); 100.998 utt | 101 falantes | não há metadado regional explícito | WAV 16 kHz (ambiente controlado) | Sem alinhamento forçado; mas cada arquivo tem transcrição |
| LaPS Benchmark (LapsBM) | FalaBrasil / espelhos | não especificado publicamente no resumo | 0h54m; 700 utt | 35 falantes | não há metadado regional explícito | 22.05 kHz (ambiente não controlado) | Sem timestamps; por utt |
| Sidney (SID) | Espelho de download / referência acadêmica | “for research purposes”; licença/termos não explicitados na página-resumo | ~7h23m (em compilações BRSD); 5.777 utt | 72 falantes | inclui “place of birth” (local de nascimento) e outros campos | 22.05 kHz; transcrição por palavra; sem alinhamento temporal | Sem time alignment; por utt |
| VoxForge (pt-BR) | Site VoxForge (português e página principal) | GPL (áudios submetidos sob GPL) | ~4h14m em compilações; 4.130 utt | ≥111 (varia; metadados incompletos) | em alguns pacotes há “Pronunciation dialect” (ex.: PT_BR) | taxa variável 16–44.1 kHz; heterogêneo | Sem timestamps; por utt |
| Multilingual LibriSpeech (MLS) – subset “Portuguese” | OpenSLR SLR94 + Hugging Face | CC BY 4.0 | “~168 h” em revisões; página OpenSLR não lista horas por idioma | speaker_id existe; #speakers não consolidado aqui | não há metadado regional; depende do narrador (LibriVox) | Original: FLAC / Comprimido: OPUS; exemplos decodificados a 16 kHz | tem `begin_time/end_time` e segmentação por utt |
| FLEURS (pt_br) | Hugging Face (google/fleurs) | CC‑BY (dataset) | ~12 h por idioma (aprox.) | não inclui speaker_id no card atual | não há metadado regional de PT-BR | WAV/16 kHz; transcrições normalizadas | Segmentado por utt; sem speaker_id; sem timestamps por padrão |

Fontes primárias consultadas para os campos acima incluem: páginas oficiais de Common Voice no Data Collective (estrutura/fields/regras/licença), anúncio oficial de release, OpenSLR para MLS, repositórios oficiais de CORAA/MuPe/NURC-SP, documentação do VoxForge e listas técnicas brasileiras (FalaBrasil/IgorMQ). citeturn24view0turn23search4turn19view0turn10view0turn16search16turn16search2turn32search0turn14view0turn18search13turn31search0turn31search2  

### Observações de consistência e “lacunas” deliberadas

- Para **Common Voice**, a estatística específica do idioma “pt” (horas/vozes) **não foi recuperada diretamente** neste levantamento por falta de uma página indexada facilmente para “Portuguese” nos resultados; porém, o próprio Data Collective publica uma página por idioma com esses números (como exemplificado para inglês). citeturn24view0turn23search4  
- Para **MLS**, a página OpenSLR enfatiza formato/licença/download, mas não lista horas por idioma; por isso, o número de horas aparece como aproximado via literatura de revisão. citeturn19view0turn18search1  

## Adequação ao experimento e ranking por utilidade

Esta seção responde diretamente aos itens (2) e (3) do seu pedido: **suitability** (mesmo texto, falantes por região, qualidade, leakage) e **ranking** (adaptação vs complemento vs impraticável).

### Avaliação dataset a dataset

**Sidney (SID)**  
- **Pontos fortes**: já foi descrito com campos como **local de nascimento** (um metadado geográfico diretamente acionável para “sotaque por região”) e número de falantes suficiente (72) para selecionar 3 grupos com 8 falantes, se a distribuição por regiões permitir. Além disso, com 5.777 enunciados, é plausível existir interseção de 30 textos comuns (ou pelo menos um conjunto repetido grande). citeturn14view0turn18search1  
- **Pontos fracos / riscos**: transcrição **sem alinhamento temporal** e “sem time alignment”, o que não impede seu experimento (porque você quer clips por texto), mas atrapalha algumas análises fonéticas finas; e sobretudo **licença/termos públicos pouco claros**, pois as descrições enfatizam “provided… for research purposes”. Isso costuma significar “dá para usar, mas não necessariamente redistribuir”. citeturn14view0turn18search1  
- **Leakage**: baixo a médio, dependendo se os 3 sotaques vierem do mesmo protocolo de gravação (parece que sim) — o canal tende a ser mais homogêneo do que misturar múltiplos corpora.

**CORAA ASR v1.1**  
- **Pontos fortes**: tem um campo `accent` com categorias regionais explícitas (Minas Gerais, Recife, São Paulo etc.). Isso é raro e valioso em dataset grande e público. citeturn10view0turn9search3  
- **Pontos fracos / riscos**: a licença divulgada no projeto é **CC BY‑NC‑ND 4.0** (o “ND” é problemático para publicar qualquer versão adaptada/derivada), e o README público não documenta *speaker_id* — o que pode exigir heurística em `file_path` e gera risco científico (erro de contagem de falantes, falas multiespeaker por segmento, etc.). citeturn10view0turn9search7turn9search3  
- **Mesmo texto?** Baixa probabilidade: é composto por múltiplos subcorpora (incluindo fala espontânea e TEDx), então os textos tendem a variar muito entre falantes. citeturn10view0turn9search3  
- **Leakage**: médio, porque “accent” pode estar correlacionado ao subcorpus de origem (p.ex., NURC-Recife vs C-ORAL). citeturn9search3turn10view0  

**MuPe Life Stories (CORAA-MUPE-ASR)**  
- **Pontos fortes**: traz `birth_state` e `speaker_code`, além de muitos segmentos. Isso torna viável formar grupos por estado/região com 8 falantes cada (em tese). citeturn16search16turn16search0  
- **Mesmo texto?** Praticamente não: entrevistas de história de vida quase nunca repetem os mesmos 30 textos em múltiplos falantes (a não ser que você re-segmente para frases muito curtas e aceite “quase mesmo texto”, o que raramente é defensável). citeturn16search0turn16search16  
- **Leakage**: alto para o seu desenho específico, porque tópicos, vocabulário, emoção e interlocução variam por falante e podem variar por região (efeito cultural).  
- **Melhor uso**: complemento para treinar/validar modelos robustos e avaliar viés por grupo; não como núcleo do seu dataset de textos controlados. citeturn16search0turn16search16  

**NURC‑SP Audio Corpus**  
- **Pontos fortes**: escala grande (239,30 h) e muitos falantes (401), com metadados úteis (idade/sexo/qualidade, e `speaker_id` atribuído automaticamente). citeturn16search2turn16search4turn16search10  
- **Pontos fracos**: `accent` é basicamente um só (“sp-city”), então não resolve 3 sotaques; “speaker_id” é automático e pode ter erro. citeturn16search2turn16search10  
- **Melhor uso**: “um sotaque fixo” (controle) e/ou complemento para modelos.  

**CETUC / LapsBM**  
- **CETUC** é o melhor “read speech repetido” em escala (1000 sentenças por falante; 101 falantes; 16 kHz, ambiente controlado). Isso é ótimo para o componente “mesmo texto”, mas o metadado regional não aparece no resumo público. citeturn14view0turn18search13turn32search6  
- **LapsBM** tem poucos minutos e 20 frases por falante; não chega em 30 textos por falante e também não traz região explicitamente. citeturn14view0turn18search13turn9search6  
- **Melhor uso**: CETUC como fonte de prompts e de controle de texto; LapsBM como sanity check/benchmark minúsculo.  

**VoxForge (pt-BR)**  
- **Pontos fortes**: “dataset comunitário” com muitos falantes e, em diversos pacotes, aparece um campo de “pronunciation dialect/PT_BR” (embora não seja padronizado). A licença GPL é explicitamente assumida para os áudios. citeturn32search0turn32search8turn32search2  
- **Pontos fracos**: extremamente heterogêneo em taxa de amostragem e qualidade, com metadados incompletos e ruído. citeturn14view0turn32search0  
- **Leakage**: médio a alto se você misturar “dialetos” que, na prática, vêm de padrões de gravação diferentes (microfone, ambiente).  
- **Licença**: GPL pode “contaminar” compatibilidade e distribuição quando combinado com outros datasets; exige atenção jurídica. citeturn32search2turn32search0  

**MLS (Portuguese) e outros audiolivros (LibriVox)**  
- **Pontos fortes**: escala e clareza de distribuição/licença (CC BY 4.0) e formato padronizado. citeturn19view0turn20view0turn21view3  
- **Pontos fracos para seu desenho**: narradores leem textos diferentes; não há metadado regional padronizado. Serve como complemento, não como núcleo. citeturn19view0turn18search1  

**FLEURS (pt_br)**  
- **Ponto forte**: áudio 16 kHz e licença CC‑BY; e o dataset é construído com sentenças paralelas (bom para montar um conjunto de 30 prompts “equilibrados”). citeturn31search0turn31search2  
- **Ponto fraco**: não expõe *speaker_id* no card atual, o que inviabiliza controle de falantes (portanto, não atende o seu desenho). citeturn31search0turn31search14  

### Ranking sintético

| Categoria | Datasets | Justificativa resumida |
|---|---|---|
| Melhor para adaptação direta ao seu mínimo (3×8×30), com forte chance de funcionar | SID | Tem falantes suficientes e metadados de “place of birth”; provável margem para selecionar 30 textos comuns; porém licença/termos precisam verificação fina. citeturn14view0turn18search1 |
| Úteis como fonte de sotaque/metadata, mas difíceis para “mesmo texto” | CORAA ASR; MuPe; NURC‑SP | Têm `accent`/região explícitos (ou `birth_state`) e grande escala, porém falham em “mesmo texto” ou têm sotaque único; CORAA/NURC‑SP têm CC BY-NC-ND. citeturn10view0turn16search16turn16search2turn16search11 |
| Complemento valioso (pré-treino, robustez, prompts), mas não atende o desenho central | Common Voice; MLS; VoxForge; CETUC; LapsBM; FLEURS | Grandes e/ou padronizados, mas sem garantia de 30 textos comuns por falante + metadados regionais confiáveis (ou faltam speaker_id). CETUC é excelente para “mesmo texto”, mas sem região explícita nos resumos. citeturn24view0turn19view0turn32search0turn18search13turn14view0turn31search0 |
| Pouco úteis para sotaque controlado (mas podem ser úteis em outras tarefas) | datasets single-speaker TTS (ex.: Cadu 1.0) | Um falante → não permite 3×8, embora sejam bons para TTS. citeturn22search0 |

## Playbooks de extração e construção do seu `metadata.csv` + `wav/`

Abaixo, um guia operacional para **Colab/Linux**, com foco em (4): download, filtros, construção de estrutura de dados, e “exemplo de consultas/comandos”.

### Estrutura recomendada do corpus final

Recomendação de estrutura “neutra” (fácil de usar em pipelines científicos):

- `wav/<speaker_id>/<text_id>.wav`  
- `metadata.csv` (UTF‑8) com colunas mínimas:
  - `relpath` (ex.: `wav/spk001/txt005.wav`)
  - `speaker_id`
  - `accent` (label controlado, ex.: `recife`, `minas_gerais`, `sp_capital`)
  - `text_id`
  - `text`
  - `split` (`train/dev/test` ou `analysis`)
  - `duration_sec`
  - `source_dataset` (para auditoria)

Isso permite: (i) assegurar que todo falante tem 30 textos, (ii) medir “text leakage” e (iii) rastrear origem/licença.

### Common Voice (Mozilla Data Collective) – download e filtragem por sotaque

1) **Baixar do Data Collective**: o próprio Common Voice anunciou que os releases atuais (ex.: Scripted Speech 24.0) estão no Data Collective e são distribuídos como `.tar.gz` por idioma. citeturn23search4turn23search18  
2) **Entender campos**: em cada TSV, há `client_id`, `path`, `text`, votos e (se opt-in) `age`, `gender`, `accent`. citeturn24view0  
3) **Restrições contratuais**: é explicitamente proibido tentar determinar identidade de falantes e proibido re-hospedar/recompartilhar o dataset. Isso deve ser refletido na sua decisão de publicar (ou não) um “dataset derivado”. citeturn24view0  

Exemplo de parsing e filtro (após baixar e descompactar o tar do idioma):

```python
import pandas as pd
from pathlib import Path

root = Path("common_voice_pt")  # onde você descompactou
tsv = pd.read_csv(root/"validated.tsv", sep="\t")
# Campos típicos: client_id, path, text, up_votes, down_votes, age, gender, accent, segment, ...
tsv["accent"] = tsv.get("accent", "").fillna("").str.lower()

# Heurística: mapear strings livres para seus 3 sotaques-alvo
def map_accent(s):
    s = s.strip()
    if any(k in s for k in ["recife", "pernambuco", "nordeste"]):
        return "recife"
    if any(k in s for k in ["minas", "mineiro", "mg"]):
        return "minas_gerais"
    if any(k in s for k in ["sao paulo", "paulist", "sp"]):
        return "sp_capital"
    return None

tsv["accent_label"] = tsv["accent"].map(map_accent)
tsv = tsv[tsv["accent_label"].notna()]

# Próximo passo: selecionar falantes e textos com interseção (ver seção Plano priorizado)
```

Criticamente, para **Common Voice** o gargalo costuma ser “**interseção de 30 textos por falante**”, porque as frases são amostradas ao longo do tempo (cada falante pode ter gravado textos diferentes). Por isso, Common Voice tende a servir melhor como **complemento** do que como base do “3×8×30”. citeturn24view0turn8search25  

### CORAA ASR – download e uso do campo `accent`

O repositório documenta explicitamente um campo `accent` com 4 sotaques + `miscellaneous`, além de `dataset` e `variety`. citeturn10view0  

- Downloads oficiais estão listados como “Train/Dev/Test audios” e “Train/Dev/Test transcriptions and metadata”. citeturn10view0  
- A licença é reportada como CC BY‑NC‑ND 4.0 em fontes do projeto/paper, então trate como **uso acadêmico** e **evite redistribuir derivados** sem análise jurídica. citeturn9search7turn9search3  

O uso típico é: baixar metadados CSV, filtrar `accent in {minas gerais, recife, sao paulo capital}`, e então tentar derivar `speaker_id` de `file_path` (heurística) ou aceitar “falante” como “fonte” (pior cientificamente).

### MuPe Life Stories (CORAA‑MUPE‑ASR) – filtragem por `birth_state`

O card do dataset descreve metadados incluindo: `speaker_code`, `speaker_gender`, e **`birth_state`** (estado de nascimento) para o entrevistado. citeturn16search16turn16search6  

Exemplo via Hugging Face (Colab):

```python
from datasets import load_dataset

ds = load_dataset("nilc-nlp/CORAA-MUPE-ASR", split="train", streaming=True)

# Exemplo: coletar apenas entrevistados (R) do Nordeste (lista simplificada)
nordeste = {"AL","BA","CE","MA","PB","PE","PI","RN","SE"}
def is_target(x):
    return (x["speaker_type"] == "R") and (x.get("birth_state") in nordeste)

subset = ds.filter(is_target)
```

Para o seu desenho, o MuPe é mais bem usado para construir um **classificador auxiliar de sotaque/região** ou para estudos de viés, não para “mesmo texto”. citeturn16search0turn16search16  

### NURC‑SP Audio Corpus – “um sotaque fixo” com muitos falantes

O README lista: `speaker_id` (automático), `age_range`, `sex`, `quality`, e `accent` fixo, além de `start_time/end_time/duration`. citeturn16search2turn16search10  
Isso é perfeito para construir um subconjunto controlado **dentro de um sotaque** (ex.: “SP capital”) e usar como “controle” ou como parte do seu conjunto “complemento”.

### CETUC e LapsBM via FalaBrasil (DVC)

O repositório do grupo FalaBrasil descreve um fluxo via **DVC + Google Drive** e publica estatísticas de duração, taxa de amostragem e #falantes (ex.: CETUC 144h39m, 16 kHz, 101 falantes; LapsBM 0h54m, 22.05 kHz, 35 falantes). citeturn18search13turn32search6  

Exemplo (alto nível):

```bash
pip install "dvc[gdrive]"
git clone https://github.com/falabrasil/speech-datasets
cd speech-datasets
dvc pull -r public datasets/cetuc
dvc pull -r public datasets/lapsbm
```

Depois, você normaliza tudo para WAV 16 kHz mono (se precisar) e monta seu `metadata.csv`.  

### VoxForge (pt-BR) – extração e metadados de “dialeto”

O VoxForge declara explicitamente que disponibiliza os áudios submetidos sob **licença GPL**. citeturn32search0turn32search1turn32search2  

Além disso, existem páginas de pacotes que mostram campos como “Language: PT_BR” e “Pronunciation dialect: Português do Brasil” (nem sempre preenchido; precisa scraping). citeturn32search8  

Fluxo típico:
1) baixar pacotes `.tgz/.tar.gz` por falante;  
2) extrair prompts/transcrições;  
3) inferir sotaque/dialeto quando houver metadado; caso contrário, “unknown”.

### MLS (OpenSLR SLR94) – download e parsing

A página oficial lista: licença **CC BY 4.0**, origem em audiolivros do LibriVox e disponibilidade em dois formatos (original FLAC e comprimido OPUS). citeturn19view0  

No Hugging Face, o dataset expõe `speaker_id` e exemplos decodificados com `sampling_rate: 16000`. citeturn20view0turn21view1turn21view3  

O MLS costuma ser usado para **pré-treino / robustez**, não para “mesmo texto por falante”.

### FLEURS (pt_br) – usar como fonte de prompts padronizados

O card do FLEURS descreve áudio com `sampling_rate: 16000` e licença CC‑BY (dataset), e o paper/abstract reporta aproximadamente 12h supervisionadas por idioma. citeturn31search0turn31search2  

Como o *speaker_id* não aparece no card atual, use principalmente para **selecionar textos/prompts** (as sentenças foram montadas para cobertura temática ampla) e não como fonte de falantes. citeturn31search14turn31search0  

## Esforço estimado e restrições legais

A tabela abaixo estima esforço (tempo humano típico) para **transformar cada dataset em um subconjunto “pronto”** (wav + metadata.csv), assumindo que você roda scripts em Colab e que seu objetivo é pesquisa científica.

| Dataset | Esforço técnico típico | Principais passos | Risco legal/licença |
|---|---:|---|---|
| SID | 1–2 dias (se download direto funcionar), + 0,5–1 dia para seleção 3×8×30 | baixar, parse metadata, mapear regiões, checar interseção de textos, normalizar áudio | **Médio/alto**: “research purposes” sem termos claros; redistribuição incerta citeturn14view0turn18search1 |
| CETUC | 1–3 dias (download + normalização + indexação) | baixar, normalizar 16k/mono, garantir IDs por falante, escolher 30 textos | **Médio**: licença pública não consolidada; em algumas descrições aparece “fins de pesquisa” citeturn32search6turn18search13 |
| VoxForge | 2–5 dias | scraping/baixa de pacotes, limpeza forte, padronização de SR, extração de metadados/dialeto | **Alto**: GPL; compatibilidade e redistribuição exigem cuidado citeturn32search0turn32search2 |
| CORAA ASR | 2–4 dias | baixar, filtrar `accent`, heurística de falante, balancear sentenças | **Alto**: CC BY‑NC‑ND limita derivados/reuso em dataset “adaptado” citeturn9search7turn10view0 |
| MuPe (CORAA-MUPE-ASR) | 1–3 dias | streaming + filtro por `birth_state`, seleção por falante, cortar/clipes | **Baixo/médio**: CC BY‑SA permite derivados sob SA; checar sua política de publicação citeturn16search6turn16search16 |
| NURC-SP Audio Corpus | 1–3 dias | baixar, filtrar qualidade, escolher 8 falantes e 30 segmentos | **Alto**: CC BY‑NC‑ND citeturn16search11turn16search2 |
| Common Voice (MDC) | 2–7 dias | download via MDC, normalização de `accent`, seleção por interseção, auditoria | **Alto (operacional)**: proibido re-host/re-share; sem publicação de derivados; opt-in demografia citeturn24view0turn23search18 |
| MLS | 1–2 dias (se já usa HF) ou 2–4 dias (download grande) | baixar, usar speaker_id, resample/normalização | **Baixo**: CC BY 4.0, mas não fornece região citeturn19view0turn21view1 |
| FLEURS | 0,5–1 dia | `load_dataset`, selecionar 30 prompts | **Baixo**: CC‑BY, mas não serve para falantes/sotaque citeturn31search0turn31search2 |

Contatos: para Common Voice, o Data Collective lista “commonvoice@mozilla.com” como ponto de contato; e alguns datasets de TTS no Data Collective listam um contato nominal (por exemplo entity["people","Michael Hansen","openhomefoundation contact"]) e e-mail, mas estes são single-speaker e não resolvem sotaque. citeturn24view0turn22search0  

## Plano priorizado para atingir 3 sotaques × 8 falantes × 30 textos

### Estratégia recomendada

**Fase de decisão (fundamental): definir “sotaque” operacionalmente**. Duas opções defensáveis:

- **Opção A: sotaques/locais específicos (cidade/estado)**: ex.: Recife‑PE, Belo Horizonte/MG, São Paulo‑SP. CORAA já sugere categorias análogas para parte dos dados. citeturn10view0turn9search3  
- **Opção B: macro-regiões** (Norte/Nordeste/Sudeste etc.), usando `birth_state` (MuPe) ou “place of birth” (SID) para agrupar. citeturn16search16turn14view0  

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["mapa regioes do Brasil norte nordeste sudeste sul centro-oeste","mapa dialetos sotaques do portugues brasileiro linguistica","mapa estados do Brasil siglas"] ,"num_per_query":1}

### Plano em duas trilhas

**Trilha A: tentar atingir o mínimo apenas com reuso de dados públicos (menor coleta nova)**  
1) **Baixar SID** e auditar:
   - verificar se existe metadado de local de nascimento por falante e quantos falantes caem em cada um dos 3 grupos;  
   - computar por script se há pelo menos **30 textos idênticos** presentes em todos os 24 falantes selecionados (interseção).  
   Se SIM, você encerra aqui: monta `wav/` + `metadata.csv`. citeturn14view0turn18search1  

2) Se SID falhar (pouca interseção ou distribuição regional insuficiente), tentar **VoxForge** como back-up:
   - filtrar pacotes com “Pronunciation dialect: Português do Brasil/PT_BR” (quando existir) e tentar extrair 8 falantes por grupo se houver metadado suficiente. citeturn32search8turn32search0  
   Observação: esta rota tem custo alto e risco legal (GPL + scraping + heterogeneidade). citeturn32search2turn14view0  

**Trilha B: mínima coleta nova para garantir o rigor do desenho “mesmo texto”**  
1) **Escolher 30 textos-prompt**:
   - usar CETUC (frases foneticamente balanceadas) se você tiver acesso a esses textos; ou  
   - usar um subconjunto do FLEURS (pt_br) para prompts tematicamente variados e limpos; ou  
   - usar um conjunto manual (30 frases curtas, sem números/abreviações) e documentar. citeturn14view0turn31search2turn31search0  

2) **Recrutar 24 falantes (8 por sotaque)** com autodeclaração e/ou local de nascimento; gravar numa plataforma única (mesma amostragem, mesmo script). Aqui você reduz drasticamente leakage de microfone/dataset.  

3) **Usar datasets públicos como complemento**:
   - MuPe para validar se seu classificador de sotaque é robusto a fala espontânea (efeito de domínio). citeturn16search16turn16search0  
   - NURC‑SP como controle “SP capital” espontâneo, se seu sotaque inclui SP. citeturn16search2turn16search4  
   - Common Voice/MLS para pré-treino e robustez, sem misturar no conjunto “controlado”. citeturn24view0turn19view0  

### Cronograma proposto

Assumindo início em **2026‑02‑11** (hoje) e foco em entregar rapidamente um corpus 3×8×30, um Gantt realista:

```mermaid
gantt
    title Plano para dataset 3×8×30 (pt-BR) — 2026
    dateFormat  YYYY-MM-DD
    axisFormat  %d/%m

    section Auditoria de dados existentes
    Baixar e inspecionar SID (viabilidade)          :a1, 2026-02-11, 3d
    Testar interseção de 30 textos e balanceamento  :a2, after a1, 2d
    Decisão: SID fecha o mínimo?                    :milestone, m1, after a2, 0d

    section Caso SID não feche
    Definir prompts (CETUC/FLEURS/manual)           :b1, 2026-02-18, 2d
    Montar pipeline de gravação + QC                :b2, after b1, 2d
    Coletar 24 falantes × 30 prompts                :b3, after b2, 7d
    Limpeza + normalização + metadata.csv           :b4, after b3, 3d

    section Complementos e validações
    Baselines (MuPe/NURC-SP/Common Voice)           :c1, 2026-03-04, 4d
    Relatório de leakage + checks finais            :c2, after c1, 2d
```

Os marcos dependem principalmente do “SID fecha o mínimo?” e das restrições de licenciamento/publicação que você aceitará (uso interno vs release público). citeturn14view0turn18search1turn24view0  

### Heurísticas quando metadado regional está ausente (e riscos)

Quando um dataset não tem “região” explícita:

- **Auto-declaração de sotaque** (Common Voice): normalizar strings livres (`accent`) para categorias controladas é possível, mas com ruído e valores ausentes; use como complemento. citeturn24view0  
- **Dialeto em páginas de pacote** (VoxForge): dá para “scrapar” campos como PT_BR/dialeto, mas isso é incompleto e heterogêneo. citeturn32search8turn32search0  
- **Inferência via fonte original do audiolivro** (MLS/LibriVox): mapear narrador → página do LibriVox/Archive.org pode dar pistas, mas é frágil e pode introduzir viés (quem lê audiolivro ≠ população geral) e também pode falhar por falta de metadados. citeturn19view0turn20view0  
- **Marcadores dialetais no texto** (“tu vs você”, léxico regional): altamente confudível, porque seu experimento justamente quer controlar texto; e, em fala espontânea, vira “topic leakage”.  

Em termos científicos, a recomendação é: se você não tem metadado regional curado, trate o sotaque como **variável latente** e faça (i) auditoria manual em amostra e (ii) análise de sensibilidade (o que muda se eu re-rotular 10%?). Isso é particularmente importante para Common Voice e VoxForge. citeturn24view0turn32search0turn14view0  

## Consultas recomendadas e fontes oficiais para continuar a busca

Como você pediu “search queries” e priorização de fontes:

- Consultas para encontrar a página do idioma no Data Collective:
  - `site:datacollective.mozillafoundation.org "Common Voice Scripted Speech" Portuguese pt v24.0`
  - `mcv-scripted-pt-v24.0.tar.gz`
  - `datacollective datasets common voice scripted pt` citeturn23search4turn24view0  

- Consultas para corpora acadêmicos regionais:
  - `nilc-nlp CORAA accent Recife Minas Gerais Sao Paulo`
  - `nilc-nlp CORAA-MUPE-ASR birth_state speaker_code`
  - `nurc-sp-audio-corpus CC BY-NC-ND` citeturn10view0turn16search16turn16search2  

- Consultas para listas brasileiras agregadoras (úteis para descobrir corpora “escondidos”):
  - `falabrasil speech-datasets dvc pull`
  - `igormq datasets CETUC sid voxforge download` citeturn18search13turn14view0  

- Para VoxForge (licença e downloads):
  - `site:voxforge.org portuguese speech files dialect PT_BR`
  - `voxforge GPL license audio corpus` citeturn32search0turn32search2turn32search8