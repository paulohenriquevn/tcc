# PERSONAS TÉCNICAS — STAGE 2 & TECHNICAL VALIDATION

---

## 1. **Lead Research Engineer — Speech & Representation Learning**

### Perfil

**Nome (persona):** Dr. Rafael Monteiro
**Nível:** Staff / Principal Research Engineer
**Formação:** PhD em Speech Processing
**Empresas:** Alibaba DAMO Academy (Qwen), Meta AI, universidade top-tier

### Experiência comprovada

* Desenvolvimento de **speech foundation models**
* Trabalhou diretamente com **Qwen Speech / TTS stack**
* Treinou modelos multimodais com **discrete speech tokens**
* Experiência prática com **LoRA, adapters, partial fine-tuning**
* Forte histórico em **representation disentanglement**

### Responsabilidades no projeto

* Aprovar o **backbone técnico**
* Avaliar se o uso de **Qwen3-TTS + LoRA** é correto
* Validar se o disentanglement proposto é **tecnicamente legítimo**
* Sugerir regularização (adversarial / information bottleneck) se necessário

### Critério de veto

> “Se o sotaque estiver emergindo apenas como prosódia superficial ou se o controle não generalizar para speakers inéditos, eu bloqueio o avanço.”

---

## 2. **Senior Speech Data Engineer — Dataset & Anti-Confound**

### Perfil

**Nome (persona):** Mariana Alves
**Nível:** Senior / Staff Data Engineer
**Empresas:** Google Speech, Apple Siri, consultora para OpenAI

### Experiência comprovada

* Curadoria de **datasets de fala em larga escala**
* Especialista em **confounds de dataset** (mic, speaker, socioeconômico)
* Trabalhou com **dados espontâneos e entrevistados**
* Experiência direta com **corpora brasileiros** (inclusive CORAA)

### Responsabilidades no projeto

* Auditar o uso do **CORAA-MUPE**
* Definir **filtros obrigatórios**
* Garantir **speaker-disjoint split real**
* Avaliar risco de **atalhos estatísticos**
* Validar se o rótulo de sotaque é aceitável como proxy

### Critério de veto

> “Se houver correlação forte entre sotaque e speaker/microfone/qualidade, o experimento técnico é inválido.”

---

## 3. **Principal ML Engineer — Systems, Infra & Reprodutibilidade**

### Perfil

**Nome (persona):** Lucas Andrade
**Nível:** Principal ML Engineer
**Empresas:** Hugging Face, NVIDIA, startups de LLM infra

### Experiência comprovada

* Deploy e treinamento de modelos **em GPU única (24GB)**
* Profundo conhecimento de **PyTorch, CUDA, memory profiling**
* Experiência com **Diffusers / Transformers-style stacks**
* Forte cultura de **reprodutibilidade e versionamento**

### Responsabilidades no projeto

* Validar que o **pipeline roda estável**
* Confirmar que **LoRA training é viável no hardware**
* Auditar uso de VRAM, tempo e checkpoints
* Garantir que notebooks são **reexecutáveis**
* Avaliar custo computacional real

### Critério de veto

> “Se o pipeline não for reexecutável ou depender de ajustes manuais implícitos, não passa.”

---

## 4. **Applied Scientist — Evaluation & Probing**

### Perfil

**Nome (persona):** Dr. Felipe Nakamura
**Nível:** Senior Applied Scientist
**Empresas:** Amazon Alexa, Uber AI, Microsoft Research

### Experiência comprovada

* Avaliação de **ASR/TTS com métricas automáticas**
* Uso de **ECAPA, x-vectors, probing classifiers**
* Design de **stress tests para representations**
* Forte viés pragmático: métricas > demos

### Responsabilidades no projeto

* Definir e revisar **todas as métricas técnicas**
* Implementar **leakage probes**
* Avaliar se os thresholds são razoáveis
* Interpretar resultados ambíguos
* Decidir PASS / ADJUST / FAIL técnico

### Critério de veto

> “Se não houver sinal mensurável claro acima do baseline, o projeto não avança.”

---

## 5. **Independent Technical Reviewer — Red Team**

### Perfil

**Nome (persona):** Ana K. Silva
**Nível:** External Principal Reviewer
**Empresas:** Ex-Meta AI, Ex-DeepMind, consultora independente

### Experiência comprovada

* Revisão de projetos de **speech + multimodal**
* Histórico de *killing bad ideas early*
* Forte ceticismo técnico
* Experiência com falhas clássicas de disentanglement

### Responsabilidades no projeto

* Atuar como **red team técnico**
* Tentar quebrar as conclusões do Stage 2
* Questionar se o efeito é real ou ilusório
* Forçar replicação mínima

### Critério de veto

> “Se o resultado não sobreviver a uma leitura hostil, ele não é confiável.”

---

