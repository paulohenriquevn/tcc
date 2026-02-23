# Regras de Hard Fail Automatico

Criterios de parada automatica. Se QUALQUER um ocorrer, o resultado e **INVALIDO**.
Nao e FAIL (hipotese refutada) — e INVALIDO (evidencia insuficiente ou contaminada).

---

## 1. Integridade de Splits

- Speaker aparece em mais de um split (train/val/test) → **INVALIDO**
- Splits nao persistidos em arquivo (efemeros em memoria) → **INVALIDO**
- Seed do split nao registrada → **INVALIDO**
- Assertions de disjointness ausentes no codigo → **INVALIDO**

## 2. Reproducibilidade

- Seed global nao documentada (`random`, `numpy`, `torch`, `torch.cuda`) → **INVALIDO**
- Config YAML ausente ou incompleto → **INVALIDO**
- Hash SHA-256 do manifest nao registrado → **INVALIDO**
- Versoes de dependencias nao pinadas → **INVALIDO**
- Commit hash do codigo nao registrado → **INVALIDO**

## 3. Metricas

- Leakage (A→speaker OU S→accent) > chance + 12 p.p. → **INVALIDO**
- CI 95% de accent probe inclui chance level → resultado **NAO SIGNIFICATIVO**
- Baseline ECAPA intra/inter speaker similarity nao medido → **INVALIDO**
- Balanced accuracy nao reportada (apenas accuracy simples) → **INVALIDO**
- Confusion matrix ausente → **INVALIDO**

## 4. Confounds

- Analise accent x gender (chi-quadrado) nao executada → **INVALIDO**
- Analise accent x duration (Kruskal-Wallis) nao executada → **INVALIDO**
- Cramer's V (accent x gender) >= 0.3 sem mitigacao documentada → **INVALIDO**

## 5. Evidencias

- Logs completos ausentes → **INVALIDO**
- Scripts/notebooks nao reproduziveis → **INVALIDO**
- Codigo critico (splits, probes, decisao) nao auditavel localmente → **INVALIDO**

---

## Consequencia

Resultado INVALIDO significa:
- Nenhum claim e permitido
- Nenhuma decisao de Gate e valida
- Re-execucao completa e obrigatoria antes de prosseguir

Resultado INVALIDO **nao** e fracasso cientifico — e ausencia de evidencia.
