# Regras para Experimentos Cientificamente Válidos

## 1. Estrutura Obrigatória
Todo experimento DEVE declarar explicitamente:
- hipótese
- variável independente
- variáveis controladas
- métricas primárias
- critério de sucesso/falha

Experimentos sem hipótese explícita são exploratórios e NÃO validam nada.

## 2. Uma Pergunta por Experimento
Cada experimento deve responder UMA pergunta técnica.

Se responder mais de uma, é inválido.

## 3. Métricas Corretas
- Classificação desbalanceada → usar balanced accuracy
- Comparações → reportar média e dispersão
- Melhorias marginais (<3pp) não sustentam claims fortes

## 4. Reprodutibilidade
Todo experimento válido deve:
- fixar seeds
- registrar versões de código
- salvar configurações (YAML ou equivalente)
- permitir rerun completo

Sem isso: NO PASS.
