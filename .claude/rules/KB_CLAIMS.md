# Regras para Claims e Conclusões

## 1. Separação Obrigatória
O Guardrails deve sempre separar:
- dados observados
- interpretação
- especulação

Misturar os três invalida a conclusão.

## 2. Claims Permitidos
Só são permitidos claims que:
- correspondam exatamente à hipótese testada
- estejam suportados por métricas explícitas
- não extrapolem o escopo do experimento

## 3. Claims Proibidos
É proibido afirmar:
- generalização além do dataset
- superioridade global
- causalidade sem ablação
- robustez sem testes adversariais

## 4. Linguagem
Palavras como:
“prova”, “resolve”, “garante”, “definitivo”
são proibidas antes do Gate final.
