# Constituição do Projeto

## 1. Objetivo Científico Central
O objetivo deste projeto é validar, de forma rigorosa e reprodutível, se é possível controlar explicitamente sotaque regional em síntese de fala (pt-BR) **sem degradar identidade do falante**, usando mecanismos de representação explícita.

Este projeto NÃO busca:
- atingir SOTA
- maximizar métricas absolutas
- produzir um produto comercial
- provar generalização ampla para múltiplos idiomas

## 2. Princípio da Validade
Nenhum resultado é considerado válido se não puder ser:
- reproduzido
- auditado
- explicado causalmente

Resultados “bons” fora do protocolo são inválidos.

## 3. Princípio do Não-Atalho
É expressamente proibido:
- inferir causalidade a partir de correlação
- pular etapas metodológicas por conveniência
- ajustar critérios após observar resultados

## 4. Princípio da Conservação da Identidade
Qualquer ganho em sotaque que resulte em degradação significativa da identidade do falante é considerado fracasso técnico.

## 5. Autoridade do Guardrails
O Guardrails tem autoridade final para:
- bloquear avanço de stage
- invalidar experimentos
- proibir conclusões
- exigir reexecução completa

Nenhuma decisão avança sem aprovação explícita do Guardrails.
