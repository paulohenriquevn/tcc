#!/bin/bash
# Hook: TeammateIdle - Verifica se teammate tem trabalho pendente
#
# Lê contexto do teammate via stdin e verifica se há tarefas pendentes.
# Informativo (não bloqueia) — exit 0 na maioria dos casos.
# Emite aviso no stderr se há tarefas pendentes.

set -e

# Lê contexto do teammate via stdin
CONTEXT_JSON=$(cat)

# Se não recebeu input, nada a fazer
if [ -z "$CONTEXT_JSON" ]; then
  exit 0
fi

# Verifica se há menção de tarefas pendentes no contexto
HAS_PENDING=$(echo "$CONTEXT_JSON" | grep -oi '"pending"' | head -1 || true)

if [ -n "$HAS_PENDING" ]; then
  TEAMMATE_NAME=$(echo "$CONTEXT_JSON" | grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"name"[[:space:]]*:[[:space:]]*"//' | sed 's/"$//' || echo "teammate")
  echo "AVISO: $TEAMMATE_NAME tem tarefas pendentes. Verifique a task list antes de encerrar." >&2
fi

# Sempre sai com 0 (informativo, não bloqueia)
exit 0
