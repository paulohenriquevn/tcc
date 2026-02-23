#!/bin/bash
# Hook: TaskCompleted - Quality Gate para validação de DoD
#
# Lê o JSON da task via stdin e valida critérios básicos de Definition of Done.
# Exit 0 = aprovado, Exit 2 = rejeitado (stderr vira feedback para o agent).

set -e

# Lê JSON da task via stdin
TASK_JSON=$(cat)

# Se não recebeu input, aprova (hook pode ser chamado sem contexto)
if [ -z "$TASK_JSON" ]; then
  exit 0
fi

# Extrai campos do JSON usando jq (com fallback para python3)
# NOTA: O Claude Code envia campos com prefixo task_ (task_subject, task_description)
if command -v jq >/dev/null 2>&1; then
  TASK_SUBJECT=$(echo "$TASK_JSON" | jq -r '.task_subject // .subject // empty' 2>/dev/null)
  TASK_STATUS=$(echo "$TASK_JSON" | jq -r '.task_status // .status // empty' 2>/dev/null)
  TASK_DESCRIPTION=$(echo "$TASK_JSON" | jq -r '.task_description // .description // empty' 2>/dev/null)
elif command -v python3 >/dev/null 2>&1; then
  TASK_SUBJECT=$(echo "$TASK_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('task_subject', d.get('subject','')))" 2>/dev/null)
  TASK_STATUS=$(echo "$TASK_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('task_status', d.get('status','')))" 2>/dev/null)
  TASK_DESCRIPTION=$(echo "$TASK_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('task_description', d.get('description','')))" 2>/dev/null)
else
  # Sem jq ou python3, aprova sem validar (melhor que falso positivo)
  echo "AVISO: Nem jq nem python3 disponiveis. Pulando validacao de task." >&2
  exit 0
fi

# Validação 1: Task deve ter subject
if [ -z "$TASK_SUBJECT" ]; then
  echo "REJEITADO: Task sem subject definido. Toda task precisa de um titulo claro." >&2
  exit 2
fi

# Validação 2: Subject deve ter tamanho minimo (evita tasks vagas)
SUBJECT_LEN=${#TASK_SUBJECT}
if [ "$SUBJECT_LEN" -lt 10 ]; then
  echo "REJEITADO: Subject muito curto ('$TASK_SUBJECT'). Descreva a task de forma clara e especifica." >&2
  exit 2
fi

# Validação 3: Status deve ser 'completed' (consistência)
if [ -n "$TASK_STATUS" ] && [ "$TASK_STATUS" != "completed" ]; then
  echo "AVISO: Task marcada como completada mas status interno e '$TASK_STATUS'." >&2
  # Não rejeita, apenas avisa
fi

# Validação 4: Descrição deve existir e ter substância
if [ -z "$TASK_DESCRIPTION" ]; then
  echo "AVISO: Task sem descricao. Considere adicionar detalhes sobre o que foi feito." >&2
fi

if [ -n "$TASK_DESCRIPTION" ]; then
  DESC_LEN=${#TASK_DESCRIPTION}
  if [ "$DESC_LEN" -lt 20 ]; then
    echo "AVISO: Descricao muito curta. Considere adicionar mais detalhes sobre o que foi feito." >&2
  fi
fi

# Validação 5: Warning se task de experimento/treinamento não menciona seed/reprodutibilidade
SUBJECT_LOWER=$(echo "$TASK_SUBJECT $TASK_DESCRIPTION" | tr '[:upper:]' '[:lower:]')
HAS_EXPERIMENT=$(echo "$SUBJECT_LOWER" | grep -oiE 'experiment|training|treinamento|treino|fine.?tun' | head -1 || true)
if [ -n "$HAS_EXPERIMENT" ]; then
  HAS_REPROD=$(echo "$SUBJECT_LOWER" | grep -oiE 'seed|reproduc|checkpoint|determin' | head -1 || true)
  if [ -z "$HAS_REPROD" ]; then
    echo "AVISO: Task menciona '$HAS_EXPERIMENT' mas nao menciona seed/reprodutibilidade. Verifique se seeds estao configurados." >&2
  fi
fi

# Aprovado
exit 0
