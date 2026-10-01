#!/usr/bin/env bash
# Detiene ordenadamente workers, Master de respaldo y Master en todos los nodos.
# Uso: bash scripts/detener_cluster.sh
set -euo pipefail
# shellcheck source=scripts/comun.sh
source "$(dirname "$0")/comun.sh"

# shellcheck disable=SC2016  # el comando se expande en cada nodo
# Los patrones usan [.] para no coincidir con la propia linea de comando de la shell
PARAR='for f in $HOME/pdn-logs/*.pid; do [ -f "$f" ] || continue; kill -TERM "$(cat "$f")" 2>/dev/null; rm -f "$f"; done; pkill -TERM -f "python.* -m pdn[.]worker" 2>/dev/null; pkill -TERM -f "python.* -m pdn[.]cli (master|respaldo)" 2>/dev/null; true'
leer_nodos
for linea in "${NODOS[@]}"; do
    campos_nodo "$linea"
    info "$N_HOST"
    if es_este_nodo "$N_IP" "$N_HOST"; then
        bash -c "$PARAR" && ok "detenido"
    elif remoto "$N_USUARIO@$N_IP" "$PARAR"; then
        ok "detenido"
    else
        falla "sin SSH"
    fi
done
