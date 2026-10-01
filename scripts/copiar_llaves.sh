#!/usr/bin/env bash
# Copia la llave SSH del Master a cada nodo y registra sus huellas en known_hosts
# (sin esto, mpirun se queda esperando la pregunta de confirmacion).
# Uso: bash scripts/copiar_llaves.sh
set -euo pipefail
# shellcheck source=scripts/comun.sh
source "$(dirname "$0")/comun.sh"

LLAVE="$HOME/.ssh/id_ed25519"
if [[ ! -f "$LLAVE" ]]; then
    info "Creando la llave $LLAVE"
    ssh-keygen -t ed25519 -N "" -f "$LLAVE"
fi
mkdir -p "$HOME/.ssh"
touch "$HOME/.ssh/known_hosts"
leer_nodos
for linea in "${NODOS[@]}"; do
    campos_nodo "$linea"
    info "$N_HOST ($N_USUARIO@$N_IP)"
    if es_este_nodo "$N_IP" "$N_HOST"; then ok "es este nodo"; continue; fi
    for nombre in "$N_IP" "$N_HOST"; do
        if ! ssh-keygen -F "$nombre" >/dev/null 2>&1; then
            ssh-keyscan -T 5 -H "$nombre" >> "$HOME/.ssh/known_hosts" 2>/dev/null || aviso "ssh-keyscan $nombre fallo"
        fi
    done
    if remoto "$N_USUARIO@$N_IP" true 2>/dev/null; then
        ok "la llave ya estaba copiada"
    elif ssh-copy-id -i "$LLAVE.pub" "$N_USUARIO@$N_IP"; then
        ok "llave copiada"
    else
        falla "no se pudo copiar la llave (revise la contrasena, la IP o el servicio ssh)"
    fi
done
