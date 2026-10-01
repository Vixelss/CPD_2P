#!/usr/bin/env bash
# Monta /cluster del Master en este worker. En la Mac se monta en ~/cluster
# (la raiz de macOS es de solo lectura) con la opcion resvport.
# Uso: bash scripts/montar_nfs.sh [--persistente]
set -euo pipefail
# shellcheck source=scripts/comun.sh
source "$(dirname "$0")/comun.sh"

leer_nodos_con --rol master
campos_nodo "${NODOS[0]}"
MASTER_IP="$N_IP"
NFS="$(cfg valor rutas.nfs)"
if es_este_nodo "$MASTER_IP" "$N_HOST"; then
    ok "este es el Master: exporta $NFS, no lo monta"
    exit 0
fi
if [[ "$(uname -s)" == "Darwin" ]]; then
    PUNTO="$HOME/cluster"
    OPCIONES="resvport,rw"
else
    PUNTO="$NFS"
    OPCIONES="rw,hard,timeo=50"
fi
info "Montando $MASTER_IP:$NFS en $PUNTO"
sudo mkdir -p "$PUNTO"
if mount | grep -q " $PUNTO "; then
    ok "ya estaba montado"
else
    sudo mount -t nfs -o "$OPCIONES" "$MASTER_IP:$NFS" "$PUNTO"
    ok "montado"
fi
if [[ "${1:-}" == "--persistente" && "$(uname -s)" != "Darwin" ]]; then
    LINEA="$MASTER_IP:$NFS $PUNTO nfs $OPCIONES,_netdev,nofail 0 0"
    grep -qsF "$LINEA" /etc/fstab || echo "$LINEA" | sudo tee -a /etc/fstab >/dev/null
    ok "agregado a /etc/fstab"
fi
if touch "$PUNTO/resultados/.prueba_$(hostname)" 2>/dev/null; then
    rm -f "$PUNTO/resultados/.prueba_$(hostname)"
    ok "se puede escribir en $PUNTO/resultados"
else
    aviso "no se puede escribir en $PUNTO/resultados"
fi
