#!/usr/bin/env bash
# Prepara el Master: todo lo de setup_nodo.sh mas el servidor NFS de /cluster.
# Uso: bash scripts/setup_master.sh
set -euo pipefail
# shellcheck source=scripts/comun.sh
source "$(dirname "$0")/comun.sh"

bash "$RAIZ/scripts/setup_nodo.sh"

SUBRED="$(cfg valor red.subred)"
NFS="$(cfg valor rutas.nfs)"
info "Servidor NFS: exportando $NFS a $SUBRED"
dpkg -s nfs-kernel-server >/dev/null 2>&1 || error "falta nfs-kernel-server: sudo apt install nfs-kernel-server"
sudo mkdir -p "$NFS/datos" "$NFS/resultados" "$NFS/referencias_resultados"
sudo chown -R "$(id -un):$(id -gn)" "$NFS"
LINEA="$NFS $SUBRED(rw,sync,no_subtree_check,no_root_squash)"
if grep -qsF "$NFS " /etc/exports; then
    sudo sed -i "\\#^$NFS #d" /etc/exports
fi
echo "$LINEA" | sudo tee -a /etc/exports >/dev/null
sudo exportfs -ra
sudo systemctl enable --now nfs-kernel-server
if sudo exportfs -v | grep -q "$NFS"; then
    ok "exportado: $LINEA"
else
    falla "no se pudo exportar $NFS"
fi
