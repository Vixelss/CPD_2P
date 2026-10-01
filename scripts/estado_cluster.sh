#!/usr/bin/env bash
# Verifica cada nodo: ping, SSH sin contrasena, hostname, Python, OpenMPI, NFS,
# datos preparados y GPU. Evidencia de conectividad del P2.1.
# Uso: bash scripts/estado_cluster.sh
set -euo pipefail
# shellcheck source=scripts/comun.sh
source "$(dirname "$0")/comun.sh"

NFS="$(cfg valor rutas.nfs)"
celda() { if [[ "$1" == "-" || "$1" == "no" || "$1" == FALLA* ]]; then printf '%s' "${ROJO}$1${NORMAL}"; else printf '%s' "${VERDE}$1${NORMAL}"; fi; }
printf '%-15s %-15s %-5s %-5s %-15s %-8s %-7s %-4s %-6s %s\n' NODO IP PING SSH HOSTNAME PYTHON OPENMPI NFS DATOS GPU
leer_nodos
for linea in "${NODOS[@]}"; do
    campos_nodo "$linea"
    PING="no"; SSHOK="no"; HOST="-"; PYV="-"; MPI="-"; MONT="-"; DATOS="-"; GPU="-"
    ping -c 1 -W 2 "$N_IP" >/dev/null 2>&1 && PING="si"
    if es_este_nodo "$N_IP" "$N_HOST"; then EJ=(bash -c); else EJ=(remoto "$N_USUARIO@$N_IP"); fi
    # Cada linea de la salida es exactamente un campo ("-" si no hay dato)
    REMOTO="uno() { v=\$(\"\$@\" 2>/dev/null | head -1); echo \"\${v:--}\"; }
uno hostname
uno '$N_ENTORNO/bin/python' -c 'import platform; print(platform.python_version())'
uno sh -c \"mpirun --version | grep -oE '[0-9]+[.][0-9]+[.][0-9]+'\"
uno sh -c \"mount | grep -q -e ' $NFS ' -e ' \$HOME/cluster ' && echo si || echo no\"
uno sh -c \"ls '$N_DATOS'/*.seq | wc -l\"
uno sh -c \"nvidia-smi -L | cut -c1-30\""
    if SALIDA="$("${EJ[@]}" "$REMOTO" 2>/dev/null)"; then
        SSHOK="si"
        mapfile -t L <<< "$SALIDA"
        HOST="${L[0]:--}"; PYV="${L[1]:--}"; MPI="${L[2]:--}"; MONT="${L[3]:--}"; DATOS="${L[4]:--}"; GPU="${L[5]:--}"
        [[ "$GPU" != "-" ]] || GPU="no"
    fi
    [[ "$N_SISTEMA" == "macos" && "$MPI" == "-" ]] && MPI="n/a"
    printf '%-15s %-15s %-14s %-14s %-15s %-8s %-7s %-13s %-6s %s\n' "$N_HOST" "$N_IP" "$(celda "$PING")" \
        "$(celda "$SSHOK")" "$HOST" "$PYV" "$MPI" "$(celda "$MONT")" "$DATOS" "$GPU"
done
