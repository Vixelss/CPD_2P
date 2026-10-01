#!/usr/bin/env bash
# Arranca el cluster: workers en cada nodo (segun 'dispositivos' de cluster.yaml),
# el Master de respaldo y el Master con el dashboard en este nodo.
# Uso: bash scripts/lanzar_cluster.sh [--sin-respaldo] [--solo-workers]
set -euo pipefail
# shellcheck source=scripts/comun.sh
source "$(dirname "$0")/comun.sh"

SIN_RESPALDO=0; SOLO_WORKERS=0
for a in "$@"; do
    case "$a" in
        --sin-respaldo) SIN_RESPALDO=1 ;;
        --solo-workers) SOLO_WORKERS=1 ;;
        *) error "opcion desconocida: $a" ;;
    esac
done
PUERTO="$(cfg valor red.puerto_tareas)"
PUERTO_DASH="$(cfg valor red.puerto_dashboard)"
leer_nodos_con --rol master
campos_nodo "${NODOS[0]}"; MASTER_IP="$N_IP"
RESPALDO_IP=""
if [[ $SIN_RESPALDO -eq 0 ]] && cfg nodos --rol respaldo | grep -q .; then
    mapfile -t R < <(cfg nodos --rol respaldo); campos_nodo "${R[0]}"; RESPALDO_IP="$N_IP"
    RESPALDO_LINEA="${R[0]}"
fi
# shellcheck disable=SC2016  # $HOME se expande en cada nodo, no aqui
LOGS='$HOME/pdn-logs'

arrancar() {  # arrancar LINEA_NODO COMANDO_EN_EL_ENTORNO NOMBRE_LOG
    campos_nodo "$1"
    # Las llaves hacen que solo el nohup quede en segundo plano: sin ellas, toda la
    # cadena de && corre en una subshell que espera al proceso y la sesion no termina
    local cmd="mkdir -p $LOGS && cd '$N_CODIGO' && { nohup '$N_ENTORNO/bin/python' $2 < /dev/null > $LOGS/$3.out 2>&1 & echo \$! > $LOGS/$3.pid; }"
    if es_este_nodo "$N_IP" "$N_HOST"; then bash -c "$cmd"; else remoto "$N_USUARIO@$N_IP" "$cmd"; fi
}

if [[ $SOLO_WORKERS -eq 0 ]]; then
    if [[ -n "$RESPALDO_IP" ]]; then
        info "Master de respaldo en $RESPALDO_IP"
        arrancar "$RESPALDO_LINEA" "-m pdn.cli respaldo --dashboard --log $LOGS/respaldo.log" respaldo && ok "lanzado"
        REPLICAR=(--replicar-a "$RESPALDO_IP")
    else
        REPLICAR=()
        aviso "sin Master de respaldo"
    fi
    info "Master y dashboard en este nodo ($MASTER_IP)"
    mkdir -p "$HOME/pdn-logs"
    nohup "$PY" -m pdn.cli master --dashboard "${REPLICAR[@]}" --log "$HOME/pdn-logs/master.log" \
        < /dev/null > "$HOME/pdn-logs/master.out" 2>&1 &
    echo $! > "$HOME/pdn-logs/master.pid"
    ok "lanzado (pid $(cat "$HOME/pdn-logs/master.pid"))"
    sleep 2
fi

MASTERS="--master $MASTER_IP:$PUERTO"
[[ -n "$RESPALDO_IP" ]] && MASTERS="$MASTERS --respaldo $RESPALDO_IP:$PUERTO"
leer_nodos
for linea in "${NODOS[@]}"; do
    campos_nodo "$linea"
    IFS=',' read -r -a DISPOSITIVOS <<< "$N_DISP"
    for d in "${DISPOSITIVOS[@]}"; do
        info "worker $N_HOST:$d"
        if arrancar "$linea" "-m pdn.worker $MASTERS --dispositivo $d --log $LOGS/worker_$d.log" "worker_$d"; then
            ok "lanzado"
        else
            falla "no se pudo lanzar (SSH?)"
        fi
    done
done
info "Dashboard: http://$MASTER_IP:$PUERTO_DASH  (respaldo: ${RESPALDO_IP:-ninguno})"
