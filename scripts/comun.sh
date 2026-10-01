#!/usr/bin/env bash
# Funciones comunes de los scripts del cluster. Se carga con: source "$(dirname "$0")/comun.sh"
# shellcheck disable=SC2034

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_YAML="${PDN_CLUSTER_YAML:-$RAIZ/cluster.yaml}"
export PDN_CLUSTER_YAML="$CLUSTER_YAML"

# Interprete: el del entorno si existe, si no python3 del sistema
if [[ -x "$HOME/pdn-env/bin/python" ]]; then
    PY="$HOME/pdn-env/bin/python"
else
    PY="python3"
fi

SSH_OPCIONES=(-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new)

if [[ -t 1 ]]; then
    VERDE=$'\e[32m'; ROJO=$'\e[31m'; AMARILLO=$'\e[33m'; NEGRITA=$'\e[1m'; NORMAL=$'\e[0m'
else
    VERDE=""; ROJO=""; AMARILLO=""; NEGRITA=""; NORMAL=""
fi

info()  { printf '%s\n' "${NEGRITA}==>${NORMAL} $*"; }
ok()    { printf '%s\n' "  ${VERDE}OK${NORMAL}  $*"; }
falla() { printf '%s\n' "  ${ROJO}FALLA${NORMAL} $*"; }
aviso() { printf '%s\n' "  ${AMARILLO}AVISO${NORMAL} $*"; }
error() { printf '%s\n' "${ROJO}ERROR:${NORMAL} $*" >&2; exit 1; }

cfg() { "$PY" "$RAIZ/scripts/cfg.py" --config "$CLUSTER_YAML" "$@"; }

# Lee todos los nodos (TSV) en el arreglo NODOS
leer_nodos() {
    mapfile -t NODOS < <(cfg nodos)
    [[ ${#NODOS[@]} -gt 0 ]] || error "cluster.yaml no tiene nodos (o falta PyYAML: sudo apt install python3-yaml)"
}

# Igual, con filtro: --rol X o --sistema Y
leer_nodos_con() {
    mapfile -t NODOS < <(cfg nodos "$@")
    [[ ${#NODOS[@]} -gt 0 ]] || error "no hay nodos que cumplan: $*"
}

# Separa una linea TSV de nodo en variables N_HOST N_IP N_USUARIO N_SISTEMA N_ROLES N_DISP N_CODIGO N_DATOS N_ENTORNO
campos_nodo() {
    IFS=$'\t' read -r N_HOST N_IP N_USUARIO N_SISTEMA N_ROLES N_DISP N_CODIGO N_DATOS N_ENTORNO <<< "$1"
}

remoto() {  # remoto USUARIO@IP comando...
    local destino="$1"; shift
    # shellcheck disable=SC2029  # el comando se arma a proposito en este lado
    ssh "${SSH_OPCIONES[@]}" "$destino" "$@"
}

es_este_nodo() {  # true si la IP o el hostname son de esta maquina
    local ip="$1" host="$2"
    [[ "$host" == "$(hostname)" ]] && return 0
    hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "$ip"
}
