#!/usr/bin/env bash
# Copia el codigo a la misma ruta en cada nodo (requisito de MPI) y compila el
# nucleo SIMD en los nodos Linux. Uso: bash scripts/desplegar_codigo.sh [hostname...]
set -euo pipefail
# shellcheck source=scripts/comun.sh
source "$(dirname "$0")/comun.sh"

EXCLUIR=(--exclude .git --exclude .venv --exclude resultados --exclude referencias_resultados
         --exclude __pycache__ --exclude .pytest_cache --exclude 'pdn/motores/simd/build' --exclude hostfile)
leer_nodos
for linea in "${NODOS[@]}"; do
    campos_nodo "$linea"
    if [[ $# -gt 0 ]] && [[ ! " $* " == *" $N_HOST "* ]]; then continue; fi
    info "$N_HOST -> $N_CODIGO"
    if es_este_nodo "$N_IP" "$N_HOST" && [[ "$(realpath "$RAIZ")" == "$(realpath -m "$N_CODIGO")" ]]; then
        ok "es este nodo y esta carpeta"; continue
    fi
    if ! remoto "$N_USUARIO@$N_IP" "mkdir -p '$N_CODIGO'"; then falla "sin SSH"; continue; fi
    if rsync -az --delete "${EXCLUIR[@]}" "$RAIZ/" "$N_USUARIO@$N_IP:$N_CODIGO/"; then
        ok "codigo copiado"
    else
        falla "rsync fallo"; continue
    fi
    if [[ "$N_SISTEMA" == "linux" ]]; then
        if remoto "$N_USUARIO@$N_IP" "bash '$N_CODIGO/scripts/compilar_simd.sh'" >/dev/null; then
            ok "nucleo SIMD compilado"
        else
            aviso "no se pudo compilar el nucleo SIMD (falta build-essential?)"
        fi
    fi
done
