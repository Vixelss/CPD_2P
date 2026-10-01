#!/usr/bin/env bash
# Copia un FASTA a ~/pdn-datos de cada nodo, lo prepara alli (.seq, .idx, .huellas)
# y verifica que la huella global coincida con la del Master.
# Uso: bash scripts/distribuir_datos.sh ~/pdn-datos/GCF_000001405.40_GRCh38.p14_genomic.fna
set -euo pipefail
# shellcheck source=scripts/comun.sh
source "$(dirname "$0")/comun.sh"

[[ $# -ge 1 ]] || error "uso: $0 archivo.fna"
FNA="$(realpath "$1")"
[[ -f "$FNA" ]] || error "no existe $FNA"
BASE="$(basename "$FNA")"
NOMBRE="${BASE%.*}"

info "Preparando en este nodo (referencia de huellas)"
"$PY" -m pdn.cli preparar "$FNA"
HUELLAS_LOCAL="$(dirname "$FNA")/$NOMBRE.huellas"
GLOBAL="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['global'])" "$HUELLAS_LOCAL")"
ok "huella global: $GLOBAL"

leer_nodos
RESUMEN=()
for linea in "${NODOS[@]}"; do
    campos_nodo "$linea"
    if es_este_nodo "$N_IP" "$N_HOST"; then RESUMEN+=("$N_HOST: este nodo"); continue; fi
    info "$N_HOST: copiando $BASE a $N_DATOS"
    if ! remoto "$N_USUARIO@$N_IP" "mkdir -p '$N_DATOS'"; then RESUMEN+=("$N_HOST: FALLA (sin SSH)"); continue; fi
    if ! rsync -P "$FNA" "$N_USUARIO@$N_IP:$N_DATOS/"; then RESUMEN+=("$N_HOST: FALLA (rsync)"); continue; fi
    info "$N_HOST: preparando"
    if ! remoto "$N_USUARIO@$N_IP" "cd '$N_CODIGO' && '$N_ENTORNO/bin/python' -m pdn.cli preparar '$N_DATOS/$BASE'"; then
        RESUMEN+=("$N_HOST: FALLA (preparacion)"); continue
    fi
    SUYA="$(remoto "$N_USUARIO@$N_IP" "'$N_ENTORNO/bin/python' -c \"import json; print(json.load(open('$N_DATOS/$NOMBRE.huellas'))['global'])\"")"
    if [[ "$SUYA" == "$GLOBAL" ]]; then RESUMEN+=("$N_HOST: OK"); else RESUMEN+=("$N_HOST: FALLA (huella $SUYA)"); fi
done
info "Resumen"
for r in "${RESUMEN[@]}"; do
    if [[ "$r" == *FALLA* ]]; then falla "$r"; else ok "$r"; fi
done
