#!/usr/bin/env bash
# Prepara un nodo Linux: paquetes, entorno ~/pdn-env, dependencias, nucleo SIMD,
# permiso de lectura de RAPL, SSH y firewall. Se corre en cada nodo Linux.
# Uso: bash scripts/setup_nodo.sh
set -euo pipefail
# shellcheck source=scripts/comun.sh
source "$(dirname "$0")/comun.sh"

ENTORNO="$HOME/pdn-env"
PAQUETES=(openssh-server openmpi-bin libopenmpi-dev python3-mpi4py python3-venv python3-pip python3-yaml
          nfs-common build-essential htop rsync)
FALTAN=0

info "Paquetes de apt"
for p in "${PAQUETES[@]}"; do
    if dpkg -s "$p" >/dev/null 2>&1; then ok "$p"; else falla "$p no instalado"; FALTAN=1; fi
done
if [[ $FALTAN -eq 1 ]]; then
    aviso "Instale lo que falta con: sudo apt install ${PAQUETES[*]}"
fi

info "Entorno virtual $ENTORNO (con --system-site-packages para ver el mpi4py de apt)"
if [[ ! -x "$ENTORNO/bin/python" ]]; then
    python3 -m venv --system-site-packages "$ENTORNO"
    ok "creado"
else
    ok "ya existe"
fi
# Sin internet (router local) pip falla: se sigue con lo ya instalado y se verifica
PIP=("$ENTORNO/bin/python" -m pip install -q --timeout 10 --retries 1)
if "${PIP[@]}" -r "$RAIZ/requirements/base.txt"; then
    ok "requirements/base.txt instalado"
else
    aviso "no se pudo descargar (sin internet?): se usan los paquetes ya instalados"
fi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    if "${PIP[@]}" -r "$RAIZ/requirements/gpu.txt"; then
        ok "GPU NVIDIA detectada: requirements/gpu.txt instalado"
    else
        aviso "no se pudo descargar requirements/gpu.txt (sin internet?)"
    fi
else
    aviso "sin GPU NVIDIA (o sin driver): se omite requirements/gpu.txt"
fi
for m in numpy psutil zmq fastapi uvicorn yaml matplotlib; do
    if "$ENTORNO/bin/python" -c "import $m" 2>/dev/null; then ok "modulo $m"; else falla "falta el modulo $m: conectese a internet y repita"; fi
done
if "$ENTORNO/bin/python" -c "import mpi4py" 2>/dev/null; then
    ok "mpi4py visible desde el entorno ($("$ENTORNO/bin/python" -c 'import mpi4py; print(mpi4py.__version__)'))"
else
    falla "mpi4py no se ve desde el entorno: instale python3-mpi4py (no lo instale con pip)"
fi

info "Nucleo SIMD"
bash "$RAIZ/scripts/compilar_simd.sh"

info "Permiso de lectura de RAPL (energia de la CPU)"
if compgen -G "/sys/class/powercap/intel-rapl:*/energy_uj" >/dev/null; then
    SERVICIO=/etc/systemd/system/pdn-rapl.service
    if [[ ! -f "$SERVICIO" ]]; then
        sudo tee "$SERVICIO" >/dev/null <<'UNIDAD'
[Unit]
Description=Permiso de lectura de RAPL para el cluster PDN

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*/*/energy_uj 2>/dev/null || true'

[Install]
WantedBy=multi-user.target
UNIDAD
        sudo systemctl daemon-reload
        sudo systemctl enable --now pdn-rapl.service
    fi
    if cat /sys/class/powercap/intel-rapl:0/energy_uj >/dev/null 2>&1; then
        ok "RAPL legible sin root"
    else
        falla "RAPL sigue sin permiso de lectura"
    fi
else
    aviso "este equipo no expone RAPL: la energia de la CPU quedara como 'sin dato'"
fi

info "SSH y firewall"
if systemctl is-active --quiet ssh; then ok "ssh activo"; else falla "ssh inactivo: sudo systemctl enable --now ssh"; fi
if command -v ufw >/dev/null 2>&1 && sudo ufw status 2>/dev/null | grep -q "Status: active"; then
    aviso "ufw activo: desactivelo (sudo ufw disable) o abra 5555, 5556, 8000 y los puertos de MPI"
else
    ok "ufw inactivo"
fi

info "Resumen del hardware detectado"
"$ENTORNO/bin/python" - <<'PYTHON'
import json, sys
sys.path.insert(0, ".")
from pdn.worker.hardware import detectar
h = detectar()
cpu = h["cpu"]
print("  CPU: %s, %d hilos%s" % (cpu["modelo"], len(cpu["logicos"]),
      " (%d P + %d E)" % (len(cpu["rendimiento"]), len(cpu["eficiencia"])) if cpu["hibrido"] else ""))
print("  RAM: %d MB   GPU: %s" % (h["ram_mb"], h["gpu"].get("nombre") if h["gpu"]["disponible"] else "no (" + str(h["gpu"]["motivo"]) + ")"))
print("  Red: %s %s, enlace %s Mb/s   Python %s   OpenMPI %s" % (h["red"]["interfaz"], h["red"]["ip"],
      h["red"]["velocidad_mbps"], h["python"], h["openmpi"]))
PYTHON
