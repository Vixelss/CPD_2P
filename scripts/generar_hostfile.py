#!/usr/bin/env python3
"""Genera el hostfile de OpenMPI a partir de cluster.yaml (CONTEXTO.md, seccion 13.2).

Solo nodos Linux con rol master o worker (la Mac no participa en MPI). Los
slots salen del campo 'slots' de cada nodo; si falta, se usa --slots-defecto.

Uso:
    python scripts/generar_hostfile.py                     # todos los slots
    python scripts/generar_hostfile.py --max-slots 1       # un proceso por nodo
    python scripts/generar_hostfile.py --nodos 3 -o hostfile_3nodos
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pdn.comun import config as cfgmod  # noqa: E402


def generar(config: dict, max_slots: int | None = None, nodos: int | None = None,
            slots_defecto: int = 4, usar_ip: bool = False) -> str:
    """Texto del hostfile."""
    lineas = []
    for n in cfgmod.nodos_mpi(config)[:nodos]:
        slots = int(n.get("slots", slots_defecto))
        if max_slots:
            slots = min(slots, max_slots)
        lineas.append("%s slots=%d" % (n["ip"] if usar_ip else n["hostname"], slots))
    return "\n".join(lineas) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config")
    ap.add_argument("--max-slots", type=int)
    ap.add_argument("--nodos", type=int, help="usar solo los primeros N nodos")
    ap.add_argument("--slots-defecto", type=int, default=4)
    ap.add_argument("--ip", action="store_true", help="escribir IPs en lugar de hostnames")
    ap.add_argument("-o", "--salida", default="hostfile")
    a = ap.parse_args()
    texto = generar(cfgmod.cargar(a.config), a.max_slots, a.nodos, a.slots_defecto, a.ip)
    with open(a.salida, "w", encoding="utf-8") as f:
        f.write(texto)
    print("Escrito %s:\n%s" % (a.salida, texto))


if __name__ == "__main__":
    main()
