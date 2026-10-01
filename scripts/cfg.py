#!/usr/bin/env python3
"""Ayudante para que los scripts de bash lean cluster.yaml.

Uso:
    python3 scripts/cfg.py valor red.subred
    python3 scripts/cfg.py nodos                    # TSV: hostname ip usuario sistema roles dispositivos codigo datos entorno
    python3 scripts/cfg.py nodos --rol worker
    python3 scripts/cfg.py nodo nodo-carranza ip
Solo necesita PyYAML (python3-yaml en Ubuntu).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pdn.comun import config as cfgmod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.environ.get("PDN_CLUSTER_YAML"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("valor")
    v.add_argument("clave")
    n = sub.add_parser("nodos")
    n.add_argument("--rol")
    n.add_argument("--sistema")
    u = sub.add_parser("nodo")
    u.add_argument("hostname")
    u.add_argument("campo")
    a = ap.parse_args()
    config = cfgmod.cargar(a.config)
    if a.cmd == "valor":
        x = config
        for parte in a.clave.split("."):
            x = x[parte]
        print(x)
    elif a.cmd == "nodos":
        for nodo in config.get("nodos", []):
            if a.rol and a.rol not in nodo.get("roles", []):
                continue
            sistema = nodo.get("sistema", "linux")
            if a.sistema and sistema != a.sistema:
                continue
            rutas = cfgmod.nodo(config, nodo["hostname"])["rutas"]
            print("\t".join([nodo["hostname"], nodo.get("ip", ""), nodo.get("usuario", "pdn"), sistema,
                             ",".join(nodo.get("roles", [])), ",".join(nodo.get("dispositivos", [])),
                             rutas["codigo"], rutas["datos"], rutas["entorno"]]))
    elif a.cmd == "nodo":
        nodo = cfgmod.nodo(config, a.hostname)
        if nodo is None:
            sys.exit("nodo desconocido: %s" % a.hostname)
        valor = nodo["rutas"][a.campo] if a.campo in nodo["rutas"] else nodo.get(a.campo, "")
        print(",".join(valor) if isinstance(valor, list) else valor)


if __name__ == "__main__":
    main()
