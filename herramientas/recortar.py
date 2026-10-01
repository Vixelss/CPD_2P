"""Recorta los primeros N MB de un FASTA, cortando en linea completa.

Uso: python -m herramientas.recortar origen.fna destino.fna --mb 200
"""

from __future__ import annotations

import argparse

from herramientas.fasta_crudo import leer_recorte


def recortar(origen: str, destino: str, mb: float) -> int:
    """Escribe el recorte y devuelve los bytes escritos."""
    datos = leer_recorte(origen, mb)
    with open(destino, "wb") as f:
        f.write(datos)
    return len(datos)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("origen")
    ap.add_argument("destino")
    ap.add_argument("--mb", type=float, required=True)
    a = ap.parse_args()
    print("Escritos %d bytes en %s" % (recortar(a.origen, a.destino, a.mb), a.destino))


if __name__ == "__main__":
    main()
