"""Genera un par A/B para comparar con diferencias conocidas por categoria.

Portado de referencias/P1.4/generar_par.py. B es una copia de un recorte de A
con sustituciones (sin inserciones ni borrados) sobre bases A C G T:
- reales: la base cambia por otra distinta;
- solo_caso: la misma base con el caso invertido;
- con_n: la base se reemplaza por N.
Escribe B y un .esperado.json con el desglose y las posiciones en el .seq.

Uso: python -m herramientas.generar_par origen.fna --mb 200 --reales 15000 --caso 2000 --con-n 3000
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from herramientas.fasta_crudo import leer_recorte, mascara_secuencia

_BASES = np.frombuffer(b"ACGTacgt", dtype=np.uint8)


def generar(origen: str, salida_a: str, salida_b: str, mb: float | None = None,
            reales: int = 1000, caso: int = 200, con_n: int = 200, semilla: int = 3) -> dict:
    """Escribe A (recorte) y B (mutado) y devuelve lo esperado."""
    crudo = np.frombuffer(leer_recorte(origen, mb), dtype=np.uint8)
    with open(salida_a, "wb") as f:
        f.write(crudo.tobytes())
    mascara = mascara_secuencia(crudo)
    candidatos = np.flatnonzero(mascara & np.isin(crudo, _BASES))
    total = reales + caso + con_n
    if total > len(candidatos):
        raise ValueError("Se piden %d diferencias y solo hay %d bases" % (total, len(candidatos)))
    rng = np.random.default_rng(semilla)
    elegidos = rng.choice(candidatos, size=total, replace=False)
    b = crudo.copy()
    cats = []
    for n, p in enumerate(elegidos):
        orig = int(b[p])
        if n < reales:
            otras = [x for x in b"ACGT" if x != (orig & 0xDF)]
            b[p] = otras[int(rng.integers(0, 3))]
            cats.append("reales")
        elif n < reales + caso:
            b[p] = orig ^ 0x20
            cats.append("solo_caso")
        else:
            b[p] = ord("N")
            cats.append("con_n")
    with open(salida_b, "wb") as f:
        f.write(b.tobytes())
    pos_seq = (np.cumsum(mascara) - 1)[elegidos]
    esperado = {"reales": reales, "solo_caso": caso, "con_n": con_n, "total": total,
                "mutaciones": sorted([int(p), c] for p, c in zip(pos_seq, cats))}
    with open(os.path.splitext(salida_b)[0] + ".esperado.json", "w", encoding="utf-8") as f:
        json.dump(esperado, f)
    return esperado


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera un par con diferencias conocidas")
    ap.add_argument("origen")
    ap.add_argument("--salida-a", default="par_A.fna")
    ap.add_argument("--salida-b", default="par_B.fna")
    ap.add_argument("--mb", type=float, default=None)
    ap.add_argument("--reales", type=int, default=1000)
    ap.add_argument("--caso", type=int, default=200)
    ap.add_argument("--con-n", type=int, default=200)
    ap.add_argument("--semilla", type=int, default=3)
    a = ap.parse_args()
    e = generar(a.origen, a.salida_a, a.salida_b, a.mb, a.reales, a.caso, a.con_n, a.semilla)
    print("Par generado: %d diferencias (%d reales, %d de caso, %d con N)"
          % (e["total"], e["reales"], e["solo_caso"], e["con_n"]))


if __name__ == "__main__":
    main()
