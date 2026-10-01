"""Benchmark del nucleo de calculo: C escalar contra C AVX2 contra numpy, en MB/s.

Medicion justa: una corrida de calentamiento descartada y la mediana de N
repeticiones. Datos aleatorios con la mezcla de un genoma (bases en ambos
casos, N y algun IUPAC), o un .seq real si se pasa --seq.

Uso: python -m herramientas.benchmark_simd --mb 64 --reps 5 --salida resultados/benchmark_simd.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import time

import numpy as np

from pdn.motores.simd import envoltorio
from pdn.operaciones.nucleo import NUMPY


def datos_prueba(mb: float, semilla: int = 1) -> np.ndarray:
    """Bytes con proporciones parecidas a un genoma."""
    n = int(mb * 1024 * 1024)
    rng = np.random.default_rng(semilla)
    d = rng.choice(np.frombuffer(b"ACGTacgtN", dtype=np.uint8), size=n,
                   p=[0.15, 0.1, 0.1, 0.15, 0.13, 0.12, 0.12, 0.08, 0.05])
    d[rng.integers(0, n, max(1, n // 1_000_000))] = ord("R")
    return d


def medir(funcion, reps: int) -> float:
    """Mediana de segundos tras un calentamiento descartado."""
    funcion()
    tiempos = []
    for _ in range(reps):
        t0 = time.perf_counter()
        funcion()
        tiempos.append(time.perf_counter() - t0)
    return statistics.median(tiempos)


def correr(mb: float = 64, reps: int = 5, seq: str | None = None) -> list[dict]:
    """Ejecuta el benchmark y devuelve las filas."""
    d = np.ascontiguousarray(np.memmap(seq, dtype=np.uint8, mode="r")[: int(mb * 1024 * 1024)]) \
        if seq else datos_prueba(mb)
    b = d.copy()
    rng = np.random.default_rng(2)
    b[rng.integers(0, b.size, b.size // 10000)] ^= 0x20
    mayus = d & 0xDF
    nucleos = {"numpy": NUMPY}
    for variante in ("escalar", "avx2"):
        try:
            nucleos["c_" + variante] = envoltorio.NucleoSIMD(variante)
        except envoltorio.SIMDNoDisponible as e:
            print("Sin %s: %s" % (variante, e))
    megas = d.size / (1024 * 1024)
    filas = []
    referencia = {}
    for nombre, nuc in nucleos.items():
        casos = {
            "conteo_histograma": lambda nuc=nuc: nuc.histograma(d),
            "comparacion": lambda nuc=nuc: nuc.contar_categorias(d, b),
            "patron_GATTACA": lambda nuc=nuc: nuc.coincidencias(mayus, "GATTACA"),
        }
        for caso, f in casos.items():
            salida = f()
            firma = salida.tobytes() if isinstance(salida, np.ndarray) else repr(salida)
            iguales = referencia.setdefault(caso, firma) == firma
            s = medir(f, reps)
            filas.append({"funcion": caso, "implementacion": nombre, "mb": round(megas, 2),
                          "segundos_mediana": round(s, 6), "mb_s": round(megas / s, 1),
                          "repeticiones": reps, "igual_a_numpy": iguales,
                          "cpu": platform.processor() or platform.machine()})
    return filas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mb", type=float, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seq", help="usar un .seq real en lugar de datos aleatorios")
    ap.add_argument("--salida", default=os.path.join("resultados", "benchmark_simd.csv"))
    a = ap.parse_args()
    filas = correr(a.mb, a.reps, a.seq)
    os.makedirs(os.path.dirname(a.salida) or ".", exist_ok=True)
    with open(a.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    for fila in filas:
        print("%-20s %-10s %9.1f MB/s  %s" % (fila["funcion"], fila["implementacion"], fila["mb_s"],
                                              "" if fila["igual_a_numpy"] else "DISTINTO"))
    print("CSV: %s" % a.salida)


if __name__ == "__main__":
    main()
