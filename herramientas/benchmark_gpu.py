"""Benchmark del motor GPU: uno contra dos streams, y tamano de lote (evidencia de solapamiento).

Mide por configuracion: MB/s, t_transferencia, t_kernel y porcentaje de
solapamiento de copia y calculo (eventos CUDA). Mediana de N repeticiones
tras un calentamiento. En el simulador los tiempos de eventos quedan vacios.

Uso (en nodo-vivanco):
    python -m herramientas.benchmark_gpu --seq ~/pdn-datos/GCF_000001405.40_GRCh38.p14_genomic.seq --mb 1024
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time

from pdn.motores.gpu_cuda import MotorGPU, simulador
from pdn.operaciones.trabajo import Trabajo
from pdn.preparacion.indice import Indice


def correr(seq: str, mb: float, lotes: list[float], reps: int, hilos: int, bloques: int) -> list[dict]:
    """Ejecuta el benchmark y devuelve una fila por configuracion."""
    indice = Indice.leer(seq[:-4] + ".idx")
    fin = min(int(mb * 1024 * 1024), indice.total)
    t = Trabajo("conteo", {"k_invalidos": 0}, indice)
    filas = []
    for lote in lotes:
        for streams in (1, 2):
            m = MotorGPU(hilos_bloque=hilos, bloques=bloques, lote_mb=lote, streams=streams)
            prep = m.preparar()
            carga = t.carga(0, fin)
            m.procesar("conteo", t.params, seq, None, carga)  # calentamiento
            medidas = []
            for _ in range(reps):
                t0 = time.perf_counter()
                _, info = m.procesar("conteo", t.params, seq, None, carga)
                medidas.append((time.perf_counter() - t0, info))
            seg = statistics.median(x[0] for x in medidas)
            info = min(medidas, key=lambda x: abs(x[0] - seg))[1]
            filas.append({"lote_mb": lote, "streams": streams, "mb": round(fin / 1048576, 1),
                          "segundos": round(seg, 5), "mb_s": round(fin / 1048576 / seg, 1),
                          "t_transferencia_s": info.get("t_transferencia_s"), "t_kernel_s": info.get("t_kernel_s"),
                          "t_copia_host_s": info.get("t_copia_host_s"),
                          "solapamiento_pct": info.get("solapamiento_pct"), "preparacion_s": round(prep, 3),
                          "simulador": simulador(), "gpu": m.describir().get("impl")})
            m.cerrar()
    return filas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seq", required=True)
    ap.add_argument("--mb", type=float, default=512)
    ap.add_argument("--lotes", default="16,64,256", help="tamanos de lote en MB")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--hilos", type=int, default=256)
    ap.add_argument("--bloques", type=int, default=1024)
    ap.add_argument("--salida", default=os.path.join("resultados", "benchmark_gpu.csv"))
    a = ap.parse_args()
    filas = correr(a.seq, a.mb, [float(x) for x in a.lotes.split(",")], a.reps, a.hilos, a.bloques)
    os.makedirs(os.path.dirname(a.salida) or ".", exist_ok=True)
    with open(a.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    for f in filas:
        print("lote %6.1f MB  streams %d  %8.1f MB/s  transf %s  kernel %s  solapamiento %s %%" % (
            f["lote_mb"], f["streams"], f["mb_s"], f["t_transferencia_s"], f["t_kernel_s"], f["solapamiento_pct"]))
    print("CSV: %s" % a.salida)


if __name__ == "__main__":
    main()
