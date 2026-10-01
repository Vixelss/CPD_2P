"""Programa MPI: reparto estatico entre ranks (CONTEXTO.md, seccion 13.1).

Se lanza con mpirun desde el Master (ver pdn/mpi/lanzador.py):
    mpirun --hostfile hostfile --map-by core --bind-to core --report-bindings \\
        ~/pdn-env/bin/python -m pdn.mpi.mpi_correr --operacion conteo --archivo NOMBRE

- Rank 0 calcula el reparto y lo envia con scatter.
- Cada rank procesa su tramo de su copia local (o NFS) con una sola hebra:
  el paralelismo lo da MPI.
- Barrier antes de arrancar el reloj; MPI.Wtime para medir.
- Consolidacion: Reduce con MPI.SUM para el histograma (numpy int64) y
  gather para el resto de los parciales.
- La preparacion (abrir archivos, leer el indice, cargar el nucleo SIMD)
  queda antes del Barrier, fuera del cronometro.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

import numpy as np


def _args(argv=None):
    ap = argparse.ArgumentParser(description="Corrida en modo MPI")
    ap.add_argument("--operacion", default="conteo")
    ap.add_argument("--archivo", required=True)
    ap.add_argument("--archivo-b")
    ap.add_argument("--datos", default=os.path.expanduser("~/pdn-datos"))
    ap.add_argument("--params", default="{}")
    ap.add_argument("--reparto", default="iguales", choices=["iguales", "proporcional"])
    ap.add_argument("--velocidades", help="JSON {hostname: MB/s por proceso}")
    ap.add_argument("--impl", default="numpy", choices=["numpy", "simd", "simd_escalar"])
    ap.add_argument("--tam-unidad", type=int)
    ap.add_argument("--trozo-mb", type=float, default=64, help="tamano de cada pasada dentro del rank")
    ap.add_argument("--precalentar", action="store_true",
                    help="toca las paginas del tramo propio antes del Barrier (medicion justa)")
    ap.add_argument("--salida", help="ruta del JSON con el resumen (lo escribe el rank 0)")
    ap.add_argument("--resultados", help="carpeta de resultados (lo escribe el rank 0)")
    ap.add_argument("--referencias", help="carpeta de referencias_resultados")
    return ap.parse_args(argv)


def _ruta_seq(datos: str, nombre: str) -> str:
    """Ruta del .seq: la dada si existe, o <datos>/<base>.seq."""
    if nombre.endswith(".seq") and os.path.exists(nombre):
        return nombre
    base = os.path.basename(nombre)
    for ext in (".seq", ".fna", ".fa", ".fasta"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    return os.path.join(datos, base + ".seq")


def tocar(trabajo, seq_a, seq_b, ini: int, fin: int) -> int:
    """Lee un byte por pagina del tramo (y su solape) para dejarlo mapeado y en cache.

    Un proceso nuevo paga los fallos de pagina del memmap la primera vez que
    lee; la referencia secuencial con calentamiento no. Para comparar en las
    mismas condiciones, esto se hace antes del Barrier, fuera del cronometro.
    """
    if fin <= ini:
        return 0
    carga = trabajo.carga(ini, fin)
    total = 0
    if "segmentos" in carga:
        for _, a, b, largo in carga["segmentos"]:
            total += int(np.asarray(seq_a[a:a + largo:4096]).sum()) + int(np.asarray(seq_b[b:b + largo:4096]).sum())
    else:
        hasta = min(carga["fin"] + carga["solape"], seq_a.shape[0])
        total += int(np.asarray(seq_a[carga["inicio"]:hasta:4096]).sum())
    return total


def main(argv=None) -> None:
    from mpi4py import MPI  # noqa: PLC0415

    from pdn.comun.huellas import leer_huellas  # noqa: PLC0415
    from pdn.mpi.reparto import pesos_para, repartir  # noqa: PLC0415
    from pdn.operaciones.trabajo import Trabajo  # noqa: PLC0415
    from pdn.preparacion.fasta_a_seq import abrir_seq  # noqa: PLC0415
    from pdn.preparacion.indice import Indice  # noqa: PLC0415

    a = _args(argv)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    host = socket.gethostname()
    t_prep = time.perf_counter()

    # -- preparacion (fuera del cronometro) --------------------------------
    error = None
    try:
        ruta_a = _ruta_seq(a.datos, a.archivo)
        ruta_b = _ruta_seq(a.datos, a.archivo_b) if a.archivo_b else None
        seq_a = abrir_seq(ruta_a)
        seq_b = abrir_seq(ruta_b) if ruta_b else None
        tam = a.tam_unidad or leer_huellas(ruta_a[:-4] + ".huellas")["tam_unidad"]
        trabajo = Trabajo(a.operacion, json.loads(a.params), Indice.leer(ruta_a[:-4] + ".idx"),
                          Indice.leer(ruta_b[:-4] + ".idx") if ruta_b else None, tam)
        if a.impl == "numpy":
            from pdn.operaciones.nucleo import NUMPY as nucleo  # noqa: PLC0415
        else:
            from pdn.motores.simd.envoltorio import nucleo_simd  # noqa: PLC0415
            nucleo = nucleo_simd("avx2" if a.impl == "simd" else "escalar")
    except Exception as e:  # se informa en el rank 0 y se aborta ordenadamente
        error = "%s: %s" % (host, e)
    errores = comm.allgather(error)
    if any(errores):
        if rank == 0:
            print("ERROR en la preparacion: %s" % [e for e in errores if e], file=sys.stderr)
        sys.exit(2)

    hosts = comm.gather(host, root=0)
    rangos = None
    if rank == 0:
        velocidades = json.loads(a.velocidades) if a.velocidades else None
        rangos = repartir(trabajo.unidades, pesos_para(hosts, a.reparto, velocidades))
    u_ini, u_fin = comm.scatter(rangos, root=0)
    afinidad = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if a.precalentar:
        tocar(trabajo, seq_a, seq_b, u_ini * tam, min(u_fin * tam, trabajo.largo))
    preparacion_s = time.perf_counter() - t_prep

    # -- medicion ------------------------------------------------------------
    comm.Barrier()
    t0 = MPI.Wtime()
    ini = u_ini * tam
    fin = min(u_fin * tam, trabajo.largo)
    parcial = trabajo.vacio()
    paso = max(tam, int(a.trozo_mb * 1024 * 1024) // tam * tam)
    for desde in range(ini, fin, paso):
        hasta = min(desde + paso, fin)
        parcial = trabajo.combinar(parcial, trabajo.procesar(seq_a, seq_b, trabajo.carga(desde, hasta), nucleo))
    t_local = MPI.Wtime() - t0

    if a.operacion == "conteo":
        hist = np.asarray(parcial["hist"], dtype=np.int64)
        total = np.zeros(256, dtype=np.int64) if rank == 0 else None
        comm.Reduce(hist, total, op=MPI.SUM, root=0)
        posiciones = comm.gather(parcial["invalidos_pos"], root=0)
    else:
        parciales = comm.gather(parcial, root=0)
    t_total = MPI.Wtime() - t0
    info = comm.gather({"rank": rank, "host": host, "u_ini": u_ini, "u_fin": u_fin,
                        "bytes": fin - ini, "t_local_s": t_local, "afinidad": afinidad,
                        "preparacion_s": preparacion_s}, root=0)
    if rank != 0:
        return

    # -- consolidacion y salida (rank 0) -----------------------------------
    if a.operacion == "conteo":
        resultado = {"hist": total.tolist(), "invalidos_pos": []}
        for p in posiciones:
            resultado = trabajo.combinar(resultado, {"hist": [0] * 256, "invalidos_pos": p})
        resultado["hist"] = total.tolist()
    else:
        resultado = trabajo.vacio()
        for p in parciales:
            resultado = trabajo.combinar(resultado, p)
    resumen = resumen_mpi(trabajo, resultado, info, t_total, size, a)
    if a.resultados:
        from pdn.master.resultados import carpeta_corrida, guardar  # noqa: PLC0415
        carpeta = carpeta_corrida(a.resultados, a.operacion + "_mpi", "np%d" % size)
        resumen["carpeta"] = carpeta
        guardar(carpeta, {k: v for k, v in resumen.items() if k not in ("resultado", "clave")},
                resumen["resultado"], info, vars(a))
    if a.salida:
        os.makedirs(os.path.dirname(os.path.abspath(a.salida)), exist_ok=True)
        with open(a.salida, "w", encoding="utf-8") as f:
            json.dump(resumen, f, ensure_ascii=False, default=str)
    print(json.dumps({k: resumen[k] for k in ("operacion", "ranks", "tiempo_s", "mb_s")}))


def resumen_mpi(trabajo, parcial: dict, info: list[dict], t_total: float, size: int, a) -> dict:
    """Resumen de una corrida MPI con validacion contra la referencia."""
    from pdn.comun.huellas import leer_huellas  # noqa: PLC0415
    from pdn.master import validacion  # noqa: PLC0415

    cob = validacion.cobertura([(r["u_ini"], r["u_fin"]) for r in info], trabajo.unidades)
    validez = {"cobertura": cob, "bytes_ok": sum(r["bytes"] for r in info) == trabajo.largo}
    if a.referencias:
        ruta_a = _ruta_seq(a.datos, a.archivo)
        huella = leer_huellas(ruta_a[:-4] + ".huellas").get("global")
        if a.archivo_b:
            huella += leer_huellas(_ruta_seq(a.datos, a.archivo_b)[:-4] + ".huellas").get("global")
        ruta = validacion.ruta_referencia(a.referencias, huella, trabajo.nombre, trabajo.params)
        ref = validacion.leer_referencia(ruta)
        if ref:
            dif = validacion.comparar(trabajo.clave(parcial), ref["clave"])
            validez.update({"referencia": ruta, "coincide_referencia": not dif, "discrepancias": dif,
                            "tiempo_referencia_s": ref.get("tiempo_s")})
    validez["valido"] = cob["ok"] and validez["bytes_ok"] and validez.get("coincide_referencia", True)
    t_ref = validez.get("tiempo_referencia_s")
    return {"modo": "mpi", "operacion": trabajo.nombre, "params": trabajo.params, "ranks": size,
            "reparto": a.reparto, "impl": a.impl, "largo_bytes": trabajo.largo, "tiempo_s": t_total,
            "mb_s": trabajo.largo / 1048576 / t_total if t_total > 0 else None,
            "preparacion_s": max(r["preparacion_s"] for r in info),
            "speedup": (t_ref / t_total) if (t_ref and t_total) else None,
            "por_rank": info, "hosts": sorted({r["host"] for r in info}), "validacion": validez,
            "clave": trabajo.clave(parcial), "resultado": trabajo.finalizar(parcial)}


if __name__ == "__main__":
    main()
