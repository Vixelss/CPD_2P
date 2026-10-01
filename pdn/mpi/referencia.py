"""Referencia secuencial (CONTEXTO.md, secciones 13.3.1 y 13.4).

Un proceso, un nucleo, un nodo, archivo completo, con la implementacion de
referencia (numpy). Su tiempo es T1, la base del speedup. El resultado se
guarda en referencias_resultados/<huella>_<operacion>_<parametros>.json y
toda corrida posterior (dinamica o MPI) se valida contra el.
"""

from __future__ import annotations

import os
import platform
import socket
import statistics
import time

from pdn.comun.huellas import leer_huellas
from pdn.master import validacion
from pdn.master.servidor import resolver_archivo
from pdn.operaciones.trabajo import Trabajo, ejecutar_secuencial
from pdn.preparacion.indice import Indice

TROZO = 64 * 1024 * 1024


def correr_referencia(cfg: dict, datos: str, carpeta: str | None = None, repeticiones: int = 1,
                      calentamiento: bool = False, nucleo_cpu: int | None = None) -> dict:
    """Ejecuta la referencia secuencial y la guarda. Devuelve el registro guardado."""
    from pdn.comun import config as cfgmod  # noqa: PLC0415

    carpeta = carpeta or cfgmod.carpeta_referencias(cfgmod.cargar())
    prep_a = resolver_archivo(cfg["archivo"], datos)
    prep_b = resolver_archivo(cfg["archivo_b"], datos) if cfg.get("archivo_b") else None
    huella = leer_huellas(prep_a.ruta_huellas)["global"] + (
        leer_huellas(prep_b.ruta_huellas)["global"] if prep_b else "")
    trabajo = Trabajo(cfg.get("operacion", "conteo"), cfg.get("params") or {}, Indice.leer(prep_a.ruta_idx),
                      Indice.leer(prep_b.ruta_idx) if prep_b else None,
                      int(cfg.get("tam_unidad") or leer_huellas(prep_a.ruta_huellas)["tam_unidad"]))
    seq_a = prep_a.mapear()
    seq_b = prep_b.mapear() if prep_b else None
    afinidad_previa = None
    if hasattr(os, "sched_setaffinity"):
        afinidad_previa = os.sched_getaffinity(0)
        nucleo_cpu = nucleo_cpu if nucleo_cpu is not None else min(afinidad_previa)
        os.sched_setaffinity(0, {nucleo_cpu})
    try:
        if calentamiento:
            ejecutar_secuencial(trabajo, seq_a, seq_b, tam_tarea=TROZO)
        tiempos = []
        parcial = None
        for _ in range(max(1, repeticiones)):
            t0 = time.perf_counter()
            parcial = ejecutar_secuencial(trabajo, seq_a, seq_b, tam_tarea=TROZO)
            tiempos.append(time.perf_counter() - t0)
    finally:
        if afinidad_previa is not None:
            os.sched_setaffinity(0, afinidad_previa)
    ruta = validacion.ruta_referencia(carpeta, huella, trabajo.nombre, trabajo.params)
    registro = {"operacion": trabajo.nombre, "params": trabajo.params, "archivo": cfg["archivo"],
                "archivo_b": cfg.get("archivo_b"), "huella": huella, "largo_bytes": trabajo.largo,
                "tiempo_s": statistics.median(tiempos), "tiempos_s": tiempos,
                "calentamiento": calentamiento, "nucleo": nucleo_cpu, "host": socket.gethostname(),
                "cpu": platform.processor() or platform.machine(), "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mb_s": trabajo.largo / 1048576 / statistics.median(tiempos)}
    validacion.guardar_referencia(ruta, trabajo.clave(parcial), registro)
    registro["ruta"] = ruta
    registro["clave"] = trabajo.clave(parcial)
    registro["resultado"] = trabajo.finalizar(parcial)
    return registro
