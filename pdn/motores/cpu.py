"""Motor CPU: pool persistente de procesos con afinidad de nucleos (seccion 10.1).

- El pool se crea una vez en preparar(), fuera de cualquier cronometro.
- Cada proceso fija su afinidad a un nucleo con os.sched_setaffinity en el
  inicializador y abre los .seq con numpy.memmap (sin copiar datos).
- Una tarea se parte en tantos tramos como procesos; cada tramo se procesa
  con la operacion de referencia y el nucleo elegido (numpy o SIMD) y luego
  se combinan los parciales.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
import time

import numpy as np

from pdn.operaciones import operacion
from pdn.worker.hardware import elegir_nucleos, topologia_cpu

IMPLEMENTACIONES = ("numpy", "simd", "simd_escalar")
TRAMO_MINIMO = 256 * 1024

# Estado de cada proceso del pool
_NUCLEO_ASIGNADO: int | None = None
_AFINIDAD_MSG: str | None = None
_MAPAS: dict[str, np.ndarray] = {}
_NUCLEO_CALCULO = None


def _abrir(ruta: str) -> np.ndarray:
    # Cache de memmaps por proceso
    if ruta not in _MAPAS:
        from pdn.preparacion.fasta_a_seq import abrir_seq  # noqa: PLC0415
        _MAPAS[ruta] = abrir_seq(ruta)
    return _MAPAS[ruta]


def _crear_nucleo(impl: str):
    if impl == "numpy":
        from pdn.operaciones.nucleo import NUMPY  # noqa: PLC0415
        return NUMPY
    from pdn.motores.simd.envoltorio import nucleo_simd  # noqa: PLC0415
    return nucleo_simd("avx2" if impl == "simd" else "escalar")


def _inicializar(cola, impl: str) -> None:
    # Inicializador de cada proceso: fija afinidad y carga el nucleo
    global _NUCLEO_ASIGNADO, _AFINIDAD_MSG, _NUCLEO_CALCULO
    nucleo = cola.get()
    if nucleo is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {nucleo})
            _NUCLEO_ASIGNADO = nucleo
        except OSError as e:
            _AFINIDAD_MSG = "no se pudo fijar afinidad al nucleo %d: %s" % (nucleo, e)
    elif nucleo is not None:
        _AFINIDAD_MSG = "afinidad no disponible en macOS"
    _NUCLEO_CALCULO = _crear_nucleo(impl)


def _info_proceso(_: int = 0) -> tuple[int, int | None, str | None]:
    time.sleep(0.05)  # para que cada proceso del pool conteste al menos una vez
    return os.getpid(), _NUCLEO_ASIGNADO, _AFINIDAD_MSG


def _procesar_tramo(args: tuple) -> tuple[dict, int, int | None, float]:
    # Procesa un tramo dentro de un proceso del pool
    nombre, params, ruta_a, ruta_b, sub = args
    t0 = time.perf_counter()
    op = operacion(nombre)
    seq_a = _abrir(ruta_a)
    if nombre == "comparacion":
        parcial = op.procesar(seq_a, _abrir(ruta_b), sub["segmentos"], params, _NUCLEO_CALCULO)
    else:
        parcial = op.procesar(seq_a, sub["inicio"], sub["fin"], params, sub["limites"],
                              _NUCLEO_CALCULO)
    return parcial, os.getpid(), _NUCLEO_ASIGNADO, time.perf_counter() - t0


def contexto_mp() -> str:
    """'fork' en Linux; 'forkserver' si el proceso ya tiene hilos (fork con hilos
    puede bloquearse); 'spawn' en macOS."""
    if not sys.platform.startswith("linux"):
        return "spawn"
    return "fork" if threading.active_count() == 1 else "forkserver"


def partir_carga(carga: dict, partes: int, minimo: int = TRAMO_MINIMO) -> list[dict]:
    """Parte la carga de una tarea en hasta 'partes' tramos contiguos."""
    if "segmentos" in carga:
        total = sum(s[3] for s in carga["segmentos"])
        partes = max(1, min(partes, total // max(minimo, 1) or 1))
        objetivo = -(-total // partes) if total else 0
        grupos: list[list[list[int]]] = [[]]
        llevado = 0
        for pid, a, b, largo in carga["segmentos"]:
            desde = 0
            while desde < largo:
                cabe = objetivo - llevado if objetivo else largo
                trozo = min(cabe, largo - desde)
                grupos[-1].append([pid, a + desde, b + desde, trozo])
                desde += trozo
                llevado += trozo
                if objetivo and llevado >= objetivo and len(grupos) < partes:
                    grupos.append([])
                    llevado = 0
        return [{"segmentos": g} for g in grupos if g]
    inicio, fin = carga["inicio"], carga["fin"]
    largo = fin - inicio
    partes = max(1, min(partes, largo // max(minimo, 1) or 1))
    cortes = [inicio + (largo * k) // partes for k in range(partes + 1)]
    return [{"inicio": cortes[k], "fin": cortes[k + 1], "limites": carga["limites"]}
            for k in range(partes) if cortes[k + 1] > cortes[k]]


class MotorCPU:
    """Motor CPU con pool persistente y afinidad de nucleos."""

    dispositivo = "cpu"

    def __init__(self, procesos: int | None = None, nucleos: str | list[int] | None = "todos",
                 impl: str = "numpy", reservar: int = 0) -> None:
        if impl not in IMPLEMENTACIONES:
            raise ValueError("Implementacion de CPU desconocida: %r (validas: %s)"
                             % (impl, ", ".join(IMPLEMENTACIONES)))
        self.topologia = topologia_cpu()
        self.nucleos = elegir_nucleos(nucleos, self.topologia)
        logicos = len(self.topologia["logicos"])
        if reservar and len(self.nucleos) > reservar:
            # Deja nucleos libres para el hilo que alimenta a la GPU
            self.nucleos = self.nucleos[:len(self.nucleos) - reservar]
        if procesos is None:
            procesos = len(self.nucleos)
        if not 1 <= procesos <= logicos:
            raise ValueError("Se pidieron %d procesos y este nodo tiene %d hilos (rango valido: 1 a %d)"
                             % (procesos, logicos, logicos))
        self.procesos = procesos
        self.impl = impl
        self.pool = None
        self.asignacion: dict[int, int | None] = {}
        self.avisos: list[str] = []
        self.preparacion_s: float | None = None

    @property
    def nombre(self) -> str:
        return "cpu-%s" % self.impl

    def preparar(self) -> float:
        """Crea el pool, fija afinidades y carga el nucleo. Devuelve los segundos."""
        t0 = time.perf_counter()
        if self.impl != "numpy":
            _crear_nucleo(self.impl)  # falla aqui, con mensaje claro, si no hay SIMD
        ctx = mp.get_context(contexto_mp())
        cola = ctx.SimpleQueue()  # sin hilo alimentador: fork seguro
        for k in range(self.procesos):
            cola.put(self.nucleos[k % len(self.nucleos)] if self.nucleos else None)
        self.pool = ctx.Pool(self.procesos, initializer=_inicializar, initargs=(cola, self.impl))
        for pid, nucleo, aviso in self.pool.map(_info_proceso, range(self.procesos * 4), chunksize=1):
            self.asignacion[pid] = nucleo
            if aviso and aviso not in self.avisos:
                self.avisos.append(aviso)
        self.preparacion_s = time.perf_counter() - t0
        return self.preparacion_s

    def procesar(self, nombre: str, params: dict, ruta_a: str, ruta_b: str | None,
                 carga: dict) -> tuple[dict, dict]:
        """Procesa una tarea. Devuelve (parcial, info) con nucleos usados y tiempos."""
        if self.pool is None:
            raise RuntimeError("El motor CPU no esta preparado: llame a preparar() antes")
        op = operacion(nombre)
        tramos = partir_carga(carga, self.procesos)
        t0 = time.perf_counter()
        resultados = self.pool.map(_procesar_tramo,
                                   [(nombre, params, ruta_a, ruta_b, t) for t in tramos],
                                   chunksize=1)
        parcial = op.vacio(params)
        usados = []
        for r, pid, nucleo, seg in resultados:
            parcial = op.combinar(parcial, r, params)
            usados.append({"pid": pid, "nucleo": nucleo, "s": round(seg, 6)})
        return parcial, {"t_calculo": time.perf_counter() - t0, "tramos": len(tramos),
                         "procesos": usados, "impl": self.impl}

    def cerrar(self) -> None:
        """Termina el pool."""
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def __enter__(self) -> "MotorCPU":
        self.preparar()
        return self

    def __exit__(self, *exc) -> None:
        self.cerrar()

    def describir(self) -> dict:
        """Configuracion efectiva del motor (para el registro y el informe)."""
        return {"dispositivo": "cpu", "impl": self.impl, "procesos": self.procesos,
                "nucleos": self.nucleos, "asignacion": {str(k): v for k, v in self.asignacion.items()},
                "avisos": self.avisos, "preparacion_s": self.preparacion_s,
                "hibrido": self.topologia["hibrido"]}
