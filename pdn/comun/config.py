"""Lectura de cluster.yaml con valores por defecto.

Cero rutas o IPs fijas en el codigo: todo sale de aqui o de la linea de
comandos.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import yaml

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUTA_POR_DEFECTO = os.path.join(RAIZ, "cluster.yaml")

DEFECTOS: dict[str, Any] = {
    "red": {"subred": "127.0.0.0/8", "puerto_tareas": 5555, "puerto_respaldo": 5556,
            "puerto_dashboard": 8000},
    "rutas": {"codigo": RAIZ, "datos": os.path.expanduser("~/pdn-datos"), "nfs": "/cluster",
              "entorno": os.path.expanduser("~/pdn-env")},
    "nodos": [],
    "master": {"respaldo_snapshot_s": 1.0, "respaldo_timeout_s": 3.0, "latido_timeout_s": 5.0,
               "tarea_vencida_min_s": 10.0, "tarea_vencida_factor": 3.0,
               "preparacion_timeout_s": 300.0, "rechazos_maximos": 2},
    "planificador": {"estrategia": "adaptativa", "tiempo_objetivo_s": 0.5, "max_tarea_mb": 128,
                     "max_tarea_gpu_mb": 512, "tam_fijo_mb": 16, "alfa_velocidad": 0.3},
    "worker": {"latido_s": 1.0, "calibracion_mb": 64, "master_timeout_s": 3.0,
               "reserva_gpu_nucleos": 1},
}


def _mezclar(base: dict, extra: dict) -> dict:
    salida = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(salida.get(k), dict):
            salida[k] = _mezclar(salida[k], v)
        else:
            salida[k] = v
    return salida


def cargar(ruta: str | None = None) -> dict:
    """Carga cluster.yaml (o la ruta dada) mezclado con los valores por defecto."""
    ruta = ruta or os.environ.get("PDN_CLUSTER_YAML") or RUTA_POR_DEFECTO
    datos: dict = {}
    if ruta and os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            datos = yaml.safe_load(f) or {}
    return _mezclar(DEFECTOS, datos)


def nodo(config: dict, hostname: str) -> dict | None:
    """Entrada de un nodo por hostname, con sus rutas propias mezcladas."""
    for n in config.get("nodos", []):
        if n.get("hostname") == hostname:
            return {**n, "rutas": _mezclar(config["rutas"], n.get("rutas", {}))}
    return None


def nodos_con_rol(config: dict, rol: str) -> list[dict]:
    """Nodos que tienen un rol (master, respaldo, worker)."""
    return [n for n in config.get("nodos", []) if rol in n.get("roles", [])]


def carpeta_resultados(config: dict) -> str:
    """Carpeta de resultados: /cluster/resultados si existe y se puede escribir; si no, ./resultados."""
    nfs = os.path.join(config["rutas"]["nfs"], "resultados")
    try:
        os.makedirs(nfs, exist_ok=True)
        if os.access(nfs, os.W_OK):
            return nfs
    except OSError:
        pass
    local = os.path.join(RAIZ, "resultados")
    os.makedirs(local, exist_ok=True)
    return local


def carpeta_referencias(config: dict) -> str:
    """Carpeta de resultados de referencia: hermana de la de resultados."""
    return os.path.join(os.path.dirname(carpeta_resultados(config).rstrip("/")), "referencias_resultados")


def nodos_mpi(config: dict) -> list[dict]:
    """Nodos Linux con rol master o worker (la Mac no participa en MPI), en orden."""
    return [n for n in config.get("nodos", []) if n.get("sistema", "linux") == "linux"
            and ({"master", "worker"} & set(n.get("roles", [])))]
