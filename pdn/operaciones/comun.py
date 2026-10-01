"""Utilidades compartidas por las operaciones de referencia."""

from __future__ import annotations

import heapq

import numpy as np

TOPE_POSICIONES = 1000


def primeras(lista: list, k: int, clave=None) -> list:
    """Los k menores elementos de una lista, ordenados."""
    if k <= 0:
        return []
    if len(lista) <= k:
        return sorted(lista, key=clave)
    return heapq.nsmallest(k, lista, key=clave)


def sumar_dict(a: dict, b: dict) -> dict:
    """Suma dos diccionarios clave -> entero (o lista de enteros)."""
    salida = dict(a)
    for k, v in b.items():
        if k in salida:
            if isinstance(v, list):
                salida[k] = [x + y for x, y in zip(salida[k], v)]
            else:
                salida[k] = salida[k] + v
        else:
            salida[k] = list(v) if isinstance(v, list) else v
    return salida


def registro_de(posiciones: np.ndarray, limites: list[list[int]]) -> np.ndarray:
    """Indice global de registro de cada posicion, segun los limites de la tarea."""
    inicios = np.asarray([l[1] for l in limites], dtype=np.int64)
    indices = np.asarray([l[0] for l in limites], dtype=np.int64)
    local = np.searchsorted(inicios, posiciones, side="right") - 1
    return indices[np.clip(local, 0, len(indices) - 1)]


def tramos_de_registro(inicio: int, fin_datos: int, limites: list[list[int]],
                       total: int) -> list[tuple[int, int, int]]:
    """Tramos (indice_registro, ini, fin) en que los registros cortan [inicio, fin_datos)."""
    tramos = []
    for n, (idx, ini) in enumerate(limites):
        sig = limites[n + 1][1] if n + 1 < len(limites) else total
        a = max(ini, inicio)
        b = min(sig, fin_datos)
        if b > a:
            tramos.append((idx, a, b))
    return tramos
