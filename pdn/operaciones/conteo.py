"""Operacion 1: conteo y validacion (CONTEXTO.md, seccion 8.1).

Referencia de un hilo: histograma de 256 casillas con numpy.bincount. Todos
los conteos se derivan del histograma con pdn.comun.formato.
"""

from __future__ import annotations

import numpy as np

from pdn.comun.formato import CLASE_INVALIDO, TABLA_CLASES, nombre_byte, resumir_histograma
from pdn.operaciones.comun import primeras

NOMBRE = "conteo"
K_INVALIDOS = 20


def validar_parametros(params: dict | None) -> dict:
    """Normaliza los parametros: k_invalidos (posiciones de invalidos a guardar)."""
    params = dict(params or {})
    k = int(params.get("k_invalidos", K_INVALIDOS))
    if not 0 <= k <= 100000:
        raise ValueError("k_invalidos debe estar entre 0 y 100000")
    params["k_invalidos"] = k
    return params


def solape(params: dict) -> int:
    """Bytes extra que necesita cada tarea: ninguno."""
    return 0


def histograma(datos: np.ndarray) -> np.ndarray:
    """Histograma de 256 casillas (int64) de un arreglo uint8."""
    return np.bincount(np.asarray(datos, dtype=np.uint8), minlength=256).astype(np.int64)


def posiciones_invalidos(datos: np.ndarray, desplazamiento: int, k: int) -> list[list[int]]:
    """Primeras k posiciones globales de bytes invalidos, con su byte."""
    if k <= 0:
        return []
    malos = np.flatnonzero(TABLA_CLASES[datos] == CLASE_INVALIDO)[:k]
    return [[int(desplazamiento + p), int(datos[p])] for p in malos]


def procesar(seq: np.ndarray, inicio: int, fin: int, params: dict,
             limites: list[list[int]] | None = None) -> dict:
    """Procesa el tramo [inicio, fin) del .seq y devuelve el resultado parcial."""
    datos = np.asarray(seq[inicio:fin])
    hist = histograma(datos)
    parcial = {"hist": hist.tolist(), "invalidos_pos": []}
    k = params.get("k_invalidos", K_INVALIDOS)
    if k and int(hist[TABLA_CLASES == CLASE_INVALIDO].sum()):
        parcial["invalidos_pos"] = posiciones_invalidos(datos, inicio, k)
    return parcial


def vacio(params: dict) -> dict:
    """Resultado parcial neutro para combinar."""
    return {"hist": [0] * 256, "invalidos_pos": []}


def combinar(a: dict, b: dict, params: dict) -> dict:
    """Combina dos resultados parciales."""
    k = params.get("k_invalidos", K_INVALIDOS)
    return {"hist": (np.asarray(a["hist"], dtype=np.int64)
                     + np.asarray(b["hist"], dtype=np.int64)).tolist(),
            "invalidos_pos": primeras(a["invalidos_pos"] + b["invalidos_pos"], k)}


def finalizar(parcial: dict, indice=None, params: dict | None = None) -> dict:
    """Resultado legible: conteos derivados y posiciones de invalidos localizadas."""
    resumen = resumir_histograma(parcial["hist"])
    posiciones = []
    for pos, byte in parcial["invalidos_pos"]:
        fila = {"posicion_global": pos, "byte": nombre_byte(byte)}
        if indice is not None:
            fila.update(indice.localizar(pos))
        posiciones.append(fila)
    resumen["primeros_invalidos"] = posiciones
    resumen["hist"] = list(parcial["hist"])
    return resumen


def clave_comparable(resultado: dict) -> dict:
    """Parte del resultado que debe coincidir exactamente con la referencia."""
    return {"hist": list(resultado["hist"]),
            "invalidos_pos": [list(x) for x in resultado.get("invalidos_pos", [])]}
