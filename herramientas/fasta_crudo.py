"""Ayudantes para manipular un FASTA crudo sin cambiar su estructura de lineas."""

from __future__ import annotations

import numpy as np


def mascara_secuencia(crudo: np.ndarray) -> np.ndarray:
    """Mascara de bytes que son secuencia (no cabecera, no \\r ni \\n)."""
    salto = crudo == 10
    # Numero de linea de cada byte y si esa linea es cabecera
    linea = np.concatenate(([0], np.cumsum(salto)[:-1]))
    inicios = np.concatenate(([0], np.flatnonzero(salto) + 1))
    inicios = inicios[inicios < crudo.shape[0]]
    es_cab = np.zeros(int(linea[-1]) + 2 if crudo.size else 1, dtype=bool)
    es_cab[linea[inicios]] = crudo[inicios] == ord(">")
    return ~es_cab[linea] & ~salto & (crudo != 13)


def leer_recorte(ruta: str, mb: float | None) -> bytes:
    """Lee el archivo completo o sus primeros mb MB cortados en linea completa."""
    with open(ruta, "rb") as f:
        if mb is None:
            return f.read()
        datos = f.read(int(mb * 1024 * 1024))
        return datos + f.readline()
