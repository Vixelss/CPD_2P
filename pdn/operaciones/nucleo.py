"""Nucleo de calculo intercambiable que usan las operaciones.

Las operaciones de referencia definen QUE se calcula (costuras, limites de
registro, consolidacion). El nucleo define COMO se calculan las tres piezas
pesadas: el histograma, las coincidencias de un patron y las diferencias por
categoria. La referencia usa NucleoNumpy; el motor CPU puede usar el nucleo
SIMD en C (pdn/motores/simd/envoltorio.py) y el motor GPU sus kernels. Asi el
algoritmo es el mismo en todas las plataformas.
"""

from __future__ import annotations

import numpy as np

_N = ord("N")


class NucleoNumpy:
    """Implementacion de referencia con numpy."""

    nombre = "numpy"

    def histograma(self, datos: np.ndarray) -> np.ndarray:
        """Histograma de 256 casillas (int64)."""
        return np.bincount(np.asarray(datos, dtype=np.uint8), minlength=256).astype(np.int64)

    def coincidencias(self, mayus: np.ndarray, patron: str) -> np.ndarray:
        """Mascara booleana (largo n - m + 1) de inicios del patron en datos en mayuscula."""
        n = mayus.shape[0] - len(patron) + 1
        if n <= 0:
            return np.zeros(0, dtype=bool)
        coincide = np.ones(n, dtype=bool)
        for i, letra in enumerate(patron):
            ventana = mayus[i:i + n]
            if letra == "N":
                coincide &= (ventana == 65) | (ventana == 67) | (ventana == 71) | (ventana == 84)
            else:
                coincide &= ventana == ord(letra)
        return coincide

    def contar_categorias(self, a: np.ndarray, b: np.ndarray) -> tuple[int, int, int]:
        """Diferencias (solo_caso, con_n, reales) entre dos tramos del mismo largo."""
        return tuple(int(m.sum()) for m in clasificar(a, b))  # type: ignore[return-value]


def clasificar(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mascaras (solo_caso, con_n, reales) de dos tramos del mismo largo."""
    dif = a != b
    ua = a & 0xDF
    ub = b & 0xDF
    solo_caso = dif & (ua == ub)
    resto = dif & ~solo_caso
    con_n = resto & ((ua == _N) | (ub == _N))
    reales = resto & ~con_n
    return solo_caso, con_n, reales


NUMPY = NucleoNumpy()
