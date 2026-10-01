"""Reglas del formato FASTA y clasificacion de bytes (CONTEXTO.md, seccion 6.1).

Todos los motores cuentan con un histograma de 256 casillas y de aqui se
derivan los conteos. Este modulo es la unica definicion de que es base, N,
IUPAC o invalido.
"""

from __future__ import annotations

import numpy as np

# Ancho de linea de los archivos reales. Solo se usa para reportar fila y
# columna, nunca para leer.
ANCHO_LINEA = 80

BASES = "ACGT"
DESCONOCIDA = "N"
IUPAC = "RYSWKMBDHV"

# Clases de byte
CLASE_INVALIDO = 0
CLASE_BASE = 1
CLASE_N = 2
CLASE_IUPAC = 3


def _tabla_clases() -> np.ndarray:
    # Construye la tabla de 256 entradas byte -> clase
    tabla = np.zeros(256, dtype=np.uint8)
    for letra in BASES:
        tabla[ord(letra)] = CLASE_BASE
        tabla[ord(letra.lower())] = CLASE_BASE
    for letra in DESCONOCIDA:
        tabla[ord(letra)] = CLASE_N
        tabla[ord(letra.lower())] = CLASE_N
    for letra in IUPAC:
        tabla[ord(letra)] = CLASE_IUPAC
        tabla[ord(letra.lower())] = CLASE_IUPAC
    return tabla


TABLA_CLASES: np.ndarray = _tabla_clases()

# Mascara de bytes que son base A C G T en cualquier caso (para zonas)
ES_ACGT: np.ndarray = TABLA_CLASES == CLASE_BASE


def nombre_byte(valor: int) -> str:
    """Representacion legible de un byte: '5', ' ' o 0xC3."""
    valor = int(valor)
    if 32 <= valor < 127:
        return repr(chr(valor))
    return "0x%02X" % valor


def histograma_vacio() -> np.ndarray:
    """Histograma de 256 casillas en cero (int64)."""
    return np.zeros(256, dtype=np.int64)


def resumir_histograma(hist: np.ndarray | list[int]) -> dict:
    """Deriva todos los conteos de un histograma de 256 casillas.

    Recibe el histograma y devuelve un diccionario con A, C, G, T (sumando
    mayuscula y minuscula), N, IUPAC por letra y total, invalidos total y su
    desglose por byte, y el total de bytes.
    """
    h = np.asarray(hist, dtype=np.int64)
    if h.shape != (256,):
        raise ValueError("el histograma debe tener 256 casillas")

    def ambos(letra: str) -> int:
        return int(h[ord(letra)] + h[ord(letra.lower())])

    bases = {b: ambos(b) for b in BASES}
    iupac = {l: ambos(l) for l in IUPAC}
    invalidos = {}
    for valor in range(256):
        if TABLA_CLASES[valor] == CLASE_INVALIDO and h[valor]:
            invalidos[nombre_byte(valor)] = int(h[valor])
    return {
        "A": bases["A"],
        "C": bases["C"],
        "G": bases["G"],
        "T": bases["T"],
        "N": ambos("N"),
        "iupac": {k: v for k, v in iupac.items() if v},
        "iupac_total": sum(iupac.values()),
        "invalidos_total": sum(invalidos.values()),
        "invalidos": invalidos,
        "minusculas": int(sum(h[ord(c)] for c in "acgtn")),
        "total": int(h.sum()),
    }


def fila_columna(pos_en_registro: int, ancho: int = ANCHO_LINEA) -> tuple[int, int]:
    """Convierte una posicion dentro de un registro en (fila, columna), base 1."""
    return pos_en_registro // ancho + 1, pos_en_registro % ancho + 1


def complemento_inverso(patron: str) -> str:
    """Complemento inverso de un patron de ADN (A<->T, C<->G, N->N)."""
    tabla = str.maketrans("ACGTN", "TGCAN")
    return patron.upper().translate(tabla)[::-1]
