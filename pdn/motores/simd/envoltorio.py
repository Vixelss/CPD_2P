"""Envoltorio ctypes del nucleo SIMD en C (simd_adn.c).

Carga libsimd_avx2.so o libsimd_escalar.so desde pdn/motores/simd/build/. Si
no existen y hay gcc, las compila con scripts/compilar_simd.sh. La version
AVX2 solo se carga si el procesador declara el flag avx2.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess

import numpy as np

from pdn.comun.formato import CLASE_INVALIDO, TABLA_CLASES

CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SIMBOLOS = np.frombuffer(b"ACGTNRYSWKMBDHVacgtnryswkmbdhv", dtype=np.uint8)
_INVALIDOS = np.flatnonzero(TABLA_CLASES == CLASE_INVALIDO)
_TROZO_BUSQUEDA = 4 * 1024 * 1024

_u8p = ctypes.POINTER(ctypes.c_uint8)
_u64p = ctypes.POINTER(ctypes.c_uint64)


class SIMDNoDisponible(RuntimeError):
    """El nucleo SIMD no se puede usar en esta maquina; el mensaje dice por que."""


def tiene_avx2() -> bool:
    """True si el procesador declara AVX2 (solo Linux x86)."""
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            return " avx2" in f.read()
    except OSError:
        return False


def _compilar() -> None:
    script = os.path.join(RAIZ, "scripts", "compilar_simd.sh")
    try:
        subprocess.run(["bash", script], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as e:
        detalle = getattr(e, "stderr", "") or str(e)
        raise SIMDNoDisponible("No se pudo compilar el nucleo SIMD: %s" % detalle.strip()) from e


def _cargar(variante: str) -> ctypes.CDLL:
    if variante not in ("avx2", "escalar"):
        raise ValueError("variante SIMD desconocida: %s" % variante)
    if variante == "avx2" and not tiene_avx2():
        raise SIMDNoDisponible("Este procesador no tiene AVX2 (en la Mac ARM no existe); "
                               "use la implementacion numpy")
    ruta = os.path.join(CARPETA, "libsimd_%s.so" % variante)
    if not os.path.exists(ruta):
        _compilar()
    if not os.path.exists(ruta):
        raise SIMDNoDisponible("No existe %s; ejecute scripts/compilar_simd.sh" % ruta)
    lib = ctypes.CDLL(ruta)
    lib.contar_simbolos_avx2.argtypes = [_u8p, ctypes.c_size_t, _u64p]
    lib.contar_simbolos_avx2.restype = ctypes.c_uint64
    lib.histograma_escalar.argtypes = [_u8p, ctypes.c_size_t, _u64p]
    lib.histograma_escalar.restype = None
    lib.comparar_avx2.argtypes = [_u8p, _u8p, ctypes.c_size_t, _u64p]
    lib.comparar_avx2.restype = ctypes.c_uint64
    lib.buscar_patron_avx2.argtypes = [_u8p, ctypes.c_size_t, _u8p, ctypes.c_size_t, _u64p,
                                       ctypes.c_uint64]
    lib.buscar_patron_avx2.restype = ctypes.c_uint64
    lib.es_avx2.restype = ctypes.c_int
    if (lib.es_avx2() == 1) != (variante == "avx2"):
        raise SIMDNoDisponible("%s no corresponde a la variante %s" % (ruta, variante))
    return lib


def _ptr8(a: np.ndarray):
    return a.ctypes.data_as(_u8p)


def _ptr64(a: np.ndarray):
    return a.ctypes.data_as(_u64p)


def _contiguo(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a if a.flags["C_CONTIGUOUS"] and a.dtype == np.uint8 else np.ascontiguousarray(a, dtype=np.uint8)


class NucleoSIMD:
    """Nucleo de calculo con la biblioteca C (avx2 o escalar)."""

    def __init__(self, variante: str = "avx2") -> None:
        self.variante = variante
        self.nombre = "simd_" + variante
        self.lib = _cargar(variante)

    def histograma(self, datos: np.ndarray) -> np.ndarray:
        """Histograma de 256 casillas, identico a numpy.bincount.

        Los 30 simbolos validos se cuentan con comparaciones vectoriales. Solo
        si hay invalidos (raro en un genoma real) se recorre el tramo con el
        histograma escalar para obtener su desglose por byte.
        """
        d = _contiguo(datos)
        n = d.shape[0]
        hist = np.zeros(256, dtype=np.int64)
        if n == 0:
            return hist
        cuentas = np.zeros(30, dtype=np.uint64)
        invalidos = self.lib.contar_simbolos_avx2(_ptr8(d), n, _ptr64(cuentas))
        hist[SIMBOLOS] = cuentas.astype(np.int64)
        if invalidos:
            completo = np.zeros(256, dtype=np.uint64)
            self.lib.histograma_escalar(_ptr8(d), n, _ptr64(completo))
            hist[_INVALIDOS] = completo[_INVALIDOS].astype(np.int64)
        return hist

    def histograma_completo(self, datos: np.ndarray) -> np.ndarray:
        """Histograma escalar de 256 casillas (para el benchmark)."""
        d = _contiguo(datos)
        completo = np.zeros(256, dtype=np.uint64)
        self.lib.histograma_escalar(_ptr8(d), d.shape[0], _ptr64(completo))
        return completo.astype(np.int64)

    def coincidencias(self, mayus: np.ndarray, patron: str) -> np.ndarray:
        """Mascara de inicios del patron (mismo contrato que NucleoNumpy)."""
        d = _contiguo(mayus)
        m = len(patron)
        n = d.shape[0] - m + 1
        if n <= 0:
            return np.zeros(0, dtype=bool)
        mascara = np.zeros(n, dtype=bool)
        p = np.frombuffer(patron.upper().encode("ascii"), dtype=np.uint8).copy()
        buf = np.empty(min(_TROZO_BUSQUEDA, n), dtype=np.uint64)
        for c in range(0, n, _TROZO_BUSQUEDA):
            sub = d[c:min(c + _TROZO_BUSQUEDA + m - 1, d.shape[0])]
            cupo = min(_TROZO_BUSQUEDA, n - c)
            hallados = self.lib.buscar_patron_avx2(_ptr8(sub), sub.shape[0], _ptr8(p), m,
                                                   _ptr64(buf), cupo)
            mascara[c + buf[:hallados].astype(np.int64)] = True
        return mascara

    def contar_categorias(self, a: np.ndarray, b: np.ndarray) -> tuple[int, int, int]:
        """Diferencias (solo_caso, con_n, reales)."""
        x, y = _contiguo(a), _contiguo(b)
        cats = np.zeros(3, dtype=np.uint64)
        self.lib.comparar_avx2(_ptr8(x), _ptr8(y), x.shape[0], _ptr64(cats))
        return int(cats[0]), int(cats[1]), int(cats[2])


_CACHE: dict[str, NucleoSIMD] = {}


def nucleo_simd(variante: str = "avx2") -> NucleoSIMD:
    """Nucleo SIMD cacheado por proceso."""
    if variante not in _CACHE:
        _CACHE[variante] = NucleoSIMD(variante)
    return _CACHE[variante]
