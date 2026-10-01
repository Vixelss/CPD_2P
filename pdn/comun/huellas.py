"""Huellas: del archivo de origen (.origen) y CRC32 por unidad (.huellas).

La huella de origen evita reutilizar una cache vieja (leccion del P1.4: una
cache con el mismo nombre dejo un indice vacio sin aviso). Los CRC por unidad
permiten al Master detectar copias locales danadas o distintas entre nodos.
"""

from __future__ import annotations

import hashlib
import json
import os
import zlib

import numpy as np

from pdn.comun.unidades import TAM_UNIDAD, num_unidades

_MB = 1024 * 1024


def huella_origen(ruta: str) -> dict:
    """Huella de un archivo: tamano, fecha (ns) y SHA-1 del primer y ultimo MB."""
    datos = os.stat(ruta)
    h = hashlib.sha1()
    with open(ruta, "rb") as f:
        h.update(f.read(_MB))
        if datos.st_size > _MB:
            f.seek(max(_MB, datos.st_size - _MB))
            h.update(f.read(_MB))
    return {"tamano": datos.st_size, "mtime_ns": datos.st_mtime_ns,
            "sha1_extremos": h.hexdigest()}


def crc_bytes(datos: np.ndarray | bytes) -> int:
    """CRC32 de un arreglo de bytes."""
    if isinstance(datos, np.ndarray):
        datos = memoryview(np.ascontiguousarray(datos, dtype=np.uint8))
    return zlib.crc32(datos) & 0xFFFFFFFF


def crc_unidades(seq: np.ndarray, u_ini: int, u_fin: int,
                 tam_unidad: int = TAM_UNIDAD) -> list[int]:
    """CRC32 de cada unidad en [u_ini, u_fin) de un .seq mapeado."""
    total = seq.shape[0]
    salida = []
    for u in range(u_ini, u_fin):
        a = u * tam_unidad
        b = min(a + tam_unidad, total)
        salida.append(crc_bytes(seq[a:b]))
    return salida


def calcular_huellas(ruta_seq: str, tam_unidad: int = TAM_UNIDAD) -> dict:
    """Calcula la tabla de CRC32 por unidad de un .seq."""
    total = os.path.getsize(ruta_seq)
    crcs = []
    with open(ruta_seq, "rb") as f:
        for _ in range(num_unidades(total, tam_unidad)):
            crcs.append(zlib.crc32(f.read(tam_unidad)) & 0xFFFFFFFF)
    return {"tam_unidad": tam_unidad, "tamano": total, "crc": crcs,
            "global": huella_global(crcs)}


def huella_global(crcs: list[int]) -> str:
    """Huella global de un .seq a partir de sus CRC por unidad."""
    return hashlib.sha1(np.asarray(crcs, dtype=np.uint32).tobytes()).hexdigest()[:16]


def guardar_huellas(tabla: dict, ruta: str) -> None:
    """Escribe la tabla de huellas en JSON (escritura atomica)."""
    temporal = ruta + ".parcial"
    with open(temporal, "w", encoding="utf-8") as f:
        json.dump(tabla, f)
    os.replace(temporal, ruta)


def leer_huellas(ruta: str) -> dict:
    """Lee una tabla de huellas escrita por guardar_huellas."""
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)
