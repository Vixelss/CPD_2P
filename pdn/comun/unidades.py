"""Unidades de reparto: el .seq se divide en unidades fijas (4 MiB por defecto).

Las tareas siempre cubren unidades completas, asi el CRC de cada unidad se
puede verificar contra la tabla .huellas.
"""

from __future__ import annotations

TAM_UNIDAD = 4 * 1024 * 1024


def num_unidades(tam_total: int, tam_unidad: int = TAM_UNIDAD) -> int:
    """Numero de unidades que cubren tam_total bytes (la ultima puede ser menor)."""
    if tam_unidad <= 0:
        raise ValueError("tam_unidad debe ser positivo")
    return (tam_total + tam_unidad - 1) // tam_unidad


def rango_unidades(u_ini: int, u_fin: int, tam_total: int,
                   tam_unidad: int = TAM_UNIDAD) -> tuple[int, int]:
    """Rango de bytes [inicio, fin) de las unidades [u_ini, u_fin)."""
    return u_ini * tam_unidad, min(u_fin * tam_unidad, tam_total)
