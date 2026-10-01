"""Operaciones de referencia (un hilo). Todo motor se valida contra ellas."""

from __future__ import annotations

from types import ModuleType

from pdn.operaciones import comparacion, conteo, patrones, zonas

OPERACIONES: dict[str, ModuleType] = {
    "conteo": conteo,
    "patrones": patrones,
    "comparacion": comparacion,
    "zonas": zonas,
}


def operacion(nombre: str) -> ModuleType:
    """Devuelve el modulo de una operacion por su nombre."""
    try:
        return OPERACIONES[nombre]
    except KeyError:
        raise ValueError("Operacion desconocida: %r (validas: %s)"
                         % (nombre, ", ".join(OPERACIONES))) from None
