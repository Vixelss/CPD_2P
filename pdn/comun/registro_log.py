"""Configuracion del log textual (archivo + consola) en espanol."""

from __future__ import annotations

import collections
import logging
import os
import threading

FORMATO = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class MemoriaLog(logging.Handler):
    """Guarda las ultimas lineas del log para mostrarlas en el dashboard."""

    def __init__(self, maximo: int = 2000) -> None:
        super().__init__()
        self.lineas: collections.deque[str] = collections.deque(maxlen=maximo)
        self.cerrojo_lineas = threading.Lock()
        self.setFormatter(logging.Formatter(FORMATO, "%H:%M:%S"))

    def emit(self, registro: logging.LogRecord) -> None:
        with self.cerrojo_lineas:
            self.lineas.append(self.format(registro))

    def ultimas(self, n: int = 200) -> list[str]:
        with self.cerrojo_lineas:
            return list(self.lineas)[-n:]


MEMORIA = MemoriaLog()


def configurar(nombre: str = "pdn", archivo: str | None = None, nivel: int = logging.INFO) -> logging.Logger:
    """Configura el logger raiz 'pdn' una sola vez. Si el archivo no se puede
    abrir (por ejemplo, NFS caido), se sigue solo con consola."""
    raiz = logging.getLogger("pdn")
    raiz.setLevel(nivel)
    if not getattr(raiz, "_pdn_configurado", False):
        consola = logging.StreamHandler()
        consola.setFormatter(logging.Formatter(FORMATO, "%H:%M:%S"))
        raiz.addHandler(consola)
        raiz.addHandler(MEMORIA)
        raiz._pdn_configurado = True  # type: ignore[attr-defined]
    if archivo:
        try:
            os.makedirs(os.path.dirname(archivo) or ".", exist_ok=True)
            manejador = logging.FileHandler(archivo, encoding="utf-8")
            manejador.setFormatter(logging.Formatter(FORMATO))
            raiz.addHandler(manejador)
        except OSError as e:
            raiz.warning("No se pudo abrir el log %s (%s); se sigue solo con consola", archivo, e)
    return logging.getLogger(nombre)
