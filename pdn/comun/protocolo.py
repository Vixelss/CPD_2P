"""Protocolo de mensajes JSON sobre ZeroMQ (CONTEXTO.md, seccion 7.2).

Canal principal (DEALER del worker -> ROUTER del Master), peticion-respuesta:
cada mensaje del worker recibe exactamente una respuesta.
    Worker -> Master: REGISTRO, CALIBRACION, PEDIR, LISTO, RESULTADO, ERROR
    Master -> Worker: ACEPTADO, PREPARAR, TAREA, ESPERAR, FIN, DETENER

Canal de latidos (otro DEALER del mismo worker), sin respuesta:
    Worker -> Master: LATIDO (cada 1 s, con metricas)
    Master -> Worker: SIMULAR_FALLO, DETENER (ordenes asincronas)

PREPARAR y LISTO se agregan al protocolo del plan: el Master entrega la
configuracion de la corrida antes del cronometro y el worker contesta cuando
ya preparo su motor y midio su calibracion sobre el archivo de la corrida.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

REGISTRO = "REGISTRO"
CALIBRACION = "CALIBRACION"
PEDIR = "PEDIR"
LISTO = "LISTO"
RESULTADO = "RESULTADO"
LATIDO = "LATIDO"
ERROR = "ERROR"

ACEPTADO = "ACEPTADO"
PREPARAR = "PREPARAR"
TAREA = "TAREA"
ESPERAR = "ESPERAR"
FIN = "FIN"
DETENER = "DETENER"
SIMULAR_FALLO = "SIMULAR_FALLO"

VERSION_PROTOCOLO = 1


def _convertir(o: Any) -> Any:
    # Convierte tipos de numpy a tipos JSON
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError("No serializable: %r" % type(o))


def crear(tipo: str, worker_id: str = "", corrida_id: str | None = None, **campos: Any) -> dict:
    """Crea un mensaje con tipo, worker_id, corrida_id y marca de tiempo."""
    msg = {"tipo": tipo, "worker_id": worker_id, "t": time.time(), "v": VERSION_PROTOCOLO}
    if corrida_id is not None:
        msg["corrida_id"] = corrida_id
    msg.update(campos)
    return msg


def codificar(msg: dict) -> bytes:
    """Mensaje a bytes JSON."""
    return json.dumps(msg, default=_convertir, separators=(",", ":")).encode("utf-8")


def decodificar(datos: bytes) -> dict:
    """Bytes JSON a mensaje; lanza ValueError si no es un mensaje valido."""
    msg = json.loads(datos.decode("utf-8"))
    if not isinstance(msg, dict) or "tipo" not in msg:
        raise ValueError("mensaje sin tipo")
    return msg
