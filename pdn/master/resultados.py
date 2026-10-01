"""Persistencia de una corrida en <carpeta>/<fecha_hora>_<operacion>/ (seccion 7.6).

Escribe resumen.json, resultado.json, tareas.csv, config.json y, si hay
datos de monitoreo, recursos.csv y energia.csv.
"""

from __future__ import annotations

import csv
import json
import os
import time


def _escribir_json(ruta: str, datos) -> None:
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=1, ensure_ascii=False, default=str)


def _escribir_csv(ruta: str, filas: list[dict]) -> None:
    if not filas:
        open(ruta, "w").close()
        return
    campos: list[str] = []
    for fila in filas:
        for k in fila:
            if k not in campos:
                campos.append(k)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def carpeta_corrida(base: str, operacion: str, corrida_id: str) -> str:
    """Crea la carpeta de una corrida y devuelve su ruta."""
    nombre = "%s_%s_%s" % (time.strftime("%Y%m%d_%H%M%S"), operacion, corrida_id[-6:])
    ruta = os.path.join(base, nombre)
    os.makedirs(ruta, exist_ok=True)
    return ruta


def guardar(carpeta: str, resumen: dict, resultado: dict, tareas: list[dict], config: dict,
            recursos: list[dict] | None = None, energia: list[dict] | None = None) -> str:
    """Guarda todos los archivos de la corrida y devuelve la carpeta."""
    _escribir_json(os.path.join(carpeta, "resumen.json"), resumen)
    _escribir_json(os.path.join(carpeta, "resultado.json"), resultado)
    _escribir_json(os.path.join(carpeta, "config.json"), config)
    _escribir_csv(os.path.join(carpeta, "tareas.csv"), tareas)
    if recursos is not None:
        _escribir_csv(os.path.join(carpeta, "recursos.csv"), recursos)
    if energia is not None:
        _escribir_csv(os.path.join(carpeta, "energia.csv"), energia)
    return carpeta
