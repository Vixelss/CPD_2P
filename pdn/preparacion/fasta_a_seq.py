"""Preparacion de un FASTA: .fna -> .seq + .idx + .huellas + .origen.

Portado de referencias/P1.4/comparador.py (preparar, cache_al_dia) con dos
cambios: el indice se construye en la misma pasada y la huella de origen
incluye un hash de los extremos del archivo. La preparacion queda fuera del
cronometro de cualquier corrida.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from pdn.comun.huellas import (calcular_huellas, guardar_huellas, huella_origen,
                               leer_huellas)
from pdn.comun.unidades import TAM_UNIDAD
from pdn.preparacion.indice import Indice, Registro

BLOQUE_LECTURA = 8 * 1024 * 1024
VERSION_PREPARACION = 1


class ErrorEntrada(Exception):
    """El archivo de entrada no sirve; el mensaje explica por que."""


@dataclass
class Preparado:
    """Rutas y metadatos de un FASTA ya preparado."""

    ruta_fna: str
    ruta_seq: str
    ruta_idx: str
    ruta_huellas: str
    largo: int
    reutilizado: bool

    @property
    def nombre(self) -> str:
        return os.path.basename(self.ruta_seq)

    def indice(self) -> Indice:
        return Indice.leer(self.ruta_idx)

    def huellas(self) -> dict:
        return leer_huellas(self.ruta_huellas)

    def mapear(self) -> np.ndarray:
        return abrir_seq(self.ruta_seq)


def abrir_seq(ruta_seq: str) -> np.ndarray:
    """Abre un .seq con numpy.memmap de solo lectura (arreglo vacio si mide 0)."""
    if os.path.getsize(ruta_seq) == 0:
        return np.zeros(0, dtype=np.uint8)
    return np.memmap(ruta_seq, dtype=np.uint8, mode="r")


def rutas_para(ruta_fna: str, destino: str | None = None) -> dict:
    """Rutas de los archivos derivados de un FASTA."""
    carpeta = destino or os.path.dirname(os.path.abspath(ruta_fna))
    base = os.path.splitext(os.path.basename(ruta_fna))[0]
    raiz = os.path.join(carpeta, base)
    return {"seq": raiz + ".seq", "idx": raiz + ".idx", "huellas": raiz + ".huellas",
            "origen": raiz + ".origen"}


def _al_dia(ruta_fna: str, rutas: dict, tam_unidad: int) -> bool:
    # La cache sirve solo si existe completa y salio de este mismo FASTA
    for clave in ("seq", "idx", "huellas", "origen"):
        if not os.path.exists(rutas[clave]):
            return False
    try:
        with open(rutas["origen"], encoding="utf-8") as f:
            guardada = json.load(f)
        if guardada.get("version") != VERSION_PREPARACION:
            return False
        if guardada.get("huella") != huella_origen(ruta_fna):
            return False
        return leer_huellas(rutas["huellas"]).get("tam_unidad") == tam_unidad
    except (OSError, ValueError):
        return False


def validar_fasta(ruta: str) -> None:
    """Comprueba que la ruta es un archivo legible y no vacio."""
    if not ruta:
        raise ErrorEntrada("No se indico ningun archivo.")
    if not os.path.exists(ruta):
        raise ErrorEntrada('El archivo "%s" no existe.' % ruta)
    if not os.path.isfile(ruta):
        raise ErrorEntrada('"%s" no es un archivo.' % ruta)
    if os.path.getsize(ruta) == 0:
        raise ErrorEntrada('El archivo "%s" esta vacio.' % ruta)


def _bloques_alineados(f) -> Iterator[bytes]:
    # Entrega bloques que terminan en linea completa
    while True:
        bloque = f.read(BLOQUE_LECTURA)
        if not bloque:
            return
        if not bloque.endswith(b"\n"):
            resto = f.readline()
            if resto:
                bloque += resto
        yield bloque


def preparar(ruta_fna: str, destino: str | None = None, forzar: bool = False,
             tam_unidad: int = TAM_UNIDAD,
             progreso: Callable[[int, int], None] | None = None) -> Preparado:
    """Prepara un FASTA y devuelve sus rutas derivadas.

    Recibe la ruta del .fna, la carpeta destino (por defecto la del .fna), si
    se fuerza la regeneracion y el tamano de unidad para las huellas. Reutiliza
    la cache si su huella de origen coincide.
    """
    validar_fasta(ruta_fna)
    rutas = rutas_para(ruta_fna, destino)
    os.makedirs(os.path.dirname(rutas["seq"]), exist_ok=True)

    if not forzar and _al_dia(ruta_fna, rutas, tam_unidad):
        return Preparado(ruta_fna, rutas["seq"], rutas["idx"], rutas["huellas"],
                         os.path.getsize(rutas["seq"]), True)

    huella = huella_origen(ruta_fna)
    tamano = os.path.getsize(ruta_fna)
    # Invalida la cache antes de tocar nada
    if os.path.exists(rutas["origen"]):
        os.remove(rutas["origen"])

    registros: list[Registro] = []
    cabecera: str | None = None
    inicio_reg = 0
    escritos = 0
    leidos = 0
    temporal = rutas["seq"] + ".parcial"
    try:
        with open(ruta_fna, "rb") as entrada, open(temporal, "wb") as salida:
            for bloque in _bloques_alineados(entrada):
                leidos += len(bloque)
                if b">" not in bloque:
                    # Ruta rapida: solo secuencia
                    limpio = bloque.replace(b"\n", b"").replace(b"\r", b"")
                    if limpio and cabecera is None:
                        cabecera, inicio_reg = ">(sin cabecera)", escritos
                    salida.write(limpio)
                    escritos += len(limpio)
                else:
                    # Ruta lenta: hay cabeceras en el bloque
                    for linea in bloque.split(b"\n"):
                        if linea.startswith(b">"):
                            if cabecera is not None:
                                registros.append(Registro(cabecera, inicio_reg,
                                                          escritos - inicio_reg))
                            cabecera = linea.rstrip(b"\r").decode("ascii", "replace").strip()
                            inicio_reg = escritos
                        else:
                            limpio = linea.replace(b"\r", b"")
                            if limpio:
                                if cabecera is None:
                                    cabecera, inicio_reg = ">(sin cabecera)", escritos
                                salida.write(limpio)
                                escritos += len(limpio)
                if progreso is not None:
                    progreso(leidos, tamano)
        if cabecera is not None:
            registros.append(Registro(cabecera, inicio_reg, escritos - inicio_reg))
        os.replace(temporal, rutas["seq"])
    except BaseException:
        if os.path.exists(temporal):
            os.remove(temporal)
        raise

    if escritos == 0:
        raise ErrorEntrada('El archivo "%s" no contiene secuencia de ADN.'
                           % os.path.basename(ruta_fna))

    Indice(registros, escritos).guardar(rutas["idx"])
    guardar_huellas(calcular_huellas(rutas["seq"], tam_unidad), rutas["huellas"])
    # El .origen se escribe al final: sin el, la cache no se da por buena
    with open(rutas["origen"], "w", encoding="utf-8") as f:
        json.dump({"version": VERSION_PREPARACION, "huella": huella,
                   "fna": os.path.basename(ruta_fna)}, f)
    return Preparado(ruta_fna, rutas["seq"], rutas["idx"], rutas["huellas"], escritos, False)
