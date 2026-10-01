"""Definicion de un trabajo: operacion + archivos + parametros.

Un trabajo define el "espacio" que se reparte en unidades: el .seq para
conteo, patrones y zonas; el espacio de comparacion para la comparacion.
Traduce un rango [inicio, fin) del espacio a la carga de una TAREA y procesa
esa carga con las operaciones de referencia. Lo usan la referencia
secuencial, el Master y los workers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pdn.comun.unidades import TAM_UNIDAD, num_unidades
from pdn.operaciones import comparacion, operacion
from pdn.preparacion.indice import Indice


@dataclass
class Trabajo:
    """Contexto de una corrida independiente de la plataforma."""

    nombre: str
    params: dict
    indice_a: Indice
    indice_b: Indice | None = None
    tam_unidad: int = TAM_UNIDAD
    espacio: comparacion.EspacioComparacion | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.op = operacion(self.nombre)
        self.params = self.op.validar_parametros(self.params)
        if self.nombre == "comparacion":
            if self.indice_b is None:
                raise ValueError("La comparacion necesita dos archivos")
            if self.espacio is None:
                self.espacio = comparacion.construir_espacio(
                    self.indice_a, self.indice_b, self.params["modo_comparacion"])

    @property
    def largo(self) -> int:
        """Bytes del espacio que se reparte."""
        return self.espacio.total if self.espacio is not None else self.indice_a.total

    @property
    def unidades(self) -> int:
        return num_unidades(self.largo, self.tam_unidad)

    @property
    def verifica_crc(self) -> bool:
        """El CRC por unidad aplica cuando el espacio es el .seq (no en comparacion emparejada)."""
        return self.espacio is None or self.espacio.modo == "posicional"

    def carga(self, inicio: int, fin: int) -> dict:
        """Carga de una tarea para el rango [inicio, fin) del espacio."""
        if self.nombre == "comparacion":
            return {"inicio": inicio, "fin": fin, "solape": 0,
                    "segmentos": self.espacio.tramos(inicio, fin)}
        solape = self.op.solape(self.params)
        fin_ext = min(fin + solape, self.largo)
        return {"inicio": inicio, "fin": fin, "solape": solape,
                "limites": self.indice_a.limites_para(inicio, max(fin_ext, inicio + 1))}

    def procesar(self, seq_a: np.ndarray, seq_b: np.ndarray | None, carga: dict,
                 nucleo=None) -> dict:
        """Procesa una carga (por defecto con el nucleo numpy de referencia)."""
        if self.nombre == "comparacion":
            return self.op.procesar(seq_a, seq_b, carga["segmentos"], self.params, nucleo)
        return self.op.procesar(seq_a, carga["inicio"], carga["fin"], self.params,
                                carga["limites"], nucleo)

    def vacio(self) -> dict:
        return self.op.vacio(self.params)

    def combinar(self, a: dict, b: dict) -> dict:
        return self.op.combinar(a, b, self.params)

    def finalizar(self, parcial: dict) -> dict:
        """Resultado legible de la operacion."""
        if self.nombre == "comparacion":
            return self.op.finalizar(parcial, self.indice_a, self.params, self.espacio,
                                     self.indice_b)
        return self.op.finalizar(parcial, self.indice_a, self.params)

    def clave(self, parcial: dict) -> dict:
        """Parte del resultado que debe coincidir exactamente con la referencia."""
        return self.op.clave_comparable(parcial)

    def rangos(self, tam_tarea: int) -> list[tuple[int, int]]:
        """Divide el espacio en tareas de tam_tarea bytes (multiplo de unidad)."""
        paso = max(1, tam_tarea // self.tam_unidad) * self.tam_unidad
        return [(i, min(i + paso, self.largo)) for i in range(0, self.largo, paso)]

    def a_dict(self) -> dict:
        """Descripcion serializable para enviar a los workers."""
        d = {"operacion": self.nombre, "params": self.params, "tam_unidad": self.tam_unidad,
             "largo": self.largo}
        return d


def ejecutar_secuencial(trabajo: Trabajo, seq_a: np.ndarray, seq_b: np.ndarray | None = None,
                        tam_tarea: int | None = None) -> dict:
    """Ejecuta el trabajo completo en un hilo, tarea por tarea, y combina."""
    total = trabajo.vacio()
    for inicio, fin in trabajo.rangos(tam_tarea or trabajo.largo or 1):
        parcial = trabajo.procesar(seq_a, seq_b, trabajo.carga(inicio, fin))
        total = trabajo.combinar(total, parcial)
    return total
