"""Indice de registros de un .seq (portado de referencias/P1.4/secuencias.py).

Cada registro guarda su cabecera y donde empieza y cuanto mide dentro de la
secuencia limpia. Permite traducir una posicion global del .seq a
(registro, posicion dentro del registro, fila, columna).
"""

from __future__ import annotations

import bisect
import json
import os
import re
from dataclasses import dataclass

from pdn.comun.formato import ANCHO_LINEA, fila_columna

# Cromosoma de ensamblaje principal; vale para GenBank y RefSeq:
#   >CM000663.2 Homo sapiens chromosome 1, GRCh38 reference primary assembly
#   >NC_000001.11 Homo sapiens chromosome 1, GRCh38.p14 Primary Assembly
_CROMOSOMA = re.compile(r"chromosome\s+([0-9]{1,2}|[XY])\s*,[^,]*primary\s+assembly", re.I)
_MITOCONDRIA = re.compile(r"mitochondrion", re.I)


@dataclass
class Registro:
    """Un registro del FASTA en coordenadas del .seq."""

    cabecera: str
    inicio: int
    largo: int

    @property
    def fin(self) -> int:
        return self.inicio + self.largo

    @property
    def identificador(self) -> str:
        return self.cabecera[1:].split(None, 1)[0] if len(self.cabecera) > 1 else ""

    @property
    def descripcion(self) -> str:
        partes = self.cabecera[1:].split(None, 1)
        return partes[1] if len(partes) > 1 else ""

    @property
    def cromosoma(self) -> str | None:
        """Clave de cromosoma ('1'..'22', 'X', 'Y', 'MT') o None."""
        encontrado = _CROMOSOMA.search(self.descripcion)
        if encontrado:
            return encontrado.group(1).upper()
        if _MITOCONDRIA.search(self.descripcion):
            return "MT"
        return None

    @property
    def nombre_corto(self) -> str:
        clave = self.cromosoma
        if clave:
            return "MT" if clave == "MT" else "Cr %s" % clave
        return self.identificador or "(sin nombre)"

    def a_dict(self) -> dict:
        return {"cabecera": self.cabecera, "inicio": self.inicio, "largo": self.largo,
                "identificador": self.identificador, "cromosoma": self.cromosoma}


class Indice:
    """Lista ordenada de registros con busqueda binaria por posicion."""

    def __init__(self, registros: list[Registro], total: int | None = None,
                 ancho: int = ANCHO_LINEA) -> None:
        self.registros = registros
        self.inicios = [r.inicio for r in registros]
        self.total = total if total is not None else (registros[-1].fin if registros else 0)
        self.ancho = ancho

    def __len__(self) -> int:
        return len(self.registros)

    def registro_de(self, pos: int) -> int:
        """Indice del registro que contiene la posicion global pos."""
        i = bisect.bisect_right(self.inicios, pos) - 1
        # Saltar registros vacios que comparten inicio
        while i > 0 and self.registros[i].largo == 0 and self.inicios[i] == pos:
            i -= 1
        return max(i, 0)

    def localizar(self, pos: int) -> dict:
        """Traduce una posicion global a registro, posicion, fila y columna."""
        i = self.registro_de(pos)
        r = self.registros[i]
        dentro = pos - r.inicio
        fila, col = fila_columna(dentro, self.ancho)
        return {"registro": i, "id": r.identificador, "nombre": r.nombre_corto,
                "posicion_global": int(pos), "posicion": int(dentro),
                "fila": fila, "columna": col}

    def limites_para(self, inicio: int, fin: int) -> list[list[int]]:
        """Pares [indice_registro, inicio] de los registros que intersecan
        [inicio, fin), incluido el que contiene a inicio (puede empezar antes)."""
        if not self.registros:
            return [[0, 0]]
        i = self.registro_de(inicio)
        j = bisect.bisect_left(self.inicios, fin)
        return [[k, self.inicios[k]] for k in range(i, max(j, i + 1))]

    def a_dict(self) -> dict:
        return {"total": self.total, "ancho": self.ancho,
                "registros": [r.a_dict() for r in self.registros]}

    @classmethod
    def desde_dict(cls, d: dict) -> "Indice":
        regs = [Registro(r["cabecera"], r["inicio"], r["largo"]) for r in d["registros"]]
        return cls(regs, d.get("total"), d.get("ancho", ANCHO_LINEA))

    def guardar(self, ruta: str) -> None:
        temporal = ruta + ".parcial"
        with open(temporal, "w", encoding="utf-8") as f:
            json.dump(self.a_dict(), f, ensure_ascii=False)
        os.replace(temporal, ruta)

    @classmethod
    def leer(cls, ruta: str) -> "Indice":
        with open(ruta, encoding="utf-8") as f:
            return cls.desde_dict(json.load(f))
