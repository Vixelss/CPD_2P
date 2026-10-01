"""Emparejamiento de registros entre dos archivos (portado de P1.4/secuencias.py).

GenBank y RefSeq son el mismo ensamblaje con los registros en distinto orden.
Se empareja primero por cromosoma y luego por longitud unica; lo que no tiene
pareja se informa aparte y no se compara.
"""

from __future__ import annotations

from dataclasses import dataclass

from pdn.preparacion.indice import Registro


@dataclass
class Pareja:
    """Dos registros que se corresponden entre los archivos A y B."""

    a: Registro
    b: Registro
    clave: str
    criterio: str  # 'cromosoma' o 'longitud'
    ia: int = -1   # indice del registro en A
    ib: int = -1   # indice del registro en B

    @property
    def largo(self) -> int:
        """Posiciones comparables: hasta donde alcanzan las dos."""
        return min(self.a.largo, self.b.largo)

    def a_dict(self) -> dict:
        return {"clave": self.clave, "criterio": self.criterio, "nombre": self.a.nombre_corto,
                "ia": self.ia, "ib": self.ib, "inicio_a": self.a.inicio,
                "inicio_b": self.b.inicio, "largo": self.largo,
                "desfase": self.a.largo - self.b.largo}


def _orden(p: Pareja) -> tuple:
    # Cromosomas primero en orden natural; el resto por posicion en A
    if p.criterio != "cromosoma":
        return (2, p.a.inicio, "")
    if p.clave.isdigit():
        return (0, int(p.clave), "")
    return (1, {"X": 0, "Y": 1, "MT": 2}.get(p.clave, 9), p.clave)


def emparejar(regs_a: list[Registro], regs_b: list[Registro]
              ) -> tuple[list[Pareja], list[int], list[int]]:
    """Empareja registros de A y B.

    Devuelve (parejas ordenadas, indices de A sin pareja, indices de B sin pareja).
    """
    parejas: list[Pareja] = []
    usados_a: set[int] = set()
    usados_b: set[int] = set()

    # 1) Por cromosoma, solo si la clave es unica en B
    por_crom_b: dict[str, list[int]] = {}
    for j, r in enumerate(regs_b):
        if r.cromosoma:
            por_crom_b.setdefault(r.cromosoma, []).append(j)
    vistos_a: dict[str, int] = {}
    for r in regs_a:
        if r.cromosoma:
            vistos_a[r.cromosoma] = vistos_a.get(r.cromosoma, 0) + 1
    for i, r in enumerate(regs_a):
        clave = r.cromosoma
        if not clave or vistos_a[clave] != 1:
            continue
        cand = por_crom_b.get(clave) or []
        if len(cand) != 1 or cand[0] in usados_b:
            continue
        usados_a.add(i)
        usados_b.add(cand[0])
        parejas.append(Pareja(r, regs_b[cand[0]], clave, "cromosoma", i, cand[0]))

    # 2) Por longitud unica entre lo que queda
    pend_a = [i for i in range(len(regs_a)) if i not in usados_a]
    pend_b = [j for j in range(len(regs_b)) if j not in usados_b]
    conteo_a: dict[int, int] = {}
    conteo_b: dict[int, int] = {}
    for i in pend_a:
        conteo_a[regs_a[i].largo] = conteo_a.get(regs_a[i].largo, 0) + 1
    for j in pend_b:
        conteo_b[regs_b[j].largo] = conteo_b.get(regs_b[j].largo, 0) + 1
    unico_b = {regs_b[j].largo: j for j in pend_b if conteo_b[regs_b[j].largo] == 1}
    for i in pend_a:
        largo = regs_a[i].largo
        if conteo_a[largo] == 1 and largo in unico_b and largo > 0:
            j = unico_b[largo]
            usados_a.add(i)
            usados_b.add(j)
            parejas.append(Pareja(regs_a[i], regs_b[j], regs_a[i].identificador,
                                  "longitud", i, j))

    parejas.sort(key=_orden)
    solo_a = [i for i in range(len(regs_a)) if i not in usados_a]
    solo_b = [j for j in range(len(regs_b)) if j not in usados_b]
    return parejas, solo_a, solo_b
