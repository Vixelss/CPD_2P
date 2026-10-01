"""Operacion 3: comparacion de dos cadenas (CONTEXTO.md, seccion 8.3).

El trabajo se define sobre un "espacio de comparacion": en modo emparejado es
la concatenacion virtual de las parejas (en su orden), en modo posicional es
[0, min(largo_A, largo_B)). Una tarea es un rango de ese espacio y se traduce
a segmentos (pareja, inicio_a, inicio_b, largo). Asi las parejas pequenas se
agrupan solas en una misma tarea.

Categorias de diferencia (mutuamente excluyentes, en este orden):
- solo_caso: a != b y (a & 0xDF) == (b & 0xDF), p. ej. 'a' contra 'A'.
- con_n: una de las dos posiciones es N (en cualquier caso).
- reales: el resto.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from pdn.operaciones.comun import SUBTRAMO, TOPE_POSICIONES, primeras
from pdn.operaciones.nucleo import NUMPY, clasificar
from pdn.preparacion.emparejar import Pareja, emparejar
from pdn.preparacion.indice import Indice

NOMBRE = "comparacion"
CATEGORIAS = ("solo_caso", "con_n", "reales")


def validar_parametros(params: dict | None) -> dict:
    """Normaliza modo ('emparejado' o 'posicional') y tope de posiciones."""
    params = dict(params or {})
    modo = params.get("modo_comparacion", "emparejado")
    if modo not in ("emparejado", "posicional"):
        raise ValueError("modo_comparacion debe ser 'emparejado' o 'posicional'")
    params["modo_comparacion"] = modo
    params["tope"] = int(params.get("tope", TOPE_POSICIONES))
    return params


def solape(params: dict) -> int:
    """La comparacion no necesita bytes extra."""
    return 0


@dataclass
class EspacioComparacion:
    """Traduce rangos del espacio de comparacion a segmentos de A y B."""

    segmentos: list[tuple[int, int, int, int]]  # (id_pareja, ini_a, ini_b, largo)
    parejas: list[dict]
    solo_a: list[int]
    solo_b: list[int]
    modo: str

    def __post_init__(self) -> None:
        self.inicios_v = []
        acumulado = 0
        for seg in self.segmentos:
            self.inicios_v.append(acumulado)
            acumulado += seg[3]
        self.total = acumulado

    def tramos(self, v_ini: int, v_fin: int) -> list[list[int]]:
        """Segmentos [id, ini_a, ini_b, largo] que cubren [v_ini, v_fin)."""
        salida = []
        i = max(bisect.bisect_right(self.inicios_v, v_ini) - 1, 0)
        while i < len(self.segmentos) and self.inicios_v[i] < v_fin:
            pid, a, b, largo = self.segmentos[i]
            desde = max(v_ini - self.inicios_v[i], 0)
            hasta = min(v_fin - self.inicios_v[i], largo)
            if hasta > desde:
                salida.append([pid, a + desde, b + desde, hasta - desde])
            i += 1
        return salida


def construir_espacio(indice_a: Indice, indice_b: Indice, modo: str) -> EspacioComparacion:
    """Construye el espacio de comparacion para el modo pedido."""
    if modo == "posicional":
        largo = min(indice_a.total, indice_b.total)
        return EspacioComparacion([(-1, 0, 0, largo)] if largo else [], [], [], [], modo)
    parejas, solo_a, solo_b = emparejar(indice_a.registros, indice_b.registros)
    segs = [(n, p.a.inicio, p.b.inicio, p.largo) for n, p in enumerate(parejas) if p.largo > 0]
    return EspacioComparacion(segs, [p.a_dict() for p in parejas], solo_a, solo_b, modo)


def procesar(seq_a: np.ndarray, seq_b: np.ndarray, segmentos: list[list[int]],
             params: dict, nucleo=None) -> dict:
    """Compara los segmentos dados y devuelve el resultado parcial."""
    nucleo = nucleo or NUMPY
    tope = params["tope"]
    parcial = vacio(params)
    trozos = []
    for pid, ia, ib, largo in segmentos:
        for d in range(0, largo, SUBTRAMO):
            trozos.append((pid, ia + d, ib + d, min(SUBTRAMO, largo - d)))
    for pid, ia, ib, largo in trozos:
        a = np.asarray(seq_a[ia:ia + largo])
        b = np.asarray(seq_b[ib:ib + largo])
        cuentas = list(nucleo.contar_categorias(a, b))
        for cat, c in zip(CATEGORIAS, cuentas):
            parcial[cat] += c
        parcial["comparadas"] += largo
        if pid >= 0:
            previo = parcial["por_pareja"].get(str(pid), [0, 0, 0, 0])
            parcial["por_pareja"][str(pid)] = [previo[0] + largo] + [
                x + c for x, c in zip(previo[1:], cuentas)]
        if sum(cuentas) and len(parcial["posiciones"]) < tope:
            # Las posiciones se buscan con numpy solo si hay diferencias
            mascaras = clasificar(a, b)
            dif = np.flatnonzero(a != b)[:tope]
            for q in dif:
                cat = 0 if mascaras[0][q] else (1 if mascaras[1][q] else 2)
                parcial["posiciones"].append([int(ia + q), int(ib + q), int(a[q]),
                                              int(b[q]), cat])
            parcial["posiciones"] = primeras(parcial["posiciones"], tope)
    return parcial


def vacio(params: dict) -> dict:
    """Resultado parcial neutro."""
    return {"solo_caso": 0, "con_n": 0, "reales": 0, "comparadas": 0,
            "por_pareja": {}, "posiciones": []}


def combinar(a: dict, b: dict, params: dict) -> dict:
    """Combina dos resultados parciales."""
    salida = {k: a[k] + b[k] for k in ("solo_caso", "con_n", "reales", "comparadas")}
    pp = {k: list(v) for k, v in a["por_pareja"].items()}
    for k, v in b["por_pareja"].items():
        pp[k] = [x + y for x, y in zip(pp[k], v)] if k in pp else list(v)
    salida["por_pareja"] = pp
    salida["posiciones"] = primeras(a["posiciones"] + b["posiciones"], params["tope"])
    return salida


def es_valido(parcial: dict) -> bool:
    """Ninguna categoria puede superar las posiciones comparadas (leccion DirectML)."""
    total = parcial["solo_caso"] + parcial["con_n"] + parcial["reales"]
    return all(0 <= parcial[c] <= parcial["comparadas"] for c in CATEGORIAS) \
        and total <= parcial["comparadas"]


def finalizar(parcial: dict, indice_a: Indice | None = None, params: dict | None = None,
              espacio: EspacioComparacion | None = None, indice_b: Indice | None = None) -> dict:
    """Resultado legible: categorias, tabla por pareja y primeras diferencias."""
    from pdn.comun.formato import nombre_byte

    total = parcial["solo_caso"] + parcial["con_n"] + parcial["reales"]
    salida = {"modo": espacio.modo if espacio else None,
              "comparadas": parcial["comparadas"], "solo_caso": parcial["solo_caso"],
              "con_n": parcial["con_n"], "reales": parcial["reales"], "total": total,
              "valido": es_valido(parcial)}
    if espacio is not None and espacio.modo == "emparejado":
        filas = []
        for k, v in sorted(parcial["por_pareja"].items(), key=lambda x: int(x[0])):
            info = dict(espacio.parejas[int(k)])
            info.update({"comparadas": v[0], "solo_caso": v[1], "con_n": v[2],
                         "reales": v[3], "total": v[1] + v[2] + v[3]})
            filas.append(info)
        salida["parejas"] = filas
        salida["num_parejas"] = len(espacio.parejas)
        salida["por_cromosoma"] = sum(1 for p in espacio.parejas if p["criterio"] == "cromosoma")
        salida["por_longitud"] = salida["num_parejas"] - salida["por_cromosoma"]
        salida["sin_pareja_a"] = [indice_a.registros[i].identificador if indice_a else i
                                  for i in espacio.solo_a]
        salida["sin_pareja_b"] = [indice_b.registros[j].identificador if indice_b else j
                                  for j in espacio.solo_b]
    difs = []
    for pa, pb, ba, bb, cat in parcial["posiciones"]:
        d = {"posicion_a": pa, "posicion_b": pb, "byte_a": nombre_byte(ba),
             "byte_b": nombre_byte(bb), "categoria": CATEGORIAS[cat]}
        if indice_a is not None:
            d.update(indice_a.localizar(pa))
        difs.append(d)
    salida["primeras"] = difs
    return salida


def clave_comparable(resultado: dict) -> dict:
    """Parte que debe coincidir exactamente con la referencia."""
    return {k: resultado[k] for k in ("solo_caso", "con_n", "reales", "comparadas")} | {
        "por_pareja": resultado["por_pareja"],
        "posiciones": [list(x) for x in resultado["posiciones"]]}


__all__ = ["Pareja", "construir_espacio", "procesar", "combinar", "finalizar"]
