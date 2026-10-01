"""Operacion 2: busqueda de patrones (CONTEXTO.md, seccion 8.2).

Reglas:
- Sin distinguir mayusculas; coincidencias solapadas.
- N en el patron es comodin de una base (A, C, G o T).
- Con complemento_inverso se busca tambien el complemento inverso. Una
  posicion que coincide en las dos hebras se cuenta una vez en el total.
- Cada tarea recibe solape = Lmax - 1 bytes y cuenta solo las coincidencias
  que empiezan en [inicio, fin).
- Una coincidencia que cruza el limite entre dos registros no cuenta.
"""

from __future__ import annotations

import numpy as np

from pdn.comun.formato import complemento_inverso
from pdn.operaciones.comun import TOPE_POSICIONES, primeras, registro_de
from pdn.operaciones.nucleo import NUMPY

NOMBRE = "patrones"
PATRONES_DEMO = ["TATAAA", "GAATTC", "GGATCC", "CCGG", "GATTACA"]
_PERMITIDAS = set("ACGTN")


def validar_parametros(params: dict | None) -> dict:
    """Valida patrones (1 a 10, longitud 2 a 64, letras ACGTN) y opciones."""
    params = dict(params or {})
    patrones = params.get("patrones") or PATRONES_DEMO
    if isinstance(patrones, str):
        patrones = [p for p in patrones.replace(",", " ").split() if p]
    patrones = [str(p).strip().upper() for p in patrones]
    if not 1 <= len(patrones) <= 10:
        raise ValueError("Se permiten de 1 a 10 patrones (se recibieron %d)" % len(patrones))
    for p in patrones:
        if not 2 <= len(p) <= 64:
            raise ValueError("El patron %r debe medir entre 2 y 64 bases" % p)
        malas = sorted(set(p) - _PERMITIDAS)
        if malas:
            raise ValueError("El patron %r tiene letras invalidas: %s (solo A C G T y N)"
                             % (p, " ".join(malas)))
    if len(set(patrones)) != len(patrones):
        raise ValueError("Hay patrones repetidos")
    params["patrones"] = patrones
    params["complemento_inverso"] = bool(params.get("complemento_inverso", True))
    params["tope"] = int(params.get("tope", TOPE_POSICIONES))
    return params


def solape(params: dict) -> int:
    """Bytes extra por tarea: longitud maxima de patron menos uno."""
    return max(len(p) for p in params["patrones"]) - 1


def coincidencias(mayus: np.ndarray, patron: str) -> np.ndarray:
    """Mascara de posiciones donde empieza el patron (datos ya en mayuscula)."""
    return NUMPY.coincidencias(mayus, patron)


def mascara_costuras(n: int, inicio: int, fin: int, largo: int,
                     limites: list[list[int]]) -> np.ndarray:
    """Mascara de inicios validos: empiezan en [inicio, fin) y no cruzan registros."""
    valido = np.zeros(n, dtype=bool)
    valido[:max(0, min(n, fin - inicio))] = True
    for _, b in limites:
        # Una coincidencia en s cruza el inicio b si s < b < s + largo
        a = max(b - largo + 1 - inicio, 0)
        z = min(b - inicio, n)
        if z > a:
            valido[a:z] = False
    return valido


def procesar(seq: np.ndarray, inicio: int, fin: int, params: dict,
             limites: list[list[int]], nucleo=None) -> dict:
    """Busca los patrones que empiezan en [inicio, fin) del .seq."""
    nucleo = nucleo or NUMPY
    total = seq.shape[0]
    fin_datos = min(fin + solape(params), total)
    mayus = np.asarray(seq[inicio:fin_datos]) & 0xDF
    tope = params["tope"]
    parcial = {"conteos": {}, "por_registro": {}, "posiciones": {}}
    for p in params["patrones"]:
        n = max(fin_datos - inicio - len(p) + 1, 0)
        valido = mascara_costuras(n, inicio, fin, len(p), limites)
        mas = nucleo.coincidencias(mayus, p) & valido
        rc = complemento_inverso(p)
        if params["complemento_inverso"]:
            menos = mas if rc == p else nucleo.coincidencias(mayus, rc) & valido
        else:
            menos = np.zeros(n, dtype=bool)
        union = mas | menos
        pos = np.flatnonzero(union)
        regs = registro_de(pos + inicio, limites) if pos.size else np.zeros(0, dtype=np.int64)
        idx, cuenta = np.unique(regs, return_counts=True)
        primeros = pos[:tope]
        parcial["conteos"][p] = {"+": int(mas.sum()), "-": int(menos.sum()),
                                 "total": int(pos.size)}
        parcial["por_registro"][p] = {str(int(i)): int(c) for i, c in zip(idx, cuenta)}
        parcial["posiciones"][p] = [
            [int(inicio + q), ("+" if mas[q] else "") + ("-" if menos[q] else "")]
            for q in primeros]
    return parcial


def vacio(params: dict) -> dict:
    """Resultado parcial neutro."""
    return {"conteos": {p: {"+": 0, "-": 0, "total": 0} for p in params["patrones"]},
            "por_registro": {p: {} for p in params["patrones"]},
            "posiciones": {p: [] for p in params["patrones"]}}


def combinar(a: dict, b: dict, params: dict) -> dict:
    """Combina dos resultados parciales."""
    salida = vacio(params)
    for p in params["patrones"]:
        for k in ("+", "-", "total"):
            salida["conteos"][p][k] = a["conteos"][p][k] + b["conteos"][p][k]
        reg = dict(a["por_registro"][p])
        for r, c in b["por_registro"][p].items():
            reg[r] = reg.get(r, 0) + c
        salida["por_registro"][p] = reg
        salida["posiciones"][p] = primeras(a["posiciones"][p] + b["posiciones"][p],
                                           params["tope"])
    return salida


def finalizar(parcial: dict, indice=None, params: dict | None = None) -> dict:
    """Resultado legible con posiciones localizadas (registro, fila, columna)."""
    salida = {"patrones": [], "complemento_inverso": params.get("complemento_inverso") if params else None}
    for p, c in parcial["conteos"].items():
        fila = {"patron": p, "complemento": complemento_inverso(p),
                "palindromo": complemento_inverso(p) == p, **c}
        por_reg = []
        for r, n in sorted(parcial["por_registro"][p].items(), key=lambda x: int(x[0])):
            nombre = indice.registros[int(r)].nombre_corto if indice is not None else r
            por_reg.append({"registro": int(r), "nombre": nombre, "coincidencias": n})
        fila["por_registro"] = por_reg
        pos = []
        for g, hebra in parcial["posiciones"][p]:
            d = {"posicion_global": g, "hebra": hebra}
            if indice is not None:
                d.update(indice.localizar(g))
            pos.append(d)
        fila["primeras"] = pos
        salida["patrones"].append(fila)
    return salida


def clave_comparable(resultado: dict) -> dict:
    """Parte que debe coincidir exactamente con la referencia."""
    return {"conteos": resultado["conteos"],
            "por_registro": resultado["por_registro"],
            "posiciones": {p: [list(x) for x in v] for p, v in resultado["posiciones"].items()}}
