"""Operacion 4: zonas de interes, islas CpG (CONTEXTO.md, seccion 8.4).

Regla de Gardiner-Garden y Frommer sobre ventanas de W bases con paso S:
    GC = (n_C + n_G) / W > 0.5   y   obs/esp = n_CG * W / (n_C * n_G) > 0.6
Se evalua con aritmetica entera para que todos los motores den exactamente
lo mismo:  2 (n_C + n_G) > W   y   10 n_CG W > 6 n_C n_G.

Las ventanas se alinean al inicio de cada registro (inicio + k S) y no cruzan
registros. Una ventana con cualquier byte que no sea A C G T (N, IUPAC o
invalido) es "no evaluable". Cada tarea recibe W - 1 bytes de solape y
evalua solo las ventanas que empiezan en [inicio, fin).
"""

from __future__ import annotations

import numpy as np

from pdn.comun.formato import ES_ACGT
from pdn.operaciones.comun import (SUBTRAMO_ZONAS, TOPE_POSICIONES, primeras, subtramos,
                                   tramos_de_registro)

NOMBRE = "zonas"
_C, _G = ord("C"), ord("G")


def validar_parametros(params: dict | None) -> dict:
    """Valida W (ventana) y S (paso)."""
    params = dict(params or {})
    w = int(params.get("ventana", 200))
    s = int(params.get("paso", w))
    if not 10 <= w <= 100000:
        raise ValueError("La ventana W debe estar entre 10 y 100000 bases")
    if not 1 <= s <= 100000:
        raise ValueError("El paso S debe estar entre 1 y 100000 bases")
    params["ventana"], params["paso"] = w, s
    params["tope"] = int(params.get("tope", TOPE_POSICIONES))
    return params


def solape(params: dict) -> int:
    """Bytes extra por tarea: W - 1."""
    return params["ventana"] - 1


def es_positiva(n_c: np.ndarray, n_g: np.ndarray, n_cg: np.ndarray, w: int) -> np.ndarray:
    """Aplica la regla CpG con enteros."""
    n_c = n_c.astype(np.int64)
    n_g = n_g.astype(np.int64)
    n_cg = n_cg.astype(np.int64)
    return (2 * (n_c + n_g) > w) & (10 * n_cg * w > 6 * n_c * n_g)


def inicios_ventanas(reg_ini: int, reg_fin: int, inicio: int, fin: int, w: int, s: int) -> np.ndarray:
    """Inicios de ventanas de un registro que empiezan en [inicio, fin) y caben en el."""
    ultimo = reg_fin - w  # ultimo inicio que cabe
    if ultimo < reg_ini:
        return np.zeros(0, dtype=np.int64)
    k0 = max(0, -(-(inicio - reg_ini) // s))
    k1 = (min(fin - 1, ultimo) - reg_ini) // s
    if k1 < k0:
        return np.zeros(0, dtype=np.int64)
    return reg_ini + s * np.arange(k0, k1 + 1, dtype=np.int64)


def ventanas(seq: np.ndarray, inicio: int, fin: int, params: dict,
             limites: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inicios globales y registros de las ventanas que empiezan en [inicio, fin), y los bytes del tramo."""
    w, s = params["ventana"], params["paso"]
    total = seq.shape[0]
    fin_datos = min(fin + w - 1, total)
    raw = np.asarray(seq[inicio:fin_datos])
    inicios, registros = [], []
    for idx, ri, rf in tramos_de_registro(inicio, fin_datos, limites, total):
        # El registro puede empezar antes de inicio: se usa su inicio real
        reg_ini = _inicio_registro(limites, idx)
        v = inicios_ventanas(reg_ini, rf, inicio, fin, w, s)
        inicios.append(v)
        registros.append(np.full(v.shape[0], idx, dtype=np.int64))
    v = np.concatenate(inicios) if inicios else np.zeros(0, dtype=np.int64)
    regs = np.concatenate(registros) if registros else np.zeros(0, dtype=np.int64)
    return v, regs, raw


def caracteristicas(seq: np.ndarray, inicio: int, fin: int, params: dict,
                    limites: list[list[int]], nucleo=None) -> dict:
    """Calcula por ventana: inicio global, registro, evaluable, n_C, n_G, n_CG."""
    w = params["ventana"]
    v, regs, raw = ventanas(seq, inicio, fin, params, limites)
    contar = getattr(nucleo, "contar_ventanas", None) or contar_ventanas
    malo, n_c, n_g, n_cg = contar(raw, v - inicio, w)
    return {"inicios": v, "registros": regs, "evaluable": malo == 0, "n_c": n_c, "n_g": n_g, "n_cg": n_cg}


def clasificar_con_modelo(seq: np.ndarray, inicio: int, fin: int, params: dict,
                          limites: list[list[int]], nucleo) -> tuple[dict, np.ndarray, np.ndarray]:
    """Ruta de la NPU: el modelo clasifica las ventanas evaluables.

    La CPU solo calcula que ventanas son evaluables (una suma acumulada) y los
    conteos de las positivas, para listarlas. Devuelve (car, positivas, prob).
    """
    w = params["ventana"]
    v, regs, raw = ventanas(seq, inicio, fin, params, limites)
    loc = v - inicio
    malo = np.concatenate(([0], np.cumsum(~ES_ACGT[raw])))
    ev = (malo[loc + w] - malo[loc]) == 0 if v.size else np.zeros(0, dtype=bool)
    prob = np.zeros(v.shape[0], dtype=np.float64)
    if ev.any():
        prob[ev] = nucleo.clasificar(raw, loc[ev], w)
    positivas = ev & (prob > 0.5)
    n_c = np.zeros(v.shape[0], dtype=np.int64)
    n_g = np.zeros_like(n_c)
    n_cg = np.zeros_like(n_c)
    if positivas.any():
        _, c, g, cg = contar_ventanas(raw, loc[positivas], w)
        n_c[positivas], n_g[positivas], n_cg[positivas] = c, g, cg
    car = {"inicios": v, "registros": regs, "evaluable": ev, "n_c": n_c, "n_g": n_g, "n_cg": n_cg}
    return car, positivas, prob


def contar_ventanas(raw: np.ndarray, loc: np.ndarray, w: int) -> tuple[np.ndarray, ...]:
    """Por ventana [loc, loc + w): bytes que no son ACGT, n_C, n_G y n_CG (sumas acumuladas)."""
    d = raw & 0xDF
    malo = (~ES_ACGT[raw]).astype(np.int64)
    es_c = (d == _C).astype(np.int64)
    es_g = (d == _G).astype(np.int64)
    es_cg = np.zeros(d.shape[0], dtype=np.int64)
    if d.shape[0] > 1:
        es_cg[:-1] = (d[:-1] == _C) & (d[1:] == _G)
    acum = [np.concatenate(([0], np.cumsum(x))) for x in (malo, es_c, es_g, es_cg)]

    def suma(a: np.ndarray, largo: int) -> np.ndarray:
        return a[loc + largo] - a[loc]

    return suma(acum[0], w), suma(acum[1], w), suma(acum[2], w), suma(acum[3], w - 1)


def _inicio_registro(limites: list[list[int]], idx: int) -> int:
    for i, ini in limites:
        if i == idx:
            return ini
    raise KeyError(idx)


def armar_parcial(car: dict, positivas: np.ndarray, prob: np.ndarray | None, params: dict) -> dict:
    """Construye el resultado parcial a partir de caracteristicas y etiquetas."""
    ev = car["evaluable"]
    pos = positivas & ev
    regs = car["registros"]
    por_reg: dict[str, list[int]] = {}
    for r in np.unique(regs):
        m = regs == r
        por_reg[str(int(r))] = [int((m & ev).sum()), int((m & pos).sum())]
    sel = np.flatnonzero(pos)[:params["tope"]]
    zonas = []
    for q in sel:
        zonas.append([int(car["inicios"][q]), int(car["registros"][q]), int(car["n_c"][q]),
                      int(car["n_g"][q]), int(car["n_cg"][q]),
                      float(prob[q]) if prob is not None else 1.0])
    return {"evaluadas": int(ev.sum()), "no_evaluables": int((~ev).sum()),
            "positivas": int(pos.sum()), "por_registro": por_reg, "zonas": zonas}


def procesar(seq: np.ndarray, inicio: int, fin: int, params: dict,
             limites: list[list[int]], nucleo=None) -> dict:
    """Evalua con la regla las ventanas que empiezan en [inicio, fin) (por subtramos)."""
    total = vacio(params)
    for a, b in subtramos(inicio, fin, SUBTRAMO_ZONAS):
        if hasattr(nucleo, "clasificar"):
            car, positivas, prob = clasificar_con_modelo(seq, a, b, params, limites, nucleo)
        else:
            car = caracteristicas(seq, a, b, params, limites, nucleo)
            positivas = es_positiva(car["n_c"], car["n_g"], car["n_cg"], params["ventana"])
            prob = None
        total = combinar(total, armar_parcial(car, positivas, prob, params), params)
    return total


def vacio(params: dict) -> dict:
    """Resultado parcial neutro."""
    return {"evaluadas": 0, "no_evaluables": 0, "positivas": 0, "por_registro": {}, "zonas": []}


def combinar(a: dict, b: dict, params: dict) -> dict:
    """Combina dos resultados parciales."""
    pr = {k: list(v) for k, v in a["por_registro"].items()}
    for k, v in b["por_registro"].items():
        pr[k] = [x + y for x, y in zip(pr[k], v)] if k in pr else list(v)
    return {"evaluadas": a["evaluadas"] + b["evaluadas"],
            "no_evaluables": a["no_evaluables"] + b["no_evaluables"],
            "positivas": a["positivas"] + b["positivas"], "por_registro": pr,
            "zonas": primeras(a["zonas"] + b["zonas"], params["tope"])}


def finalizar(parcial: dict, indice=None, params: dict | None = None) -> dict:
    """Resultado legible: densidad por registro y primeras zonas."""
    w = params["ventana"] if params else 200
    salida = {k: parcial[k] for k in ("evaluadas", "no_evaluables", "positivas")}
    filas = []
    for r, (ev, po) in sorted(parcial["por_registro"].items(), key=lambda x: int(x[0])):
        nombre = indice.registros[int(r)].nombre_corto if indice is not None else r
        filas.append({"registro": int(r), "nombre": nombre, "evaluadas": ev, "positivas": po,
                      "densidad": (po / ev) if ev else None})
    salida["por_registro"] = filas
    zonas = []
    for ini, reg, nc, ng, ncg, prob in parcial["zonas"]:
        z = {"inicio_global": ini, "fin_global": ini + w, "registro": reg,
             "gc": (nc + ng) / w, "obs_esp": (ncg * w / (nc * ng)) if nc * ng else 0.0,
             "probabilidad": prob}
        if indice is not None:
            loc = indice.localizar(ini)
            z.update({"nombre": loc["nombre"], "inicio": loc["posicion"],
                      "fila": loc["fila"], "columna": loc["columna"]})
        zonas.append(z)
    salida["primeras"] = zonas
    return salida


def clave_comparable(resultado: dict) -> dict:
    """Parte que debe coincidir exactamente con la referencia (sin probabilidades)."""
    return {k: resultado[k] for k in ("evaluadas", "no_evaluables", "positivas", "por_registro")} | {
        "zonas": [list(z[:5]) for z in resultado["zonas"]]}
