"""Prueba 16.3.2 y costuras de la referencia: operaciones contra lo esperado."""

from __future__ import annotations

import itertools

import pytest

from pdn.operaciones.trabajo import Trabajo, ejecutar_secuencial

# Combinaciones de (tam_unidad, tam_tarea en unidades) para mover las costuras
COSTURAS = [(4 << 20, 1), (1 << 20, 1), (65536, 1), (65536, 3), (4096, 1), (4096, 7),
            (1000, 1), (333, 2), (97, 5), (64, 1), (4096, 64), (1 << 20, 2)]


def _trabajo(nombre, prep, params=None, tam_unidad=4 << 20, prep_b=None):
    return Trabajo(nombre, params or {}, prep.indice(),
                   prep_b.indice() if prep_b else None, tam_unidad)


def _esp_patrones(esp, tope):
    e = esp["patrones"]
    return {"conteos": e["conteos"], "por_registro": e["por_registro"],
            "posiciones": {p: v[:tope] for p, v in e["posiciones"].items()}}


def _esp_zonas(esp, tope):
    e = esp["zonas"]
    return {"evaluadas": e["evaluadas"], "no_evaluables": e["no_evaluables"],
            "positivas": e["positivas"], "por_registro": e["por_registro"],
            "zonas": e["zonas"][:tope]}


@pytest.mark.parametrize("fixture", ["sintetico", "sintetico_crlf", "sintetico_grande"])
def test_conteo(fixture, request):
    prep, esp = request.getfixturevalue(fixture)
    t = _trabajo("conteo", prep, {"k_invalidos": 1000})
    r = ejecutar_secuencial(t, prep.mapear())
    assert r["hist"] == esp["conteo"]["hist"]
    assert r["invalidos_pos"] == esp["conteo"]["invalidos_pos"][:1000]
    fin = t.finalizar(r)
    assert fin["invalidos_total"] == len(esp["conteo"]["invalidos_pos"])
    assert fin["total"] == esp["seq_total"]


@pytest.mark.parametrize("fixture", ["sintetico", "sintetico_grande"])
def test_patrones(fixture, request):
    prep, esp = request.getfixturevalue(fixture)
    params = {"patrones": esp["patrones"]["patrones"], "tope": 50}
    t = _trabajo("patrones", prep, params)
    r = ejecutar_secuencial(t, prep.mapear())
    assert t.clave(r) == _esp_patrones(esp, 50)


def test_patrones_sin_complemento(sintetico):
    prep, esp = sintetico
    t = _trabajo("patrones", prep, {"patrones": ["GATTACA"], "complemento_inverso": False})
    r = ejecutar_secuencial(t, prep.mapear())
    assert r["conteos"]["GATTACA"]["total"] == esp["patrones"]["conteos"]["GATTACA"]["+"]
    assert r["conteos"]["GATTACA"]["-"] == 0


@pytest.mark.parametrize("fixture", ["sintetico", "sintetico_grande"])
def test_zonas(fixture, request):
    prep, esp = request.getfixturevalue(fixture)
    t = _trabajo("zonas", prep, {"ventana": 200, "paso": 200, "tope": 100})
    r = ejecutar_secuencial(t, prep.mapear())
    assert t.clave(r) == _esp_zonas(esp, 100)


@pytest.mark.parametrize("tam_unidad,unidades_tarea", COSTURAS)
def test_costuras_todas_las_operaciones(sintetico, tam_unidad, unidades_tarea):
    """Las costuras caen en sitios distintos y el resultado no cambia."""
    prep, esp = sintetico
    seq = prep.mapear()
    tam_tarea = tam_unidad * unidades_tarea
    t = _trabajo("conteo", prep, {"k_invalidos": 1000}, tam_unidad)
    r = ejecutar_secuencial(t, seq, tam_tarea=tam_tarea)
    assert r["hist"] == esp["conteo"]["hist"]
    assert r["invalidos_pos"] == esp["conteo"]["invalidos_pos"]

    t = _trabajo("patrones", prep, {"patrones": esp["patrones"]["patrones"], "tope": 30}, tam_unidad)
    assert t.clave(ejecutar_secuencial(t, seq, tam_tarea=tam_tarea)) == _esp_patrones(esp, 30)

    t = _trabajo("zonas", prep, {"tope": 40}, tam_unidad)
    assert t.clave(ejecutar_secuencial(t, seq, tam_tarea=tam_tarea)) == _esp_zonas(esp, 40)


def test_zonas_paso_distinto_de_ventana(sintetico):
    """Con S != W el resultado tambien es independiente del reparto."""
    prep, _ = sintetico
    seq = prep.mapear()
    base = None
    for tam_unidad, n in itertools.islice(COSTURAS, 0, None, 2):
        t = _trabajo("zonas", prep, {"ventana": 120, "paso": 37, "tope": 200}, tam_unidad)
        r = t.clave(ejecutar_secuencial(t, seq, tam_tarea=tam_unidad * n))
        base = base or r
        assert r == base


def test_comparacion_emparejada(par_mutado):
    a, b, esp = par_mutado
    for tam_unidad, n in [(4 << 20, 1), (4096, 1), (1000, 3), (97, 2)]:
        t = _trabajo("comparacion", a, {"tope": 5000}, tam_unidad, prep_b=b)
        r = ejecutar_secuencial(t, a.mapear(), b.mapear(), tam_tarea=tam_unidad * n)
        for k in ("reales", "solo_caso", "con_n", "comparadas"):
            assert r[k] == esp[k], k
        cats = {0: "solo_caso", 1: "con_n", 2: "reales"}
        assert sorted([p[0], cats[p[4]]] for p in r["posiciones"]) == esp["mutaciones"]
        fin = t.finalizar(r)
        assert fin["valido"] and fin["num_parejas"] == esp["parejas"]
        assert sum(f["total"] for f in fin["parejas"]) == esp["total"]


def test_comparacion_reordenada(par_reordenado):
    a, b, esp = par_reordenado
    t = _trabajo("comparacion", a, {"modo_comparacion": "emparejado"}, 4096, prep_b=b)
    r = ejecutar_secuencial(t, a.mapear(), b.mapear(), tam_tarea=4096 * 3)
    fin = t.finalizar(r)
    assert fin["total"] == 0
    assert fin["comparadas"] == esp["emparejado"]["comparadas"]
    assert len(fin["sin_pareja_a"]) == 1
    t = _trabajo("comparacion", a, {"modo_comparacion": "posicional"}, 4096, prep_b=b)
    r = ejecutar_secuencial(t, a.mapear(), b.mapear(), tam_tarea=4096 * 5)
    assert r["comparadas"] == esp["posicional"]["comparadas"]
    assert r["solo_caso"] + r["con_n"] + r["reales"] == esp["posicional"]["total"]


@pytest.mark.parametrize("params,mensaje", [
    ({"patrones": ["ACGU"]}, "letras invalidas"),
    ({"patrones": ["A"]}, "entre 2 y 64"),
    ({"patrones": ["A" * 65]}, "entre 2 y 64"),
    ({"patrones": ["AC"] * 2}, "repetidos"),
    ({"patrones": ["ACG%d" % i for i in range(11)]}, "1 a 10"),
])
def test_validacion_patrones(params, mensaje):
    from pdn.operaciones import patrones
    with pytest.raises(ValueError, match=mensaje):
        patrones.validar_parametros(params)
