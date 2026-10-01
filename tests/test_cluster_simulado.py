"""Pruebas 16.3.4 a 16.3.7 sobre el cluster simulado local."""

from __future__ import annotations

import itertools
import time

import pytest

from pdn.sim.cluster import ClusterSimulado, config_rapida

TAM = 16 * 1024  # unidades pequenas: muchas tareas sobre 1 MiB


def _esp_patrones(esp, tope):
    e = esp["patrones"]
    return {"conteos": e["conteos"], "por_registro": e["por_registro"],
            "posiciones": {p: v[:tope] for p, v in e["posiciones"].items()}}


def _verificar(resumen, esp, operacion):
    """Comprueba que el resultado distribuido es exactamente el esperado."""
    assert resumen["estado"] == "TERMINADA", resumen.get("error")
    assert resumen["validacion"]["valido"]
    r = resumen["resultado"]
    if operacion == "conteo":
        assert r["hist"] == esp["conteo"]["hist"]
        assert [x["posicion_global"] for x in r["primeros_invalidos"]] == \
            [p for p, _ in esp["conteo"]["invalidos_pos"][:1000]]
    elif operacion == "patrones":
        for fila in r["patrones"]:
            c = esp["patrones"]["conteos"][fila["patron"]]
            assert (fila["+"], fila["-"], fila["total"]) == (c["+"], c["-"], c["total"])
            assert [[x["posicion_global"], x["hebra"]] for x in fila["primeras"]] == \
                esp["patrones"]["posiciones"][fila["patron"]][:1000]
    elif operacion == "zonas":
        z = esp["zonas"]
        assert (r["evaluadas"], r["no_evaluables"], r["positivas"]) == \
            (z["evaluadas"], z["no_evaluables"], z["positivas"])
        assert [x["inicio_global"] for x in r["primeras"]] == [x[0] for x in z["zonas"][:1000]]


@pytest.fixture(scope="module")
def cluster3(sintetico, par_mutado):
    prep, _ = sintetico
    a, b, _ = par_mutado
    with ClusterSimulado([prep, a, b], retardos=[0.0, 0.3, 1.0]) as c:
        yield c


def _cfg(op, prep, **extra):
    import os
    base = {"operacion": op, "archivo": os.path.basename(prep.ruta_seq)[:-4], "tam_unidad": TAM,
            "tiempo_objetivo_s": 0.2, "calibracion_mb": 0.1}
    if op == "conteo":
        base["params"] = {"k_invalidos": 1000}
    base.update(extra)
    return base


@pytest.mark.parametrize("op", ["conteo", "patrones", "zonas"])
def test_resultado_correcto(cluster3, sintetico, op):
    prep, esp = sintetico
    params = {"patrones": esp["patrones"]["patrones"]} if op == "patrones" else {}
    cfg = _cfg(op, prep)
    cfg.setdefault("params", {}).update(params)
    _verificar(cluster3.correr(cfg), esp, op)


def test_comparacion_distribuida(cluster3, par_mutado):
    import os
    a, b, esp = par_mutado
    r = cluster3.correr({"operacion": "comparacion", "archivo": os.path.basename(a.ruta_seq)[:-4],
                         "archivo_b": os.path.basename(b.ruta_seq)[:-4], "tam_unidad": TAM,
                         "params": {"tope": 5000}, "calibracion_mb": 0.1})
    assert r["estado"] == "TERMINADA" and r["validacion"]["valido"]
    res = r["resultado"]
    assert (res["reales"], res["solo_caso"], res["con_n"]) == (esp["reales"], esp["solo_caso"], esp["con_n"])
    assert sorted([d["posicion_a"], d["categoria"]] for d in res["primeras"]) == esp["mutaciones"]
    assert r["validacion"]["crc_verificado"] is False  # emparejado: integridad por huella global


def test_balanceo_rapido_procesa_mas(sintetico):
    """Un 'i9' (0,5 s/MB), un nodo medio (2 s/MB) y un 'i3' (8 s/MB)."""
    prep, esp = sintetico
    with ClusterSimulado([prep], retardos=[0.5, 2.0, 8.0]) as c:
        r = c.correr(_cfg("conteo", prep))
    _verificar(r, esp, "conteo")
    bytes_ = {p["worker"]: p["bytes"] for p in r["por_worker"]}
    assert bytes_["sim-0:cpu"] > bytes_["sim-1:cpu"] > bytes_["sim-2:cpu"], bytes_
    assert bytes_["sim-0:cpu"] > 4 * bytes_["sim-2:cpu"], bytes_
    # Fase final decreciente: nadie se queda esperando mucho al final
    for p in r["por_worker"]:
        assert p["ocioso_final_s"] is not None and p["ocioso_final_s"] < 0.5, p


@pytest.mark.parametrize("estrategia", ["fija", "proporcional"])
def test_otras_estrategias(cluster3, sintetico, estrategia):
    prep, esp = sintetico
    r = cluster3.correr(_cfg("conteo", prep, estrategia=estrategia, tam_fijo_mb=0.0625))
    _verificar(r, esp, "conteo")


def test_costuras_distribuidas(sintetico):
    """Muchas combinaciones de unidad, tiempo objetivo y numero de workers."""
    prep, esp = sintetico
    combos_unidad = [4096, 7000, 16384, 65536, 1 << 20]
    combos_obj = [0.1, 0.3]
    for retardos in ([0.0, 0.2, 0.5], [0.0, 0.0, 0.1, 0.3, 0.6]):
        with ClusterSimulado([prep], retardos=retardos) as c:
            for tam, obj in itertools.islice(itertools.product(combos_unidad, combos_obj), 0, None, 1):
                for op in ("conteo", "patrones", "zonas"):
                    cfg = _cfg(op, prep, tam_unidad=tam, tiempo_objetivo_s=obj)
                    if op == "patrones":
                        cfg["params"] = {"patrones": esp["patrones"]["patrones"]}
                    _verificar(c.correr(cfg), esp, op)


def _esperar_progreso(master, minimo=0.15, timeout=30):
    limite = time.time() + timeout
    while time.time() < limite:
        c = master.estado().get("corrida")
        if c and c["estado"] == "EJECUTANDO" and c["progreso"] >= minimo:
            return
        time.sleep(0.05)
    raise TimeoutError("la corrida no avanzo")


def test_tolerancia_caida_y_congelado(sintetico):
    prep, esp = sintetico
    with ClusterSimulado([prep], retardos=[1.0, 1.5, 2.0, 2.5]) as c:
        cid = c.master.iniciar_corrida(_cfg("conteo", prep, tiempo_objetivo_s=0.3))
        _esperar_progreso(c.master)
        c.master.simular_fallo("sim-1:cpu", "caida")
        c.master.simular_fallo("sim-2:cpu", "congelado")
        r = c.master.esperar_corrida(cid, timeout=120)
        _verificar(r, esp, "conteo")
        perdidas = [x for x in r["reasignaciones"] if "perdido" in x["motivo"]]
        assert perdidas and {x["worker"] for x in perdidas} <= {"sim-1:cpu", "sim-2:cpu"}
        # La corrida puede terminar antes del plazo de latido: se espera la deteccion
        limite = time.time() + 10
        while time.time() < limite:
            estados = {w["wid"]: w["estado"] for w in c.master.estado()["workers"]}
            if estados["sim-1:cpu"] == estados["sim-2:cpu"] == "perdido":
                break
            time.sleep(0.2)
        assert estados["sim-1:cpu"] == "perdido" and estados["sim-2:cpu"] == "perdido", estados
        assert c.procs["sim-1:cpu"].wait(5) == 1  # termino con os._exit(1)
        assert c.procs["sim-2:cpu"].poll() is None  # congelado: sigue vivo


def test_tolerancia_sigkill(sintetico):
    prep, esp = sintetico
    with ClusterSimulado([prep], retardos=[1.0, 2.0, 2.0]) as c:
        cid = c.master.iniciar_corrida(_cfg("zonas", prep, tiempo_objetivo_s=0.3))
        _esperar_progreso(c.master)
        c.matar("sim-0:cpu")
        r = c.master.esperar_corrida(cid, timeout=120)
        _verificar(r, esp, "zonas")


def test_integridad_copia_danada(sintetico):
    prep, esp = sintetico
    # El worker 1 tiene el .seq con un byte cambiado en una de cada dos unidades:
    # asi recibe seguro alguna unidad danada (con un solo byte, a veces la
    # unidad danada la procesaba otro worker y no habia nada que rechazar)
    danados = list(range(5_000, prep.largo, 2 * TAM))
    with ClusterSimulado([prep], retardos=[0.5, 0.0, 0.5], corruptos={1: danados}) as c:
        r = c.correr(_cfg("conteo", prep))
        _verificar(r, esp, "conteo")
        rechazos = [x for x in r["reasignaciones"] if "rechazada" in x["motivo"]]
        assert rechazos and all(x["worker"] == "sim-1:cpu" for x in rechazos)
        w1 = next(p for p in r["por_worker"] if p["worker"] == "sim-1:cpu")
        assert w1["excluido"] and "sospechosa" in w1["excluido"]


def test_copia_distinta_excluida(tmp_path, sintetico):
    """Un worker cuya copia tiene otras huellas no participa."""
    prep, esp = sintetico
    import json
    import os
    import shutil
    with ClusterSimulado([prep], retardos=[0.0, 0.0]) as c:
        # Falsear la huella global del nodo 1 y forzar su re-registro
        d = os.path.join(c.carpeta, "nodo1")
        nombre = os.path.basename(prep.ruta_seq)[:-4]
        h = os.path.join(d, nombre + ".huellas")
        tabla = json.load(open(h))
        os.remove(h)
        tabla["global"] = "ffffffffffffffff"
        json.dump(tabla, open(h, "w"))
        c.matar("sim-1:cpu")
        c.procs["sim-1:cpu"].wait(5)
        c.nombres.remove("sim-1:cpu")
        c.lanzar_worker(1, 0.0, rehacer_datos=False)
        time.sleep(2.5)
        r = c.correr(_cfg("conteo", prep))
        _verificar(r, esp, "conteo")
        assert "distinta" in r["excluidos"]["sim-1:cpu"]
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_resultados_persistidos(cluster3, sintetico):
    import csv
    import json
    import os
    prep, _ = sintetico
    r = cluster3.correr(_cfg("conteo", prep))
    carpeta = r["carpeta"]
    for nombre in ("resumen.json", "resultado.json", "tareas.csv", "config.json"):
        assert os.path.exists(os.path.join(carpeta, nombre)), nombre
    filas = list(csv.DictReader(open(os.path.join(carpeta, "tareas.csv"))))
    assert sum(int(f["bytes"]) for f in filas if f["estado"] == "HECHA") == prep.largo
    assert json.load(open(os.path.join(carpeta, "resumen.json")))["validacion"]["valido"]


def test_errores_de_configuracion(cluster3, sintetico):
    from pdn.master.servidor import ErrorCorrida
    prep, _ = sintetico
    with pytest.raises(ErrorCorrida, match="No se encontro"):
        cluster3.master.iniciar_corrida({"operacion": "conteo", "archivo": "no_existe"})
    with pytest.raises(ErrorCorrida, match="dos archivos"):
        cluster3.master.iniciar_corrida({"operacion": "comparacion", "archivo": "sint_lf"})
    with pytest.raises(ErrorCorrida, match="letras invalidas"):
        cluster3.master.iniciar_corrida(_cfg("patrones", prep, params={"patrones": ["AXG"]}))
    with pytest.raises(ErrorCorrida, match="Estrategia"):
        cluster3.master.iniciar_corrida(_cfg("conteo", prep, estrategia="magica"))
    with pytest.raises(ErrorCorrida, match="Ningun worker"):
        cluster3.master.iniciar_corrida(_cfg("conteo", prep, nodos=["nadie"]))


def test_config_rapida():
    cfg = config_rapida()
    assert cfg["master"]["latido_timeout_s"] < 5
