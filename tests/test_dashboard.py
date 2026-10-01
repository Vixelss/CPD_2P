"""E7: API del dashboard (FastAPI TestClient) sobre el cluster simulado."""

from __future__ import annotations

import os
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from pdn.dashboard.app import crear_app
from pdn.master.estado import WorkerInfo
from pdn.master.servidor import validar_motor
from pdn.sim.cluster import ClusterSimulado


@pytest.fixture(scope="module")
def entorno(sintetico, par_mutado):
    prep, esp = sintetico
    a, b, _ = par_mutado
    with ClusterSimulado([prep, a, b], retardos=[0.0, 0.5]) as c:
        yield c, TestClient(crear_app(c.master)), prep, esp


def _nombre(prep):
    return os.path.basename(prep.ruta_seq)[:-4]


def _esperar(cliente, cid, timeout=60):
    limite = time.time() + timeout
    while time.time() < limite:
        r = cliente.get("/api/corridas/" + cid).json()
        if r["terminada"]:
            return r["resumen"]
        time.sleep(0.2)
    raise TimeoutError(cid)


def test_pagina_y_estaticos(entorno):
    _, cli, _, _ = entorno
    r = cli.get("/")
    assert r.status_code == 200 and "Clúster PDN" in r.text
    assert cli.get("/static/chart.umd.min.js").status_code == 200  # Chart.js local, sin CDN
    assert "cdn" not in r.text.lower()


def test_estado_y_archivos(entorno, sintetico):
    _, cli, prep, _ = entorno
    e = cli.get("/api/estado").json()
    assert len(e["workers"]) == 2
    w = e["workers"][0]
    assert w["rangos"]["procesos_max"] >= 1 and w["hardware"]["cpu"]["logicos"]
    archivos = {a["nombre"]: a for a in cli.get("/api/archivos").json()}
    assert len(archivos[_nombre(prep)]["workers"]) == 2


def test_corrida_completa_por_api(entorno):
    c, cli, prep, esp = entorno
    r = cli.post("/api/corridas", json={"operacion": "conteo", "archivo": _nombre(prep), "tam_unidad": 16384,
                                        "calibracion_mb": 0.1, "nodos": ["sim-0", "sim-1"],
                                        "motor_nodos": {"sim-0": {"cpu": {"procesos": 1, "impl": "numpy"}}}})
    assert r.status_code == 200, r.text
    res = _esperar(cli, r.json()["corrida_id"])
    assert res["resultado"]["hist"] == esp["conteo"]["hist"]
    lista = cli.get("/api/resultados").json()
    assert any(f["corrida_id"] == res["corrida_id"] for f in lista)
    det = cli.get("/api/resultados/" + res["corrida_id"]).json()
    assert det["validacion"]["valido"]
    j = cli.get("/api/resultados/%s/exportar?formato=json" % res["corrida_id"])
    assert j.status_code == 200 and j.json()["corrida_id"] == res["corrida_id"]
    csvr = cli.get("/api/resultados/%s/exportar?formato=csv" % res["corrida_id"])
    assert csvr.status_code == 200 and "tarea_id" in csvr.text


def test_errores_de_validacion_claros(entorno):
    _, cli, prep, _ = entorno
    casos = [
        ({"motor": {"cpu": {"procesos": 999}}}, "procesos"),
        ({"motor": {"cpu": {"nucleos": "0,999"}}}, "inexistentes"),
        ({"operacion": "patrones", "params": {"patrones": ["ACGU"]}}, "letras invalidas"),
        ({"tiempo_objetivo_s": 50}, "tiempo objetivo"),
        ({"archivo": "no_existe"}, "No se encontro"),
    ]
    for extra, msg in casos:
        cfg = {"operacion": "conteo", "archivo": _nombre(prep), **extra}
        r = cli.post("/api/corridas", json=cfg)
        assert r.status_code == 400 and msg in r.json()["detail"], (extra, r.text)


def test_validar_motor_gpu_y_simd():
    w = WorkerInfo("n:gpu", "n", "gpu", hardware={"gpu": {"vram_libre_mb": 1800}})
    assert "VRAM" in validar_motor(w, {"lote_mb": 512, "streams": 2})
    assert validar_motor(w, {"lote_mb": 256, "streams": 1}) is None
    c = WorkerInfo("m:cpu", "m", "cpu", hardware={"cpu": {"logicos": [0, 1], "flags": ["popcnt"]}})
    assert "AVX2" in validar_motor(c, {"impl": "simd"})
    assert "rendimiento" in validar_motor(c, {"nucleos": "rendimiento"}) or "hibrido" in validar_motor(c, {"nucleos": "rendimiento"})


def test_fallo_simulado_por_api(entorno):
    c, cli, prep, _ = entorno
    r = cli.post("/api/fallo", json={"worker": "nadie:cpu", "modo": "caida"})
    assert r.status_code == 400
    r = cli.post("/api/master/caida")
    assert r.status_code == 400 and "respaldo" in r.json()["detail"]


def test_evidencias_y_registro(entorno):
    c, cli, _, _ = entorno
    filas = cli.get("/api/evidencias").json()
    criterios = {f["criterio"] for f in filas}
    assert {"CPU: SIMD", "Balanceo de carga", "Integridad", "Tolerancia a fallos"} <= criterios
    for f in filas:
        assert (f["valor"] is None) == (f["estado"] == "sin dato")
        assert f["valor"] is not None or f["motivo"]
    ruta = cli.post("/api/evidencias/exportar").json()["ruta"]
    assert os.path.exists(ruta) and "Evidencias" in open(ruta).read()
    assert "Registrado" in cli.get("/api/registro").text


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun no disponible")
def test_mpi_y_escalabilidad_desde_api(entorno):
    _, cli, prep, esp = entorno
    r = cli.post("/api/mpi", json={"operacion": "conteo", "archivo": _nombre(prep), "tam_unidad": 16384,
                                   "mpi_nodos": [{"hostname": "sim-0", "slots": 2}]})
    assert r.status_code == 200, r.text
    limite = time.time() + 60
    while cli.get("/api/mpi").json()["corriendo"] and time.time() < limite:
        time.sleep(0.2)
    e = cli.get("/api/mpi").json()
    assert e["error"] is None and e["resumen"]["resultado"]["hist"] == esp["conteo"]["hist"]
    r = cli.post("/api/escalabilidad", json={"operacion": "conteo", "archivo": _nombre(prep), "nucleos": [1, 2],
                                             "repeticiones": 1, "tam_unidad": 16384})
    assert r.status_code == 200
    limite = time.time() + 120
    while cli.get("/api/escalabilidad").json()["corriendo"] and time.time() < limite:
        time.sleep(0.3)
    e = cli.get("/api/escalabilidad").json()
    assert e["error"] is None and any(f["serie"] == "2_nucleos_un_nodo" for f in e["resultado"]["filas"])


def test_websocket(entorno):
    _, cli, _, _ = entorno
    with cli.websocket_connect("/ws") as ws:
        e = ws.receive_json()
        assert "workers" in e and "rangos" in e["workers"][0]
