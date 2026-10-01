"""E9: modelo CpG, cuantizacion INT8, exportacion a Core ML y worker NPU (alternativa en CPU)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from pdn.motores.npu import construir_modelo as C
from pdn.motores.npu import modelo as M
from pdn.motores.npu.motor_npu import MotorNPU
from pdn.operaciones.trabajo import Trabajo, ejecutar_secuencial
from pdn.operaciones.zonas import contar_ventanas


def test_one_hot_y_extraccion_exactas():
    rng = np.random.default_rng(1)
    raw = rng.choice(np.frombuffer(b"ACGTacgtNRx", dtype=np.uint8), size=5000)
    loc = np.arange(0, 4700, 13, dtype=np.int64)
    x = M.one_hot(raw, loc, 200)
    assert x.shape == (loc.size, 4, 200)
    assert np.all(x.sum(axis=1) <= 1)  # N, IUPAC e invalidos quedan en cero
    n_c, n_g, n_cg = M.extraer(x)
    _, c, g, cg = contar_ventanas(raw, loc, 200)
    assert np.array_equal(n_c, c) and np.array_equal(n_g, g) and np.array_equal(n_cg, cg)


def test_caracteristicas_separan_la_regla():
    """El signo de f1 y f2 reproduce la regla (empates incluidos) en aritmetica float32."""
    n_c, n_g, n_cg = M.datos_entrenamiento(50_000, semilla=3)
    f = M.caracteristicas(n_c, n_g, n_cg)
    regla = M.regla(n_c, n_g, n_cg)
    assert np.array_equal((f[:, 0] > 0) & (f[:, 1] > 0), regla)


def test_modelo_guardado_y_metricas():
    r = C.rutas()
    p, w = M.cargar(r["pesos"])
    assert w == 200 and p["w1"].shape == (M.OCULTAS, 2)
    met = json.load(open(r["metricas"]))
    real = met["evaluacion"]["ventanas_reales"]
    assert real["fp32_vs_regla"]["concordancia"] >= 0.9999
    assert real["int8_vs_regla"]["concordancia"] >= 0.9999
    assert met["coreml"]["ops_cuantizacion"]["constexpr_affine_dequantize"] == 4


def test_cuantizacion_int8():
    p, _ = M.cargar(C.rutas()["pesos"])
    dq, ent = M.cuantizar_int8(p)
    for k in p:
        assert ent[k]["q"].dtype == np.int8
        assert np.abs(dq[k] - p[k]).max() <= ent[k]["escala"] / 2 + 1e-7


def test_entrenamiento_reproduce_la_regla():
    p, info = M.entrenar(n=20_000, epocas=1500, semilla=5)
    n_c, n_g, n_cg = M.datos_entrenamiento(20_000, semilla=11)
    acierto = ((M.densa(M.caracteristicas(n_c, n_g, n_cg), p) > 0.5) == M.regla(n_c, n_g, n_cg)).mean()
    assert acierto > 0.995  # conjunto concentrado en la frontera: es el caso dificil


def test_programa_coreml_se_construye(tmp_path):
    ct = pytest.importorskip("coremltools")
    p, w = M.cargar(C.rutas()["pesos"])
    info = C.exportar_coreml(p, w, str(tmp_path / "a.mlpackage"), str(tmp_path / "b.mlpackage"), lote=64)
    assert info["exportado"] and info["ops_cuantizacion"] == {"constexpr_affine_dequantize": 4}
    assert ct.utils.load_spec(str(tmp_path / "b.mlpackage")).WhichOneof("Type") == "mlProgram"


def test_motor_npu_alternativa_cpu(sintetico):
    prep, esp = sintetico
    m = MotorNPU(backend="auto", precision="int8")
    m.preparar()
    if os.uname().sysname != "Darwin":
        assert m.backend == "cpu" and "alternativa en CPU" in m.aviso
        assert m.describir()["pide_neural_engine"] is False
    t = Trabajo("zonas", {}, prep.indice(), tam_unidad=65536)
    total = t.vacio()
    for ini, fin in t.rangos(65536 * 3):
        parcial, info = m.procesar("zonas", t.params, prep.ruta_seq, None, t.carga(ini, fin))
        total = t.combinar(total, parcial)
    z = esp["zonas"]
    assert (total["evaluadas"], total["no_evaluables"]) == (z["evaluadas"], z["no_evaluables"])
    # El modelo puede discrepar de la regla en empates: se mide, no se oculta
    assert abs(total["positivas"] - z["positivas"]) <= max(2, 0.002 * z["positivas"])
    assert info["ventanas"] > 0 and info["ops_estimadas"] > 0
    with pytest.raises(ValueError, match="solo ejecuta"):
        m.procesar("conteo", {}, prep.ruta_seq, None, {"inicio": 0, "fin": 1, "limites": []})
    with pytest.raises(ValueError, match="ventanas de 200"):
        m.inferir(np.zeros(100, np.uint8), np.array([0]), 50)


def test_worker_npu_en_cluster(sintetico):
    """Un worker npu (alternativa en CPU) y uno cpu reparten zonas; la NPU no recibe conteo."""
    from pdn.sim.cluster import ClusterSimulado
    prep, esp = sintetico
    nombre = os.path.basename(prep.ruta_seq)[:-4]
    with ClusterSimulado([prep], retardos=[0.5]) as c:
        c.lanzar_worker(7, 0.0, extra=["--dispositivo", "npu"])
        c.master.esperar_workers(2, timeout=60)
        r = c.correr({"operacion": "zonas", "archivo": nombre, "tam_unidad": 16384, "calibracion_mb": 0.1})
        npu = next(p for p in r["por_worker"] if p["dispositivo"] == "npu")
        assert npu["tareas"] > 0 and npu["motor"]["backend"] in ("cpu", "coreml")
        assert r["resultado"]["evaluadas"] == esp["zonas"]["evaluadas"]
        r2 = c.correr({"operacion": "conteo", "archivo": nombre, "tam_unidad": 16384, "calibracion_mb": 0.1})
        assert "zonas" in r2["excluidos"]["sim-7:npu"]
        assert r2["resultado"]["hist"] == esp["conteo"]["hist"]
