"""E5: motor GPU (simulador de CUDA en la nube, GPU real con PDN_GPU_REAL=1)."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("numba", reason="numba no instalado (solo requirements/gpu.txt lo trae)")

from pdn.motores import gpu_cuda  # noqa: E402
from pdn.operaciones.trabajo import Trabajo, ejecutar_secuencial  # noqa: E402

pytestmark = pytest.mark.skipif(
    not gpu_cuda.simulador() and os.environ.get("PDN_GPU_REAL") != "1",
    reason="sin simulador ni GPU real")

TAM = 8192


@pytest.fixture(scope="module")
def motor():
    m = gpu_cuda.MotorGPU(hilos_bloque=32, bloques=4, lote_mb=0.01, streams=2)
    m.preparar()
    yield m
    m.cerrar()


def _correr(motor, t, prep_a, prep_b=None, unidades=1):
    total = t.vacio()
    for ini, fin in t.rangos(TAM * unidades):
        parcial, info = motor.procesar(t.nombre, t.params, prep_a.ruta_seq,
                                       prep_b.ruta_seq if prep_b else None, t.carga(ini, fin))
        total = t.combinar(total, parcial)
    return total, info


@pytest.mark.parametrize("unidades", [1, 3])
def test_conteo_gpu(motor, sintetico_mini, unidades):
    prep, esp = sintetico_mini
    t = Trabajo("conteo", {"k_invalidos": 1000}, prep.indice(), tam_unidad=TAM)
    r, info = _correr(motor, t, prep, unidades=unidades)
    assert r["hist"] == esp["conteo"]["hist"]
    assert r["invalidos_pos"] == esp["conteo"]["invalidos_pos"][:1000]
    assert info["lotes"] >= 1 and info["streams"] == 2


@pytest.mark.parametrize("op", ["patrones", "zonas"])
def test_patrones_y_zonas_gpu(motor, sintetico_mini, op):
    prep, esp = sintetico_mini
    params = {"patrones": esp["patrones"]["patrones"]} if op == "patrones" else {}
    t = Trabajo(op, params, prep.indice(), tam_unidad=TAM)
    ref = t.clave(ejecutar_secuencial(t, prep.mapear(), tam_tarea=TAM))
    r, _ = _correr(motor, t, prep, unidades=2)
    assert t.clave(r) == ref
    if op == "patrones":
        assert r["conteos"] == esp["patrones"]["conteos"]
    else:
        assert r["positivas"] == esp["zonas"]["positivas"]


def test_comparacion_gpu(motor, par_mini):
    a, b, esp = par_mini
    t = Trabajo("comparacion", {}, a.indice(), b.indice(), TAM)
    r, _ = _correr(motor, t, a, b)
    assert (r["reales"], r["solo_caso"], r["con_n"]) == (esp["reales"], esp["solo_caso"], esp["con_n"])


def test_nucleo_gpu_contra_numpy(motor):
    from pdn.operaciones.nucleo import NUMPY
    from pdn.operaciones.zonas import contar_ventanas
    rng = np.random.default_rng(3)
    d = rng.choice(np.frombuffer(b"ACGTacgtNRx ", dtype=np.uint8), size=3000)
    e = d.copy()
    e[rng.integers(0, 3000, 200)] = rng.choice(np.frombuffer(b"ACGTNacgtn", dtype=np.uint8), size=200)
    g = motor.nucleo
    assert np.array_equal(g.histograma(d), NUMPY.histograma(d))
    assert g.contar_categorias(d, e) == NUMPY.contar_categorias(d, e)
    for p in ("CG", "ACNT", "NNA"):
        assert np.array_equal(g.coincidencias(d & 0xDF, p), NUMPY.coincidencias(d & 0xDF, p))
    loc = np.arange(0, 2800, 37, dtype=np.int64)
    for x, y in zip(g.contar_ventanas(d, loc, 60), contar_ventanas(d, loc, 60)):
        assert np.array_equal(x, y)


def test_un_stream_igual_resultado(sintetico_mini):
    prep, esp = sintetico_mini
    m = gpu_cuda.MotorGPU(hilos_bloque=32, bloques=2, lote_mb=0.005, streams=1)
    m.preparar()
    t = Trabajo("conteo", {}, prep.indice(), tam_unidad=TAM)
    r, info = _correr(m, t, prep, unidades=2)
    assert r["hist"] == esp["conteo"]["hist"] and info["streams"] == 1


def test_validaciones_gpu():
    with pytest.raises(ValueError, match="hilos_bloque"):
        gpu_cuda.MotorGPU(hilos_bloque=100)
    with pytest.raises(ValueError, match="streams"):
        gpu_cuda.MotorGPU(streams=3)
    m = gpu_cuda.MotorGPU(lote_mb=512, streams=2)
    with pytest.raises(ValueError, match="no cabe en la VRAM"):
        m.lote_valido(2048)  # MX450 de 2 GB
    gpu_cuda.MotorGPU(lote_mb=256, streams=1).lote_valido(2048)
    with pytest.raises(RuntimeError, match="preparar"):
        gpu_cuda.MotorGPU().procesar("conteo", {}, "x", None, {"inicio": 0, "fin": 1})


def test_worker_gpu_en_cluster_simulado(sintetico_mini):
    """Un worker GPU (simulador) y uno CPU reparten una corrida."""
    from pdn.sim.cluster import ClusterSimulado
    prep, esp = sintetico_mini
    with ClusterSimulado([prep], retardos=[0.0]) as c:
        c.lanzar_worker(5, 0.0, extra=["--dispositivo", "gpu", "--motor",
                                       '{"hilos_bloque": 32, "bloques": 2, "lote_mb": 0.01}'])
        c.master.esperar_workers(2, timeout=60)
        r = c.correr({"operacion": "conteo", "archivo": os.path.basename(prep.ruta_seq)[:-4],
                      "tam_unidad": 4096, "calibracion_mb": 0.004}, timeout=180)
    assert r["resultado"]["hist"] == esp["conteo"]["hist"]
    disp = {p["worker"]: p["dispositivo"] for p in r["por_worker"]}
    assert "gpu" in disp.values()
    gpu = next(p for p in r["por_worker"] if p["dispositivo"] == "gpu")
    assert gpu["motor"]["simulador"] is True


def test_benchmark_gpu_simulador(sintetico_mini):
    from herramientas.benchmark_gpu import correr
    prep, _ = sintetico_mini
    filas = correr(prep.ruta_seq, 0.01, [0.005], 1, 32, 2)
    assert [f["streams"] for f in filas] == [1, 2]
    assert all(f["solapamiento_pct"] is None for f in filas)  # simulador: sin tiempos reales
