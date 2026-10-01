"""E2: nucleo SIMD y motor CPU dan exactamente lo mismo que la referencia."""

from __future__ import annotations

import os

import numpy as np
import pytest

from pdn.motores.cpu import MotorCPU, partir_carga
from pdn.motores.simd import envoltorio
from pdn.operaciones.nucleo import NUMPY
from pdn.operaciones.trabajo import Trabajo, ejecutar_secuencial

VARIANTES = ["escalar"] + (["avx2"] if envoltorio.tiene_avx2() else [])
TODOS = np.frombuffer(b"ACGTNacgtnRYSWKMBDHVryswkmbdhv", dtype=np.uint8)


def _datos_aleatorios(n: int, semilla: int) -> np.ndarray:
    """Bytes con todos los casos: bases en ambos casos, N, IUPAC e invalidos."""
    rng = np.random.default_rng(semilla)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    d = rng.choice(np.frombuffer(b"ACGTacgt", dtype=np.uint8), size=n)
    k = max(1, n // 50)
    d[rng.integers(0, n, k)] = rng.choice(TODOS, size=k)
    d[rng.integers(0, n, max(1, n // 200))] = rng.integers(0, 256, max(1, n // 200), dtype=np.uint8)
    return d


@pytest.mark.parametrize("variante", VARIANTES)
@pytest.mark.parametrize("n", [0, 1, 31, 32, 33, 255 * 32, 255 * 32 + 7, 100_003, 1 << 20])
def test_histograma_simd(variante, n):
    nucleo = envoltorio.NucleoSIMD(variante)
    d = _datos_aleatorios(n, n)
    assert np.array_equal(nucleo.histograma(d), NUMPY.histograma(d))
    assert np.array_equal(nucleo.histograma_completo(d), NUMPY.histograma(d))


@pytest.mark.parametrize("variante", VARIANTES)
@pytest.mark.parametrize("n", [0, 5, 31, 32, 64, 1000, 100_001])
def test_comparar_simd(variante, n):
    nucleo = envoltorio.NucleoSIMD(variante)
    a = _datos_aleatorios(n, 1)
    b = a.copy()
    rng = np.random.default_rng(2)
    if n:
        idx = rng.integers(0, n, max(1, n // 10))
        b[idx] = rng.choice(TODOS, size=idx.size)
        b[idx[::3]] ^= 0x20
    assert nucleo.contar_categorias(a, b) == NUMPY.contar_categorias(a, b)


@pytest.mark.parametrize("variante", VARIANTES)
@pytest.mark.parametrize("patron", ["AC", "TATAAA", "NNA", "ACNGT", "GATTACA", "A" * 40])
def test_buscar_simd(variante, patron):
    nucleo = envoltorio.NucleoSIMD(variante)
    for n in (0, 3, 33, 100_000):
        d = _datos_aleatorios(n, 7) & 0xDF
        d[: min(n, 10)] = ord("A")
        assert np.array_equal(nucleo.coincidencias(d, patron), NUMPY.coincidencias(d, patron))


def test_buscar_simd_trozos(monkeypatch):
    """La busqueda por trozos no pierde coincidencias en los cortes."""
    monkeypatch.setattr(envoltorio, "_TROZO_BUSQUEDA", 1000)
    nucleo = envoltorio.NucleoSIMD(VARIANTES[-1])
    d = np.frombuffer(b"GATTACA" * 5000, dtype=np.uint8).copy()
    for p in ("GATTACA", "ACAG", "AG"):
        assert np.array_equal(nucleo.coincidencias(d, p), NUMPY.coincidencias(d, p))


def test_partir_carga_cubre_todo():
    c = {"inicio": 100, "fin": 10_000_000, "limites": [[0, 0]]}
    tramos = partir_carga(c, 7)
    assert tramos[0]["inicio"] == 100 and tramos[-1]["fin"] == 10_000_000
    assert all(a["fin"] == b["inicio"] for a, b in zip(tramos, tramos[1:]))
    segs = {"segmentos": [[0, 0, 5, 1000], [1, 2000, 3000, 3_000_000], [2, 9, 9, 7]]}
    partes = partir_carga(segs, 4, minimo=1000)
    assert sum(s[3] for p in partes for s in p["segmentos"]) == 3_001_007
    assert len(partes) <= 4


@pytest.fixture(scope="module")
def motores():
    """Motores con 1 y N procesos, numpy y SIMD."""
    n = min(4, len(os.sched_getaffinity(0)))
    lista = [MotorCPU(1, impl="numpy"), MotorCPU(n, impl="numpy"),
             MotorCPU(n, impl="simd" if envoltorio.tiene_avx2() else "simd_escalar"),
             MotorCPU(2, impl="simd_escalar")]
    for m in lista:
        m.preparar()
    yield lista
    for m in lista:
        m.cerrar()


def _correr(motor, trabajo, prep_a, prep_b, tam_tarea):
    total = trabajo.vacio()
    for ini, fin in trabajo.rangos(tam_tarea):
        parcial, info = motor.procesar(trabajo.nombre, trabajo.params, prep_a.ruta_seq,
                                       prep_b.ruta_seq if prep_b else None, trabajo.carga(ini, fin))
        total = trabajo.combinar(total, parcial)
    return total


@pytest.mark.parametrize("tam_unidad,unidades", [(4 << 20, 1), (65536, 3), (4096, 5), (1000, 7)])
def test_motor_cpu_equivale_a_referencia(motores, sintetico, par_mutado, tam_unidad, unidades):
    prep, esp = sintetico
    casos = [("conteo", {"k_invalidos": 1000}), ("patrones", {"patrones": esp["patrones"]["patrones"]}),
             ("zonas", {})]
    for nombre, params in casos:
        t = Trabajo(nombre, params, prep.indice(), tam_unidad=tam_unidad)
        ref = t.clave(ejecutar_secuencial(t, prep.mapear()))
        for m in motores:
            assert t.clave(_correr(m, t, prep, None, tam_unidad * unidades)) == ref, (nombre, m.nombre)
    a, b, _ = par_mutado
    t = Trabajo("comparacion", {}, a.indice(), b.indice(), tam_unidad)
    ref = t.clave(ejecutar_secuencial(t, a.mapear(), b.mapear()))
    for m in motores:
        assert t.clave(_correr(m, t, a, b, tam_unidad * unidades)) == ref, m.nombre


def test_afinidad_registrada(motores):
    m = motores[1]
    desc = m.describir()
    assert sorted(v for v in m.asignacion.values()) == sorted(m.nucleos[:m.procesos])
    assert desc["preparacion_s"] is not None


def test_validaciones_de_rango():
    logicos = len(os.sched_getaffinity(0))
    with pytest.raises(ValueError, match="procesos"):
        MotorCPU(logicos + 100)
    with pytest.raises(ValueError, match="inexistentes"):
        MotorCPU(1, nucleos=[999])
    with pytest.raises(ValueError, match="desconocida"):
        MotorCPU(1, impl="cuda")
    with pytest.raises(ValueError, match="no es hibrido"):
        from pdn.worker.hardware import elegir_nucleos
        elegir_nucleos("rendimiento", {"logicos": [0, 1], "rendimiento": [], "eficiencia": [],
                                       "sin_hermanos": [0, 1]})
    with pytest.raises(RuntimeError, match="preparar"):
        MotorCPU(1).procesar("conteo", {}, "x", None, {"inicio": 0, "fin": 1, "limites": []})


def test_nucleos_hibridos_simulados():
    from pdn.worker.hardware import elegir_nucleos
    topo = {"logicos": list(range(12)), "rendimiento": list(range(8)),
            "eficiencia": list(range(8, 12)), "sin_hermanos": [0, 2, 4, 6, 8, 9, 10, 11]}
    assert elegir_nucleos("rendimiento", topo) == list(range(8))
    assert elegir_nucleos("eficiencia", topo) == [8, 9, 10, 11]
    assert elegir_nucleos("sin_hermanos", topo) == [0, 2, 4, 6, 8, 9, 10, 11]
    assert elegir_nucleos("0-2,5", topo) == [0, 1, 2, 5]
