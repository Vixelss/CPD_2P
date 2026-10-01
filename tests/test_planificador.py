"""Planificador: raciones adaptativas, fase final y estrategias."""

from __future__ import annotations

import pytest

from pdn.master.planificador import Planificador

U = 4 << 20


def _plan(n=100, **kw) -> Planificador:
    return Planificador(n, U, n * U, **kw)


def test_racion_minima_al_inicio():
    p = _plan()
    p.registrar_worker("rapido", "cpu", 1000 * U)
    p.registrar_worker("lento", "cpu", 1 * U)
    assert p.pedir("rapido", ["rapido", "lento"]) == (0, 1)
    assert p.pedir("lento", ["rapido", "lento"]) == (1, 2)


def test_tamano_adaptativo_y_maximo():
    p = _plan(1000, tiempo_objetivo=0.5, max_bytes=128 << 20)
    p.registrar_worker("a", "cpu", None)
    p.actualizar_velocidad("a", 40 * U, 1.0)  # 40 unidades/s -> 20 por 0,5 s
    a, b = p.pedir("a", ["a"])
    assert b - a == 20
    p.actualizar_velocidad("a", 10_000 * U, 1.0)
    a, b = p.pedir("a", ["a"])
    assert b - a == 32  # 128 MB / 4 MB


def test_maximo_gpu_mayor():
    p = _plan(1000, max_bytes=128 << 20, max_bytes_gpu=512 << 20)
    p.registrar_worker("g", "gpu", None)
    p.actualizar_velocidad("g", 100_000 * U, 1.0)
    a, b = p.pedir("g", ["g"])
    assert b - a == 128


def test_fase_final_decreciente():
    p = _plan(40, tiempo_objetivo=0.5)
    for w in ("a", "b"):
        p.registrar_worker(w, "cpu", None)
        p.actualizar_velocidad(w, 40 * U, 1.0)
    # Pendiente 40 unidades < suma(80 u/s) * 0,5 * 2 = 80 -> tope 40 / (2*2) = 10
    a, b = p.pedir("a", ["a", "b"])
    assert b - a == 10


def test_devolver_fusiona():
    p = _plan(10)
    p.pendientes = []
    p.devolver(5, 7)
    p.devolver(2, 3)
    p.devolver(3, 5)
    assert p.pendientes == [[2, 7]]
    assert p.pendiente_unidades() == 5


def test_fija():
    p = _plan(100, estrategia="fija", tam_fijo=16 << 20)
    p.registrar_worker("a", "cpu", 1)
    assert p.pedir("a", ["a"]) == (0, 4)


def test_proporcional():
    p = _plan(100, estrategia="proporcional")
    p.registrar_worker("a", "cpu", 3 * U)
    p.registrar_worker("b", "cpu", 1 * U)
    p.iniciar_proporcional(["a", "b"])
    assert p.pedir("b", ["a", "b"]) == (75, 100)
    assert p.pedir("a", ["a", "b"]) == (0, 75)
    assert p.pedir("a", ["a", "b"]) is None
    assert p.terminado()


def test_proporcional_reserva_liberada():
    p = _plan(10, estrategia="proporcional")
    p.registrar_worker("a", "cpu", 1 * U)
    p.registrar_worker("b", "cpu", 1 * U)
    p.iniciar_proporcional(["a", "b"])
    p.liberar_reserva("b")
    assert p.pedir("a", ["a"]) == (0, 5)
    assert p.pedir("a", ["a"]) == (5, 10)


def test_zonas_prefiere_npu():
    p = _plan(10, operacion="zonas")
    p.registrar_worker("npu", "npu", None)
    p.registrar_worker("cpu", "cpu", None)
    p.actualizar_velocidad("npu", 100 * U, 1.0)
    p.actualizar_velocidad("cpu", 100 * U, 1.0)
    # Lo pendiente cabe en la capacidad de la NPU: la CPU cede
    assert p.pedir("cpu", ["npu", "cpu"]) is None
    assert p.pedir("npu", ["npu", "cpu"]) is not None


@pytest.mark.parametrize("kw,msg", [({"estrategia": "rara"}, "Estrategia"),
                                    ({"tiempo_objetivo": 9}, "tiempo objetivo")])
def test_validaciones(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _plan(**kw)
