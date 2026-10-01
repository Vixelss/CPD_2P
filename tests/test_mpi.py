"""E4: reparto MPI, referencia secuencial, mpirun local y escalabilidad."""

from __future__ import annotations

import os
import shutil

import pytest

from pdn.comun import config as cfgmod
from pdn.mpi.escalabilidad import ajustar_amdahl, speedup_amdahl
from pdn.mpi.reparto import pesos_para, repartir

hay_mpi = pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun no disponible")


def test_cluster_yaml_valido():
    config = cfgmod.cargar()
    assert len(config["nodos"]) == 6
    assert [n["hostname"] for n in cfgmod.nodos_mpi(config)] == [
        "nodo-vivanco", "nodo-carranza", "nodo-ocampo", "nodo-naranjo", "nodo-palomo"]
    assert config["planificador"]["estrategia"] == "adaptativa"


@pytest.mark.parametrize("n,pesos", [(10, [1, 1, 1]), (7, [1, 1, 1, 1]), (100, [3, 1]),
                                     (3, [1, 1, 1, 1, 1]), (0, [1, 2]), (50, [0, 0])])
def test_repartir_cubre_todo(n, pesos):
    rangos = repartir(n, pesos)
    assert rangos[0][0] == 0 and rangos[-1][1] == n
    assert all(a[1] == b[0] for a, b in zip(rangos, rangos[1:]))
    assert all(b >= a for a, b in rangos)


def test_repartir_proporcional():
    assert repartir(100, [3, 1]) == [(0, 75), (75, 100)]
    assert pesos_para(["a", "b", "a"], "proporcional", {"a": 2.0, "b": 1.0}) == [2.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="calibrada"):
        pesos_para(["a"], "proporcional", {"b": 1})
    with pytest.raises(ValueError, match="necesita"):
        pesos_para(["a"], "proporcional", None)


def test_amdahl_recupera_fraccion():
    f = 0.08
    puntos = [(p, speedup_amdahl(p, f)) for p in (1, 2, 4, 8, 16)]
    fit = ajustar_amdahl(puntos)
    assert abs(fit["fraccion_secuencial"] - f) < 1e-9
    assert abs(fit["speedup_maximo"] - 12.5) < 1e-6
    assert ajustar_amdahl([(1, 1.0)])["fraccion_secuencial"] is None


def test_referencia_y_validacion_del_master(tmp_path, sintetico):
    """La referencia se guarda y el modo dinamico se valida contra ella."""
    from pdn.mpi.referencia import correr_referencia
    from pdn.sim.cluster import ClusterSimulado

    prep, esp = sintetico
    nombre = os.path.basename(prep.ruta_seq)[:-4]
    datos = os.path.dirname(prep.ruta_seq)
    with ClusterSimulado([prep], retardos=[0.0, 0.0], carpeta=str(tmp_path / "sim")) as c:
        ref = correr_referencia({"operacion": "conteo", "archivo": nombre, "tam_unidad": 16384},
                                datos, c.master.carpeta_referencias, repeticiones=2, calentamiento=True)
        assert ref["clave"]["hist"] == esp["conteo"]["hist"]
        assert len(ref["tiempos_s"]) == 2 and os.path.exists(ref["ruta"])
        r = c.correr({"operacion": "conteo", "archivo": nombre, "tam_unidad": 16384, "calibracion_mb": 0.1})
        v = r["validacion"]
        assert v["coincide_referencia"] is True and v["valido"]
        assert r["speedup"] is not None
        import json
        vel = json.load(open(os.path.join(c.master.carpeta_resultados, "velocidades.json")))
        assert set(vel["conteo"]) == {"sim-0", "sim-1"}


@hay_mpi
@pytest.mark.parametrize("np_", [1, 2, 4])
@pytest.mark.parametrize("op", ["conteo", "patrones", "zonas"])
def test_mpirun_local_da_la_referencia(tmp_path, sintetico, op, np_):
    from pdn.mpi.lanzador import correr_mpi
    from pdn.mpi.referencia import correr_referencia

    prep, esp = sintetico
    nombre = os.path.basename(prep.ruta_seq)[:-4]
    datos = os.path.dirname(prep.ruta_seq)
    np_ = min(np_, len(os.sched_getaffinity(0)))
    cfg = {"operacion": op, "archivo": nombre, "tam_unidad": 8192,
           "params": {"patrones": esp["patrones"]["patrones"]} if op == "patrones" else {}}
    refs = str(tmp_path / "refs")
    correr_referencia(cfg, datos, refs)
    r = correr_mpi(cfg, np_, datos=datos, referencias=refs, resultados=str(tmp_path / "res"))
    assert r["validacion"]["coincide_referencia"] is True, r["validacion"]
    assert r["validacion"]["valido"] and r["ranks"] == np_
    assert len(r["bindings"]) == np_  # salida de --report-bindings
    assert os.path.exists(os.path.join(r["carpeta"], "bindings.txt"))
    if op == "conteo":
        assert r["resultado"]["hist"] == esp["conteo"]["hist"]


@hay_mpi
def test_mpirun_comparacion_y_simd(tmp_path, par_mutado):
    from pdn.mpi.lanzador import correr_mpi
    from pdn.motores.simd.envoltorio import tiene_avx2

    a, b, esp = par_mutado
    datos = os.path.dirname(a.ruta_seq)
    cfg = {"operacion": "comparacion", "archivo": os.path.basename(a.ruta_seq)[:-4],
           "archivo_b": os.path.basename(b.ruta_seq)[:-4], "tam_unidad": 8192}
    r = correr_mpi(cfg, 2, datos=datos, impl="simd" if tiene_avx2() else "simd_escalar")
    res = r["resultado"]
    assert (res["reales"], res["solo_caso"], res["con_n"]) == (esp["reales"], esp["solo_caso"], esp["con_n"])


@hay_mpi
def test_mpirun_proporcional(tmp_path, sintetico):
    import socket

    from pdn.mpi.lanzador import correr_mpi
    prep, esp = sintetico
    cfg = {"operacion": "conteo", "archivo": os.path.basename(prep.ruta_seq)[:-4], "tam_unidad": 8192}
    r = correr_mpi(cfg, 2, datos=os.path.dirname(prep.ruta_seq), reparto="proporcional",
                   velocidades={socket.gethostname(): 100.0})
    assert r["resultado"]["hist"] == esp["conteo"]["hist"]


@hay_mpi
def test_series_de_escalabilidad(tmp_path, sintetico):
    from pdn.mpi.escalabilidad import correr_series

    prep, _ = sintetico
    salida = str(tmp_path / "esc")
    r = correr_series({"operacion": "conteo", "archivo": os.path.basename(prep.ruta_seq)[:-4],
                       "tam_unidad": 8192}, os.path.dirname(prep.ruta_seq), salida, [1, 2], repeticiones=2)
    for archivo in ("escalabilidad.csv", "amdahl.json", "escalabilidad.png"):
        assert os.path.exists(os.path.join(salida, archivo)), archivo
    series = {f["serie"] for f in r["filas"]}
    assert {"1_referencia", "2_nucleos_un_nodo"} <= series
    assert all(f["valido"] for f in r["filas"] if "valido" in f)


def test_lanzador_comando():
    from pdn.mpi.lanzador import construir_comando, extraer_bindings
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun no disponible")
    cmd = construir_comando({"operacion": "conteo", "archivo": "X"}, 8, hostfile="hf",
                            python="/home/pdn/pdn-env/bin/python", interfaz="192.168.1.0/24")
    texto = " ".join(cmd)
    assert "--map-by core --bind-to core --report-bindings" in texto
    assert "--mca btl_tcp_if_include 192.168.1.0/24" in texto
    assert "--hostfile hf" in texto and "/home/pdn/pdn-env/bin/python -m pdn.mpi.mpi_correr" in texto
    assert extraer_bindings("[h:1] MCW rank 0 bound to socket 0[core 0[hwt 0]]: [B/.]\nbasura") == [
        "[h:1] MCW rank 0 bound to socket 0[core 0[hwt 0]]: [B/.]"]
    with pytest.raises(ValueError):
        construir_comando({"archivo": "X"}, 0)
