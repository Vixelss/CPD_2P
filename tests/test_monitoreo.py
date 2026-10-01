"""E6: monitoreo de recursos y energia (lo no medido queda en None con motivo)."""

from __future__ import annotations

import builtins
import csv
import os
import plistlib
import time

import pytest

from pdn.monitoreo import energia
from pdn.monitoreo.recursos import Monitor


def _zona(base, nombre, valor, rango):
    d = base / nombre
    d.mkdir()
    (d / "energy_uj").write_text(str(valor))
    (d / "max_energy_range_uj").write_text(str(rango))
    (d / "name").write_text("package-0")
    return d


def test_rapl_acumula_y_maneja_desborde(tmp_path):
    z = _zona(tmp_path, "intel-rapl:0", 900_000, 1_000_000)
    lector = energia.LectorRAPL(str(tmp_path))
    assert lector.disponible
    assert lector.leer_j() == 0.0
    (z / "energy_uj").write_text("950000")       # +0,05 J
    assert abs(lector.leer_j() - 0.05) < 1e-9
    (z / "energy_uj").write_text("50000")        # desborde: +0,1 J
    assert abs(lector.leer_j() - 0.15) < 1e-9


def test_rapl_varios_paquetes(tmp_path):
    a = _zona(tmp_path, "intel-rapl:0", 0, 10**9)
    b = _zona(tmp_path, "intel-rapl:1", 0, 10**9)
    lector = energia.LectorRAPL(str(tmp_path))
    (a / "energy_uj").write_text("1000000")
    (b / "energy_uj").write_text("2000000")
    assert lector.leer_j() == 3.0


def test_rapl_sin_permiso(tmp_path, monkeypatch):
    _zona(tmp_path, "intel-rapl:0", 1, 10)
    abrir = builtins.open

    def falso(ruta, *a, **k):
        if str(ruta).endswith("energy_uj"):
            raise PermissionError(ruta)
        return abrir(ruta, *a, **k)

    monkeypatch.setattr(builtins, "open", falso)
    lector = energia.LectorRAPL(str(tmp_path))
    assert not lector.disponible and "sin permiso" in lector.motivo
    assert lector.leer_j() is None


def test_rapl_ausente(tmp_path):
    lector = energia.LectorRAPL(str(tmp_path / "no_existe"))
    assert not lector.disponible and lector.leer_j() is None and lector.motivo


def test_parsear_powermetrics():
    muestra = plistlib.dumps({"elapsed_ns": 1_000_000_000,
                              "processor": {"cpu_power": 2500.0, "gpu_power": 300.0, "ane_power": 1200.0}})
    m = energia.parsear_powermetrics(muestra + b"\x00")
    assert (m["cpu_w"], m["gpu_w"], m["ane_w"]) == (2.5, 0.3, 1.2)


def test_medidor_sin_hardware_deja_none():
    m = energia.MedidorEnergia(gpu=True, mac=False).muestra()
    for clave in ("gpu_j", "ane_j"):
        assert m[clave] is None
    if not os.path.exists(energia.RAPL_BASE):
        assert m["cpu_j"] is None and m["motivos_energia"]["cpu"]
    assert m["motivos_energia"].get("gpu")


def test_monitor_muestras_y_resumen():
    mon = Monitor("cpu")
    marca = mon.marcar()
    primera = mon.muestra()
    assert 0 <= primera["cpu_pct"] <= 100 and len(primera["cpu_nucleos"]) >= 1
    r1 = mon.resumen(marca)
    assert r1["muestras"] == 1 and r1["cpu_pct_medio"] is None  # una sola muestra: sin dato
    time.sleep(0.2)
    mon.muestra()
    r2 = mon.resumen(marca)
    assert r2["cpu_pct_medio"] is not None and r2["cpu_por_nucleo"] is not None
    assert r2["gpu_uso_pct_medio"] is None  # no hay GPU: vacio, nunca cero


def test_cluster_produce_recursos_y_energia(sintetico):
    """La corrida guarda recursos.csv coherente y energia.csv con None donde no se mide."""
    from pdn.sim.cluster import ClusterSimulado
    prep, _ = sintetico
    with ClusterSimulado([prep], retardos=[2.0, 2.0]) as c:
        r = c.correr({"operacion": "conteo", "archivo": os.path.basename(prep.ruta_seq)[:-4],
                      "tam_unidad": 16384, "calibracion_mb": 0.1})
    filas = list(csv.DictReader(open(os.path.join(r["carpeta"], "recursos.csv"))))
    assert len(filas) >= 4
    for f in filas:
        assert 0 <= float(f["cpu_pct"]) <= 100
        assert f["worker"] in ("sim-0:cpu", "sim-1:cpu")
    t = [float(f["t_abs"]) for f in filas if f["worker"] == "sim-0:cpu"]
    assert t == sorted(t)
    e = list(csv.DictReader(open(os.path.join(r["carpeta"], "energia.csv"))))
    assert {f["worker"] for f in e} == {"sim-0:cpu", "sim-1:cpu"}
    if not os.path.exists(energia.RAPL_BASE):
        assert all(f["energia_j"] == "" for f in e)  # no medido: vacio, nunca cero
        assert r["energia_por_arquitectura"]["cpu"]["energia_j"] is None


def test_energia_por_arquitectura():
    from pdn.master.servidor import Master
    filas = [{"dispositivo": "cpu", "energia_j": 10.0, "energia_tareas_j": None, "bytes": 1048576 * 20},
             {"dispositivo": "cpu", "energia_j": None, "energia_tareas_j": 5.0, "bytes": 1048576 * 10},
             {"dispositivo": "gpu", "energia_j": None, "energia_tareas_j": None, "bytes": 1048576}]
    r = Master.energia_por_arquitectura(filas)
    assert r["cpu"]["energia_j"] == 15.0 and r["cpu"]["mb_por_j"] == 2.0
    assert r["gpu"]["energia_j"] is None and r["gpu"]["mb_por_j"] is None
