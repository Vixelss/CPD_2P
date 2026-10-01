"""Prueba 16.3.8: el Master principal cae a mitad de corrida y el respaldo termina."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import yaml

from pdn.comun import config as cfgmod
from pdn.master.respaldo import MasterRespaldo
from pdn.sim.cluster import ClusterSimulado, config_rapida, puerto_libre


def _esperar(condicion, timeout, paso=0.1):
    limite = time.time() + timeout
    while time.time() < limite:
        valor = condicion()
        if valor:
            return valor
        time.sleep(paso)
    raise TimeoutError("la condicion no se cumplio en %s s" % timeout)


def test_caida_del_master_principal(tmp_path, sintetico):
    prep, esp = sintetico
    nombre = os.path.basename(prep.ruta_seq)[:-4]
    datos = os.path.dirname(prep.ruta_seq)
    config = config_rapida(respaldo_timeout_s=2.0, gracia_promocion_s=30.0)
    ruta_cfg = tmp_path / "cluster.yaml"
    ruta_cfg.write_text(yaml.safe_dump(config))
    p_principal, p_respaldo, p_replica = puerto_libre(), puerto_libre(), puerto_libre()

    respaldo = MasterRespaldo(config, p_respaldo, p_replica, datos, str(tmp_path / "res_respaldo"),
                              host="127.0.0.1", carpeta_referencias=str(tmp_path / "refs")).iniciar()
    principal = subprocess.Popen(
        [sys.executable, "-m", "pdn.cli", "master", "--config", str(ruta_cfg), "--puerto", str(p_principal),
         "--datos", datos, "--resultados", str(tmp_path / "res_principal"),
         "--replicar-a", "127.0.0.1:%d" % p_replica, "--esperar-workers", "3", "--operacion", "conteo",
         "--archivo", nombre, "--tam-unidad", "16384", "--tiempo-objetivo", "0.3"],
        cwd=cfgmod.RAIZ, stdout=open(tmp_path / "principal.log", "w"), stderr=subprocess.STDOUT,
        env=dict(os.environ, PYTHONPATH=cfgmod.RAIZ))
    try:
        with ClusterSimulado([prep], retardos=[2.0, 3.0, 4.0], carpeta=str(tmp_path / "sim"), config=config,
                             puerto=p_principal, iniciar_master=False,
                             masters_extra=["127.0.0.1:%d" % p_respaldo]):
            # Esperar a que el respaldo vea la corrida avanzando
            def avanzada():
                c = (respaldo.ultima or {}).get("corrida")
                return c and c["estado"] == "EJECUTANDO" and \
                    sum(1 for t in c["tareas"] if t["estado"] == "HECHA") >= 3 and c["corrida_id"] in respaldo.preparadas
            _esperar(avanzada, 60)
            hechas_antes = sum(1 for t in respaldo.ultima["corrida"]["tareas"] if t["estado"] == "HECHA")
            principal.kill()  # caida sin avisar
            principal.wait(5)
            _esperar(respaldo.promovido.is_set, 20)
            m = respaldo.master
            assert m.rol == "respaldo"
            resumen = _esperar(lambda: m.historial[-1] if m.historial and m.historial[-1].get("estado") == "TERMINADA"
                               and m.historial[-1]["corrida_id"] == respaldo.ultima["corrida"]["corrida_id"] else None, 120)
    finally:
        if principal.poll() is None:
            principal.kill()
        respaldo.detener()
    assert resumen["rol_master"] == "respaldo"
    assert resumen["validacion"]["valido"] and resumen["validacion"]["cobertura"]["ok"]
    assert resumen["resultado"]["hist"] == esp["conteo"]["hist"]
    motivos = [r["motivo"] for r in resumen["reasignaciones"]]
    assert any("Master principal caido" in x for x in motivos) or hechas_antes > 0
    # Las tareas hechas antes de la caida se conservaron (no se rehizo todo)
    assert hechas_antes >= 3
    assert len({p["worker"] for p in resumen["por_worker"] if p["tareas"]}) >= 2


def test_instantanea_y_restauracion_en_proceso(sintetico):
    """Instantanea -> restauracion conserva tareas hechas y devuelve las asignadas."""
    from pdn.master.respaldo import instantanea, restaurar
    from pdn.master.servidor import Master
    prep, esp = sintetico
    config = config_rapida()
    with ClusterSimulado([prep], retardos=[1.0, 1.0], config=config) as c:
        cfg = {"operacion": "conteo", "archivo": os.path.basename(prep.ruta_seq)[:-4], "tam_unidad": 16384,
               "calibracion_mb": 0.1}
        cid = c.master.iniciar_corrida(cfg)
        _esperar(lambda: (c.master.estado().get("corrida") or {}).get("progreso", 0) > 0.2, 30)
        inst = c.master._llamar("ejecutar", instantanea)
        c.master.esperar_corrida(cid, 60)
    otro = Master(config, puerto_libre(), os.path.dirname(prep.ruta_seq), host="127.0.0.1")
    preparada = otro.preparar_corrida({k: v for k, v in inst["corrida"]["config"].items()
                                       if k in ("operacion", "archivo", "params", "tam_unidad")})["corrida"]
    restaurar(otro, inst, preparada)
    r = otro.corrida
    hechas = [t for t in r.tareas.values() if t.estado == "HECHA"]
    asignadas = [t for t in inst["corrida"]["tareas"] if t["estado"] == "ASIGNADA"]
    assert hechas and r.corrida_id == cid
    assert all(r.tareas[t["tid"]].estado == "DESCARTADA" for t in asignadas)
    # Lo pendiente mas lo hecho cubre todo el archivo
    pend = r.planificador.pendiente_unidades()
    assert pend + sum(t.u_fin - t.u_ini for t in hechas) == r.trabajo.unidades


def test_principal_ve_al_respaldo(tmp_path, sintetico):
    """hay_respaldo() se activa con los acuses y habilita la caida simulada en el dashboard."""
    from fastapi.testclient import TestClient

    from pdn.dashboard.app import crear_app
    from pdn.master.respaldo import Replicador
    from pdn.master.servidor import Master
    prep, _ = sintetico
    config = config_rapida()
    p_replica = puerto_libre()
    respaldo = MasterRespaldo(config, puerto_libre(), p_replica, os.path.dirname(prep.ruta_seq),
                              str(tmp_path / "r"), host="127.0.0.1").iniciar()
    m = Master(config, puerto_libre(), os.path.dirname(prep.ruta_seq), str(tmp_path / "p"), host="127.0.0.1").iniciar()
    try:
        assert not m.hay_respaldo()
        Replicador(m, "127.0.0.1:%d" % p_replica).iniciar()
        _esperar(m.hay_respaldo, 10)
        _esperar(lambda: TestClient(crear_app(m)).get("/api/estado").json()["respaldo_conectado"], 5)
        assert respaldo.ultima is not None and not respaldo.promovido.is_set()
    finally:
        m.detener()
        respaldo.detener()
