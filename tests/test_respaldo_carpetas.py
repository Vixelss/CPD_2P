"""El respaldo promovido no se cuelga si la carpeta compartida (NFS del Master) no responde."""

from __future__ import annotations

import os
import threading
import time

from pdn.comun import config as cfgmod
from pdn.master import respaldo


def test_carpeta_que_responde(tmp_path):
    cfg = cfgmod.cargar(os.devnull)
    r, ref, local = respaldo.carpetas_tras_promocion(cfg, str(tmp_path / "res"), "REF")
    assert (r, ref, local) == (str(tmp_path / "res"), "REF", False)


def test_carpeta_colgada_pasa_a_local(monkeypatch, tmp_path):
    # Simula un NFS sin servidor: el acceso a la carpeta compartida no vuelve nunca
    bloqueo = threading.Event()
    original = os.makedirs

    def makedirs(ruta, *a, **k):
        if str(ruta).startswith("/colgado"):
            bloqueo.wait(30)
        return original(ruta, *a, **k)

    monkeypatch.setattr(os, "makedirs", makedirs)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = cfgmod.cargar(os.devnull)
    t0 = time.time()
    r, ref, local = respaldo.carpetas_tras_promocion(cfg, "/colgado/resultados", "/colgado/ref")
    bloqueo.set()
    assert time.time() - t0 < 5
    assert local and ref is None
    assert r == str(tmp_path / "pdn-resultados")
