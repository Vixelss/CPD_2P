"""Dos workers con el mismo nombre y el mismo Master no pueden correr a la vez."""

from __future__ import annotations

import os
import subprocess
import sys

from pdn.worker.__main__ import tomar_candado

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_candado_unico_por_worker_y_master():
    wid = "prueba-candado-%d:cpu" % os.getpid()
    c1 = tomar_candado(wid, "127.0.0.1:1")
    assert c1 is not None
    assert tomar_candado(wid, "127.0.0.1:2") is not None  # otro Master: se permite
    codigo = ("import sys; from pdn.worker.__main__ import tomar_candado; "
              "sys.exit(0 if tomar_candado(%r, '127.0.0.1:1') is None else 1)" % wid)
    r = subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ)
    assert r.returncode == 0  # otro proceso no consigue el candado
    c1.close()
    r = subprocess.run([sys.executable, "-c", codigo.replace("is None else 1", "is not None else 1")], cwd=RAIZ)
    assert r.returncode == 0  # liberado: ahora si
