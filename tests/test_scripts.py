"""E10: scripts de despliegue (shellcheck) y lectura de cluster.yaml desde bash."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

import pytest

from pdn.comun.config import RAIZ

SCRIPTS = sorted(glob.glob(os.path.join(RAIZ, "scripts", "*.sh")))


def test_estan_todos_los_scripts():
    nombres = {os.path.basename(s) for s in SCRIPTS}
    assert {"setup_nodo.sh", "setup_master.sh", "setup_mac.sh", "copiar_llaves.sh", "montar_nfs.sh",
            "desplegar_codigo.sh", "distribuir_datos.sh", "estado_cluster.sh", "lanzar_cluster.sh",
            "detener_cluster.sh", "compilar_simd.sh"} <= nombres
    for s in SCRIPTS:
        if s.endswith("comun.sh"):  # biblioteca que se carga con source
            continue
        texto = open(s).read()
        assert "set -euo pipefail" in texto, s


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck no instalado")
def test_shellcheck():
    r = subprocess.run(["shellcheck", "-x"] + [os.path.relpath(s, RAIZ) for s in SCRIPTS], cwd=RAIZ,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_cfg_nodos():
    r = subprocess.run([sys.executable, "scripts/cfg.py", "nodos", "--rol", "respaldo"], cwd=RAIZ,
                       capture_output=True, text=True, check=True)
    campos = r.stdout.strip().split("\t")
    assert campos[0] == "nodo-carranza" and len(campos) == 9
    r = subprocess.run([sys.executable, "scripts/cfg.py", "valor", "red.puerto_tareas"], cwd=RAIZ,
                       capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "5555"


def test_generar_hostfile(tmp_path):
    salida = tmp_path / "hf"
    subprocess.run([sys.executable, "scripts/generar_hostfile.py", "--max-slots", "1", "--nodos", "3",
                    "-o", str(salida)], cwd=RAIZ, check=True, capture_output=True)
    assert salida.read_text().splitlines() == ["nodo-vivanco slots=1", "nodo-carranza slots=1",
                                               "nodo-ocampo slots=1"]
