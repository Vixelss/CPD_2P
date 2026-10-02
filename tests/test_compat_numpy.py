"""numba-cuda usa np.row_stack, que numpy 2.4 quito; el motor GPU le da el alias."""

from __future__ import annotations

import subprocess
import sys

import pytest

CODIGO = ("from pdn.motores.gpu_cuda import compat_numpy; compat_numpy(); "
          "import numpy as np; assert np.row_stack is not None; "
          "import numba.cuda.np.arrayobj")


def test_numba_cuda_carga_con_numpy_nuevo():
    pytest.importorskip("numba.cuda")
    # En un proceso aparte, para que el alias no venga de otra prueba
    r = subprocess.run([sys.executable, "-c", CODIGO], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
