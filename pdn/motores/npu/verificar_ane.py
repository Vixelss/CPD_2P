"""Verifica en la Mac si Core ML usa de verdad el Neural Engine (CONTEXTO.md, seccion 10.4).

Ejecuta el modelo con CPU_AND_NE y con CPU_ONLY mientras sudo -n powermetrics
mide la potencia del Neural Engine (ANE). Si la potencia del ANE sube con
CPU_AND_NE y no con CPU_ONLY, el modelo corre en el Neural Engine. Si no se
puede confirmar, se informa tal cual.

Uso (en nodo-hidalgo): python -m pdn.motores.npu.verificar_ane --segundos 10
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from pdn.monitoreo.energia import LectorPowermetrics
from pdn.motores.npu import modelo as M
from pdn.motores.npu.motor_npu import MotorNPU


def medir(unidades: str, precision: str, segundos: float) -> dict:
    """Inferencias continuas durante 'segundos' con la potencia del ANE."""
    motor = MotorNPU(backend="coreml", precision=precision, unidades=unidades)
    motor.preparar()
    rng = np.random.default_rng(0)
    w = motor.ventana
    raw = rng.choice(np.frombuffer(b"ACGTCG", dtype=np.uint8), size=8192 * w)
    loc = np.arange(8192, dtype=np.int64) * w
    pm = LectorPowermetrics(intervalo_ms=500)
    time.sleep(1.5)
    base = dict(pm.acumulado)
    t0 = time.perf_counter()
    ventanas = 0
    while time.perf_counter() - t0 < segundos:
        motor.inferir(raw, loc, w)
        ventanas += loc.size
    seg = time.perf_counter() - t0
    time.sleep(1.0)
    fin = dict(pm.acumulado)
    pm.cerrar()
    energia = {k: round(fin[k] - base[k], 3) for k in fin}
    return {"unidades": unidades, "precision": precision, "segundos": round(seg, 2), "ventanas": ventanas,
            "ventanas_s": round(ventanas / seg, 1), "energia_j": energia, "muestras_powermetrics": pm.muestras,
            "motivo": pm.motivo,
            "j_por_ventana_ane": (energia["ane_j"] / ventanas) if ventanas else None,
            "ops_por_w_estimado": (ventanas * M.operaciones_por_ventana(w) / (energia["ane_j"] + energia["cpu_j"]))
            if (energia["ane_j"] + energia["cpu_j"]) > 0 else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--segundos", type=float, default=10)
    ap.add_argument("--salida", default="resultados/verificacion_ane.json")
    a = ap.parse_args()
    filas = [medir(u, p, a.segundos) for u in ("cpu_y_ne", "solo_cpu") for p in ("int8", "fp16")]
    ne = next(f for f in filas if f["unidades"] == "cpu_y_ne" and f["precision"] == "int8")
    cpu = next(f for f in filas if f["unidades"] == "solo_cpu" and f["precision"] == "int8")
    usa = None
    if ne["muestras_powermetrics"] and cpu["muestras_powermetrics"]:
        usa = ne["energia_j"]["ane_j"] > max(0.05, 5 * cpu["energia_j"]["ane_j"])
    resultado = {"mediciones": filas, "usa_neural_engine": usa,
                 "conclusion": "sin dato: powermetrics no dio muestras (ver scripts/setup_mac.sh)" if usa is None else
                 ("el ANE consume energia con CPU_AND_NE y no con CPU_ONLY: el modelo corre en el Neural Engine"
                  if usa else "no se observo consumo del ANE: Core ML ejecuto el modelo en CPU o GPU")}
    with open(a.salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=1)
    print(json.dumps(resultado, indent=1))


if __name__ == "__main__":
    main()
