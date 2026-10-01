"""Fabrica de motores por dispositivo.

Todo motor expone: preparar() -> segundos, procesar(nombre, params, ruta_a,
ruta_b, carga) -> (parcial, info), cerrar(), describir() -> dict.
"""

from __future__ import annotations

DISPOSITIVOS = ("cpu", "gpu", "npu", "opencl")


def crear_motor(dispositivo: str, opciones: dict | None = None):
    """Crea (sin preparar) el motor de un dispositivo con sus opciones."""
    o = dict(opciones or {})
    if dispositivo == "cpu":
        from pdn.motores.cpu import MotorCPU  # noqa: PLC0415
        return MotorCPU(procesos=o.get("procesos"), nucleos=o.get("nucleos", "todos"),
                        impl=o.get("impl", "numpy"), reservar=int(o.get("reservar", 0) or 0))
    if dispositivo == "gpu":
        from pdn.motores.gpu_cuda import MotorGPU  # noqa: PLC0415
        return MotorGPU(**{k: v for k, v in o.items() if k in MotorGPU.OPCIONES})
    if dispositivo == "npu":
        from pdn.motores.npu.motor_npu import MotorNPU  # noqa: PLC0415
        return MotorNPU(**{k: v for k, v in o.items() if k in MotorNPU.OPCIONES})
    if dispositivo == "opencl":
        from pdn.motores.opencl import MotorOpenCL  # noqa: PLC0415
        return MotorOpenCL(**o)
    raise ValueError("Dispositivo desconocido: %r (validos: %s)" % (dispositivo, ", ".join(DISPOSITIVOS)))
