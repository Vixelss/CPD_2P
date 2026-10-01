"""Motor NPU: modelo CpG cuantizado en Core ML (CONTEXTO.md, seccion 10.4).

- En macOS con Apple Silicon: Core ML con compute_units=CPU_AND_NE (pide el
  Neural Engine; Core ML decide capa por capa si de verdad lo usa, y eso se
  verifica con powermetrics, ver verificar_ane.py).
- En cualquier otro sistema: la MISMA red en numpy (alternativa en CPU), con
  los mismos pesos (INT8 decuantizados o float32). Se informa siempre como
  "alternativa en CPU": nunca se afirma que algo corre en la NPU si no es asi.
Solo ejecuta la operacion de zonas de interes.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import time

import numpy as np

from pdn.motores.npu import modelo as M
from pdn.motores.npu.construir_modelo import LOTE, rutas
from pdn.operaciones import zonas
from pdn.preparacion.fasta_a_seq import abrir_seq

log = logging.getLogger("pdn.npu")


def hay_neural_engine() -> tuple[bool, str | None]:
    """True si es una Mac con Apple Silicon y coremltools puede cargar modelos."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False, "no es una Mac con Apple Silicon"
    try:
        import coremltools  # noqa: F401,PLC0415
    except Exception as e:
        return False, "coremltools no disponible: %s" % e
    return True, None


class NucleoNPU:
    """Clasifica ventanas con el modelo (Core ML o numpy)."""

    nombre = "npu"

    def __init__(self, motor: "MotorNPU") -> None:
        self.motor = motor

    def clasificar(self, raw: np.ndarray, loc: np.ndarray, w: int) -> np.ndarray:
        return self.motor.inferir(raw, loc, w)


class MotorNPU:
    """Motor del dispositivo 'npu'."""

    dispositivo = "npu"
    OPCIONES = ("backend", "precision", "lote", "unidades", "carpeta_modelos")

    def __init__(self, backend: str = "auto", precision: str = "int8", lote: int = LOTE,
                 unidades: str = "cpu_y_ne", carpeta_modelos: str | None = None) -> None:
        if backend not in ("auto", "coreml", "cpu"):
            raise ValueError("backend debe ser auto, coreml o cpu")
        if precision not in ("int8", "fp16", "fp32"):
            raise ValueError("precision debe ser int8, fp16 o fp32")
        if unidades not in ("cpu_y_ne", "solo_cpu", "todas"):
            raise ValueError("unidades debe ser cpu_y_ne, solo_cpu o todas")
        self.backend_pedido = backend
        self.precision = precision
        self.lote = int(lote)
        self.unidades = unidades
        self.carpeta = carpeta_modelos
        self.backend: str | None = None
        self.aviso: str | None = None
        self.pesos: dict | None = None
        self.ventana: int | None = None
        self.mlmodel = None
        self.nucleo = NucleoNPU(self)
        self.preparacion_s: float | None = None
        self.ventanas = 0
        self.inferencias = 0
        self.t_inferencia = 0.0
        self._mapas: dict[str, np.ndarray] = {}

    @property
    def nombre(self) -> str:
        return "npu-%s-%s" % (self.backend or "?", self.precision)

    def preparar(self) -> float:
        """Carga el modelo (fuera del cronometro) y lo precalienta."""
        t0 = time.perf_counter()
        kw = {"carpeta": self.carpeta} if self.carpeta else {}
        r = rutas(**kw)
        if not os.path.exists(r["pesos"]):
            raise FileNotFoundError("No existe %s: ejecute python -m pdn.motores.npu.construir_modelo" % r["pesos"])
        p, self.ventana = M.cargar(r["pesos"])
        self.pesos = M.cuantizar_int8(p)[0] if self.precision == "int8" else p
        ne, motivo = hay_neural_engine()
        backend = self.backend_pedido
        if backend == "auto":
            backend = "coreml" if ne else "cpu"
        if backend == "coreml" and not ne:
            raise RuntimeError("Core ML no disponible: %s" % motivo)
        if backend == "coreml":
            import coremltools as ct  # noqa: PLC0415
            unidades = {"cpu_y_ne": ct.ComputeUnit.CPU_AND_NE, "solo_cpu": ct.ComputeUnit.CPU_ONLY,
                        "todas": ct.ComputeUnit.ALL}[self.unidades]
            paquete = r["int8"] if self.precision == "int8" else r["fp16"]
            self.mlmodel = ct.models.MLModel(paquete, compute_units=unidades)
        else:
            self.aviso = "Neural Engine no disponible (%s): alternativa en CPU, misma red en numpy" % motivo
            log.warning(self.aviso)
        self.backend = backend
        # Precalentamiento: una inferencia de un lote completo
        muestra = np.frombuffer(b"ACGTCGCGAT" * (self.ventana // 10 + 1), dtype=np.uint8)[:self.ventana]
        self.inferir(np.tile(muestra, 2), np.array([0, 0], dtype=np.int64), self.ventana)
        self.ventanas = self.inferencias = 0
        self.t_inferencia = 0.0
        self.preparacion_s = time.perf_counter() - t0
        return self.preparacion_s

    def inferir(self, raw: np.ndarray, loc: np.ndarray, w: int) -> np.ndarray:
        """Probabilidad de zona de interes para cada ventana."""
        if w != self.ventana:
            raise ValueError("El modelo NPU es para ventanas de %d bases y la corrida pide %d; construya otro "
                             "con python -m pdn.motores.npu.construir_modelo --ventana %d" % (self.ventana, w, w))
        t0 = time.perf_counter()
        salida = []
        for a in range(0, loc.shape[0], self.lote):
            x = M.one_hot(raw, loc[a:a + self.lote], w)
            if self.backend == "coreml":
                n = x.shape[0]
                if n < LOTE:  # el modelo tiene lote fijo
                    x = np.concatenate([x, np.zeros((LOTE - n, 4, w), np.float32)])
                for b in range(0, x.shape[0], LOTE):
                    out = self.mlmodel.predict({"x": x[b:b + LOTE]})["probabilidad"].reshape(-1)
                    salida.append(out[:max(0, min(LOTE, n - b))])
                    self.inferencias += 1
            else:
                salida.append(M.predecir(x, self.pesos, w))
                self.inferencias += 1
        self.t_inferencia += time.perf_counter() - t0
        self.ventanas += int(loc.shape[0])
        return np.concatenate(salida) if salida else np.zeros(0)

    def _abrir(self, ruta: str) -> np.ndarray:
        if ruta not in self._mapas:
            self._mapas[ruta] = abrir_seq(ruta)
        return self._mapas[ruta]

    def procesar(self, nombre: str, params: dict, ruta_a: str, ruta_b: str | None,
                 carga: dict) -> tuple[dict, dict]:
        """Procesa una tarea de zonas con el modelo."""
        if self.pesos is None:
            raise RuntimeError("El motor NPU no esta preparado: llame a preparar() antes")
        if nombre != "zonas":
            raise ValueError("La NPU solo ejecuta la operacion de zonas de interes (se pidio %s)" % nombre)
        v0, i0, t_inf0 = self.ventanas, self.inferencias, self.t_inferencia
        t0 = time.perf_counter()
        parcial = zonas.procesar(self._abrir(ruta_a), carga["inicio"], carga["fin"], params, carga["limites"],
                                 self.nucleo)
        seg = time.perf_counter() - t0
        n = self.ventanas - v0
        t_inf = self.t_inferencia - t_inf0
        return parcial, {"t_calculo": seg, "impl": self.nombre, "ventanas": n, "inferencias": self.inferencias - i0,
                         "t_inferencia_s": round(t_inf, 5),
                         "ventanas_s": round(n / t_inf, 1) if t_inf > 0 else None,
                         "ops_estimadas": n * M.operaciones_por_ventana(self.ventana), "aviso": self.aviso}

    def cerrar(self) -> None:
        self._mapas.clear()
        self.mlmodel = None

    def describir(self) -> dict:
        return {"dispositivo": "npu", "impl": self.nombre, "backend": self.backend, "precision": self.precision,
                "unidades": self.unidades if self.backend == "coreml" else None, "ventana": self.ventana,
                "lote": self.lote, "preparacion_s": self.preparacion_s, "aviso": self.aviso,
                # Core ML solo PIDE el Neural Engine; que lo use se verifica con powermetrics
                "pide_neural_engine": self.backend == "coreml" and self.unidades != "solo_cpu"}
