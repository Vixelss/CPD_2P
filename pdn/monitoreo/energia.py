"""Medicion de energia por dispositivo (CONTEXTO.md, seccion 11.2).

- CPU en Linux: contadores RAPL (/sys/class/powercap/intel-rapl:*/energy_uj),
  acumulados en microjoules que se reinician al llegar a max_energy_range_uj.
  Solo los lee root salvo que scripts/setup_nodo.sh les de permiso; si no se
  pueden leer, el valor queda en None con el motivo "sin permiso".
  RAPL mide el paquete completo: si en el nodo corren un worker de CPU y uno
  de GPU, los dos ven la misma energia de paquete.
- GPU NVIDIA: potencia instantanea por NVML, integrada en el tiempo.
- Mac: sudo -n powermetrics (CPU, GPU y Neural Engine) en segundo plano.
Lo que no se mide queda en None, nunca en cero.
"""

from __future__ import annotations

import glob
import os
import plistlib
import shutil
import subprocess
import sys
import threading
import time

RAPL_BASE = "/sys/class/powercap"


# ---------------------------------------------------------------------------
# RAPL
# ---------------------------------------------------------------------------

class LectorRAPL:
    """Energia acumulada de los paquetes de CPU (zonas RAPL de primer nivel)."""

    def __init__(self, base: str = RAPL_BASE) -> None:
        self.zonas: list[dict] = []
        self.motivo: str | None = None
        candidatas = sorted(glob.glob(os.path.join(base, "intel-rapl:[0-9]")) +
                            glob.glob(os.path.join(base, "intel-rapl:[0-9][0-9]")))
        if not candidatas:
            self.motivo = "RAPL no disponible en este equipo"
            return
        for z in candidatas:
            try:
                with open(os.path.join(z, "max_energy_range_uj")) as f:
                    rango = int(f.read())
                with open(os.path.join(z, "energy_uj")) as f:
                    previo = int(f.read())
            except PermissionError:
                self.motivo = "sin permiso para leer RAPL (ejecute scripts/setup_nodo.sh)"
                self.zonas = []
                return
            except (OSError, ValueError):
                continue
            nombre = ""
            try:
                with open(os.path.join(z, "name")) as f:
                    nombre = f.read().strip()
            except OSError:
                pass
            self.zonas.append({"ruta": os.path.join(z, "energy_uj"), "rango": rango, "previo": previo,
                               "acumulado_uj": 0, "nombre": nombre})
        if not self.zonas and self.motivo is None:
            self.motivo = "RAPL sin zonas legibles"

    @property
    def disponible(self) -> bool:
        return bool(self.zonas)

    def leer_j(self) -> float | None:
        """Energia acumulada desde que se creo el lector, en joules (con desbordes)."""
        if not self.zonas:
            return None
        for z in self.zonas:
            try:
                with open(z["ruta"]) as f:
                    actual = int(f.read())
            except (OSError, ValueError):
                return None
            delta = actual - z["previo"]
            if delta < 0:  # el contador se reinicio al llegar al maximo
                delta += z["rango"]
            z["acumulado_uj"] += delta
            z["previo"] = actual
        return sum(z["acumulado_uj"] for z in self.zonas) / 1e6


# ---------------------------------------------------------------------------
# NVML
# ---------------------------------------------------------------------------

class LectorNVML:
    """Uso, memoria, temperatura y potencia de una GPU NVIDIA (pynvml)."""

    def __init__(self, indice: int = 0) -> None:
        self.motivo: str | None = None
        self.h = None
        self.acumulado_j = 0.0
        self._t_previo: float | None = None
        self._w_previo: float | None = None
        try:
            import pynvml  # noqa: PLC0415
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(indice)
        except Exception as e:
            self.motivo = "NVML no disponible: %s" % e

    @property
    def disponible(self) -> bool:
        return self.h is not None

    def leer(self) -> dict:
        """Muestra de la GPU; la energia se integra con la regla del trapecio."""
        if self.h is None:
            return {}
        nv = self.nv
        m: dict = {}
        try:
            u = nv.nvmlDeviceGetUtilizationRates(self.h)
            m["gpu_uso_pct"] = float(u.gpu)
        except Exception:
            pass
        try:
            mem = nv.nvmlDeviceGetMemoryInfo(self.h)
            m["gpu_mem_mb"] = round(mem.used / 1048576)
        except Exception:
            pass
        try:
            m["gpu_temp_c"] = float(nv.nvmlDeviceGetTemperature(self.h, nv.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
        try:
            w = nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0
            ahora = time.time()
            if self._t_previo is not None and self._w_previo is not None:
                self.acumulado_j += (w + self._w_previo) / 2 * (ahora - self._t_previo)
            self._t_previo, self._w_previo = ahora, w
            m["gpu_w"] = round(w, 2)
            m["gpu_j"] = round(self.acumulado_j, 3)
        except Exception:
            pass
        return m


# ---------------------------------------------------------------------------
# powermetrics (macOS)
# ---------------------------------------------------------------------------

def parsear_powermetrics(bloque: bytes) -> dict:
    """Extrae potencias (W) de una muestra plist de powermetrics."""
    datos = plistlib.loads(bloque.strip(b"\x00").strip())
    proc = datos.get("processor", {})

    def watts(*claves):
        for c in claves:
            if c in proc and proc[c] is not None:
                return float(proc[c]) / 1000.0  # powermetrics informa mW
        return None

    return {"cpu_w": watts("cpu_power", "combined_power"), "gpu_w": watts("gpu_power"),
            "ane_w": watts("ane_power"), "intervalo_ns": datos.get("elapsed_ns")}


class LectorPowermetrics:
    """Lanza sudo -n powermetrics en segundo plano e integra las potencias."""

    def __init__(self, intervalo_ms: int = 1000) -> None:
        self.motivo: str | None = None
        self.proc = None
        self.acumulado = {"cpu_j": 0.0, "gpu_j": 0.0, "ane_j": 0.0}
        self.ultima: dict = {}
        self.muestras = 0
        if sys.platform != "darwin":
            self.motivo = "powermetrics solo existe en macOS"
            return
        if not shutil.which("powermetrics"):
            self.motivo = "no se encontro powermetrics"
            return
        cmd = ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power,ane_power",
               "-i", str(intervalo_ms), "-f", "plist"]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            self.motivo = "no se pudo lanzar powermetrics: %s" % e
            return
        self._hilo = threading.Thread(target=self._leer, daemon=True)
        self._hilo.start()

    def _leer(self) -> None:
        buf = b""
        while self.proc is not None and self.proc.poll() is None:
            trozo = self.proc.stdout.read1(65536) if hasattr(self.proc.stdout, "read1") else self.proc.stdout.read(4096)
            if not trozo:
                break
            buf += trozo
            while b"\x00" in buf:
                bloque, buf = buf.split(b"\x00", 1)
                try:
                    m = parsear_powermetrics(bloque)
                except Exception:
                    continue
                seg = (m.get("intervalo_ns") or 1e9) / 1e9
                for clave in ("cpu", "gpu", "ane"):
                    if m.get(clave + "_w") is not None:
                        self.acumulado[clave + "_j"] += m[clave + "_w"] * seg
                self.ultima = m
                self.muestras += 1
        if self.proc is not None and self.proc.returncode not in (None, 0):
            err = self.proc.stderr.read().decode(errors="replace")
            self.motivo = "powermetrics termino: %s (configure sudoers, ver scripts/setup_mac.sh)" % err.strip()[:200]

    @property
    def disponible(self) -> bool:
        return self.proc is not None and self.muestras > 0

    def leer(self) -> dict:
        if not self.disponible:
            return {}
        return {**{k: round(v, 3) for k, v in self.acumulado.items()},
                **{k: self.ultima.get(k) for k in ("cpu_w", "gpu_w", "ane_w")}}

    def cerrar(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None


# ---------------------------------------------------------------------------
# Medidor combinado
# ---------------------------------------------------------------------------

class MedidorEnergia:
    """Energia acumulada por dispositivo (cpu_j, gpu_j, ane_j) con sus motivos."""

    def __init__(self, gpu: bool = True, mac: bool = True) -> None:
        self.rapl = LectorRAPL() if sys.platform.startswith("linux") else None
        self.nvml = LectorNVML() if gpu else None
        self.pm = LectorPowermetrics() if (mac and sys.platform == "darwin") else None
        self._t_rapl: float | None = None
        self._j_rapl: float | None = None

    def muestra(self) -> dict:
        """Lectura actual: energias acumuladas (J) y potencias (W); None si no se mide."""
        m: dict = {"cpu_j": None, "gpu_j": None, "ane_j": None, "cpu_w": None, "gpu_w": None,
                   "ane_w": None, "motivos_energia": {}}
        if self.rapl is not None:
            j = self.rapl.leer_j()
            if j is None:
                m["motivos_energia"]["cpu"] = self.rapl.motivo
            else:
                ahora = time.time()
                if self._t_rapl is not None and ahora > self._t_rapl:
                    m["cpu_w"] = round((j - self._j_rapl) / (ahora - self._t_rapl), 2)
                self._t_rapl, self._j_rapl = ahora, j
                m["cpu_j"] = round(j, 3)
        if self.nvml is not None:
            if self.nvml.disponible:
                m.update({k: v for k, v in self.nvml.leer().items()})
            else:
                m["motivos_energia"]["gpu"] = self.nvml.motivo
        if self.pm is not None:
            if self.pm.disponible:
                m.update(self.pm.leer())
            else:
                m["motivos_energia"]["mac"] = self.pm.motivo or "powermetrics sin muestras todavia"
        return m

    def cerrar(self) -> None:
        if self.pm is not None:
            self.pm.cerrar()
