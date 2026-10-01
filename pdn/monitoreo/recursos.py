"""Monitoreo de recursos de un nodo (CONTEXTO.md, seccion 11.1).

Adaptado de referencias/P1.4/monitor.py (marcar, resumen(desde)) sin la
parte de PowerShell: en Linux la temperatura se lee con
psutil.sensors_temperatures().

Un solo muestreador de psutil.cpu_percent por proceso: la referencia de
interval=None es global al proceso y dos muestreadores se la corrompen. Por
eso solo Monitor.muestra() llama a cpu_percent.

Recursos no medidos quedan vacios (None), nunca en cero: en resumen(), una
magnitud con menos de MINIMO_MUESTRAS lecturas no se publica.
"""

from __future__ import annotations

import collections
import threading
import time

import psutil

from pdn.monitoreo.energia import MedidorEnergia

MINIMO_MUESTRAS = 2
SENSORES_CPU = ("coretemp", "k10temp", "zenpower", "cpu_thermal", "acpitz")


def temperatura_cpu() -> tuple[float | None, str | None]:
    """Temperatura del paquete de CPU y el sensor usado (None si no hay lectura)."""
    try:
        sensores = psutil.sensors_temperatures()
    except (AttributeError, OSError):
        return None, None
    for nombre in SENSORES_CPU:
        lecturas = sensores.get(nombre) or []
        valores = [l.current for l in lecturas if l.current is not None and l.current > 0]
        if valores:
            # El primer sensor de coretemp es el paquete; si no, el maximo
            paquete = [l.current for l in lecturas if (l.label or "").lower().startswith(("package", "tctl", "tdie"))]
            return float(paquete[0] if paquete else max(valores)), nombre
    return None, None


class Monitor:
    """Muestreador de recursos y energia del nodo."""

    def __init__(self, dispositivo: str = "cpu", energia: bool = True, historial: int = 3600) -> None:
        self.dispositivo = dispositivo
        self._cerrojo = threading.Lock()
        self._proceso = psutil.Process()
        self._red_previa: tuple[float, int, int] | None = None
        self.medidor = MedidorEnergia(gpu=dispositivo == "gpu") if energia else None
        self.muestras: collections.deque[dict] = collections.deque(maxlen=historial)
        self._indice = 0
        psutil.cpu_percent(None, percpu=True)  # fija la referencia del muestreador unico

    def muestra(self) -> dict:
        """Toma una muestra (valores no medidos en None) y la guarda en el historial."""
        with self._cerrojo:
            ahora = time.time()
            por_nucleo = psutil.cpu_percent(None, percpu=True)
            mem = psutil.virtual_memory()
            red = psutil.net_io_counters()
            m: dict = {"t": ahora,
                       "cpu_pct": round(sum(por_nucleo) / len(por_nucleo), 1) if por_nucleo else None,
                       "cpu_nucleos": por_nucleo, "ram_pct": mem.percent,
                       "ram_usada_mb": int((mem.total - mem.available) / 1048576),
                       "rss_worker_mb": int(self._proceso.memory_info().rss / 1048576),
                       "red_tx_b": red.bytes_sent, "red_rx_b": red.bytes_recv,
                       "red_tx_mb_s": None, "red_rx_mb_s": None}
            if self._red_previa is not None and ahora > self._red_previa[0]:
                dt = ahora - self._red_previa[0]
                m["red_tx_mb_s"] = round((red.bytes_sent - self._red_previa[1]) / dt / 1048576, 3)
                m["red_rx_mb_s"] = round((red.bytes_recv - self._red_previa[2]) / dt / 1048576, 3)
            self._red_previa = (ahora, red.bytes_sent, red.bytes_recv)
            try:
                frec = psutil.cpu_freq()
                m["frecuencia_mhz"] = round(frec.current) if frec else None
            except Exception:
                m["frecuencia_mhz"] = None
            m["temp_cpu_c"], m["sensor_temp"] = temperatura_cpu()
            if self.medidor is not None:
                m.update(self.medidor.muestra())
            self.muestras.append(m)
            self._indice += 1
            return m

    def energia_actual(self) -> dict:
        """Energia acumulada por dispositivo sin tomar muestra de CPU (para medir una tarea)."""
        if self.medidor is None:
            return {}
        with self._cerrojo:
            e = self.medidor.muestra()
        return {k: e.get(k) for k in ("cpu_j", "gpu_j", "ane_j")}

    def marcar(self) -> int:
        """Indice de la proxima muestra (para resumir desde aqui)."""
        return self._indice

    def resumen(self, desde: int = 0) -> dict:
        """Promedio y maximo de cada magnitud desde el indice dado (None si faltan muestras)."""
        with self._cerrojo:
            todas = list(self.muestras)
        primero = self._indice - len(todas)
        serie = todas[max(0, desde - primero):]
        datos: dict = {"muestras": len(serie)}
        for clave in ("cpu_pct", "temp_cpu_c", "ram_pct", "rss_worker_mb", "gpu_uso_pct", "gpu_mem_mb",
                      "gpu_temp_c", "cpu_w", "gpu_w", "ane_w", "frecuencia_mhz"):
            valores = [m[clave] for m in serie if m.get(clave) is not None]
            if len(valores) >= MINIMO_MUESTRAS:
                datos[clave + "_medio"] = sum(valores) / len(valores)
                datos[clave + "_max"] = max(valores)
            else:
                datos[clave + "_medio"] = datos[clave + "_max"] = None
        for clave in ("cpu_j", "gpu_j", "ane_j"):
            valores = [m[clave] for m in serie if m.get(clave) is not None]
            datos[clave] = (valores[-1] - valores[0]) if len(valores) >= MINIMO_MUESTRAS else None
        nucleos = [m["cpu_nucleos"] for m in serie if m.get("cpu_nucleos")]
        if len(nucleos) >= MINIMO_MUESTRAS:
            n = len(nucleos[0])
            datos["cpu_por_nucleo"] = [round(sum(s[i] for s in nucleos if len(s) > i) / len(nucleos), 1)
                                       for i in range(n)]
        else:
            datos["cpu_por_nucleo"] = None
        return datos

    def cerrar(self) -> None:
        if self.medidor is not None:
            self.medidor.cerrar()
