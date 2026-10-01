"""Monitoreo de recursos de un nodo (CONTEXTO.md, seccion 11.1).

Un solo muestreador de psutil.cpu_percent por proceso: la referencia de
interval=None es global al proceso y dos muestreadores se la corrompen. Por
eso solo Monitor.muestra() llama a cpu_percent, y solo desde el hilo de
latidos.
"""

from __future__ import annotations

import threading
import time

import psutil


class Monitor:
    """Muestreador de recursos del nodo (se completa en la etapa E6)."""

    def __init__(self, dispositivo: str = "cpu") -> None:
        self.dispositivo = dispositivo
        self._cerrojo = threading.Lock()
        self._proceso = psutil.Process()
        self._red_previa = None
        psutil.cpu_percent(None, percpu=True)  # fija la referencia

    def muestra(self) -> dict:
        """Una muestra de recursos (valores no medidos quedan en None)."""
        with self._cerrojo:
            por_nucleo = psutil.cpu_percent(None, percpu=True)
            mem = psutil.virtual_memory()
            red = psutil.net_io_counters()
            ahora = time.time()
            m = {"t": ahora, "cpu_pct": round(sum(por_nucleo) / len(por_nucleo), 1) if por_nucleo else None,
                 "cpu_nucleos": por_nucleo, "ram_pct": mem.percent,
                 "ram_usada_mb": int((mem.total - mem.available) / 1048576),
                 "rss_worker_mb": int(self._proceso.memory_info().rss / 1048576),
                 "red_tx_b": red.bytes_sent, "red_rx_b": red.bytes_recv}
            try:
                frec = psutil.cpu_freq()
                m["frecuencia_mhz"] = round(frec.current) if frec else None
            except Exception:
                m["frecuencia_mhz"] = None
            return m
