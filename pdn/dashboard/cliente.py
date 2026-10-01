"""Cliente HTTP de la API del dashboard (para la linea de comandos y las series)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class ErrorAPI(RuntimeError):
    """La API devolvio un error; el mensaje es el del Master."""


class Cliente:
    """Habla con el Master activo por su API HTTP."""

    def __init__(self, url: str = "http://127.0.0.1:8000", timeout: float = 30) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _pedir(self, metodo: str, ruta: str, cuerpo: dict | None = None):
        datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
        req = urllib.request.Request(self.url + ruta, data=datos, method=metodo,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            try:
                detalle = json.loads(e.read().decode()).get("detail")
            except Exception:
                detalle = str(e)
            raise ErrorAPI(detalle) from None
        except urllib.error.URLError as e:
            raise ErrorAPI("No se pudo conectar con el Master en %s: %s" % (self.url, e.reason)) from None

    def estado(self) -> dict:
        return self._pedir("GET", "/api/estado")

    def iniciar(self, cfg: dict) -> str:
        return self._pedir("POST", "/api/corridas", cfg)["corrida_id"]

    def resumen(self, corrida_id: str) -> dict | None:
        r = self._pedir("GET", "/api/corridas/%s" % corrida_id)
        return r if r.get("terminada") else None

    def correr(self, cfg: dict, timeout: float = 3600, intervalo: float = 0.5) -> dict:
        """Inicia una corrida y espera su resumen."""
        cid = self.iniciar(cfg)
        limite = time.time() + timeout
        while time.time() < limite:
            r = self.resumen(cid)
            if r is not None:
                return r["resumen"]
            time.sleep(intervalo)
        raise TimeoutError("La corrida %s no termino en %s s" % (cid, timeout))

    def simular_fallo(self, worker: str, modo: str) -> dict:
        return self._pedir("POST", "/api/fallo", {"worker": worker, "modo": modo})
