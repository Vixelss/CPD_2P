"""Cluster simulado local (CONTEXTO.md, seccion 16.3.5).

Levanta un Master en este proceso (127.0.0.1) y de 3 a 6 workers como
procesos aparte, cada uno con su propia carpeta de datos (enlaces a los
archivos preparados) y un retardo artificial por MB para simular nodos
rapidos y lentos (un i9 y un i3). Sirve para las pruebas y para la demo sin
hardware.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

from pdn.comun import config as cfgmod
from pdn.master.servidor import Master
from pdn.preparacion.fasta_a_seq import Preparado

RAIZ = cfgmod.RAIZ


def puerto_libre() -> int:
    """Un puerto TCP libre en 127.0.0.1."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def config_rapida(**master) -> dict:
    """Configuracion con tiempos cortos, para pruebas."""
    cfg = cfgmod.cargar(os.devnull)
    cfg["master"].update({"latido_timeout_s": 2.0, "tarea_vencida_min_s": 4.0,
                          "preparacion_timeout_s": 60.0, **master})
    cfg["worker"].update({"latido_s": 0.3, "master_timeout_s": 3.0, "calibracion_mb": 1})
    return cfg


class ClusterSimulado:
    """Master local + workers en procesos aparte."""

    def __init__(self, archivos: list[Preparado], retardos: list[float] | None = None,
                 carpeta: str | None = None, config: dict | None = None, procesos: int = 1,
                 impl: str = "numpy", corruptos: dict[int, int | list[int]] | None = None,
                 puerto: int | None = None, iniciar_master: bool = True,
                 masters_extra: list[str] | None = None) -> None:
        self.archivos = archivos
        self.retardos = retardos if retardos is not None else [0.0, 0.0, 0.0]
        self.carpeta = carpeta or tempfile.mkdtemp(prefix="pdn_sim_")
        self.config = config or config_rapida()
        self.procesos = procesos
        self.impl = impl
        self.corruptos = corruptos or {}
        self.puerto = puerto or puerto_libre()
        self.iniciar_master = iniciar_master
        self.masters_extra = masters_extra or []
        self.master: Master | None = None
        self.procs: dict[str, subprocess.Popen] = {}
        self.nombres: list[str] = []

    def _carpeta_worker(self, k: int) -> str:
        d = os.path.join(self.carpeta, "nodo%d" % k)
        os.makedirs(d, exist_ok=True)
        for prep in self.archivos:
            raiz = os.path.splitext(prep.ruta_seq)[0]
            nombre = os.path.basename(raiz)
            for ext in (".seq", ".huellas", ".idx"):
                destino = os.path.join(d, nombre + ext)
                if os.path.lexists(destino):
                    os.remove(destino)
                if ext == ".seq" and k in self.corruptos:
                    # Copia con bytes cambiados (las huellas siguen siendo las buenas)
                    shutil.copy(raiz + ext, destino)
                    posiciones = self.corruptos[k]
                    with open(destino, "r+b") as f:
                        for pos in (posiciones if isinstance(posiciones, list) else [posiciones]):
                            f.seek(pos)
                            b = f.read(1)
                            f.seek(pos)
                            f.write(bytes([b[0] ^ 0x01]))
                else:
                    os.symlink(raiz + ext, destino)
        return d

    def lanzar_worker(self, k: int, retardo: float, extra: list[str] | None = None,
                      rehacer_datos: bool = True) -> str:
        """Lanza el worker k y devuelve su worker_id."""
        nombre = "sim-%d" % k
        datos = self._carpeta_worker(k) if rehacer_datos else os.path.join(self.carpeta, "nodo%d" % k)
        cmd = [sys.executable, "-m", "pdn.worker", "--master", "127.0.0.1:%d" % self.puerto,
               "--nombre", nombre, "--datos", datos, "--procesos", str(self.procesos),
               "--impl", self.impl, "--retardo", str(retardo),
               "--latido", str(self.config["worker"]["latido_s"]),
               "--timeout-master", str(self.config["worker"]["master_timeout_s"]),
               "--config", os.devnull, "--log", os.path.join(self.carpeta, nombre + ".log")]
        for m in self.masters_extra:
            cmd += ["--respaldo", m]
        cmd += extra or []
        entorno = dict(os.environ, PYTHONPATH=RAIZ + os.pathsep + os.environ.get("PYTHONPATH", ""))
        salida = open(os.path.join(self.carpeta, nombre + ".salida"), "w")
        extra = extra or []
        dispositivo = extra[extra.index("--dispositivo") + 1] if "--dispositivo" in extra else "cpu"
        wid = "%s:%s" % (nombre, dispositivo)
        self.procs[wid] = subprocess.Popen(cmd, cwd=RAIZ, env=entorno, stdout=salida, stderr=subprocess.STDOUT)
        self.nombres.append(wid)
        return wid

    def __enter__(self) -> "ClusterSimulado":
        if self.iniciar_master:
            self.master = Master(self.config, self.puerto, datos=os.path.dirname(self.archivos[0].ruta_seq),
                                 carpeta_resultados=os.path.join(self.carpeta, "resultados"),
                                 host="127.0.0.1",
                                 carpeta_referencias=os.path.join(self.carpeta, "referencias")).iniciar()
        for k, r in enumerate(self.retardos):
            self.lanzar_worker(k, r)
        if self.master is not None:
            self.master.esperar_workers(len(self.retardos), timeout=60)
        return self

    def correr(self, cfg: dict, timeout: float = 120) -> dict:
        """Ejecuta una corrida en el Master local y devuelve su resumen."""
        return self.master.correr(cfg, timeout=timeout)

    def matar(self, wid: str) -> None:
        """Mata un worker sin avisar (SIGKILL)."""
        p = self.procs.get(wid)
        if p is not None:
            p.kill()

    def __exit__(self, *exc) -> None:
        for p in self.procs.values():
            if p.poll() is None:
                p.terminate()
        limite = time.time() + 5
        for p in self.procs.values():
            try:
                p.wait(max(0.1, limite - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
        if self.master is not None:
            self.master.detener()
