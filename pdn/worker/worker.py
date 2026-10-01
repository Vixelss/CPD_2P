"""Proceso worker del modo dinamico (CONTEXTO.md, seccion 9.1).

1. Detecta hardware y archivos locales con sus huellas.
2. Prepara el motor antes de cualquier cronometro.
3. Se registra en el Master (lista ordenada de Masters: principal, respaldo).
4. Bucle peticion-respuesta: PEDIR -> PREPARAR/TAREA/ESPERAR/FIN.
5. Un hilo aparte envia LATIDO cada segundo y recibe ordenes asincronas
   (SIMULAR_FALLO, DETENER).
Todos los modulos se importan al arrancar, para que una caida de NFS no
rompa un worker en marcha.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import socket
import tempfile
import threading
import time

import numpy as np
import psutil
import zmq

from pdn import __version__ as VERSION_PDN
from pdn.comun import protocolo as P
from pdn.comun.huellas import crc_unidades
from pdn.operaciones import comparacion, conteo, patrones, zonas  # noqa: F401  (import temprano)
from pdn.preparacion.fasta_a_seq import abrir_seq
from pdn.worker import hardware
from pdn.worker.motores import crear_motor

log = logging.getLogger("pdn.worker")
MB = 1024 * 1024


def escanear_archivos(carpeta: str) -> dict:
    """Archivos preparados en una carpeta: nombre -> tamano, huella global, unidad."""
    salida = {}
    for ruta in sorted(glob.glob(os.path.join(carpeta, "*.huellas"))):
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        seq = os.path.join(carpeta, nombre + ".seq")
        if not os.path.exists(seq):
            continue
        try:
            with open(ruta, encoding="utf-8") as f:
                tabla = json.load(f)
        except (OSError, ValueError):
            continue
        salida[nombre] = {"tamano": os.path.getsize(seq), "global": tabla.get("global"),
                          "tam_unidad": tabla.get("tam_unidad")}
    return salida


class Worker:
    """Un worker: un proceso por dispositivo de computo."""

    def __init__(self, masters: list[str], dispositivo: str = "cpu", datos: str | None = None,
                 nfs_datos: str | None = None, nombre: str | None = None,
                 motor_opciones: dict | None = None, retardo_s_mb: float = 0.0,
                 latido_s: float = 1.0, timeout_master: float = 3.0,
                 calibracion_sintetica_mb: float = 0.0, monitor=None, reserva_gpu: int = 1) -> None:
        if not masters:
            raise ValueError("Hace falta al menos un Master")
        self.masters = [m if ":" in m else "%s:5555" % m for m in masters]
        self.i_master = 0
        self.dispositivo = dispositivo
        self.hostname = nombre or socket.gethostname()
        self.wid = "%s:%s" % (self.hostname, dispositivo)
        self.datos = datos or os.path.expanduser("~/pdn-datos")
        self.nfs_datos = nfs_datos or "/cluster/datos"
        self.motor_base = dict(motor_opciones or {})
        self.motor_opciones: dict | None = None
        self.reserva_gpu = reserva_gpu
        self.retardo_s_mb = retardo_s_mb
        self.latido_s = latido_s
        self.timeout_master = timeout_master
        self.calibracion_sintetica_mb = calibracion_sintetica_mb
        self.monitor = monitor
        self.motor = None
        self.hw: dict = {}
        self._ctx = zmq.Context()
        self._sock = None
        self._registrado = False
        self._fin = threading.Event()
        self._congelado = threading.Event()
        self._cerrojo_master = threading.Lock()
        self._mapas: dict[str, np.ndarray] = {}
        self.corrida: dict | None = None
        self.corrida_id: str | None = None
        self.tareas_hechas = 0

    # -- utilidades ---------------------------------------------------------
    @property
    def master_actual(self) -> str:
        with self._cerrojo_master:
            return self.masters[self.i_master]

    def _rotar_master(self) -> None:
        with self._cerrojo_master:
            self.i_master = (self.i_master + 1) % len(self.masters)
        log.warning("%s: el Master no responde; se pasa a %s", self.wid, self.master_actual)

    def _abrir_socket(self) -> None:
        if self._sock is not None:
            self._sock.close(0)
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect("tcp://" + self.master_actual)

    def _peticion(self, msg: dict, timeout: float | None = None) -> dict | None:
        """Envia un mensaje y espera la respuesta; None si el Master no contesta."""
        if self._sock is None:
            self._abrir_socket()
        self._sock.send(P.codificar(msg))
        if self._sock.poll(int((timeout or self.timeout_master) * 1000)):
            return P.decodificar(self._sock.recv())
        self._abrir_socket()
        return None

    def _ruta(self, nombre: str | None, origen: str) -> str | None:
        if not nombre:
            return None
        base = self.nfs_datos if origen == "nfs" else self.datos
        return os.path.join(base, nombre + ".seq")

    def _mapa(self, ruta: str) -> np.ndarray:
        if ruta not in self._mapas:
            self._mapas[ruta] = abrir_seq(ruta)
        return self._mapas[ruta]

    # -- ciclo de vida --------------------------------------------------------
    def preparar_motor(self, opciones: dict | None = None) -> float:
        """Crea y prepara el motor (fuera del cronometro). Devuelve segundos.

        Las opciones de la corrida se combinan con las del arranque (que
        incluyen, por ejemplo, el nucleo reservado para alimentar la GPU).
        """
        opciones = {**self.motor_base, **(opciones or {})}
        if self.motor is not None and opciones == self.motor_opciones:
            return 0.0
        if self.motor is not None:
            self.motor.cerrar()
        self.motor_opciones = opciones
        self.motor = crear_motor(self.dispositivo, opciones)
        return self.motor.preparar()

    def _registrar(self) -> bool:
        msg = P.crear(P.REGISTRO, self.wid, hostname=self.hostname, dispositivo=self.dispositivo,
                      hardware=self.hw, archivos=escanear_archivos(self.datos), version=VERSION_PDN)
        r = self._peticion(msg)
        if r is None or r["tipo"] != P.ACEPTADO:
            self._rotar_master()
            return False
        self.latido_s = float(r.get("latido_s", self.latido_s))
        self._registrado = True
        log.info("%s registrado en %s (rol %s)", self.wid, self.master_actual, r.get("rol"))
        if self.calibracion_sintetica_mb > 0:
            r2 = self._peticion(P.crear(P.CALIBRACION, self.wid, calibracion=self._calibracion_sintetica()))
            if r2 is None:
                self._registrado = False
                return False
        return True

    def _calibracion_sintetica(self) -> dict:
        # Conteo sobre un bloque sintetico, para mostrar en la topologia
        n = int(self.calibracion_sintetica_mb * MB)
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "calibracion.seq")
            rng.choice(np.frombuffer(b"ACGTacgtN", dtype=np.uint8), size=n).tofile(ruta)
            carga = {"inicio": 0, "fin": n, "limites": [[0, 0]]}
            try:
                self.motor.procesar("conteo", conteo.validar_parametros({}), ruta, None, carga)
                t0 = time.perf_counter()
                self.motor.procesar("conteo", conteo.validar_parametros({}), ruta, None, carga)
                return {"conteo_mb_s": round(n / MB / (time.perf_counter() - t0), 1)}
            except Exception as e:  # la calibracion sintetica es informativa
                return {"error": str(e)}

    def correr(self) -> None:
        """Bucle principal del worker."""
        self.hw = hardware.detectar(completo=True, ip_master=self.master_actual.split(":")[0])
        gpu = self.hw.get("gpu") or {}
        if self.dispositivo == "cpu" and gpu.get("disponible") and not gpu.get("simulador") \
                and "reservar" not in self.motor_base and self.reserva_gpu:
            # Leccion 5 del P1: un nucleo queda libre para el hilo que alimenta a la GPU
            self.motor_base["reservar"] = self.reserva_gpu
            log.info("%s: GPU en el nodo, se reservan %d nucleos para alimentarla", self.wid, self.reserva_gpu)
        self.preparar_motor()
        hilo = threading.Thread(target=self._latidos, name="latidos", daemon=True)
        hilo.start()
        siguiente: dict | None = None
        while not self._fin.is_set():
            self._quizas_congelado()
            if not self._registrado:
                if not self._registrar():
                    time.sleep(0.5)
                continue
            msg = siguiente or P.crear(P.PEDIR, self.wid, self.corrida_id)
            siguiente = None
            r = self._peticion(msg)
            if r is None:
                self._registrado = False
                self._rotar_master()
                continue
            siguiente = self._manejar(r)
        self.cerrar()

    def cerrar(self) -> None:
        self._fin.set()
        if self.motor is not None:
            self.motor.cerrar()
            self.motor = None
        if self._sock is not None:
            self._sock.close(0)
            self._sock = None

    def _quizas_congelado(self) -> None:
        # Fallo simulado "congelado": el proceso sigue vivo pero no responde
        while self._congelado.is_set() and not self._fin.is_set():
            time.sleep(0.5)

    # -- mensajes -------------------------------------------------------------
    def _manejar(self, r: dict) -> dict | None:
        tipo = r["tipo"]
        if tipo == P.ESPERAR:
            time.sleep(float(r.get("pausa", 0.2)))
            return None
        if tipo == P.ACEPTADO and r.get("reregistrar"):
            self._registrado = False
            return None
        if tipo == P.PREPARAR:
            return self._preparar(r)
        if tipo == P.TAREA:
            return self._tarea(r)
        if tipo == P.FIN:
            log.info("%s: fin de la corrida %s (%d tareas)", self.wid, r.get("corrida_id"), self.tareas_hechas)
            time.sleep(0.1)
            return None
        if tipo == P.DETENER:
            self._fin.set()
            return None
        time.sleep(0.2)
        return None

    def _preparar(self, r: dict) -> dict:
        cid = r["corrida_id"]
        cfg = r["corrida"]
        t0 = time.perf_counter()
        try:
            prep_motor = self.preparar_motor(cfg.get("motor"))
            ruta_a = self._ruta(cfg["nombre_a"], cfg.get("origen_datos", "local"))
            ruta_b = self._ruta(cfg.get("nombre_b"), cfg.get("origen_datos", "local"))
            for ruta, largo in ((ruta_a, cfg.get("largo_a")), (ruta_b, cfg.get("largo_b"))):
                if ruta is None:
                    continue
                if not os.path.exists(ruta):
                    raise FileNotFoundError("No existe %s en este nodo" % ruta)
                if largo is not None and os.path.getsize(ruta) != largo:
                    raise ValueError("%s mide %d bytes y el Master espera %d" % (ruta, os.path.getsize(ruta), largo))
            self._mapas.clear()
            calib = None
            if cfg.get("calibracion"):
                carga = cfg["calibracion"]
                nbytes = carga["fin"] - carga["inicio"]
                self.motor.procesar(cfg["operacion"], cfg["params"], ruta_a, ruta_b, carga)  # calentamiento
                t1 = time.perf_counter()
                self.motor.procesar(cfg["operacion"], cfg["params"], ruta_a, ruta_b, carga)
                seg = time.perf_counter() - t1
                if self.retardo_s_mb:
                    seg += nbytes / MB * self.retardo_s_mb
                calib = nbytes / seg if seg > 0 else None
        except Exception as e:
            log.error("%s: error al preparar la corrida %s: %s", self.wid, cid, e)
            return P.crear(P.ERROR, self.wid, cid, fase="preparacion", motivo=str(e))
        self.corrida = dict(cfg, ruta_a=ruta_a, ruta_b=ruta_b)
        self.corrida_id = cid
        self.tareas_hechas = 0
        return P.crear(P.LISTO, self.wid, cid, preparacion_s=round(time.perf_counter() - t0, 4),
                       preparacion_motor_s=prep_motor, calibracion_bps=calib,
                       motor=self.motor.describir())

    def _tarea(self, r: dict) -> dict:
        cfg = self.corrida
        tid = r["tarea_id"]
        if cfg is None or r.get("corrida_id") != self.corrida_id:
            return P.crear(P.ERROR, self.wid, r.get("corrida_id"), tarea_id=tid,
                           motivo="tarea de una corrida que este worker no preparo")
        carga = r["carga"]
        e0 = self.monitor.energia_actual() if self.monitor is not None else {}
        t_inicio = time.time()
        try:
            parcial, info = self.motor.procesar(cfg["operacion"], cfg["params"], cfg["ruta_a"],
                                                cfg.get("ruta_b"), carga)
            t_calculo = time.time() - t_inicio
            crc_a = crc_b = None
            t_crc = None
            if cfg.get("verificar_crc"):
                t1 = time.perf_counter()
                tam = cfg["tam_unidad"]
                crc_a = crc_unidades(self._mapa(cfg["ruta_a"]), r["u_ini"], r["u_fin"], tam)
                if cfg.get("ruta_b"):
                    crc_b = crc_unidades(self._mapa(cfg["ruta_b"]), r["u_ini"], r["u_fin"], tam)
                t_crc = time.perf_counter() - t1
            if self.retardo_s_mb:
                time.sleep((carga["fin"] - carga["inicio"]) / MB * self.retardo_s_mb)
        except Exception as e:
            log.exception("%s: error en la tarea %s", self.wid, tid)
            return P.crear(P.ERROR, self.wid, self.corrida_id, tarea_id=tid, motivo=str(e))
        if self.monitor is not None:
            e1 = self.monitor.energia_actual()
            info = dict(info, energia_j={k: (round(e1[k] - e0[k], 4) if e1.get(k) is not None
                                             and e0.get(k) is not None else None) for k in e1})
        self._quizas_congelado()
        self.tareas_hechas += 1
        return P.crear(P.RESULTADO, self.wid, self.corrida_id, tarea_id=tid, parcial=parcial,
                       crc_a=crc_a, crc_b=crc_b, bytes=carga["fin"] - carga["inicio"],
                       t_inicio=t_inicio, t_fin=time.time(), t_calculo=t_calculo, t_crc=t_crc,
                       dispositivo=self.dispositivo, info=info)

    # -- latidos --------------------------------------------------------------
    def _metricas(self) -> dict:
        if self.monitor is not None:
            return self.monitor.muestra()
        mem = psutil.virtual_memory()
        return {"cpu_pct": psutil.cpu_percent(None), "ram_pct": mem.percent}

    def _latidos(self) -> None:
        sock = None
        conectado_a = None
        while not self._fin.is_set():
            if self._congelado.is_set():
                time.sleep(0.2)
                continue
            destino = self.master_actual
            if destino != conectado_a:
                if sock is not None:
                    sock.close(0)
                sock = self._ctx.socket(zmq.DEALER)
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect("tcp://" + destino)
                conectado_a = destino
            try:
                sock.send(P.codificar(P.crear(P.LATIDO, self.wid, self.corrida_id,
                                              registrado=self._registrado, metricas=self._metricas())),
                          zmq.NOBLOCK)
            except zmq.ZMQError:
                pass
            limite = time.time() + self.latido_s
            while time.time() < limite and not self._fin.is_set():
                if sock.poll(int(max(0.0, limite - time.time()) * 1000) + 1):
                    try:
                        self._orden(P.decodificar(sock.recv()))
                    except ValueError:
                        pass
        if sock is not None:
            sock.close(0)

    def _orden(self, msg: dict) -> None:
        tipo = msg["tipo"]
        if tipo == P.SIMULAR_FALLO:
            modo = msg.get("modo")
            log.warning("%s: fallo simulado '%s'", self.wid, modo)
            if modo == "caida":
                os._exit(1)
            if modo == "congelado":
                self._congelado.set()
        elif tipo == P.DETENER:
            log.info("%s: orden de detenerse", self.wid)
            self._fin.set()
