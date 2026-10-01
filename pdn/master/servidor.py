"""Master del modo dinamico (CONTEXTO.md, seccion 7).

Un hilo propio atiende el socket ROUTER de ZeroMQ: registra workers, reparte
tareas con el planificador, recibe resultados, detecta caidas y vencimientos,
valida integridad, consolida y persiste. El resto del programa (CLI,
dashboard) habla con el Master por una API segura entre hilos: los comandos
entran por una cola y el estado sale en una instantanea protegida por un
cerrojo. Ningun otro hilo toca los sockets ni el estado interno.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import queue
import threading
import time
import uuid
from collections import deque

import zmq

from pdn.comun import config as cfgmod
from pdn.comun import protocolo as P
from pdn.comun.huellas import calcular_huellas, leer_huellas
from pdn.master import validacion
from pdn.master.estado import (ASIGNADA, CANCELADA, DESCARTADA, EJECUTANDO, FALLIDA, HECHA,
                               PREPARANDO, TERMINADA, Corrida, WorkerInfo)
from pdn.master.planificador import Planificador
from pdn.master.resultados import carpeta_corrida, guardar
from pdn.operaciones.trabajo import Trabajo
from pdn.preparacion.fasta_a_seq import Preparado, preparar, rutas_para
from pdn.preparacion.indice import Indice

log = logging.getLogger("pdn.master")

EXT_FASTA = (".fna", ".fa", ".fasta", ".fas")
MB = 1024 * 1024


class ErrorCorrida(ValueError):
    """La corrida pedida no se puede iniciar; el mensaje explica por que."""


def nombre_archivo(ruta_seq: str) -> str:
    """Nombre con que los workers buscan un archivo: base del .seq sin extension."""
    return os.path.splitext(os.path.basename(ruta_seq))[0]


def resolver_archivo(archivo: str, datos: str) -> Preparado:
    """Encuentra y, si hace falta, prepara un archivo para el Master."""
    candidatos = [archivo] if os.path.isabs(archivo) or os.path.exists(archivo) else []
    base = os.path.join(datos, archivo)
    candidatos += [base, base + ".seq"] + [base + e for e in EXT_FASTA]
    for c in candidatos:
        if not os.path.isfile(c):
            continue
        if c.endswith(".seq"):
            raiz = c[:-4]
            rutas = {"seq": c, "idx": raiz + ".idx", "huellas": raiz + ".huellas"}
            if not (os.path.exists(rutas["idx"]) and os.path.exists(rutas["huellas"])):
                raise ErrorCorrida("Falta el .idx o el .huellas de %s; prepare el .fna" % c)
            return Preparado("", c, rutas["idx"], rutas["huellas"], os.path.getsize(c), True)
        return preparar(c)
    raise ErrorCorrida("No se encontro el archivo %r en %s" % (archivo, datos))


class Master:
    """Servidor del modo dinamico."""

    def __init__(self, config: dict | None = None, puerto: int | None = None, datos: str | None = None,
                 carpeta_resultados: str | None = None, host: str = "*", rol: str = "principal",
                 carpeta_referencias: str | None = None) -> None:
        self.config = config or cfgmod.cargar()
        self.puerto = puerto or self.config["red"]["puerto_tareas"]
        self.host = host
        self.datos = datos or self.config["rutas"]["datos"]
        self.carpeta_resultados = carpeta_resultados or cfgmod.carpeta_resultados(self.config)
        self.carpeta_referencias = carpeta_referencias or os.path.join(
            os.path.dirname(self.carpeta_resultados.rstrip("/")) or ".", "referencias_resultados")
        self.rol = rol
        self.m = self.config["master"]
        self.workers: dict[str, WorkerInfo] = {}
        self.corrida: Corrida | None = None
        self.historial: list[dict] = []
        self.eventos: deque[dict] = deque(maxlen=500)
        self._comandos: queue.Queue = queue.Queue()
        self._cerrojo = threading.Lock()
        self._instantanea: dict = {}
        self._esperas: dict[str, list[concurrent.futures.Future]] = {}
        self._fin_notificar: set[str] = set()
        self._parar = threading.Event()
        self._hilo: threading.Thread | None = None
        self._ctx: zmq.Context | None = None
        self._sock = None
        self.t_arranque = time.time()
        self.activo_desde = time.time()
        self.ganchos_estado: list = []  # funciones llamadas con cada instantanea (HA, monitoreo)
        self.monitoreo: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------
    def iniciar(self) -> "Master":
        """Abre el puerto y arranca el hilo del Master."""
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.ROUTER_HANDOVER, 1)
        self._sock.bind("tcp://%s:%d" % (self.host, self.puerto))
        self._parar.clear()
        self._hilo = threading.Thread(target=self._bucle, name="master", daemon=True)
        self._hilo.start()
        self._evento("master", "Master %s escuchando en el puerto %d" % (self.rol, self.puerto))
        return self

    def detener(self, avisar_workers: bool = False) -> None:
        """Detiene el hilo y cierra el socket."""
        if avisar_workers:
            self._comandos.put(("detener_workers", None, None))
            time.sleep(0.3)
        self._parar.set()
        if self._hilo is not None:
            self._hilo.join(timeout=5)
        if self._sock is not None:
            self._sock.close(0)
        if self._ctx is not None:
            self._ctx.term()
        self._sock = None
        self._ctx = None

    def __enter__(self) -> "Master":
        return self.iniciar()

    def __exit__(self, *exc) -> None:
        self.detener()

    # ------------------------------------------------------------------
    # API segura entre hilos
    # ------------------------------------------------------------------
    def _llamar(self, nombre: str, arg=None, timeout: float = 30.0):
        futuro: concurrent.futures.Future = concurrent.futures.Future()
        self._comandos.put((nombre, arg, futuro))
        return futuro.result(timeout=timeout)

    def preparar_corrida(self, cfg: dict) -> dict:
        """Valida la configuracion y prepara archivos y trabajo (fuera del bucle)."""
        t0 = time.perf_counter()
        cfg = dict(cfg)
        op = cfg.get("operacion", "conteo")
        if cfg.get("modo", "dinamico") != "dinamico":
            raise ErrorCorrida("El Master solo ejecuta el modo dinamico; el modo MPI se lanza con mpirun")
        if not cfg.get("archivo"):
            raise ErrorCorrida("Falta el archivo de la corrida")
        prep_a = resolver_archivo(cfg["archivo"], self.datos)
        prep_b = None
        if op == "comparacion":
            if not cfg.get("archivo_b"):
                raise ErrorCorrida("La comparacion necesita dos archivos (archivo_b)")
            prep_b = resolver_archivo(cfg["archivo_b"], self.datos)
        p = self.config["planificador"]
        tam_unidad = int(cfg.get("tam_unidad") or leer_huellas(prep_a.ruta_huellas)["tam_unidad"])
        try:
            trabajo = Trabajo(op, cfg.get("params") or {}, Indice.leer(prep_a.ruta_idx),
                              Indice.leer(prep_b.ruta_idx) if prep_b else None, tam_unidad)
            estrategia = cfg.get("estrategia", p["estrategia"])
            plan = Planificador(trabajo.unidades, tam_unidad, trabajo.largo, estrategia,
                                float(cfg.get("tiempo_objetivo_s", p["tiempo_objetivo_s"])),
                                int(float(cfg.get("max_tarea_mb", p["max_tarea_mb"])) * MB),
                                int(float(cfg.get("max_tarea_gpu_mb", p["max_tarea_gpu_mb"])) * MB),
                                int(float(cfg.get("tam_fijo_mb", p["tam_fijo_mb"])) * MB),
                                float(p["alfa_velocidad"]), op)
        except ValueError as e:
            raise ErrorCorrida(str(e)) from e
        verificar = bool(cfg.get("verificar_crc", True)) and trabajo.verifica_crc

        def tabla(prep: Preparado | None) -> dict | None:
            if prep is None or not verificar:
                return None
            t = leer_huellas(prep.ruta_huellas)
            return t if t["tam_unidad"] == tam_unidad else calcular_huellas(prep.ruta_seq, tam_unidad)

        cid = time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:6]
        cfg.update({"operacion": op, "params": trabajo.params, "tam_unidad": tam_unidad,
                    "estrategia": plan.estrategia, "verificar_crc": verificar,
                    "nombre_a": nombre_archivo(prep_a.ruta_seq),
                    "nombre_b": nombre_archivo(prep_b.ruta_seq) if prep_b else None,
                    "huella_a": leer_huellas(prep_a.ruta_huellas).get("global"),
                    "huella_b": leer_huellas(prep_b.ruta_huellas).get("global") if prep_b else None,
                    "largo": trabajo.largo, "unidades": trabajo.unidades,
                    "tamano_a": trabajo.indice_a.total,
                    "tamano_b": trabajo.indice_b.total if trabajo.indice_b else None,
                    "origen_datos": cfg.get("origen_datos", "local")})
        corrida = Corrida(cid, cfg, trabajo, plan, tabla(prep_a), tabla(prep_b))
        corrida.parcial = trabajo.vacio()
        corrida.preparacion_master_s = time.perf_counter() - t0
        return {"corrida": corrida}

    def iniciar_corrida(self, cfg: dict) -> str:
        """Prepara e inicia una corrida. Devuelve su corrida_id."""
        preparado = self.preparar_corrida(cfg)
        return self._llamar("nueva_corrida", preparado["corrida"])

    def esperar_corrida(self, corrida_id: str, timeout: float | None = None) -> dict:
        """Bloquea hasta que la corrida termine y devuelve su resumen."""
        futuro = self._llamar("esperar", corrida_id)
        return futuro.result(timeout=timeout)

    def correr(self, cfg: dict, timeout: float | None = None) -> dict:
        """Inicia una corrida y espera su resumen."""
        return self.esperar_corrida(self.iniciar_corrida(cfg), timeout)

    def cancelar_corrida(self) -> None:
        self._llamar("cancelar")

    def simular_fallo(self, wid: str, modo: str) -> None:
        """Ordena a un worker caerse ('caida') o congelarse ('congelado')."""
        if modo not in ("caida", "congelado"):
            raise ValueError("modo de fallo desconocido: %s" % modo)
        self._llamar("fallo", (wid, modo))

    def estado(self) -> dict:
        """Instantanea del estado (copia, segura entre hilos)."""
        with self._cerrojo:
            return dict(self._instantanea)

    def workers_listos(self) -> list[str]:
        return [w["wid"] for w in self.estado().get("workers", []) if w["estado"] == "conectado"]

    def esperar_workers(self, n: int, timeout: float = 30.0) -> list[str]:
        """Espera a que haya n workers conectados."""
        limite = time.time() + timeout
        while time.time() < limite:
            listos = self.workers_listos()
            if len(listos) >= n:
                return listos
            time.sleep(0.1)
        raise TimeoutError("Solo se conectaron %d de %d workers" % (len(self.workers_listos()), n))

    # ------------------------------------------------------------------
    # Bucle
    # ------------------------------------------------------------------
    def _bucle(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        ultimo_chequeo = 0.0
        ultima_inst = 0.0
        while not self._parar.is_set():
            try:
                eventos = dict(poller.poll(50))
            except zmq.ZMQError:
                break
            if self._sock in eventos:
                for _ in range(200):
                    try:
                        ident, datos = self._sock.recv_multipart(zmq.NOBLOCK)[:2]
                    except zmq.Again:
                        break
                    except ValueError:
                        continue
                    self._atender(ident, datos)
            self._procesar_comandos()
            ahora = time.time()
            if ahora - ultimo_chequeo >= 0.2:
                self._chequear(ahora)
                ultimo_chequeo = ahora
            if ahora - ultima_inst >= 0.25:
                self._publicar()
                ultima_inst = ahora
        self._publicar()

    def _enviar(self, ident: bytes, msg: dict) -> None:
        try:
            self._sock.send_multipart([ident, P.codificar(msg)], zmq.NOBLOCK)
        except zmq.ZMQError as e:
            log.warning("No se pudo enviar %s: %s", msg.get("tipo"), e)

    def _evento(self, tipo: str, texto: str, **extra) -> None:
        e = {"t": time.time(), "tipo": tipo, "texto": texto, **extra}
        self.eventos.append(e)
        if tipo in ("reasignacion", "caida", "rechazo", "error"):
            log.warning(texto)
        else:
            log.info(texto)

    # -- comandos de la API ---------------------------------------------
    def _procesar_comandos(self) -> None:
        while True:
            try:
                nombre, arg, futuro = self._comandos.get_nowait()
            except queue.Empty:
                return
            try:
                r = getattr(self, "_cmd_" + nombre)(arg)
                if futuro is not None:
                    futuro.set_result(r)
            except Exception as e:  # el error vuelve al que llamo
                if futuro is not None:
                    futuro.set_exception(e)
                else:
                    log.exception("Error en comando %s", nombre)

    def _cmd_nueva_corrida(self, corrida: Corrida) -> str:
        if self.corrida is not None and self.corrida.estado in (PREPARANDO, EJECUTANDO):
            raise ErrorCorrida("Ya hay una corrida en curso (%s)" % self.corrida.corrida_id)
        cfg = corrida.config
        nodos = cfg.get("nodos", "todos")
        dispositivos = cfg.get("dispositivos") or {}
        for w in self.workers.values():
            w.corrida_lista = None
            w.corrida_preparando = None
            w.rechazos = 0
            w.errores = 0
            w.excluido = None
            w.bytes_corrida = 0
            w.tareas_corrida = 0
            w.calibracion_corrida = None
            if w.estado == "sospechoso":
                w.estado = "conectado"
        for w in self.workers.values():
            motivo = self._motivo_no_participa(w, cfg, nodos, dispositivos)
            if motivo:
                corrida.excluidos[w.wid] = motivo
                if w.estado == "conectado" and motivo.startswith("copia"):
                    w.estado = "sospechoso"
            else:
                corrida.participantes.append(w.wid)
        if not corrida.participantes:
            raise ErrorCorrida("Ningun worker conectado puede participar: %s" % (corrida.excluidos or "no hay workers"))
        self.corrida = corrida
        self._fin_notificar.clear()
        self._evento("corrida", "Corrida %s (%s) en preparacion con %d workers" % (
            corrida.corrida_id, cfg["operacion"], len(corrida.participantes)))
        return corrida.corrida_id

    def _motivo_no_participa(self, w: WorkerInfo, cfg: dict, nodos, dispositivos: dict) -> str | None:
        if w.estado == "perdido":
            return "worker perdido"
        if nodos not in (None, "todos") and w.hostname not in nodos and w.wid not in nodos:
            return "nodo no seleccionado"
        permitidos = dispositivos.get(w.hostname)
        if permitidos is not None and w.dispositivo not in permitidos:
            return "dispositivo no seleccionado"
        if cfg.get("origen_datos", "local") == "local":
            for clave in ("a", "b"):
                nombre = cfg.get("nombre_" + clave)
                if not nombre:
                    continue
                suyo = w.archivos.get(nombre)
                if suyo is None:
                    return "no tiene el archivo %s" % nombre
                if suyo.get("tamano") != cfg.get("tamano_" + clave):
                    return "copia de datos distinta (tamano %s, se esperaba %s)" % (
                        suyo.get("tamano"), cfg.get("tamano_" + clave))
                huella = cfg.get("huella_" + clave)
                if huella and suyo.get("global") and suyo["global"] != huella:
                    return "copia de datos distinta (huella global)"
        return None

    def _cmd_esperar(self, corrida_id: str) -> concurrent.futures.Future:
        futuro: concurrent.futures.Future = concurrent.futures.Future()
        terminada = next((h for h in self.historial if h["corrida_id"] == corrida_id), None)
        if terminada is not None:
            futuro.set_result(terminada)
        else:
            self._esperas.setdefault(corrida_id, []).append(futuro)
        return futuro

    def _cmd_cancelar(self, _arg) -> None:
        c = self.corrida
        if c is not None and c.estado in (PREPARANDO, EJECUTANDO):
            c.error = "cancelada por el usuario"
            self._cerrar_corrida(CANCELADA)

    def _cmd_fallo(self, arg) -> None:
        wid, modo = arg
        w = self.workers.get(wid)
        if w is None:
            raise ValueError("Worker desconocido: %s" % wid)
        if not w.identidad_latido:
            raise ValueError("El worker %s no tiene canal de latidos" % wid)
        self._enviar(w.identidad_latido, P.crear(P.SIMULAR_FALLO, wid, modo=modo))
        self._evento("fallo", "Fallo simulado en %s: %s" % (wid, modo), worker=wid, modo=modo)

    def _cmd_detener_workers(self, _arg) -> None:
        for w in self.workers.values():
            if w.identidad_latido:
                self._enviar(w.identidad_latido, P.crear(P.DETENER, w.wid))

    def _cmd_ejecutar(self, funcion):
        # Ejecuta una funcion dentro del hilo del Master (para pruebas y HA)
        return funcion(self)

    # -- mensajes ----------------------------------------------------------
    def _atender(self, ident: bytes, datos: bytes) -> None:
        try:
            msg = P.decodificar(datos)
        except ValueError as e:
            log.warning("Mensaje ilegible descartado: %s", e)
            return
        tipo = msg["tipo"]
        wid = msg.get("worker_id", "")
        ahora = time.time()
        if tipo == P.LATIDO:
            w = self.workers.get(wid)
            if w is not None:
                w.identidad_latido = ident
                w.t_ultimo_latido = ahora
                if msg.get("metricas"):
                    w.metricas = msg["metricas"]
                    self._guardar_metricas(w, msg["metricas"])
                if w.estado == "perdido" and msg.get("registrado"):
                    # Volvio tras perderse: debe registrarse otra vez por el canal principal
                    pass
            return
        if tipo == P.REGISTRO:
            self._registro(ident, msg, ahora)
            return
        w = self.workers.get(wid)
        if w is None:
            # Worker desconocido (p. ej. tras un cambio de Master): que se registre
            self._enviar(ident, P.crear(P.ACEPTADO, wid, reregistrar=True))
            return
        w.identidad = ident
        w.t_ultimo_mensaje = ahora
        if w.estado == "perdido":
            self._evento("worker", "%s volvio a responder" % wid, worker=wid)
            w.estado = "conectado"
            w.t_ultimo_latido = ahora
        if tipo == P.CALIBRACION:
            w.calibracion_sintetica = msg.get("calibracion") or {}
            self._enviar(ident, P.crear(P.ESPERAR, wid))
        elif tipo == P.PEDIR:
            self._responder_pedido(w)
        elif tipo == P.LISTO:
            self._listo(w, msg)
            self._responder_pedido(w)
        elif tipo == P.RESULTADO:
            self._resultado(w, msg, ahora)
            self._responder_pedido(w)
        elif tipo == P.ERROR:
            self._error(w, msg)
            self._enviar(ident, P.crear(P.ESPERAR, wid, pausa=0.5))
        else:
            log.warning("Tipo de mensaje desconocido %s de %s", tipo, wid)
            self._enviar(ident, P.crear(P.ESPERAR, wid))

    def _registro(self, ident: bytes, msg: dict, ahora: float) -> None:
        wid = msg["worker_id"]
        previo = self.workers.get(wid)
        if previo is not None:
            # Re-registro: lo que tenia asignado se perdio con su proceso anterior
            self._liberar_tareas(wid, "worker re-registrado")
            previo.reconexiones += 1
            w = previo
            w.estado = "conectado"
            w.corrida_lista = None
            w.corrida_preparando = None
        else:
            w = WorkerInfo(wid, msg.get("hostname", wid.split(":")[0]), msg.get("dispositivo", "cpu"))
            self.workers[wid] = w
        w.identidad = ident
        w.hardware = msg.get("hardware") or {}
        w.archivos = msg.get("archivos") or {}
        w.version = msg.get("version", "")
        w.t_ultimo_latido = ahora
        w.t_ultimo_mensaje = ahora
        self._evento("worker", "Registrado %s (%s) con %d archivos" % (wid, w.dispositivo, len(w.archivos)),
                     worker=wid)
        self._advertir_versiones()
        c = self.corrida
        if c is not None and c.estado in (PREPARANDO, EJECUTANDO) and wid not in c.participantes \
                and wid not in c.excluidos:
            motivo = self._motivo_no_participa(w, c.config, c.config.get("nodos", "todos"),
                                               c.config.get("dispositivos") or {})
            if motivo is None:
                c.participantes.append(wid)
                self._evento("worker", "%s se une a la corrida en curso" % wid, worker=wid)
            else:
                c.excluidos[wid] = motivo
        self._enviar(ident, P.crear(P.ACEPTADO, wid, rol=self.rol, latido_s=self.config["worker"]["latido_s"],
                                    master_timeout_s=self.config["worker"]["master_timeout_s"]))

    def _advertir_versiones(self) -> None:
        linux = [w for w in self.workers.values() if (w.hardware.get("sistema") == "Linux")]
        pys = {w.hardware.get("python", "")[:4] for w in linux if w.hardware.get("python")}
        mpis = {w.hardware.get("openmpi") for w in linux if w.hardware.get("openmpi")}
        if len(pys) > 1:
            self._evento("aviso", "Versiones de Python distintas entre nodos Linux: %s" % sorted(pys))
        if len(mpis) > 1:
            self._evento("aviso", "Versiones de OpenMPI distintas entre nodos Linux: %s" % sorted(mpis))

    def _guardar_metricas(self, w: WorkerInfo, metricas: dict) -> None:
        c = self.corrida
        if c is None or c.estado not in (PREPARANDO, EJECUTANDO) or w.wid not in c.participantes:
            return
        fila = {"t": round(time.time() - (c.t_inicio or c.t_creacion), 3), "worker": w.wid, **{
            k: v for k, v in metricas.items() if not isinstance(v, (dict, list))}}
        self.monitoreo.setdefault(c.corrida_id, []).append(fila)

    def _config_worker(self, c: Corrida, w: WorkerInfo) -> dict:
        cfg = c.config
        motor = dict((cfg.get("motor") or {}).get(w.dispositivo) or {})
        motor.update(((cfg.get("motor_nodos") or {}).get(w.hostname) or {}).get(w.dispositivo) or {})
        calib = int(float(cfg.get("calibracion_mb", self.config["worker"]["calibracion_mb"])) * MB)
        u_cal = max(1, min(c.trabajo.unidades, -(-calib // c.trabajo.tam_unidad)))
        fin_cal = min(u_cal * c.trabajo.tam_unidad, c.trabajo.largo)
        return {"operacion": cfg["operacion"], "params": cfg["params"], "nombre_a": cfg["nombre_a"],
                "nombre_b": cfg.get("nombre_b"), "origen_datos": cfg.get("origen_datos", "local"),
                "tam_unidad": c.trabajo.tam_unidad, "verificar_crc": cfg["verificar_crc"],
                "motor": motor, "calibracion": c.trabajo.carga(0, fin_cal) if c.trabajo.largo else None,
                "largo_a": c.trabajo.indice_a.total,
                "largo_b": c.trabajo.indice_b.total if c.trabajo.indice_b else None}

    def _responder_pedido(self, w: WorkerInfo) -> None:
        c = self.corrida
        wid = w.wid
        if c is None or c.estado not in (PREPARANDO, EJECUTANDO) or wid not in c.participantes:
            if wid in self._fin_notificar:
                self._fin_notificar.discard(wid)
                self._enviar(w.identidad, P.crear(P.FIN, wid, c.corrida_id if c else None))
            else:
                self._enviar(w.identidad, P.crear(P.ESPERAR, wid, pausa=0.2))
            return
        if w.excluido or w.estado == "perdido":
            self._enviar(w.identidad, P.crear(P.ESPERAR, wid, c.corrida_id, pausa=0.5,
                                              motivo=w.excluido))
            return
        if w.corrida_lista != c.corrida_id:
            if w.corrida_preparando != c.corrida_id:
                w.corrida_preparando = c.corrida_id
                self._enviar(w.identidad, P.crear(P.PREPARAR, wid, c.corrida_id,
                                                  corrida=self._config_worker(c, w)))
            else:
                self._enviar(w.identidad, P.crear(P.ESPERAR, wid, c.corrida_id, pausa=0.2))
            return
        if c.estado != EJECUTANDO:
            self._enviar(w.identidad, P.crear(P.ESPERAR, wid, c.corrida_id, pausa=0.05))
            return
        tarea = self._asignar(c, w)
        if tarea is None:
            self._enviar(w.identidad, P.crear(P.ESPERAR, wid, c.corrida_id, pausa=0.05))
            return
        self._enviar(w.identidad, P.crear(P.TAREA, wid, c.corrida_id, tarea_id=tarea.tid,
                                          u_ini=tarea.u_ini, u_fin=tarea.u_fin,
                                          carga=c.trabajo.carga(tarea.inicio, tarea.fin)))

    def _activos(self, c: Corrida) -> list[str]:
        return [wid for wid in c.participantes
                if (w := self.workers.get(wid)) is not None and w.estado != "perdido"
                and not w.excluido and w.corrida_lista == c.corrida_id]

    def _asignar(self, c: Corrida, w: WorkerInfo):
        rango = c.planificador.pedir(w.wid, self._activos(c))
        if rango is None:
            return None
        origen = None
        # Si el rango viene de una tarea devuelta, se anota de donde viene
        for t in c.tareas.values():
            if t.estado == DESCARTADA and t.u_ini <= rango[0] < t.u_fin:
                origen = t.tid
        tarea = c.nueva_tarea(rango[0], rango[1], origen)
        tarea.estado = ASIGNADA
        tarea.worker = w.wid
        tarea.dispositivo = w.dispositivo
        tarea.t_asignacion = time.time()
        vel = c.planificador.velocidad.get(w.wid)
        tarea.t_esperado = tarea.nbytes / vel if vel else None
        return tarea

    def _listo(self, w: WorkerInfo, msg: dict) -> None:
        c = self.corrida
        if c is None or msg.get("corrida_id") != c.corrida_id:
            return
        w.corrida_lista = c.corrida_id
        w.corrida_preparando = None
        w.preparacion_s = msg.get("preparacion_s")
        w.calibracion_corrida = msg.get("calibracion_bps")
        w.motor = msg.get("motor") or {}
        c.planificador.registrar_worker(w.wid, w.dispositivo, w.calibracion_corrida)
        self._evento("worker", "%s listo: %.1f MB/s calibrados, preparacion %.2f s" % (
            w.wid, (w.calibracion_corrida or 0) / MB, w.preparacion_s or 0), worker=w.wid)
        if c.estado == PREPARANDO:
            self._quizas_arrancar(c)

    def _quizas_arrancar(self, c: Corrida, forzar: bool = False) -> None:
        pendientes = [wid for wid in c.participantes
                      if self.workers[wid].corrida_lista != c.corrida_id
                      and self.workers[wid].estado != "perdido" and not self.workers[wid].excluido]
        listos = self._activos(c)
        minimo = int(c.config.get("min_workers", 1))
        if (not pendientes or forzar) and len(listos) >= minimo:
            c.estado = EJECUTANDO
            c.t_inicio = time.time()
            if c.planificador.estrategia == "proporcional":
                c.planificador.iniciar_proporcional(listos)
            self._evento("corrida", "Corrida %s en ejecucion con %d workers" % (c.corrida_id, len(listos)))
            if c.trabajo.largo == 0:
                self._finalizar(c)

    def _resultado(self, w: WorkerInfo, msg: dict, ahora: float) -> None:
        c = self.corrida
        tid = msg.get("tarea_id")
        if c is None or msg.get("corrida_id") != c.corrida_id or c.estado != EJECUTANDO:
            self._descartar(c, w, tid, "resultado de una corrida que ya no esta en curso")
            return
        t = c.tareas.get(tid)
        if t is None:
            self._descartar(c, w, tid, "tarea desconocida")
            return
        if t.estado != ASIGNADA or t.worker != w.wid:
            self._descartar(c, w, tid, "la tarea ya estaba %s (resultado tardio o duplicado)" % t.estado)
            return
        # 1) Integridad por unidad
        motivo = validacion.verificar_crc(c.huellas_a, t.u_ini, t.u_fin, msg.get("crc_a")) \
            or validacion.verificar_crc(c.huellas_b, t.u_ini, t.u_fin, msg.get("crc_b"))
        parcial = msg.get("parcial")
        if motivo is None and parcial is None:
            motivo = "resultado sin parcial"
        if motivo is None and c.config["operacion"] == "comparacion":
            from pdn.operaciones.comparacion import es_valido  # noqa: PLC0415
            if not es_valido(parcial):
                motivo = "categorias incoherentes (mayores que las posiciones comparadas)"
        if motivo is not None:
            w.rechazos += 1
            self._evento("rechazo", "Resultado de la tarea %d de %s rechazado: %s" % (tid, w.wid, motivo),
                         worker=w.wid, tarea=tid)
            self._devolver(c, t, "rechazada: " + motivo)
            if w.rechazos >= int(self.m["rechazos_maximos"]):
                w.excluido = "copia de datos sospechosa (%d rechazos)" % w.rechazos
                w.estado = "sospechoso"
                c.planificador.liberar_reserva(w.wid)
                self._evento("rechazo", "%s deja de recibir tareas: %s" % (w.wid, w.excluido), worker=w.wid)
            return
        # 2) Aceptar
        t.estado = HECHA
        t.t_resultado = ahora
        t.t_calculo = msg.get("t_calculo")
        t.t_crc = msg.get("t_crc")
        t.info = msg.get("info") or {}
        duracion = ahora - (t.t_asignacion or ahora)
        t.mb_s = (t.nbytes / MB) / duracion if duracion > 0 else None
        c.parcial = c.trabajo.combinar(c.parcial, parcial)
        c.planificador.actualizar_velocidad(w.wid, t.nbytes, duracion)
        w.bytes_corrida += t.nbytes
        w.tareas_corrida += 1
        if c.planificador.terminado() and not c.en_vuelo():
            self._finalizar(c)

    def _descartar(self, c: Corrida | None, w: WorkerInfo, tid, motivo: str) -> None:
        if c is not None:
            c.descartados.append({"t": time.time(), "worker": w.wid, "tarea": tid, "motivo": motivo})
        self._evento("descarte", "Resultado de %s (tarea %s) descartado: %s" % (w.wid, tid, motivo),
                     worker=w.wid)

    def _error(self, w: WorkerInfo, msg: dict) -> None:
        c = self.corrida
        w.errores += 1
        texto = msg.get("motivo", "error desconocido")
        self._evento("error", "Error en %s: %s" % (w.wid, texto), worker=w.wid)
        if c is None:
            return
        tid = msg.get("tarea_id")
        t = c.tareas.get(tid) if tid is not None else None
        if t is not None and t.estado == ASIGNADA and t.worker == w.wid:
            self._devolver(c, t, "error en el worker: " + texto)
        if msg.get("fase") == "preparacion" or w.errores >= 3:
            w.excluido = "error: " + texto
            w.corrida_preparando = None
            c.planificador.liberar_reserva(w.wid)
            if c.estado == PREPARANDO:
                self._quizas_arrancar(c)

    def _devolver(self, c: Corrida, t, motivo: str) -> None:
        """Devuelve una tarea a la fila y registra la reasignacion."""
        t.estado = DESCARTADA
        t.motivo = motivo
        c.planificador.devolver(t.u_ini, t.u_fin)
        c.reasignaciones.append({"t": round(time.time() - (c.t_inicio or c.t_creacion), 3),
                                 "tarea": t.tid, "worker": t.worker, "motivo": motivo,
                                 "u_ini": t.u_ini, "u_fin": t.u_fin})
        self._evento("reasignacion", "Tarea %d de %s vuelve a la fila: %s" % (t.tid, t.worker, motivo),
                     worker=t.worker, tarea=t.tid)

    def _liberar_tareas(self, wid: str, motivo: str) -> None:
        c = self.corrida
        if c is None or c.estado != EJECUTANDO:
            return
        for t in list(c.en_vuelo()):
            if t.worker == wid:
                self._devolver(c, t, motivo)
        c.planificador.liberar_reserva(wid)

    # -- chequeos periodicos -------------------------------------------
    def _chequear(self, ahora: float) -> None:
        limite_latido = float(self.m["latido_timeout_s"])
        for w in self.workers.values():
            if w.estado != "perdido" and ahora - w.t_ultimo_latido > limite_latido:
                w.estado = "perdido"
                w.corrida_preparando = None
                self._evento("caida", "%s perdido: %.1f s sin latido" % (w.wid, ahora - w.t_ultimo_latido),
                             worker=w.wid)
                self._liberar_tareas(w.wid, "worker perdido (sin latido)")
                c = self.corrida
                if c is not None and c.estado == PREPARANDO:
                    self._quizas_arrancar(c)
        c = self.corrida
        if c is None:
            return
        if c.estado == PREPARANDO and ahora - c.t_creacion > float(self.m["preparacion_timeout_s"]):
            if self._activos(c):
                self._evento("aviso", "Tiempo de preparacion agotado: se arranca con los workers listos")
                self._quizas_arrancar(c, forzar=True)
            else:
                c.error = "ningun worker termino la preparacion"
                self._cerrar_corrida(FALLIDA)
        if c.estado == EJECUTANDO:
            for t in list(c.en_vuelo()):
                esperado = t.t_esperado or 0.0
                limite = max(float(self.m["tarea_vencida_factor"]) * esperado,
                             float(self.m["tarea_vencida_min_s"]))
                if ahora - (t.t_asignacion or ahora) > limite:
                    self._devolver(c, t, "vencida (%.1f s sin resultado)" % (ahora - t.t_asignacion))
            if not self._activos(c) and not c.en_vuelo() and not c.planificador.terminado():
                pend = [wid for wid in c.participantes if self.workers[wid].estado != "perdido"
                        and not self.workers[wid].excluido]
                if not pend:
                    c.error = "no quedan workers activos"
                    self._cerrar_corrida(FALLIDA)

    # -- cierre ------------------------------------------------------------
    def _finalizar(self, c: Corrida) -> None:
        c.t_fin = time.time()
        hechas = [t for t in c.tareas.values() if t.estado == HECHA]
        cob = validacion.cobertura([(t.u_ini, t.u_fin) for t in hechas], c.trabajo.unidades)
        bytes_ok = sum(t.nbytes for t in hechas) == c.trabajo.largo
        clave = c.trabajo.clave(c.parcial)
        ref_ruta = validacion.ruta_referencia(
            self.carpeta_referencias, "%s%s" % (c.config.get("huella_a"), c.config.get("huella_b") or ""),
            c.config["operacion"], c.trabajo.params)
        ref = validacion.leer_referencia(ref_ruta)
        discrepancias = validacion.comparar(clave, ref["clave"]) if ref else None
        c.validacion = {"cobertura": cob, "bytes_ok": bytes_ok,
                        "crc_verificado": c.config["verificar_crc"],
                        "referencia": ref_ruta if ref else None,
                        "coincide_referencia": (not discrepancias) if ref else None,
                        "discrepancias": discrepancias or [],
                        "tiempo_referencia_s": ref.get("tiempo_s") if ref else None}
        valido = cob["ok"] and bytes_ok and (discrepancias in (None, []))
        if c.config["operacion"] == "comparacion":
            from pdn.operaciones.comparacion import es_valido  # noqa: PLC0415
            valido = valido and es_valido(c.parcial)
        c.validacion["valido"] = bool(valido)
        try:
            c.resultado = c.trabajo.finalizar(c.parcial)
        except Exception as e:  # el resultado se guarda igual, con el error
            log.exception("Error al finalizar")
            c.resultado = {"error": str(e)}
        self._cerrar_corrida(TERMINADA)

    def resumen_corrida(self, c: Corrida) -> dict:
        """Resumen con tiempos, reparto por worker y validacion."""
        t0 = c.t_inicio
        duracion = (c.t_fin - t0) if (c.t_fin and t0) else None
        por_worker = []
        for wid in c.participantes:
            w = self.workers.get(wid)
            hechas = [t for t in c.tareas.values() if t.worker == wid and t.estado == HECHA]
            ultimo = max((t.t_resultado for t in hechas), default=None)
            por_worker.append({
                "worker": wid, "dispositivo": w.dispositivo if w else None,
                "bytes": sum(t.nbytes for t in hechas), "tareas": len(hechas),
                "mb_s_calibrado": round(w.calibracion_corrida / MB, 2) if w and w.calibracion_corrida else None,
                "mb_s_final": round(c.planificador.velocidad.get(wid, 0) / MB, 2) if wid in c.planificador.velocidad else None,
                "ocioso_final_s": round(c.t_fin - ultimo, 3) if (ultimo and c.t_fin) else None,
                "preparacion_s": w.preparacion_s if w else None,
                "excluido": w.excluido if w else None, "motor": w.motor if w else None})
        prep = [p["preparacion_s"] for p in por_worker if p["preparacion_s"] is not None]
        val = c.validacion or {}
        t_ref = val.get("tiempo_referencia_s")
        return {
            "corrida_id": c.corrida_id, "estado": c.estado, "operacion": c.config["operacion"],
            "error": c.error, "rol_master": self.rol,
            "config": {k: v for k, v in c.config.items() if k not in ("params",)},
            "params": c.trabajo.params, "largo_bytes": c.trabajo.largo, "unidades": c.trabajo.unidades,
            "tam_unidad": c.trabajo.tam_unidad, "tiempo_s": duracion,
            "mb_s": (c.trabajo.largo / MB / duracion) if duracion else None,
            "preparacion_s": round(max(prep, default=0.0) + c.preparacion_master_s, 4),
            "speedup": (t_ref / duracion) if (t_ref and duracion) else None,
            "tareas_total": len(c.tareas), "reasignaciones": c.reasignaciones,
            "descartados": c.descartados, "excluidos": c.excluidos,
            "por_worker": por_worker, "validacion": c.validacion, "carpeta": c.carpeta,
            "t_inicio": c.t_inicio, "t_fin": c.t_fin}

    def _cerrar_corrida(self, estado: str) -> None:
        c = self.corrida
        if c is None:
            return
        c.estado = estado
        c.t_fin = c.t_fin or time.time()
        for wid in c.participantes:
            self._fin_notificar.add(wid)
        resumen = self.resumen_corrida(c)
        if c.config.get("guardar", True):
            try:
                c.carpeta = carpeta_corrida(self.carpeta_resultados, c.config["operacion"], c.corrida_id)
                resumen["carpeta"] = c.carpeta
                guardar(c.carpeta, resumen, c.resultado or {}, [t.fila(c.t_inicio) for t in
                        sorted(c.tareas.values(), key=lambda t: t.tid)], c.config,
                        recursos=self.monitoreo.get(c.corrida_id),
                        energia=self._energia(c))
            except OSError as e:
                log.error("No se pudieron guardar los resultados: %s", e)
        resumen["resultado"] = c.resultado
        self.historial.append(resumen)
        self.historial = self.historial[-50:]
        nivel = "corrida" if estado == TERMINADA else "error"
        self._evento(nivel, "Corrida %s %s en %s s (valida: %s)" % (
            c.corrida_id, estado.lower(), None if resumen["tiempo_s"] is None else round(resumen["tiempo_s"], 3),
            (c.validacion or {}).get("valido")))
        for futuro in self._esperas.pop(c.corrida_id, []):
            futuro.set_result(resumen)

    def _energia(self, c: Corrida) -> list[dict] | None:
        # Se completa en la etapa de monitoreo; aqui no se inventan valores
        filas = [f for f in self.monitoreo.get(c.corrida_id, []) if f.get("energia_j") is not None]
        return filas or None

    # -- instantanea -------------------------------------------------------
    def _publicar(self) -> None:
        ahora = time.time()
        c = self.corrida
        corrida = None
        if c is not None:
            hechos = c.bytes_hechos()
            dur = ((c.t_fin or ahora) - c.t_inicio) if c.t_inicio else None
            corrida = {"corrida_id": c.corrida_id, "estado": c.estado, "operacion": c.config["operacion"],
                       "config": {k: v for k, v in c.config.items()},
                       "largo": c.trabajo.largo, "bytes_hechos": hechos,
                       "progreso": (hechos / c.trabajo.largo) if c.trabajo.largo else 1.0,
                       "tiempo_s": dur, "mb_s": (hechos / MB / dur) if dur else None,
                       "participantes": c.participantes, "excluidos": c.excluidos,
                       "tareas": [t.fila(c.t_inicio) for t in c.tareas.values()][-3000:],
                       "reasignaciones": c.reasignaciones[-200:], "descartados": c.descartados[-100:],
                       "velocidades": {k: v / MB for k, v in c.planificador.velocidad.items()},
                       "pendiente_bytes": c.planificador.pendiente_bytes(), "error": c.error}
        inst = {"t": ahora, "rol": self.rol, "activo_desde": self.activo_desde, "puerto": self.puerto,
                "workers": [w.publico(ahora) for w in self.workers.values()],
                "corrida": corrida, "historial": [{k: v for k, v in h.items() if k != "resultado"}
                                                  for h in self.historial[-20:]],
                "ultimo_resultado": self.historial[-1] if self.historial else None,
                "eventos": list(self.eventos)[-200:]}
        with self._cerrojo:
            self._instantanea = inst
        for gancho in self.ganchos_estado:
            try:
                gancho(self)
            except Exception:
                log.exception("Error en gancho de estado")

    def ruta_seq(self, nombre: str) -> str:
        """Ruta local del .seq de un archivo por nombre (para la referencia)."""
        return rutas_para(os.path.join(self.datos, nombre + ".fna"))["seq"]
