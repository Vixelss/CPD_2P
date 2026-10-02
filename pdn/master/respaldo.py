"""Alta disponibilidad: Master de respaldo (CONTEXTO.md, seccion 7.7).

Principal: un Replicador envia cada segundo una instantanea del estado al
respaldo (configuracion de la corrida, registro de tareas, parcial
consolidado, planificador, participantes) y sabe si el respaldo esta vivo por
sus acuses de recibo.

Respaldo: escucha en el puerto de replicacion (5556). Cuando ve una corrida
nueva, la prepara en segundo plano (archivos, indices, huellas) para que la
promocion sea inmediata. Si pasan 3 s sin instantaneas, se promueve: abre el
puerto de tareas, marca como PENDIENTE todo lo ASIGNADA y continua. Los
workers, al no tener respuesta del principal en 3 s, pasan al siguiente
Master de su lista y se registran de nuevo.

Funciona porque cada nodo tiene su copia local de los datos. En modo NFS la
caida del Master (que sirve /cluster) es fatal; el modo MPI no tiene HA.
"""

from __future__ import annotations

import copy
import logging
import os
import queue
import threading
import time
from dataclasses import asdict

import zmq

from pdn.comun import protocolo as P
from pdn.master.estado import ASIGNADA, EJECUTANDO, PREPARANDO, Corrida, Tarea

log = logging.getLogger("pdn.respaldo")
INSTANTANEA = "INSTANTANEA"
ACUSE = "ACUSE"


def accesible(ruta: str, timeout: float = 2.0) -> bool:
    """True si la carpeta responde a tiempo y se puede escribir.

    Con el Master caido de verdad (desconectado de la red), /cluster es un NFS
    sin servidor: con montaje hard cualquier acceso se bloquea para siempre, asi
    que se prueba en un hilo aparte y se abandona si no contesta.
    """
    res: list[bool] = []

    def probar() -> None:
        try:
            os.makedirs(ruta, exist_ok=True)
            res.append(os.access(ruta, os.W_OK))
        except OSError:
            res.append(False)

    h = threading.Thread(target=probar, daemon=True, name="probar-carpeta")
    h.start()
    h.join(timeout)
    return bool(res and res[0])


def carpetas_tras_promocion(config: dict, resultados: str | None,
                            referencias: str | None) -> tuple[str, str | None, bool]:
    """Carpetas del respaldo promovido: las compartidas si responden; si no, ~/pdn-resultados.

    Devuelve (resultados, referencias, local). Sin carpeta explicita el Master
    usaria /cluster/resultados, que es justo la que puede estar colgada.
    """
    destino = resultados or os.path.join(config["rutas"]["nfs"], "resultados")
    if accesible(destino):
        return destino, referencias, False
    return os.path.join(os.path.expanduser("~"), "pdn-resultados"), None, True


# ---------------------------------------------------------------------------
# Instantanea del estado
# ---------------------------------------------------------------------------

def instantanea(master) -> dict:
    """Estado replicable del Master (se llama en el hilo del Master)."""
    c = master.corrida
    inst = {"t": time.time(), "rol": master.rol, "historial": [
        {k: v for k, v in h.items() if k != "resultado"} for h in master.historial[-5:]], "corrida": None}
    if c is not None and c.estado in (PREPARANDO, EJECUTANDO):
        p = c.planificador
        inst["corrida"] = {
            "corrida_id": c.corrida_id, "config": {k: v for k, v in c.config.items()}, "estado": c.estado,
            "t_creacion": c.t_creacion, "t_inicio": c.t_inicio, "participantes": list(c.participantes),
            "excluidos": dict(c.excluidos), "sig_tid": c.sig_tid, "parcial": c.parcial,
            "tareas": [asdict(t) for t in c.tareas.values()], "reasignaciones": c.reasignaciones[-500:],
            "descartados": c.descartados[-200:], "preparacion_master_s": c.preparacion_master_s,
            "planificador": {"pendientes": p.pendientes, "velocidad": p.velocidad, "dispositivo": p.dispositivo,
                             "con_resultado": sorted(p.con_resultado), "reservas": p.reservas}}
    # Copia profunda: la instantanea no puede compartir listas con el estado vivo
    return copy.deepcopy(inst)


def restaurar(master, inst: dict, preparada: Corrida | None) -> None:
    """Restaura en 'master' la corrida de una instantanea (en el hilo del Master)."""
    master.historial = list(inst.get("historial") or [])
    ic = inst.get("corrida")
    if ic is None or preparada is None:
        return
    c = preparada
    c.corrida_id = ic["corrida_id"]
    c.estado = ic["estado"]
    c.t_creacion = ic["t_creacion"]
    c.t_inicio = ic["t_inicio"]
    c.participantes = list(ic["participantes"])
    c.excluidos = dict(ic["excluidos"])
    c.sig_tid = ic["sig_tid"]
    c.parcial = ic["parcial"]
    c.reasignaciones = list(ic["reasignaciones"])
    c.descartados = list(ic["descartados"])
    c.preparacion_master_s = ic.get("preparacion_master_s", 0.0)
    c.tareas = {t["tid"]: Tarea(**t) for t in ic["tareas"]}
    p = c.planificador
    ip = ic["planificador"]
    p.pendientes = [list(x) for x in ip["pendientes"]]
    p.velocidad = {k: float(v) for k, v in ip["velocidad"].items()}
    p.dispositivo = dict(ip["dispositivo"])
    p.con_resultado = set(ip["con_resultado"])
    p.reservas = {k: list(v) for k, v in ip["reservas"].items()}
    master.corrida = c
    # Lo que estaba asignado se perdio con el Master principal
    for t in list(c.tareas.values()):
        if t.estado == ASIGNADA:
            master._devolver(c, t, "Master principal caido: la tarea estaba asignada")
    master._evento("ha", "Corrida %s restaurada en el respaldo: %d tareas hechas, %d unidades pendientes" % (
        c.corrida_id, sum(1 for t in c.tareas.values() if t.estado == "HECHA"), p.pendiente_unidades()))


# ---------------------------------------------------------------------------
# Lado del principal
# ---------------------------------------------------------------------------

class Replicador:
    """Envia instantaneas del Master principal a su respaldo."""

    def __init__(self, master, destino: str, intervalo: float = 1.0, timeout_acuse: float = 3.0) -> None:
        self.master = master
        self.destino = destino if ":" in destino else "%s:5556" % destino
        self.intervalo = intervalo
        self.timeout_acuse = timeout_acuse
        self._cola: queue.Queue = queue.Queue(maxsize=2)
        self._ultimo_envio = 0.0
        self._ultimo_acuse = 0.0
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._enviar, daemon=True, name="replicador")

    def iniciar(self) -> "Replicador":
        self.master.replicador = self
        self.master.ganchos_estado.append(self.gancho)
        self._hilo.start()
        log.info("Replicando el estado hacia el respaldo %s", self.destino)
        return self

    def conectado(self) -> bool:
        return time.time() - self._ultimo_acuse < self.timeout_acuse

    def gancho(self, master) -> None:
        # Corre en el hilo del Master: la instantanea es consistente
        ahora = time.time()
        if ahora - self._ultimo_envio < self.intervalo:
            return
        self._ultimo_envio = ahora
        try:
            self._cola.put_nowait(P.codificar(P.crear(INSTANTANEA, "master", estado=instantanea(master))))
        except queue.Full:
            pass

    def _enviar(self) -> None:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDHWM, 4)
        sock.connect("tcp://" + self.destino)
        while not self._parar.is_set():
            try:
                datos = self._cola.get(timeout=0.2)
                sock.send(datos, zmq.NOBLOCK)
            except queue.Empty:
                pass
            except zmq.Again:
                pass
            while sock.poll(0):
                try:
                    if P.decodificar(sock.recv())["tipo"] == ACUSE:
                        self._ultimo_acuse = time.time()
                except ValueError:
                    pass
        sock.close(0)

    def detener(self) -> None:
        self._parar.set()


# ---------------------------------------------------------------------------
# Lado del respaldo
# ---------------------------------------------------------------------------

class MasterRespaldo:
    """Recibe instantaneas y se promueve a Master si el principal deja de enviarlas."""

    def __init__(self, config: dict, puerto_tareas: int | None = None, puerto_replica: int | None = None,
                 datos: str | None = None, carpeta_resultados: str | None = None, host: str = "*",
                 carpeta_referencias: str | None = None, al_promover=None) -> None:
        self.config = config
        self.puerto_tareas = puerto_tareas or config["red"]["puerto_tareas"]
        self.puerto_replica = puerto_replica or config["red"]["puerto_respaldo"]
        self.datos = datos
        self.carpeta_resultados = carpeta_resultados
        self.carpeta_referencias = carpeta_referencias
        self.host = host
        self.timeout = float(config["master"]["respaldo_timeout_s"])
        self.al_promover = al_promover
        self.ultima: dict | None = None
        self.t_ultima: float | None = None
        self.preparadas: dict[str, Corrida] = {}
        self._preparando: set[str] = set()
        self.master = None
        self.promovido = threading.Event()
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, daemon=True, name="respaldo")
        # Master sin arrancar: solo para preparar corridas igual que el principal
        from pdn.master.servidor import Master  # noqa: PLC0415
        self._preparador = Master(config, self.puerto_tareas, datos, carpeta_resultados, host, "respaldo",
                                  carpeta_referencias)

    def iniciar(self) -> "MasterRespaldo":
        self._hilo.start()
        log.info("Master de respaldo escuchando instantaneas en el puerto %d", self.puerto_replica)
        return self

    def _preparar(self, cfg: dict, cid: str) -> None:
        # Prepara la corrida en segundo plano (puede tardar si hay que leer archivos)
        try:
            base = {k: v for k, v in cfg.items() if k not in ("nombre_a", "nombre_b", "huella_a", "huella_b",
                                                               "largo", "unidades", "tamano_a", "tamano_b")}
            self.preparadas[cid] = self._preparador.preparar_corrida(base)["corrida"]
            log.info("Corrida %s preparada en el respaldo", cid)
        except Exception as e:
            log.error("El respaldo no pudo preparar la corrida %s: %s", cid, e)
        finally:
            self._preparando.discard(cid)

    def _bucle(self) -> None:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind("tcp://%s:%d" % (self.host, self.puerto_replica))
        while not self._parar.is_set():
            if sock.poll(200):
                try:
                    ident, datos = sock.recv_multipart()[:2]
                    msg = P.decodificar(datos)
                except (ValueError, zmq.ZMQError):
                    continue
                if msg["tipo"] == INSTANTANEA:
                    self.ultima = msg["estado"]
                    self.t_ultima = time.time()
                    sock.send_multipart([ident, P.codificar(P.crear(ACUSE, "respaldo"))])
                    c = self.ultima.get("corrida")
                    if c and c["corrida_id"] not in self.preparadas and c["corrida_id"] not in self._preparando:
                        self._preparando.add(c["corrida_id"])
                        threading.Thread(target=self._preparar, args=(c["config"], c["corrida_id"]),
                                         daemon=True).start()
            if self.t_ultima is not None and time.time() - self.t_ultima > self.timeout:
                sock.close(0)
                self._promover()
                return
        sock.close(0)

    def _promover(self) -> None:
        from pdn.master.servidor import Master  # noqa: PLC0415

        log.warning("%.1f s sin instantaneas del principal: el respaldo se promueve a Master",
                    time.time() - self.t_ultima)
        c = (self.ultima or {}).get("corrida")
        if c is not None:
            limite = time.time() + 120
            while c["corrida_id"] not in self.preparadas and c["corrida_id"] in self._preparando \
                    and time.time() < limite:
                time.sleep(0.05)
        resultados, referencias, local = carpetas_tras_promocion(self.config, self.carpeta_resultados,
                                                                 self.carpeta_referencias)
        if local:
            log.warning("La carpeta compartida no responde (el Master cayo con el NFS): resultados en %s",
                        resultados)
        m = Master(self.config, self.puerto_tareas, self.datos, resultados, self.host, "respaldo",
                   referencias or os.path.join(os.path.dirname(resultados.rstrip("/")) or ".",
                                               "referencias_resultados"))
        m.activo_desde = time.time()
        m.gracia_hasta = time.time() + float(self.config["master"].get("gracia_promocion_s", 30.0))
        m.iniciar()
        preparada = self.preparadas.get(c["corrida_id"]) if c else None
        if c is not None and preparada is None:
            log.error("La corrida %s no se pudo preparar en el respaldo: no se puede continuar", c["corrida_id"])
        m._llamar("ejecutar", lambda mm: restaurar(mm, self.ultima or {}, preparada))
        m._evento("ha", "Master de respaldo activo desde %s" % time.strftime("%H:%M:%S"))
        if local:
            m._evento("aviso", "La carpeta compartida no responde: los resultados se guardan en %s de este nodo"
                      % resultados)
        self.master = m
        self.promovido.set()
        if self.al_promover is not None:
            self.al_promover(m)

    def estado(self) -> dict:
        return {"promovido": self.promovido.is_set(), "ultima_instantanea": self.t_ultima,
                "corrida": ((self.ultima or {}).get("corrida") or {}).get("corrida_id"),
                "preparadas": list(self.preparadas)}

    def detener(self) -> None:
        self._parar.set()
        if self.master is not None:
            self.master.detener()
