"""Arranque de un worker.

Uso:
    python -m pdn.worker --master IP_PRINCIPAL --respaldo IP_RESPALDO --dispositivo cpu
    python -m pdn.worker --master 127.0.0.1:5555 --nombre sim-1 --datos /tmp/d1 --retardo 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

from pdn.comun import config as cfgmod
from pdn.comun.registro_log import configurar
from pdn.worker.motores import DISPOSITIVOS


def _con_puerto(host: str | None, puerto: int) -> str | None:
    if not host:
        return None
    return host if ":" in host else "%s:%d" % (host, puerto)


def tomar_candado(wid: str, master: str):
    """Candado exclusivo por worker_id y Master en este equipo; None si ya hay otro igual corriendo.

    Dos procesos con el mismo worker_id se pisan en el Master (uno prepara la
    corrida y al otro le llegan tareas que no preparo), asi que el segundo sale.
    """
    try:
        import fcntl  # noqa: PLC0415
    except ImportError:  # sin fcntl (Windows): no se comprueba
        return True
    usuario = os.environ.get("USER") or str(os.getuid())
    clave = re.sub(r"[^A-Za-z0-9.-]", "_", "%s-%s" % (wid, master))
    ruta = os.path.join(tempfile.gettempdir(), "pdn-worker-%s-%s.lock" % (usuario, clave))
    f = open(ruta, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    f.seek(0)
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    return f  # el candado dura lo que viva el proceso


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Worker del cluster PDN")
    ap.add_argument("--master", required=True, help="IP[:puerto] del Master principal")
    ap.add_argument("--respaldo", help="IP[:puerto] del Master de respaldo")
    ap.add_argument("--dispositivo", default="cpu", choices=DISPOSITIVOS)
    ap.add_argument("--config", help="ruta de cluster.yaml")
    ap.add_argument("--datos", help="carpeta de datos local (por defecto rutas.datos)")
    ap.add_argument("--nfs-datos", help="carpeta de datos en NFS (por defecto rutas.nfs/datos)")
    ap.add_argument("--nombre", help="nombre del nodo (por defecto el hostname)")
    ap.add_argument("--procesos", type=int, help="procesos del motor CPU")
    ap.add_argument("--nucleos", default=None, help="todos | rendimiento | eficiencia | sin_hermanos | 0-3,6")
    ap.add_argument("--impl", default=None, choices=["numpy", "simd", "simd_escalar"])
    ap.add_argument("--reservar", type=int, default=None, help="nucleos a reservar (alimentar la GPU)")
    ap.add_argument("--motor", default=None, help="opciones del motor en JSON")
    ap.add_argument("--retardo", type=float, default=0.0, help="segundos extra por MB (simulacion)")
    ap.add_argument("--latido", type=float, default=None)
    ap.add_argument("--timeout-master", type=float, default=None)
    ap.add_argument("--calibracion-mb", type=float, default=0.0, help="calibracion sintetica al registrarse")
    ap.add_argument("--log", default=None, help="archivo de log")
    a = ap.parse_args(argv)

    config = cfgmod.cargar(a.config)
    configurar("pdn.worker", a.log)
    puerto = config["red"]["puerto_tareas"]
    masters = [m for m in (_con_puerto(a.master, puerto), _con_puerto(a.respaldo, puerto)) if m]
    opciones = json.loads(a.motor) if a.motor else {}
    for clave in ("procesos", "nucleos", "impl", "reservar"):
        valor = getattr(a, clave)
        if valor is not None:
            opciones[clave] = valor

    from pdn.monitoreo.recursos import Monitor  # noqa: PLC0415
    from pdn.worker.worker import Worker  # noqa: PLC0415

    w = Worker(masters, a.dispositivo, a.datos or config["rutas"]["datos"],
               a.nfs_datos or os.path.join(config["rutas"]["nfs"], "datos"), a.nombre, opciones,
               a.retardo, a.latido or config["worker"]["latido_s"],
               a.timeout_master or config["worker"]["master_timeout_s"], a.calibracion_mb,
               monitor=Monitor(dispositivo=a.dispositivo),
               reserva_gpu=int(config["worker"].get("reserva_gpu_nucleos", 1)))
    candado = tomar_candado(w.wid, masters[0])
    if candado is None:
        print("ERROR: ya hay un worker %s corriendo en este equipo (detener_cluster.sh lo detiene)" % w.wid,
              file=sys.stderr)
        sys.exit(3)
    import signal  # noqa: PLC0415

    def _terminar(*_):
        raise KeyboardInterrupt  # SIGTERM (detener_cluster.sh) cierra igual que Ctrl+C
    signal.signal(signal.SIGTERM, _terminar)
    try:
        w.correr()
    except KeyboardInterrupt:
        w.cerrar()
    except Exception as e:
        print("ERROR: %s" % e, file=sys.stderr)
        w.cerrar()
        sys.exit(2)


if __name__ == "__main__":
    main()
