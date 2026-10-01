"""Dashboard web del Master activo (CONTEXTO.md, seccion 12).

FastAPI + uvicorn en el puerto 8000. HTML, CSS y JavaScript sin framework en
static/, con Chart.js copiado en el repositorio (sin CDN). El estado en vivo
viaja por WebSocket cada segundo. Todo lo que hace la interfaz tiene su
equivalente en la API, que es la que usa la linea de comandos.
"""

from __future__ import annotations

import asyncio
import csv
import glob
import io
import json
import logging
import os
import threading
import time

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from pdn import evidencias
from pdn.comun.registro_log import MEMORIA
from pdn.master.servidor import ErrorCorrida, Master

log = logging.getLogger("pdn.dashboard")
ESTATICOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _json(x):
    return json.loads(json.dumps(x, default=str))


def rangos_hardware(w: dict) -> dict:
    """Rangos validos de cada campo segun el hardware real de un worker."""
    hw = w.get("hardware") or {}
    cpu = hw.get("cpu") or {}
    gpu = hw.get("gpu") or {}
    return {"procesos_max": len(cpu.get("logicos") or []) or None, "logicos": cpu.get("logicos"),
            "rendimiento": cpu.get("rendimiento"), "eficiencia": cpu.get("eficiencia"),
            "sin_hermanos": cpu.get("sin_hermanos"), "hibrido": cpu.get("hibrido"),
            "avx2": "avx2" in (cpu.get("flags") or []), "vram_libre_mb": gpu.get("vram_libre_mb"),
            "lote_max_mb": round(gpu["vram_libre_mb"] * 0.35) if gpu.get("vram_libre_mb") else None}


def archivos_disponibles(master: Master) -> list[dict]:
    """Archivos preparados en el Master y que workers tienen copia identica."""
    from pdn.worker.worker import escanear_archivos  # noqa: PLC0415
    propios = escanear_archivos(master.datos)
    workers = master.estado().get("workers", [])
    salida = []
    for nombre, info in sorted(propios.items()):
        con = [w["wid"] for w in workers if (w.get("archivos") or {}).get(nombre, {}).get("global") == info["global"]]
        salida.append({"nombre": nombre, "tamano": info["tamano"], "huella": info["global"], "workers": con})
    return salida


class Escalabilidad:
    """Serie de escalabilidad lanzada desde el dashboard, en segundo plano."""

    def __init__(self) -> None:
        self.estado = {"corriendo": False, "error": None, "resultado": None, "inicio": None}
        self.hilo: threading.Thread | None = None

    def lanzar(self, master: Master, cfg: dict) -> None:
        if self.estado["corriendo"]:
            raise ErrorCorrida("Ya hay una serie de escalabilidad en curso")
        from pdn.comun import config as cfgmod  # noqa: PLC0415
        from pdn.mpi.escalabilidad import correr_series  # noqa: PLC0415

        nucleos = [int(x) for x in cfg.get("nucleos", [1, 2, 4])]
        salida = os.path.join(master.carpeta_resultados,
                              time.strftime("%Y%m%d_%H%M%S") + "_escalabilidad_" + cfg.get("operacion", "conteo"))
        nodos = None if cfg.get("solo_local", True) else cfgmod.nodos_mpi(master.config)
        dinamico = (lambda c: master.correr(c)) if cfg.get("dinamico") else None
        corrida = {k: cfg[k] for k in ("operacion", "archivo", "archivo_b", "params", "tam_unidad") if cfg.get(k)}

        def trabajo():
            try:
                r = correr_series(corrida, master.datos, salida, nucleos, nodos, int(cfg.get("repeticiones", 1)),
                                  bool(cfg.get("calentamiento", False)), cfg.get("impl", "numpy"),
                                  referencias=master.carpeta_referencias, dinamico=dinamico)
                self.estado["resultado"] = _json({**r, "carpeta": salida})
            except Exception as e:  # el error se muestra en la pantalla
                log.exception("Serie de escalabilidad fallida")
                self.estado["error"] = str(e)
            finally:
                self.estado["corriendo"] = False

        self.estado.update({"corriendo": True, "error": None, "resultado": None, "inicio": time.time()})
        self.hilo = threading.Thread(target=trabajo, daemon=True)
        self.hilo.start()


class CorridaMPI:
    """Corrida MPI lanzada desde el dashboard, en segundo plano."""

    def __init__(self) -> None:
        self.estado = {"corriendo": False, "error": None, "resumen": None}

    def lanzar(self, master: Master, cfg: dict) -> None:
        if self.estado["corriendo"]:
            raise ErrorCorrida("Ya hay una corrida MPI en curso")
        import socket  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        from pdn.mpi.lanzador import correr_mpi  # noqa: PLC0415

        nodos = cfg.get("mpi_nodos") or []
        if not nodos:
            raise ErrorCorrida("Seleccione al menos un nodo con CPU para MPI")
        local = socket.gethostname()
        es_local = all(n["hostname"] in (local, "localhost", "127.0.0.1") or n["hostname"].startswith("sim-")
                       for n in nodos)
        np_ = sum(int(n.get("slots") or 1) for n in nodos)
        opciones = {"datos": master.datos, "resultados": master.carpeta_resultados,
                    "referencias": master.carpeta_referencias, "impl": "numpy"}
        impls = {(n.get("impl") or "numpy") for n in nodos}
        if len(impls) == 1:  # MPI usa la misma implementacion en todos los ranks
            opciones["impl"] = impls.pop()
        if es_local:
            # Cluster simulado o un solo nodo: sin hostfile, como mucho los hilos locales
            np_ = min(np_, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else np_)
        else:
            hf = tempfile.NamedTemporaryFile("w", suffix=".hostfile", delete=False)
            hf.write("\n".join("%s slots=%d" % (n["hostname"], int(n.get("slots") or 1)) for n in nodos) + "\n")
            hf.close()
            opciones.update({"hostfile": hf.name, "interfaz": master.config["red"]["subred"],
                             "python": os.path.join(master.config["rutas"]["entorno"], "bin", "python")})
        corrida = {k: cfg[k] for k in ("operacion", "archivo", "archivo_b", "params", "tam_unidad") if cfg.get(k)}

        def trabajo():
            try:
                self.estado["resumen"] = _json({k: v for k, v in correr_mpi(corrida, np_, **opciones).items()
                                               if k != "clave"})
                log.info("Corrida MPI terminada con %d procesos", np_)
            except Exception as e:
                log.exception("Corrida MPI fallida")
                self.estado["error"] = str(e)
            finally:
                self.estado["corriendo"] = False

        self.estado.update({"corriendo": True, "error": None, "resumen": None})
        threading.Thread(target=trabajo, daemon=True).start()


def crear_app(master: Master) -> FastAPI:
    """Aplicacion FastAPI conectada a un Master."""
    app = FastAPI(title="Cluster PDN", docs_url="/api/docs")
    escala = Escalabilidad()
    mpi = CorridaMPI()
    app.mount("/static", StaticFiles(directory=ESTATICOS), name="static")

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.get("/")
    def inicio():
        return FileResponse(os.path.join(ESTATICOS, "index.html"))

    @app.get("/api/estado")
    def estado():
        e = master.estado()
        for w in e.get("workers", []):
            w["rangos"] = rangos_hardware(w)
        return _json(e)

    @app.get("/api/archivos")
    def archivos():
        return archivos_disponibles(master)

    @app.post("/api/corridas")
    def iniciar(cfg: dict):
        try:
            cid = master.iniciar_corrida(cfg)
        except (ErrorCorrida, ValueError) as e:
            raise HTTPException(400, str(e)) from None
        return {"corrida_id": cid}

    @app.get("/api/corridas/{corrida_id}")
    def corrida(corrida_id: str):
        for h in reversed(master.historial):
            if h["corrida_id"] == corrida_id:
                return _json({"terminada": True, "resumen": h})
        # La instantanea se refresca cada 0,25 s: la corrida recien creada se busca en el Master
        actual = master.corrida
        if actual is not None and actual.corrida_id == corrida_id:
            largo = actual.trabajo.largo
            return _json({"terminada": False, "estado": actual.estado,
                          "progreso": (actual.bytes_hechos() / largo) if largo else 1.0})
        raise HTTPException(404, "Corrida desconocida: %s" % corrida_id)

    @app.post("/api/corridas/actual/cancelar")
    def cancelar():
        master.cancelar_corrida()
        return {"ok": True}

    @app.post("/api/fallo")
    def fallo(pedido: dict):
        try:
            master.simular_fallo(pedido.get("worker", ""), pedido.get("modo", "caida"))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return {"ok": True}

    @app.post("/api/master/caida")
    def caida_master():
        if not master.hay_respaldo():
            raise HTTPException(400, "No hay un Master de respaldo conectado: la caida no se puede simular")
        master.simular_caida()
        return {"ok": True, "aviso": "El Master principal se detiene; el respaldo tomara el control en unos segundos"}

    @app.get("/api/resultados")
    def resultados():
        propios = {h["corrida_id"]: h for h in master.historial}
        guardados = evidencias.cargar_historial(master.carpeta_resultados, 100)
        filas = []
        for h in list(propios.values())[::-1] + guardados:
            filas.append({k: h.get(k) for k in ("corrida_id", "operacion", "estado", "tiempo_s", "mb_s", "modo",
                                                "carpeta", "speedup", "ranks")}
                         | {"valido": (h.get("validacion") or {}).get("valido")})
        vistos, unicos = set(), []
        for f in filas:
            clave = f.get("carpeta") or f.get("corrida_id")
            if clave not in vistos:
                vistos.add(clave)
                unicos.append(f)
        return _json(unicos)

    def _buscar(corrida_id: str) -> dict:
        for h in reversed(master.historial):
            if h["corrida_id"] == corrida_id:
                return h
        for h in evidencias.cargar_historial(master.carpeta_resultados, 300):
            if h.get("corrida_id") == corrida_id or os.path.basename(h.get("carpeta", "")) == corrida_id:
                ruta = os.path.join(h["carpeta"], "resultado.json")
                if os.path.exists(ruta):
                    with open(ruta, encoding="utf-8") as f:
                        h["resultado"] = json.load(f)
                return h
        raise HTTPException(404, "Corrida desconocida: %s" % corrida_id)

    @app.get("/api/resultados/{corrida_id}")
    def resultado(corrida_id: str):
        return _json(_buscar(corrida_id))

    @app.get("/api/resultados/{corrida_id}/exportar")
    def exportar(corrida_id: str, formato: str = "json"):
        h = _buscar(corrida_id)
        if formato == "json":
            return Response(json.dumps(h, ensure_ascii=False, indent=1, default=str), media_type="application/json",
                            headers={"Content-Disposition": "attachment; filename=%s.json" % corrida_id})
        if formato == "csv":
            carpeta = h.get("carpeta")
            ruta = os.path.join(carpeta, "tareas.csv") if carpeta else None
            if ruta and os.path.exists(ruta):
                return FileResponse(ruta, media_type="text/csv", filename="%s_tareas.csv" % corrida_id)
            buf = io.StringIO()
            filas = h.get("por_worker") or []
            if filas:
                w = csv.DictWriter(buf, fieldnames=list(filas[0]))
                w.writeheader()
                w.writerows(filas)
            return Response(buf.getvalue(), media_type="text/csv")
        raise HTTPException(400, "Formato desconocido: %s (json o csv)" % formato)

    @app.post("/api/mpi")
    def lanzar_mpi(cfg: dict):
        try:
            mpi.lanzar(master, cfg)
        except ErrorCorrida as e:
            raise HTTPException(400, str(e)) from None
        return {"ok": True}

    @app.get("/api/mpi")
    def estado_mpi():
        return _json(mpi.estado)

    @app.post("/api/escalabilidad")
    def lanzar_escalabilidad(cfg: dict):
        try:
            escala.lanzar(master, cfg)
        except ErrorCorrida as e:
            raise HTTPException(400, str(e)) from None
        return {"ok": True}

    @app.get("/api/escalabilidad")
    def estado_escalabilidad():
        e = dict(escala.estado)
        if e["resultado"] is None and not e["corriendo"]:
            # Ultima serie guardada, si existe
            rutas = sorted(glob.glob(os.path.join(master.carpeta_resultados, "*escalabilidad*",
                                                  "escalabilidad.csv")), reverse=True)
            if rutas:
                with open(rutas[0], encoding="utf-8") as f:
                    filas = list(csv.DictReader(f))
                amdahl = os.path.join(os.path.dirname(rutas[0]), "amdahl.json")
                e["resultado"] = {"filas": filas, "carpeta": os.path.dirname(rutas[0]),
                                  "amdahl": json.load(open(amdahl)).get("amdahl") if os.path.exists(amdahl) else {}}
        return _json(e)

    @app.get("/api/evidencias")
    def ver_evidencias():
        hist = list(reversed(master.historial)) + evidencias.cargar_historial(master.carpeta_resultados)
        return _json(evidencias.armar(master.carpeta_resultados, hist, master.estado().get("workers")))

    @app.post("/api/evidencias/exportar")
    def exportar_evidencias():
        hist = list(reversed(master.historial)) + evidencias.cargar_historial(master.carpeta_resultados)
        filas = evidencias.armar(master.carpeta_resultados, hist, master.estado().get("workers"))
        ruta = evidencias.exportar_md(filas, os.path.join(master.carpeta_resultados, "evidencias.md"))
        return {"ruta": ruta}

    @app.get("/api/registro", response_class=PlainTextResponse)
    def registro(n: int = 300):
        return "\n".join(MEMORIA.ultimas(n))

    @app.get("/api/archivo")
    def archivo(ruta: str):
        # Solo archivos dentro de la carpeta de resultados (graficas, CSV)
        real = os.path.realpath(ruta)
        base = os.path.realpath(master.carpeta_resultados)
        if not real.startswith(base + os.sep) or not os.path.isfile(real):
            raise HTTPException(404, "Archivo no disponible")
        return FileResponse(real)

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        try:
            while True:
                e = master.estado()
                for w in e.get("workers", []):
                    w["rangos"] = rangos_hardware(w)
                await socket.send_text(json.dumps(e, default=str))
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, RuntimeError):
            return

    @app.exception_handler(ErrorCorrida)
    async def _error_corrida(_req, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    return app


def servir(master: Master, puerto: int = 8000, host: str = "0.0.0.0", bloquear: bool = True):
    """Arranca uvicorn con la app del Master (en un hilo si bloquear=False)."""
    import uvicorn  # noqa: PLC0415

    servidor = uvicorn.Server(uvicorn.Config(crear_app(master), host=host, port=puerto, log_level="warning"))
    if bloquear:
        servidor.run()
        return servidor
    hilo = threading.Thread(target=servidor.run, daemon=True, name="dashboard")
    hilo.start()
    limite = time.time() + 10
    while not servidor.started and time.time() < limite:
        time.sleep(0.05)
    log.info("Dashboard en http://%s:%d", host, puerto)
    return servidor
