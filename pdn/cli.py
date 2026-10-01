"""Linea de comandos: todo lo que hace el dashboard se puede hacer desde aqui.

Ejemplos:
    python -m pdn.cli preparar ~/pdn-datos/GCF_000001405.40_GRCh38.p14_genomic.fna
    python -m pdn.cli master --dashboard
    python -m pdn.cli correr --operacion conteo --archivo GCF_000001405.40_GRCh38.p14_genomic --nodos todos
    python -m pdn.cli sim --workers 4 --retardos 0,0.5,1,2 --operacion patrones --mb 20
    python -m pdn.cli referencia --operacion conteo --archivo GCF_000001405.40_GRCh38.p14_genomic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from pdn.comun import config as cfgmod
from pdn.comun.registro_log import configurar


def _params(a) -> dict:
    p: dict = {}
    if getattr(a, "patrones", None):
        p["patrones"] = [x for x in a.patrones.replace(",", " ").split() if x]
    if getattr(a, "sin_complemento", False):
        p["complemento_inverso"] = False
    if getattr(a, "ventana", None):
        p["ventana"] = a.ventana
    if getattr(a, "paso", None):
        p["paso"] = a.paso
    if getattr(a, "modo_comparacion", None):
        p["modo_comparacion"] = a.modo_comparacion
    if getattr(a, "params", None):
        p.update(json.loads(a.params))
    return p


def _config_corrida(a) -> dict:
    cfg = {"operacion": a.operacion, "archivo": a.archivo, "params": _params(a), "modo": a.modo,
           "origen_datos": a.origen, "estrategia": a.estrategia,
           "verificar_crc": not a.sin_crc}
    if a.archivo_b:
        cfg["archivo_b"] = a.archivo_b
    if a.nodos and a.nodos != "todos":
        cfg["nodos"] = [n for n in a.nodos.split(",") if n]
    if a.tiempo_objetivo:
        cfg["tiempo_objetivo_s"] = a.tiempo_objetivo
    if a.max_tarea_mb:
        cfg["max_tarea_mb"] = a.max_tarea_mb
    if getattr(a, "tam_unidad", None):
        cfg["tam_unidad"] = a.tam_unidad
    motor = {}
    if a.procesos or a.nucleos or a.impl:
        motor["cpu"] = {k: v for k, v in (("procesos", a.procesos), ("nucleos", a.nucleos),
                                          ("impl", a.impl)) if v}
    if motor:
        cfg["motor"] = motor
    return cfg


def _args_corrida(sp) -> None:
    sp.add_argument("--operacion", default="conteo", choices=["conteo", "patrones", "comparacion", "zonas"])
    sp.add_argument("--archivo", help="nombre del archivo (sin extension) o ruta")
    sp.add_argument("--archivo-b", help="segundo archivo (comparacion)")
    sp.add_argument("--modo", default="dinamico", choices=["dinamico", "mpi"])
    sp.add_argument("--origen", default="local", choices=["local", "nfs"])
    sp.add_argument("--nodos", default="todos")
    sp.add_argument("--estrategia", default="adaptativa", choices=["adaptativa", "fija", "proporcional"])
    sp.add_argument("--tiempo-objetivo", type=float)
    sp.add_argument("--max-tarea-mb", type=float)
    sp.add_argument("--patrones", help="lista separada por comas")
    sp.add_argument("--sin-complemento", action="store_true")
    sp.add_argument("--ventana", type=int)
    sp.add_argument("--paso", type=int)
    sp.add_argument("--modo-comparacion", choices=["emparejado", "posicional"])
    sp.add_argument("--params", help="parametros extra en JSON")
    sp.add_argument("--procesos", type=int)
    sp.add_argument("--nucleos")
    sp.add_argument("--impl", choices=["numpy", "simd", "simd_escalar"])
    sp.add_argument("--sin-crc", action="store_true", help="desactiva la verificacion por CRC")
    sp.add_argument("--tam-unidad", type=int, help="bytes por unidad (por defecto la de las huellas, 4 MiB)")


def imprimir_resumen(r: dict) -> None:
    """Resumen legible de una corrida."""
    print("\nCorrida %s: %s" % (r.get("corrida_id"), r.get("estado")))
    if r.get("error"):
        print("  Error: %s" % r["error"])
    t = r.get("tiempo_s")
    print("  Operacion: %s   bytes: %s   tiempo: %s s   MB/s: %s   preparacion: %s s" % (
        r.get("operacion"), r.get("largo_bytes"), None if t is None else round(t, 3),
        None if r.get("mb_s") is None else round(r["mb_s"], 1), r.get("preparacion_s")))
    v = r.get("validacion") or {}
    print("  Validacion: valido=%s cobertura=%s crc=%s referencia=%s" % (
        v.get("valido"), (v.get("cobertura") or {}).get("ok"), v.get("crc_verificado"),
        "sin dato (no hay referencia)" if v.get("coincide_referencia") is None else
        ("coincide" if v["coincide_referencia"] else "DISCREPA: %s" % v.get("discrepancias"))))
    print("  %-22s %-6s %12s %7s %10s %10s" % ("worker", "disp", "bytes", "tareas", "MB/s fin", "ocioso s"))
    for p in r.get("por_worker", []):
        print("  %-22s %-6s %12d %7d %10s %10s%s" % (p["worker"], p["dispositivo"], p["bytes"], p["tareas"],
                                                    p["mb_s_final"], p["ocioso_final_s"],
                                                    "  EXCLUIDO: %s" % p["excluido"] if p["excluido"] else ""))
    if r.get("reasignaciones"):
        print("  Reasignaciones: %d" % len(r["reasignaciones"]))
    res = r.get("resultado") or {}
    if r.get("operacion") == "conteo":
        print("  A=%s C=%s G=%s T=%s N=%s IUPAC=%s invalidos=%s" % tuple(
            res.get(k) for k in ("A", "C", "G", "T", "N", "iupac_total", "invalidos_total")))
    elif r.get("operacion") == "patrones":
        for f in res.get("patrones", []):
            print("  %-12s total=%d (+%d, -%d)" % (f["patron"], f["total"], f["+"], f["-"]))
    elif r.get("operacion") == "comparacion":
        print("  comparadas=%s reales=%s solo_caso=%s con_n=%s" % tuple(
            res.get(k) for k in ("comparadas", "reales", "solo_caso", "con_n")))
    elif r.get("operacion") == "zonas":
        print("  ventanas evaluadas=%s positivas=%s no evaluables=%s" % tuple(
            res.get(k) for k in ("evaluadas", "positivas", "no_evaluables")))
    if r.get("carpeta"):
        print("  Resultados en %s" % r["carpeta"])


def cmd_preparar(a) -> None:
    from pdn.preparacion.fasta_a_seq import preparar  # noqa: PLC0415

    for ruta in a.archivos:
        t0 = time.perf_counter()
        p = preparar(ruta, a.destino, a.forzar,
                     progreso=lambda h, t: print("\r  %s: %5.1f %%" % (os.path.basename(ruta), 100 * h / t),
                                                 end="", flush=True))
        print("\r  %s -> %s (%d bytes, %s en %.1f s)" % (os.path.basename(ruta), p.ruta_seq, p.largo,
                                                         "reutilizado" if p.reutilizado else "preparado",
                                                         time.perf_counter() - t0))


def cmd_master(a) -> None:
    from pdn.master.servidor import Master  # noqa: PLC0415

    config = cfgmod.cargar(a.config)
    configurar("pdn.master", a.log)
    m = Master(config, a.puerto, a.datos, a.resultados).iniciar()
    if a.replicar_a:
        from pdn.master.respaldo import Replicador  # noqa: PLC0415
        Replicador(m, a.replicar_a, float(config["master"]["respaldo_snapshot_s"]),
                   float(config["master"]["respaldo_timeout_s"])).iniciar()
    if a.dashboard:
        from pdn.dashboard.app import servir  # noqa: PLC0415
        servir(m, puerto=a.puerto_dashboard or config["red"]["puerto_dashboard"], bloquear=False)
    try:
        if a.archivo:
            m.esperar_workers(a.esperar_workers, timeout=a.timeout_workers)
            imprimir_resumen(m.correr(_config_corrida(a)))
            if not a.dashboard:
                return
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        m.detener()


def cmd_respaldo(a) -> None:
    from pdn.master.respaldo import MasterRespaldo  # noqa: PLC0415

    config = cfgmod.cargar(a.config)
    configurar("pdn.respaldo", a.log)

    def al_promover(m):
        if a.dashboard:
            from pdn.dashboard.app import servir  # noqa: PLC0415
            servir(m, puerto=a.puerto_dashboard or config["red"]["puerto_dashboard"], bloquear=False)

    r = MasterRespaldo(config, a.puerto, a.puerto_replica, a.datos or config["rutas"]["datos"], a.resultados,
                       al_promover=al_promover).iniciar()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        r.detener()


def cmd_sim(a) -> None:
    from herramientas import generar_sintetico  # noqa: PLC0415
    from pdn.preparacion.fasta_a_seq import preparar  # noqa: PLC0415
    from pdn.sim.cluster import ClusterSimulado  # noqa: PLC0415

    configurar("pdn.sim")
    import signal  # noqa: PLC0415

    def _terminar(*_):
        raise KeyboardInterrupt  # SIGTERM limpia igual que Ctrl+C (detiene los workers)
    signal.signal(signal.SIGTERM, _terminar)
    carpeta = a.carpeta or os.path.join(cfgmod.RAIZ, "resultados", "sim")
    os.makedirs(carpeta, exist_ok=True)
    retardos = [float(x) for x in a.retardos.split(",")] if a.retardos else [0.0] * a.workers
    retardos = (retardos * a.workers)[:a.workers]
    archivos = []
    if a.archivo:
        archivos.append(preparar(a.archivo))
    else:
        ruta = os.path.join(carpeta, "sim_%gmb.fna" % a.mb)
        if not os.path.exists(ruta):
            generar_sintetico.generar(ruta, a.mb, semilla=7)
        archivos.append(preparar(ruta))
    b = None
    if a.operacion == "comparacion":
        ra, rb = os.path.join(carpeta, "sim_par_A.fna"), os.path.join(carpeta, "sim_par_B.fna")
        if not os.path.exists(rb):
            generar_sintetico.generar_par(ra, rb, a.mb, semilla=11)
        archivos = [preparar(ra), preparar(rb)]
        b = "sim_par_B"
    cfg = _config_corrida(a)
    cfg.setdefault("tam_unidad", 65536)  # archivos pequenos: unidades pequenas para que haya reparto
    cfg["archivo"] = os.path.basename(archivos[0].ruta_seq)[:-4]
    if b:
        cfg["archivo_b"] = b
    from pdn.sim.cluster import config_rapida  # noqa: PLC0415
    config = config_rapida(tam_unidad=cfg["tam_unidad"])  # tambien para las corridas del dashboard
    with ClusterSimulado(archivos, retardos, carpeta=os.path.join(carpeta, "cluster"), config=config,
                         procesos=a.procesos or 1, impl=a.impl or "numpy") as c:
        if a.dashboard:
            from pdn.dashboard.app import servir  # noqa: PLC0415
            servir(c.master, puerto=a.puerto_dashboard, bloquear=False)
            print("Dashboard en http://127.0.0.1:%d (Ctrl+C para salir)" % a.puerto_dashboard)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                return
        imprimir_resumen(c.correr(cfg, timeout=a.timeout))


def cmd_correr(a) -> None:
    from pdn.dashboard.cliente import Cliente  # noqa: PLC0415

    cfg = _config_corrida(a)
    if a.modo == "mpi":
        from pdn.mpi.lanzador import lanzar_desde_cli  # noqa: PLC0415
        lanzar_desde_cli(cfg, a)
        return
    cli = Cliente(a.url)
    imprimir_resumen(cli.correr(cfg, timeout=a.timeout))


def cmd_referencia(a) -> None:
    from pdn.mpi.referencia import correr_referencia  # noqa: PLC0415

    config = cfgmod.cargar(a.config)
    r = correr_referencia(_config_corrida(a), a.datos or config["rutas"]["datos"],
                          a.salida or cfgmod.carpeta_referencias(config), a.repeticiones, a.calentamiento)
    print(json.dumps({k: v for k, v in r.items() if k not in ("clave", "resultado")}, indent=1,
                     ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="pdn", description="Cluster heterogeneo para procesamiento de ADN")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preparar", help="prepara FASTA (.seq, .idx, .huellas)")
    sp.add_argument("archivos", nargs="+")
    sp.add_argument("--destino")
    sp.add_argument("--forzar", action="store_true")
    sp.set_defaults(func=cmd_preparar)

    sp = sub.add_parser("master", help="arranca el Master (y opcionalmente el dashboard)")
    sp.add_argument("--config")
    sp.add_argument("--puerto", type=int)
    sp.add_argument("--datos")
    sp.add_argument("--resultados")
    sp.add_argument("--log")
    sp.add_argument("--dashboard", action="store_true")
    sp.add_argument("--puerto-dashboard", type=int)
    sp.add_argument("--esperar-workers", type=int, default=1)
    sp.add_argument("--replicar-a", help="IP[:puerto] del Master de respaldo (alta disponibilidad)")
    sp.add_argument("--timeout-workers", type=float, default=120)
    _args_corrida(sp)
    sp.set_defaults(func=cmd_master)

    sp = sub.add_parser("respaldo", help="Master de respaldo: recibe instantaneas y se promueve si el principal cae")
    sp.add_argument("--config")
    sp.add_argument("--puerto", type=int, help="puerto de tareas que abre al promoverse")
    sp.add_argument("--puerto-replica", type=int, help="puerto donde recibe las instantaneas")
    sp.add_argument("--datos")
    sp.add_argument("--resultados")
    sp.add_argument("--log")
    sp.add_argument("--dashboard", action="store_true", help="servir el dashboard al promoverse")
    sp.add_argument("--puerto-dashboard", type=int)
    sp.set_defaults(func=cmd_respaldo)

    sp = sub.add_parser("sim", help="corre una operacion en el cluster simulado local")
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--retardos", help="segundos extra por MB de cada worker, separados por comas")
    sp.add_argument("--mb", type=float, default=8)
    sp.add_argument("--carpeta")
    sp.add_argument("--timeout", type=float, default=600)
    sp.add_argument("--dashboard", action="store_true")
    sp.add_argument("--puerto-dashboard", type=int, default=8000)
    _args_corrida(sp)
    sp.set_defaults(func=cmd_sim)

    sp = sub.add_parser("correr", help="lanza una corrida en el Master activo (via su API)")
    sp.add_argument("--url", default="http://127.0.0.1:8000", help="URL del dashboard del Master")
    sp.add_argument("--timeout", type=float, default=3600)
    sp.add_argument("--hostfile", default="hostfile")
    sp.add_argument("--np", type=int)
    sp.add_argument("--reparto", default="iguales", choices=["iguales", "proporcional"],
                    help="modo MPI: partes iguales o proporcionales a velocidades.json")
    sp.add_argument("--precalentar", action="store_true", help="modo MPI: mapear el tramo antes del reloj")
    _args_corrida(sp)
    sp.set_defaults(func=cmd_correr)

    sp = sub.add_parser("referencia", help="corrida secuencial de referencia (1 proceso, 1 nucleo)")
    sp.add_argument("--config")
    sp.add_argument("--repeticiones", type=int, default=1)
    sp.add_argument("--calentamiento", action="store_true")
    sp.add_argument("--datos")
    sp.add_argument("--salida", help="carpeta de referencias_resultados")
    _args_corrida(sp)
    sp.set_defaults(func=cmd_referencia)

    a = ap.parse_args(argv)
    try:
        a.func(a)
    except (ValueError, FileNotFoundError, TimeoutError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
