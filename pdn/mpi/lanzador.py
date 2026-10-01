"""Lanzador de corridas MPI desde el Master (CONTEXTO.md, seccion 13.1).

Construye el comando mpirun con afinidad (--map-by core --bind-to core
--report-bindings), la interfaz de red correcta (--mca btl_tcp_if_include) y
el interprete del entorno, lo ejecuta y devuelve el resumen que escribe el
rank 0, junto con la salida de --report-bindings como evidencia.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from pdn.comun import config as cfgmod


def construir_comando(cfg: dict, np_: int, hostfile: str | None = None, python: str | None = None,
                      datos: str | None = None, reparto: str = "iguales", velocidades: dict | None = None,
                      impl: str = "numpy", interfaz: str | None = None, afinidad: bool = True,
                      salida: str | None = None, resultados: str | None = None,
                      referencias: str | None = None, sobresuscribir: bool = False,
                      codigo: str | None = None, precalentar: bool = False) -> list[str]:
    """Comando mpirun completo para una corrida."""
    if not shutil.which("mpirun"):
        raise FileNotFoundError("No se encontro mpirun; instale openmpi-bin")
    if np_ < 1:
        raise ValueError("El numero de procesos MPI debe ser al menos 1")
    cmd = ["mpirun", "-np", str(np_)]
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        cmd.append("--allow-run-as-root")
    if hostfile:
        cmd += ["--hostfile", hostfile]
    if afinidad:
        cmd += ["--map-by", "core", "--bind-to", "core", "--report-bindings"]
    else:
        cmd += ["--bind-to", "none"]
    if sobresuscribir:
        cmd.append("--oversubscribe")
    if interfaz:
        cmd += ["--mca", "btl_tcp_if_include", interfaz, "--mca", "oob_tcp_if_include", interfaz]
    codigo = codigo or cfgmod.RAIZ
    cmd += ["-x", "PYTHONPATH=%s" % codigo, "-x", "OMP_NUM_THREADS=1", "-x", "OPENBLAS_NUM_THREADS=1"]
    cmd += [python or sys.executable, "-m", "pdn.mpi.mpi_correr", "--operacion", cfg.get("operacion", "conteo"),
            "--archivo", cfg["archivo"], "--params", json.dumps(cfg.get("params") or {}),
            "--reparto", reparto, "--impl", impl]
    if cfg.get("archivo_b"):
        cmd += ["--archivo-b", cfg["archivo_b"]]
    if datos:
        cmd += ["--datos", datos]
    if cfg.get("tam_unidad"):
        cmd += ["--tam-unidad", str(cfg["tam_unidad"])]
    if velocidades:
        cmd += ["--velocidades", json.dumps(velocidades)]
    if salida:
        cmd += ["--salida", salida]
    if resultados:
        cmd += ["--resultados", resultados]
    if referencias:
        cmd += ["--referencias", referencias]
    if precalentar:
        cmd.append("--precalentar")
    return cmd


def extraer_bindings(texto: str) -> list[str]:
    """Lineas de --report-bindings (una por rank)."""
    return [l.strip() for l in texto.splitlines() if re.search(r"MCW rank \d+ (bound|is not bound)", l)]


def correr_mpi(cfg: dict, np_: int, timeout: float = 3600, **opciones) -> dict:
    """Ejecuta una corrida MPI y devuelve el resumen del rank 0 con evidencias."""
    with tempfile.TemporaryDirectory() as d:
        salida = os.path.join(d, "resumen.json")
        cmd = construir_comando(cfg, np_, salida=salida, **opciones)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 or not os.path.exists(salida):
            raise RuntimeError("mpirun fallo (codigo %d):\n%s\n%s" % (proc.returncode, proc.stdout[-2000:],
                                                                      proc.stderr[-3000:]))
        with open(salida, encoding="utf-8") as f:
            resumen = json.load(f)
    resumen["comando"] = " ".join(cmd)
    resumen["bindings"] = extraer_bindings(proc.stderr + proc.stdout)
    if resumen.get("carpeta"):
        with open(os.path.join(resumen["carpeta"], "bindings.txt"), "w", encoding="utf-8") as f:
            f.write(resumen["comando"] + "\n\n" + "\n".join(resumen["bindings"]) + "\n")
    return resumen


def lanzar_desde_cli(cfg: dict, args) -> None:
    """Punto de entrada de 'python -m pdn.cli correr --modo mpi'."""
    config = cfgmod.cargar(getattr(args, "config", None))
    np_ = args.np or 4
    hostfile = args.hostfile if args.hostfile and os.path.exists(args.hostfile) else None
    velocidades = None
    if getattr(args, "reparto", "iguales") == "proporcional":
        ruta = os.path.join(cfgmod.carpeta_resultados(config), "velocidades.json")
        if not os.path.exists(ruta):
            raise ValueError("No hay %s: corra antes la misma operacion en modo dinamico" % ruta)
        with open(ruta, encoding="utf-8") as f:
            velocidades = json.load(f).get(cfg.get("operacion", "conteo"))
    r = correr_mpi(cfg, np_, hostfile=hostfile, python=os.path.join(config["rutas"]["entorno"], "bin", "python")
                   if hostfile else None, datos=config["rutas"]["datos"],
                   impl=(cfg.get("motor") or {}).get("cpu", {}).get("impl", "numpy"),
                   interfaz=config["red"]["subred"] if hostfile else None,
                   resultados=cfgmod.carpeta_resultados(config),
                   referencias=cfgmod.carpeta_referencias(config),
                   reparto=getattr(args, "reparto", "iguales"), velocidades=velocidades,
                   precalentar=getattr(args, "precalentar", False))
    print(json.dumps({k: v for k, v in r.items() if k not in ("resultado", "clave")}, indent=1,
                     ensure_ascii=False, default=str))
