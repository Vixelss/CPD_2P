"""Series de escalabilidad y ajuste de la Ley de Amdahl (CONTEXTO.md, seccion 13.3).

Series:
  1. referencia secuencial (T1)
  2. nucleos en un nodo: 1, 2, 4, 8, todos
  3. nodos con un nucleo cada uno: 1..5 nodos
  4. completa: 1..5 nodos con todos sus nucleos
  5. modo dinamico: solo CPU, CPU+GPU, CPU+GPU+NPU (via el Master)
Por punto: tiempo (mediana de N, con calentamiento opcional), speedup T1/T,
eficiencia speedup/nucleos y energia (si se midio; si no, vacio).

Ley de Amdahl con fraccion secuencial s:  S(p) = 1 / (s + (1 - s) / p).
Como 1/S - 1/p = s (1 - 1/p), s se ajusta por minimos cuadrados sobre los
puntos medidos con x = 1 - 1/p, y = 1/S - 1/p.

Uso:
    python -m pdn.mpi.escalabilidad --archivo GCF_000001405.40_GRCh38.p14_genomic --operacion conteo
    python -m pdn.mpi.escalabilidad --archivo sim --solo-local --nucleos 1,2,4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import tempfile
import time

from pdn.comun import config as cfgmod


def ajustar_amdahl(puntos: list[tuple[float, float]]) -> dict:
    """Ajusta la fraccion secuencial s a puntos (p, speedup)."""
    xs, ys = [], []
    for p, s in puntos:
        if p > 1 and s and s > 0:
            xs.append(1 - 1 / p)
            ys.append(1 / s - 1 / p)
    if not xs:
        return {"fraccion_secuencial": None, "speedup_maximo": None, "puntos": 0,
                "motivo": "hacen falta puntos con mas de un proceso"}
    f = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    f = min(max(f, 0.0), 1.0)
    pred = [1 / (f + (1 - f) / p) for p, _ in puntos if p > 1]
    reales = [s for p, s in puntos if p > 1]
    media = statistics.mean(reales)
    ss_tot = sum((r - media) ** 2 for r in reales)
    ss_res = sum((r - q) ** 2 for r, q in zip(reales, pred))
    return {"fraccion_secuencial": f, "speedup_maximo": (1 / f) if f > 0 else None,
            "r2": (1 - ss_res / ss_tot) if ss_tot > 0 else None, "puntos": len(xs)}


def speedup_amdahl(p: float, f: float) -> float:
    """Speedup teorico para p procesos y fraccion secuencial f."""
    return 1 / (f + (1 - f) / p)


def medir(funcion, repeticiones: int = 1, calentamiento: bool = False) -> tuple[float, dict]:
    """Ejecuta funcion() (que devuelve un resumen con 'tiempo_s') y toma la mediana."""
    if calentamiento:
        funcion()
    resumenes = [funcion() for _ in range(max(1, repeticiones))]
    tiempos = [r["tiempo_s"] for r in resumenes]
    mediana = statistics.median(tiempos)
    elegido = min(resumenes, key=lambda r: abs(r["tiempo_s"] - mediana))
    elegido["tiempos_s"] = tiempos
    return mediana, elegido


def fila(serie: str, punto: str, nucleos: int, nodos: int, t: float, t1: float | None, r: dict) -> dict:
    """Fila de la tabla de escalabilidad."""
    speedup = (t1 / t) if (t1 and t) else None
    v = r.get("validacion") or {}
    return {"serie": serie, "punto": punto, "nucleos": nucleos, "nodos": nodos, "tiempo_s": round(t, 5),
            "speedup": None if speedup is None else round(speedup, 4),
            "eficiencia": None if speedup is None else round(speedup / max(nucleos, 1), 4),
            "mb_s": None if not r.get("largo_bytes") else round(r["largo_bytes"] / 1048576 / t, 2),
            "energia_j": r.get("energia_j"), "valido": v.get("valido"),
            "tiempos_s": json.dumps([round(x, 5) for x in r.get("tiempos_s", [t])])}


def _hostfile(lineas: list[str]) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".hostfile", delete=False)
    f.write("\n".join(lineas) + "\n")
    f.close()
    return f.name


def correr_series(cfg: dict, datos: str, salida: str, nucleos_local: list[int], nodos: list[dict] | None = None,
                  repeticiones: int = 1, calentamiento: bool = False, impl: str = "numpy",
                  python: str | None = None, interfaz: str | None = None, referencias: str | None = None,
                  dinamico=None) -> dict:
    """Ejecuta las series y escribe escalabilidad.csv, amdahl.json y las graficas."""
    from pdn.mpi.lanzador import correr_mpi  # noqa: PLC0415
    from pdn.mpi.referencia import correr_referencia  # noqa: PLC0415

    os.makedirs(salida, exist_ok=True)
    referencias = referencias or os.path.join(salida, "referencias")
    filas: list[dict] = []
    # 1) referencia secuencial
    ref = correr_referencia(cfg, datos, referencias, repeticiones, calentamiento)
    t1 = ref["tiempo_s"]
    filas.append(fila("1_referencia", "1 proceso", 1, 1, t1, t1, {"largo_bytes": ref["largo_bytes"],
                                                                   "validacion": {"valido": True},
                                                                   "tiempos_s": ref["tiempos_s"]}))
    # Con calentamiento, cada rank precalienta su tramo: mismas condiciones que la referencia
    comunes = {"datos": datos, "impl": impl, "referencias": referencias, "precalentar": calentamiento}

    def punto(np_, hostfile=None, **extra):
        return lambda: correr_mpi(cfg, np_, hostfile=hostfile, python=python, interfaz=interfaz,
                                  **comunes, **extra)

    # 2) nucleos en un nodo (local)
    for n in nucleos_local:
        t, r = medir(punto(n), repeticiones, calentamiento)
        filas.append(fila("2_nucleos_un_nodo", "%d nucleos" % n, n, 1, t, t1, r))
    # 3) y 4) varios nodos
    if nodos:
        for k in range(1, len(nodos) + 1):
            hf = _hostfile(["%s slots=1" % n["hostname"] for n in nodos[:k]])
            t, r = medir(punto(k, hf), repeticiones, calentamiento)
            filas.append(fila("3_nodos_un_nucleo", "%d nodos" % k, k, k, t, t1, r))
        for k in range(1, len(nodos) + 1):
            slots = [int(n.get("slots", 4)) for n in nodos[:k]]
            hf = _hostfile(["%s slots=%d" % (n["hostname"], s) for n, s in zip(nodos[:k], slots)])
            t, r = medir(punto(sum(slots), hf), repeticiones, calentamiento)
            filas.append(fila("4_completa", "%d nodos" % k, sum(slots), k, t, t1, r))
    # 5) modo dinamico (opcional): dinamico(nombre) -> resumen
    if dinamico is not None:
        for nombre, extra in (("solo CPU", {"solo_dispositivos": ["cpu"]}),
                              ("CPU+GPU", {"solo_dispositivos": ["cpu", "gpu"]}),
                              ("CPU+GPU+NPU", {"solo_dispositivos": ["cpu", "gpu", "npu"]})):
            try:
                t, r = medir(lambda: dinamico(dict(cfg, **extra)), repeticiones, calentamiento)
                nucleos = sum((w.get("motor") or {}).get("procesos") or 1 for w in r.get("por_worker", []))
                filas.append(fila("5_dinamico", nombre, nucleos, len({w["worker"].split(":")[0]
                                                                     for w in r.get("por_worker", [])}),
                                  t, t1, r))
            except Exception as e:  # una combinacion sin hardware no detiene la serie
                filas.append({"serie": "5_dinamico", "punto": nombre, "error": str(e)})

    amdahl = {}
    for serie in ("2_nucleos_un_nodo", "3_nodos_un_nucleo", "4_completa"):
        pts = [(f["nucleos"], f["speedup"]) for f in filas if f.get("serie") == serie and f.get("speedup")]
        if pts:
            amdahl[serie] = ajustar_amdahl(pts)
    escribir(salida, filas, amdahl, ref)
    return {"filas": filas, "amdahl": amdahl, "referencia": {k: v for k, v in ref.items()
                                                             if k not in ("clave", "resultado")}}


def escribir(salida: str, filas: list[dict], amdahl: dict, ref: dict) -> None:
    """CSV, JSON y graficas de la serie."""
    campos: list[str] = []
    for f in filas:
        campos += [k for k in f if k not in campos]
    with open(os.path.join(salida, "escalabilidad.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    with open(os.path.join(salida, "amdahl.json"), "w", encoding="utf-8") as fh:
        json.dump({"amdahl": amdahl, "t1_s": ref["tiempo_s"], "fecha": time.strftime("%Y-%m-%d %H:%M:%S")},
                  fh, indent=1)
    graficar(salida, filas, amdahl)


def graficar(salida: str, filas: list[dict], amdahl: dict) -> list[str]:
    """Graficas de tiempo, speedup (con ideal y Amdahl) y eficiencia por serie."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rutas = []
    series = [s for s in ("2_nucleos_un_nodo", "3_nodos_un_nucleo", "4_completa", "5_dinamico")
              if any(f.get("serie") == s and f.get("tiempo_s") for f in filas)]
    if not series:
        return rutas
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5))
    for s in series:
        pts = sorted([f for f in filas if f.get("serie") == s and f.get("tiempo_s")], key=lambda f: f["nucleos"])
        x = [f["nucleos"] for f in pts]
        ejes[0].plot(x, [f["tiempo_s"] for f in pts], "o-", label=s)
        ejes[1].plot(x, [f["speedup"] for f in pts], "o-", label=s)
        ejes[2].plot(x, [f["eficiencia"] for f in pts], "o-", label=s)
        fit = amdahl.get(s) or {}
        if fit.get("fraccion_secuencial") is not None and x:
            xs = list(range(1, max(x) + 1))
            ejes[1].plot(xs, [speedup_amdahl(p, fit["fraccion_secuencial"]) for p in xs], "--",
                         label="Amdahl %s (s=%.3f)" % (s.split("_", 1)[0], fit["fraccion_secuencial"]))
    maximo = max(f["nucleos"] for f in filas if f.get("tiempo_s"))
    ejes[1].plot([1, maximo], [1, maximo], ":", color="gray", label="ideal")
    for eje, titulo in zip(ejes, ("Tiempo (s)", "Speedup", "Eficiencia")):
        eje.set_title(titulo)
        eje.set_xlabel("nucleos / procesos")
        eje.grid(alpha=0.3)
    ejes[1].legend(fontsize=7)
    ejes[0].legend(fontsize=7)
    fig.tight_layout()
    ruta = os.path.join(salida, "escalabilidad.png")
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
    rutas.append(ruta)
    return rutas


def main() -> None:
    ap = argparse.ArgumentParser(description="Series de escalabilidad con ajuste de Amdahl")
    ap.add_argument("--config")
    ap.add_argument("--operacion", default="conteo")
    ap.add_argument("--archivo", required=True)
    ap.add_argument("--archivo-b")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--datos")
    ap.add_argument("--salida")
    ap.add_argument("--nucleos", default="1,2,4,8", help="serie 2: lista de procesos en el nodo local")
    ap.add_argument("--solo-local", action="store_true", help="omite las series 3 y 4 (varios nodos)")
    ap.add_argument("--repeticiones", type=int, default=3)
    ap.add_argument("--calentamiento", action="store_true")
    ap.add_argument("--impl", default="numpy", choices=["numpy", "simd", "simd_escalar"])
    ap.add_argument("--url-master", help="URL del dashboard para la serie 5 (modo dinamico)")
    a = ap.parse_args()
    config = cfgmod.cargar(a.config)
    import psutil  # noqa: PLC0415

    maximo = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else psutil.cpu_count()
    nucleos = sorted({min(int(x), maximo) for x in a.nucleos.split(",")} | {maximo})
    cfg = {"operacion": a.operacion, "archivo": a.archivo, "params": json.loads(a.params)}
    if a.archivo_b:
        cfg["archivo_b"] = a.archivo_b
    dinamico = None
    if a.url_master:
        from pdn.dashboard.cliente import Cliente  # noqa: PLC0415
        cliente = Cliente(a.url_master)
        dinamico = lambda c: cliente.correr(c)  # noqa: E731
    salida = a.salida or os.path.join(cfgmod.carpeta_resultados(config),
                                      time.strftime("%Y%m%d_%H%M%S") + "_escalabilidad_" + a.operacion)
    r = correr_series(cfg, a.datos or config["rutas"]["datos"], salida, nucleos,
                      None if a.solo_local else cfgmod.nodos_mpi(config), a.repeticiones, a.calentamiento,
                      a.impl, os.path.join(config["rutas"]["entorno"], "bin", "python") if not a.solo_local else None,
                      config["red"]["subred"] if not a.solo_local else None,
                      cfgmod.carpeta_referencias(config), dinamico)
    for f in r["filas"]:
        print("%-20s %-14s %s" % (f.get("serie"), f.get("punto"),
                                  f.get("error") or "t=%.3f s  S=%s  E=%s" % (f["tiempo_s"], f["speedup"], f["eficiencia"])))
    print("Amdahl:", json.dumps(r["amdahl"], indent=1))
    print("Resultados en %s" % salida)

if __name__ == "__main__":
    main()
