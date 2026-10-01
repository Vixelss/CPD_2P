"""Matriz de evidencias de la rubrica del P2.3 (CONTEXTO.md, seccion 14).

Arma, con datos reales de las corridas guardadas, una fila por criterio con
su estado ("con dato" o "sin dato" y el motivo). Nunca rellena valores de
ejemplo: si no hay corrida que lo demuestre, lo dice.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import time


def cargar_historial(carpeta: str, maximo: int = 300) -> list[dict]:
    """Resumenes guardados en <carpeta>/*/resumen.json (dinamicos y MPI), del mas nuevo al mas viejo."""
    rutas = sorted(glob.glob(os.path.join(carpeta, "*", "resumen.json")), reverse=True)[:maximo]
    salida = []
    for r in rutas:
        try:
            with open(r, encoding="utf-8") as f:
                d = json.load(f)
            d["carpeta"] = os.path.dirname(r)
            salida.append(d)
        except (OSError, ValueError):
            continue
    return salida


def _csv(ruta: str) -> list[dict] | None:
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fila(criterio: str, evidencia: str, valor=None, motivo: str | None = None, archivo: str | None = None) -> dict:
    return {"criterio": criterio, "evidencia": evidencia, "estado": "con dato" if valor is not None else "sin dato",
            "valor": valor, "motivo": None if valor is not None else motivo, "archivo": archivo}


def armar(carpeta_resultados: str, historial: list[dict] | None = None, workers: list[dict] | None = None) -> list[dict]:
    """Filas de la matriz de evidencias."""
    hist = historial if historial is not None else cargar_historial(carpeta_resultados)
    dinamicas = [h for h in hist if h.get("modo") != "mpi" and h.get("estado") == "TERMINADA"]
    mpi = [h for h in hist if h.get("modo") == "mpi"]
    filas = []

    # Infraestructura: estabilidad bajo carga
    largas = [h for h in dinamicas if (h.get("tiempo_s") or 0) >= 300]
    if largas:
        h = largas[0]
        caidas = [r for r in h.get("reasignaciones", []) if "perdido" in r.get("motivo", "")]
        filas.append(_fila("Infraestructura: estabilidad bajo carga", "Corrida de 5 minutos o mas con todos los nodos",
                           {"corrida": h["corrida_id"], "tiempo_s": round(h["tiempo_s"], 1),
                            "workers": len(h.get("por_worker", [])), "caidas": len(caidas),
                            "valida": (h.get("validacion") or {}).get("valido")}, archivo=h.get("carpeta")))
    else:
        filas.append(_fila("Infraestructura: estabilidad bajo carga", "Corrida de 5 minutos o mas con todos los nodos",
                           motivo="no hay ninguna corrida de 5 minutos o mas"))

    # Infraestructura: NFS
    por_origen: dict[str, dict] = {}
    for h in dinamicas:
        origen = (h.get("config") or {}).get("origen_datos", "local")
        clave = (h.get("operacion"), (h.get("config") or {}).get("nombre_a"))
        por_origen.setdefault(clave, {}).setdefault(origen, h)
    pares = [(k, v) for k, v in por_origen.items() if "local" in v and "nfs" in v]
    montajes = {w["wid"]: (w.get("hardware") or {}).get("nfs_montado") for w in (workers or [])}
    if pares:
        (op, arch), v = pares[0]
        filas.append(_fila("Infraestructura: NFS", "Montaje de /cluster y corrida NFS contra local",
                           {"operacion": op, "archivo": arch, "mb_s_local": v["local"].get("mb_s"),
                            "mb_s_nfs": v["nfs"].get("mb_s"), "montajes": montajes or None}))
    else:
        filas.append(_fila("Infraestructura: NFS", "Montaje de /cluster y corrida NFS contra local",
                           {"montajes": montajes} if any(montajes.values()) else None,
                           motivo="falta una corrida en modo NFS y otra en modo local del mismo archivo"))

    # CPU: SIMD
    simd = _csv(os.path.join(carpeta_resultados, "benchmark_simd.csv"))
    filas.append(_fila("CPU: SIMD", "Benchmark escalar contra AVX2 contra numpy (MB/s)",
                       [{k: f[k] for k in ("funcion", "implementacion", "mb_s")} for f in simd] if simd else None,
                       motivo="falta resultados/benchmark_simd.csv (python -m herramientas.benchmark_simd)",
                       archivo=os.path.join(carpeta_resultados, "benchmark_simd.csv") if simd else None))

    # CPU: afinidad
    bindings = sorted(glob.glob(os.path.join(carpeta_resultados, "*_mpi_*", "bindings.txt")), reverse=True)
    asignaciones = None
    for h in dinamicas:
        cpu = [p for p in h.get("por_worker", []) if p.get("dispositivo") == "cpu" and (p.get("motor") or {}).get("asignacion")]
        if cpu:
            asignaciones = {p["worker"]: (p["motor"] or {}).get("asignacion") for p in cpu}
            break
    if bindings or asignaciones:
        texto = ""
        if bindings:
            with open(bindings[0], encoding="utf-8") as f:
                texto = f.read()
        filas.append(_fila("CPU: afinidad", "--report-bindings y nucleo de cada proceso",
                           {"report_bindings": texto.strip().splitlines()[-16:] if texto else None,
                            "asignacion_dinamico": asignaciones},
                           archivo=bindings[0] if bindings else None))
    else:
        filas.append(_fila("CPU: afinidad", "--report-bindings y nucleo de cada proceso",
                           motivo="no hay corridas MPI ni dinamicas con afinidad registrada"))

    # GPU: transferencia sin bloquear
    gpu = _csv(os.path.join(carpeta_resultados, "benchmark_gpu.csv"))
    reales = [f for f in (gpu or []) if f.get("simulador") in ("False", "false", "0") and f.get("solapamiento_pct")]
    filas.append(_fila("GPU: transferencia sin bloquear", "Transferencia, kernel y solapamiento con 2 streams contra 1",
                       reales or None,
                       motivo="falta resultados/benchmark_gpu.csv medido en una GPU real (el simulador no mide tiempos)",
                       archivo=os.path.join(carpeta_resultados, "benchmark_gpu.csv") if reales else None))

    # NPU: modelo cuantizado
    ruta_npu = os.path.join(os.path.dirname(__file__), "motores", "npu", "modelos", "metricas.json")
    npu = None
    if os.path.exists(ruta_npu):
        with open(ruta_npu, encoding="utf-8") as f:
            npu = json.load(f)
    filas.append(_fila("NPU: modelo cuantizado", "Tamano y precision float32 contra INT8, concordancia, energia",
                       npu, motivo="falta pdn/motores/npu/modelos/metricas.json (python -m pdn.motores.npu.construir_modelo)",
                       archivo=ruta_npu if npu else None))

    # Balanceo
    if dinamicas:
        h = dinamicas[0]
        estrategias = {}
        for x in dinamicas:
            estrategias.setdefault((x.get("config") or {}).get("estrategia"), x)
        filas.append(_fila("Balanceo de carga", "Bytes por worker contra velocidad, ocioso final, adaptativa contra fija",
                           {"corrida": h["corrida_id"],
                            "por_worker": [{k: p.get(k) for k in ("worker", "dispositivo", "bytes", "mb_s_final", "ocioso_final_s")}
                                           for p in h.get("por_worker", [])],
                            "tiempo_por_estrategia": {k: v.get("tiempo_s") for k, v in estrategias.items() if k}},
                           archivo=h.get("carpeta")))
    else:
        filas.append(_fila("Balanceo de carga", "Bytes por worker contra velocidad", motivo="no hay corridas dinamicas"))

    # Tolerancia a fallos
    con_caida = [h for h in dinamicas if any("perdido" in r.get("motivo", "") for r in h.get("reasignaciones", []))]
    con_respaldo = [h for h in dinamicas if h.get("rol_master") == "respaldo"]
    if con_caida or con_respaldo:
        filas.append(_fila("Tolerancia a fallos", "Caida de un worker y del Master con resultado correcto",
                           {"caida_worker": con_caida[0]["corrida_id"] if con_caida else None,
                            "valida_tras_caida": (con_caida[0].get("validacion") or {}).get("valido") if con_caida else None,
                            "caida_master": con_respaldo[0]["corrida_id"] if con_respaldo else None,
                            "valida_tras_caida_master": (con_respaldo[0].get("validacion") or {}).get("valido")
                            if con_respaldo else None}))
    else:
        filas.append(_fila("Tolerancia a fallos", "Caida de un worker y del Master",
                           motivo="no hay corridas con una caida simulada"))

    # Integridad
    verificadas = [h for h in dinamicas + mpi if (h.get("validacion") or {}).get("coincide_referencia")]
    if verificadas:
        h = verificadas[0]
        filas.append(_fila("Integridad", "CRC por unidad y coincidencia exacta con la referencia secuencial",
                           {"corrida": h.get("corrida_id") or h.get("carpeta"), "crc": (h.get("validacion") or {}).get("crc_verificado"),
                            "coincide_referencia": True}, archivo=h.get("carpeta")))
    else:
        filas.append(_fila("Integridad", "CRC y coincidencia con la referencia",
                           motivo="no hay corridas validadas contra una referencia (python -m pdn.cli referencia ...)"))

    # Informe: rentabilidad y Amdahl
    amdahl = sorted(glob.glob(os.path.join(carpeta_resultados, "*escalabilidad*", "amdahl.json")), reverse=True)
    energia = next((h.get("energia_por_arquitectura") for h in dinamicas if h.get("energia_por_arquitectura")), None)
    valor = {}
    if amdahl:
        with open(amdahl[0], encoding="utf-8") as f:
            valor["amdahl"] = json.load(f)
    if energia and any(v.get("energia_j") is not None for v in energia.values()):
        valor["energia_por_arquitectura"] = energia
    filas.append(_fila("Informe: rentabilidad y Amdahl", "Tiempo, energia y MB/J por arquitectura; ajuste de Amdahl",
                       valor or None, motivo="faltan series de escalabilidad y energia medida",
                       archivo=amdahl[0] if amdahl else None))
    return filas


def exportar_md(filas: list[dict], ruta: str) -> str:
    """Escribe evidencias.md con la matriz."""
    lineas = ["# Evidencias para la rúbrica del P2.3", "",
              "Generado el %s a partir de las corridas guardadas." % time.strftime("%Y-%m-%d %H:%M:%S"), "",
              "| Criterio | Evidencia | Estado | Detalle |", "|---|---|---|---|"]
    for f in filas:
        detalle = f["motivo"] if f["valor"] is None else json.dumps(f["valor"], ensure_ascii=False)[:400]
        lineas.append("| %s | %s | %s | %s |" % (f["criterio"], f["evidencia"], f["estado"],
                                                 detalle.replace("|", "/")))
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    return ruta
