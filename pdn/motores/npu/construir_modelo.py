"""Construye, evalua, exporta y cuantiza el modelo CpG (CONTEXTO.md, seccion 8.4).

Pasos:
1. Entrena la parte densa con etiquetas de la regla (pdn/motores/npu/modelo.py).
2. Evalua la concordancia float32 contra regla, INT8 contra regla e INT8
   contra float32: sobre conteos sinteticos cargados de casos al borde y,
   si se pasa --seq, sobre ventanas reales de un .seq (one-hot -> extraccion
   -> clasificacion, la red completa).
3. Exporta a Core ML (mlprogram fp16) con el constructor MIL de coremltools
   y cuantiza los pesos a INT8 (linear_quantize_weights). Esto funciona en
   Linux. La ejecucion en el Neural Engine y la cuantizacion de activaciones
   solo se pueden hacer en la Mac: con --verificar-coreml se compara la
   prediccion de Core ML contra la red en numpy.
4. Guarda modelos/cpg_w<W>.npz (pesos), los .mlpackage y metricas.json.

Uso:
    python -m pdn.motores.npu.construir_modelo
    python -m pdn.motores.npu.construir_modelo --seq ~/pdn-datos/GCF_000001405.40_GRCh38.p14_genomic.seq --verificar-coreml
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time

import numpy as np

from pdn.motores.npu import modelo as M

CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos")
LOTE = 1024


def rutas(w: int = M.VENTANA, carpeta: str = CARPETA) -> dict:
    base = os.path.join(carpeta, "cpg_w%d" % w)
    return {"pesos": base + ".npz", "fp16": base + "_fp16.mlpackage", "int8": base + "_int8.mlpackage",
            "metricas": os.path.join(carpeta, "metricas.json")}


def ventanas_de_seq(ruta_seq: str, n: int, w: int, semilla: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Bytes crudos y inicios de n ventanas evaluables (solo A C G T) tomadas al azar de un .seq."""
    seq = np.memmap(ruta_seq, dtype=np.uint8, mode="r")
    rng = np.random.default_rng(semilla)
    trozos, inicios, acumulado = [], [], 0
    intentos = 0
    while len(inicios) < n and intentos < n * 50:
        intentos += 1
        a = int(rng.integers(0, max(1, seq.shape[0] - w)))
        v = np.asarray(seq[a:a + w])
        if v.shape[0] == w and np.isin(v & 0xDF, np.frombuffer(b"ACGT", dtype=np.uint8)).all():
            trozos.append(v)
            inicios.append(acumulado)
            acumulado += w
    raw = np.concatenate(trozos) if trozos else np.zeros(0, dtype=np.uint8)
    return raw, np.asarray(inicios, dtype=np.int64)


def concordancia(a: np.ndarray, b: np.ndarray) -> dict:
    return {"iguales": int((a == b).sum()), "total": int(a.size),
            "concordancia": float((a == b).mean()) if a.size else None, "desacuerdos": int((a != b).sum())}


def evaluar(p: dict, pq: dict, w: int, seq: str | None, n_seq: int) -> dict:
    """Concordancias sobre conteos al borde y (opcional) ventanas reales."""
    salida = {}
    n_c, n_g, n_cg = M.datos_entrenamiento(100_000, w, semilla=99)
    f = M.caracteristicas(n_c, n_g, n_cg, w)
    regla = M.regla(n_c, n_g, n_cg, w)
    f32 = M.densa(f, p) > M.UMBRAL
    i8 = M.densa(f, pq) > M.UMBRAL
    malos = np.flatnonzero(f32 != regla)[:10]
    salida["frontera_sintetica"] = {
        "descripcion": "conteos sinteticos concentrados cerca de la frontera de la regla (caso dificil)",
        "fp32_vs_regla": concordancia(f32, regla), "int8_vs_regla": concordancia(i8, regla),
        "int8_vs_fp32": concordancia(i8, f32), "positivas_regla": int(regla.sum()),
        "ejemplos_desacuerdo_fp32": [{"n_c": int(n_c[k]), "n_g": int(n_g[k]), "n_cg": int(n_cg[k]),
                                      "regla": bool(regla[k]),
                                      # distancia entera a cada condicion (0 = empate exacto)
                                      "10_ncg_w_menos_6_nc_ng": int(10 * n_cg[k] * w - 6 * n_c[k] * n_g[k]),
                                      "2_ncg_mas_ng_menos_w": int(2 * (n_c[k] + n_g[k]) - w)} for k in malos],
        "nota": "los desacuerdos son empates o casi empates de la regla (obs/esp = 0,6): la diferencia relativa "
                "entre enteros vecinos es del orden de 1e-5, menor que la precision de un clasificador entrenado"}
    if seq:
        raw, ini = ventanas_de_seq(seq, n_seq, w)
        x = M.one_hot(raw, ini, w)
        n_c, n_g, n_cg = M.extraer(x)
        regla = M.regla(n_c, n_g, n_cg, w)
        f32 = M.predecir(x, p, w) > M.UMBRAL
        i8 = M.predecir(x, pq, w) > M.UMBRAL
        salida["ventanas_reales"] = {"archivo": os.path.basename(seq), "ventanas": int(ini.size),
                                     "fp32_vs_regla": concordancia(f32, regla),
                                     "int8_vs_regla": concordancia(i8, regla), "int8_vs_fp32": concordancia(i8, f32),
                                     "positivas_regla": int(regla.sum())}
    return salida


# ---------------------------------------------------------------------------
# Core ML
# ---------------------------------------------------------------------------

def programa_mil(p: dict, w: int, lote: int = LOTE, int8: bool = False):
    """Programa MIL de la red completa (lote fijo, como prefiere el Neural Engine).

    Con int8=True los cuatro tensores de pesos (dos convoluciones y dos capas
    densas) se guardan en INT8 con constexpr_affine_dequantize (simetrico por
    tensor, igual que modelo.cuantizar_int8); los sesgos quedan en fp16.
    """
    import coremltools as ct  # noqa: PLC0415,F401
    from coremltools.converters.mil import Builder as mb  # noqa: PLC0415

    e = M.pesos_extraccion()
    _, enteros = M.cuantizar_int8({"w_gc": e["w_gc"], "w_cg": e["w_cg"], "w1": p["w1"], "w2": p["w2"]})

    def peso(nombre: str, valor: np.ndarray):
        if not int8:
            return valor
        q = enteros[nombre]
        return mb.constexpr_affine_dequantize(quantized_data=q["q"], zero_point=np.int8(0),
                                              scale=np.float32(q["escala"]), axis=0, name=nombre + "_int8")

    @mb.program(input_specs=[mb.TensorSpec(shape=(lote, 4, w))], opset_version=ct.target.macOS13)
    def red(x):
        gc = mb.conv(x=x, weight=peso("w_gc", e["w_gc"]), pad_type="valid", name="conv_gc")    # [L, 2, W]
        sumas = mb.reduce_sum(x=gc, axes=[2], keep_dims=False)                                  # [L, 2]
        cg = mb.conv(x=x, weight=peso("w_cg", e["w_cg"]), bias=e["b_cg"], pad_type="valid",
                     name="conv_cg")                                                             # [L, 1, W-1]
        cg = mb.relu(x=cg)
        n_cg = mb.reduce_sum(x=cg, axes=[2], keep_dims=False)                           # [L, 1]
        n_c, n_g = mb.split(x=sumas, num_splits=2, axis=1)
        inv = np.float32(1.0 / w)
        c = mb.mul(x=n_c, y=inv)
        g = mb.mul(x=n_g, y=inv)
        cgf = mb.mul(x=n_cg, y=inv)
        f1 = mb.sub(x=mb.mul(x=mb.add(x=c, y=g), y=np.float32(2.0)), y=np.float32(1.0 + 0.5 / w))
        num = mb.log(x=mb.add(x=mb.mul(x=cgf, y=np.float32(10.0)), y=np.float32(M.EPS)))
        den = mb.log(x=mb.add(x=mb.mul(x=mb.mul(x=c, y=g), y=np.float32(6.0)),
                              y=np.float32(M.EPS + 0.5 / (w * w))))
        f2 = mb.sub(x=num, y=den)
        f = mb.concat(values=[mb.mul(x=f1, y=np.float32(M.ESCALA_F1)),
                              mb.mul(x=f2, y=np.float32(M.ESCALA_F2))], axis=1)       # [L, 2]
        h = mb.relu(x=mb.linear(x=f, weight=peso("w1", p["w1"]), bias=p["b1"]))
        z = mb.linear(x=h, weight=peso("w2", p["w2"]), bias=p["b2"])
        return mb.sigmoid(x=z, name="probabilidad")                                      # [L, 1]

    return red


def exportar_coreml(p: dict, w: int, destino_fp16: str, destino_int8: str, lote: int = LOTE) -> dict:
    """Convierte a mlprogram fp16 y a mlprogram con pesos INT8."""
    try:
        import coremltools as ct  # noqa: PLC0415
    except Exception as e:
        return {"exportado": False, "motivo": "coremltools no disponible: %s" % e}
    info: dict = {"coremltools": ct.__version__, "lote": lote}
    for d in (destino_fp16, destino_int8):
        if os.path.exists(d):
            shutil.rmtree(d)
    for destino, int8 in ((destino_fp16, False), (destino_int8, True)):
        modelo = ct.convert(programa_mil(p, w, lote, int8), convert_to="mlprogram",
                            minimum_deployment_target=ct.target.macOS13,
                            compute_precision=ct.precision.FLOAT16)
        modelo.save(destino)
    info["ops_cuantizacion"] = ops_cuantizadas(destino_int8)
    info.update({"exportado": True, "fp16": os.path.basename(destino_fp16), "int8": os.path.basename(destino_int8),
                 "bytes_fp16": tamano(destino_fp16), "bytes_int8": tamano(destino_int8),
                 "pesos_int8": "las dos convoluciones y las dos capas densas en INT8 (constexpr_affine_dequantize); "
                               "sesgos en fp16; el calculo en el Neural Engine es fp16",
                 "nota_tamano": "el modelo tiene 33 parametros entrenados: el paquete lo ocupan los metadatos y "
                                "la cuantizacion casi no cambia su tamano; su efecto se mide en concordancia y energia",
                 "activaciones_int8": "pendiente: linear_quantize_activations necesita ejecutar el modelo (solo en macOS)"})
    return info


def ops_cuantizadas(ruta: str) -> dict:
    """Cuenta las operaciones de decuantizacion INT8 del programa guardado (evidencia de la cuantizacion)."""
    import coremltools as ct  # noqa: PLC0415
    spec = ct.utils.load_spec(ruta)
    cuenta: dict = {}
    for f in spec.mlProgram.functions.values():
        for bloque in f.block_specializations.values():
            for op in bloque.operations:
                if "quant" in op.type or "constexpr" in op.type:
                    cuenta[op.type] = cuenta.get(op.type, 0) + 1
    return cuenta


def tamano(ruta: str) -> int:
    if os.path.isfile(ruta):
        return os.path.getsize(ruta)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(ruta) for f in fs)


def verificar_coreml(p: dict, rutas_modelo: dict, w: int, seq: str | None, n: int = 8192) -> dict:
    """En la Mac: prediccion de Core ML (CPU+NE y solo CPU) contra la red en numpy."""
    if sys.platform != "darwin":
        return {"verificado": False, "motivo": "Core ML solo ejecuta modelos en macOS"}
    import coremltools as ct  # noqa: PLC0415

    if seq:
        raw, ini = ventanas_de_seq(seq, n, w)
    else:
        rng = np.random.default_rng(5)
        raw = rng.choice(np.frombuffer(b"ACGTCGCG", dtype=np.uint8), size=n * w)
        ini = np.arange(n, dtype=np.int64) * w
    x = M.one_hot(raw, ini, w)
    ref = M.predecir(x, p, w) > M.UMBRAL
    n_c, n_g, n_cg = M.extraer(x)
    regla = M.regla(n_c, n_g, n_cg, w)
    salida = {"verificado": True, "ventanas": int(ini.size)}
    for etiqueta, unidades in (("cpu_y_ne", ct.ComputeUnit.CPU_AND_NE), ("solo_cpu", ct.ComputeUnit.CPU_ONLY)):
        for prec in ("fp16", "int8"):
            m = ct.models.MLModel(rutas_modelo[prec], compute_units=unidades)
            pred = []
            t0 = time.perf_counter()
            for a in range(0, x.shape[0], LOTE):
                bloque = x[a:a + LOTE]
                relleno = LOTE - bloque.shape[0]
                if relleno:
                    bloque = np.concatenate([bloque, np.zeros((relleno, 4, w), np.float32)])
                out = m.predict({"x": bloque})["probabilidad"].reshape(-1)
                pred.append(out[:LOTE - relleno])
            seg = time.perf_counter() - t0
            pred = np.concatenate(pred) > M.UMBRAL
            salida["%s_%s" % (etiqueta, prec)] = {"vs_numpy": concordancia(pred, ref), "vs_regla": concordancia(pred, regla),
                                                 "ventanas_s": round(x.shape[0] / seg, 1)}
    return salida


def construir(w: int = M.VENTANA, n: int = 200_000, epocas: int = 4000, seq: str | None = None,
              n_seq: int = 50_000, carpeta: str = CARPETA, exportar: bool = True,
              verificar: bool = False) -> dict:
    """Entrena, evalua, exporta y guarda. Devuelve las metricas."""
    os.makedirs(carpeta, exist_ok=True)
    r = rutas(w, carpeta)
    t0 = time.perf_counter()
    p, info = M.entrenar(n, w, epocas)
    pq, enteros = M.cuantizar_int8(p)
    M.guardar(r["pesos"], p, w)
    metricas = {"ventana": w, "fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
                "entrenamiento": {**info, "segundos": round(time.perf_counter() - t0, 1)},
                "arquitectura": "one-hot [L,4,W] -> conv fijas (n_C, n_G, n_CG) -> f1, f2 -> densa 2-8-1 (ReLU, sigmoide)",
                "parametros_entrenados": int(sum(v.size for v in p.values())),
                "operaciones_por_ventana": M.operaciones_por_ventana(w),
                "bytes_pesos_fp32": int(sum(v.nbytes for v in p.values())),
                "bytes_pesos_int8": int(sum(e["q"].nbytes + 4 for e in enteros.values())),
                "evaluacion": evaluar(p, pq, w, seq, n_seq)}
    if exportar:
        metricas["coreml"] = exportar_coreml(p, w, r["fp16"], r["int8"])
    if verificar:
        metricas["coreml_verificacion"] = verificar_coreml(p, r, w, seq)
    with open(r["metricas"], "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=1, ensure_ascii=False)
    return metricas


def main() -> None:
    ap = argparse.ArgumentParser(description="Construye el modelo CpG para la NPU")
    ap.add_argument("--ventana", type=int, default=M.VENTANA)
    ap.add_argument("--muestras", type=int, default=200_000)
    ap.add_argument("--epocas", type=int, default=4000)
    ap.add_argument("--seq", help=".seq para evaluar con ventanas reales")
    ap.add_argument("--ventanas-seq", type=int, default=50_000)
    ap.add_argument("--sin-coreml", action="store_true")
    ap.add_argument("--verificar-coreml", action="store_true", help="solo en macOS")
    a = ap.parse_args()
    m = construir(a.ventana, a.muestras, a.epocas, a.seq, a.ventanas_seq, exportar=not a.sin_coreml,
                  verificar=a.verificar_coreml)
    print(json.dumps({k: v for k, v in m.items() if k != "entrenamiento"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
