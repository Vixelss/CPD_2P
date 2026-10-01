"""Modelo de clasificacion de islas CpG para la NPU (CONTEXTO.md, seccion 8.4).

Entrada: lote de ventanas en one-hot de 4 canales (A, C, G, T; mayuscula y
minuscula igual; N y otros en cero), forma [lote, 4, W].

1. Extraccion con pesos fijos:
   - conv1d 4->2 de nucleo 1 que copia los canales C y G; suma a lo largo de
     la ventana: n_C, n_G.
   - conv1d 4->1 de nucleo 2 con peso 1 en (C, t) y (G, t+1) y sesgo -1,
     seguida de ReLU: vale 1 solo donde hay un dinucleotido CG; suma: n_CG.
2. Caracteristicas (escaladas por W para que quepan en fp16, que es como
   calcula el Neural Engine):
     c = n_C/W, g = n_G/W, cg = n_CG/W
     f1 = 2 (c + g) - 1 - 0,5/W           > 0  sii  2 (n_C + n_G) > W
     f2 = ln(10 cg + e) - ln(6 c g + e + 0,5/W^2)
                                           > 0  sii  10 n_CG W > 6 n_C n_G
   (el 0,5/W^2 es medio paso entero: un empate exacto, que la regla estricta
   da como negativo, queda negativo con margen en lugar de en cero)
   (en aritmetica exacta; en fp16 los casos al borde pueden cambiar).
3. Clasificador denso de 2 capas (2 -> 8 -> 1, ReLU y sigmoide) ENTRENADO con
   etiquetas de la regla de referencia.

La misma red se ejecuta aqui en numpy (alternativa en CPU, float32 o con los
pesos INT8) y se exporta a Core ML (pdn/motores/npu/construir_modelo.py).
"""

from __future__ import annotations

import numpy as np

VENTANA = 200
EPS = 1e-4
ESCALA_F1 = 4.0
ESCALA_F2 = 1.0
OCULTAS = 8
UMBRAL = 0.5

# Canal de cada byte: A=0, C=1, G=2, T=3 (ambos casos); cualquier otro = 4 (fuera)
_CANAL = np.full(256, 4, dtype=np.int64)
for _i, _b in enumerate(b"ACGT"):
    _CANAL[_b] = _i
    _CANAL[_b | 0x20] = _i
_OJO = np.vstack([np.eye(4, dtype=np.float32), np.zeros((1, 4), dtype=np.float32)])


def one_hot(raw: np.ndarray, inicios: np.ndarray, w: int = VENTANA, dtype=np.float32) -> np.ndarray:
    """Ventanas [lote, 4, W] en one-hot a partir de bytes crudos y sus inicios locales."""
    if inicios.size == 0:
        return np.zeros((0, 4, w), dtype=dtype)
    idx = inicios[:, None] + np.arange(w)[None, :]
    canales = _CANAL[np.asarray(raw)[idx]]
    return np.ascontiguousarray(_OJO[canales].transpose(0, 2, 1), dtype=dtype)


def pesos_extraccion() -> dict:
    """Pesos fijos de las dos convoluciones (formato Core ML: [salida, entrada, nucleo])."""
    w_gc = np.zeros((2, 4, 1), dtype=np.float32)
    w_gc[0, 1, 0] = 1.0  # C
    w_gc[1, 2, 0] = 1.0  # G
    w_cg = np.zeros((1, 4, 2), dtype=np.float32)
    w_cg[0, 1, 0] = 1.0  # C en t
    w_cg[0, 2, 1] = 1.0  # G en t+1
    return {"w_gc": w_gc, "w_cg": w_cg, "b_cg": np.array([-1.0], dtype=np.float32)}


def extraer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n_C, n_G, n_CG por ventana con las convoluciones fijas (x: [lote, 4, W])."""
    x = np.asarray(x, dtype=np.float32)
    n_c = x[:, 1, :].sum(axis=1)
    n_g = x[:, 2, :].sum(axis=1)
    cg = np.maximum(x[:, 1, :-1] + x[:, 2, 1:] - 1.0, 0.0)
    return n_c, n_g, cg.sum(axis=1)


def caracteristicas(n_c, n_g, n_cg, w: int = VENTANA) -> np.ndarray:
    """Las dos caracteristicas escaladas [lote, 2]."""
    c = np.asarray(n_c, dtype=np.float32) / w
    g = np.asarray(n_g, dtype=np.float32) / w
    cg = np.asarray(n_cg, dtype=np.float32) / w
    f1 = 2.0 * (c + g) - 1.0 - 0.5 / w
    f2 = np.log(10.0 * cg + EPS) - np.log(6.0 * c * g + EPS + 0.5 / (w * w))
    return np.stack([f1 * ESCALA_F1, f2 * ESCALA_F2], axis=1).astype(np.float32)


def densa(f: np.ndarray, p: dict) -> np.ndarray:
    """Clasificador 2 -> 8 -> 1: probabilidad de zona de interes."""
    h = np.maximum(f @ p["w1"].T + p["b1"], 0.0)
    z = h @ p["w2"].T + p["b2"]
    return (1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))).reshape(-1)


def predecir(x: np.ndarray, p: dict, w: int = VENTANA) -> np.ndarray:
    """Red completa en numpy: one-hot [lote, 4, W] -> probabilidad [lote]."""
    return densa(caracteristicas(*extraer(x), w), p)


def regla(n_c, n_g, n_cg, w: int = VENTANA) -> np.ndarray:
    """Etiqueta de referencia (enteros, igual que pdn/operaciones/zonas.py)."""
    from pdn.operaciones.zonas import es_positiva  # noqa: PLC0415
    return es_positiva(np.asarray(n_c), np.asarray(n_g), np.asarray(n_cg), w)


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def datos_entrenamiento(n: int, w: int = VENTANA, semilla: int = 0) -> tuple[np.ndarray, ...]:
    """Conteos sinteticos (n_C, n_G, n_CG) con muchos casos cerca de la frontera."""
    rng = np.random.default_rng(semilla)
    gc = np.clip(rng.normal(0.5, 0.12, n), 0.05, 0.95)
    frac_c = np.clip(rng.normal(0.5, 0.08, n), 0.1, 0.9)
    n_c = np.round(gc * w * frac_c)
    n_g = np.round(gc * w * (1 - frac_c))
    # n_CG alrededor del valor que hace obs/esp = 0,6, para poblar la frontera
    esperado = n_c * n_g / w
    n_cg = np.round(np.clip(esperado * rng.lognormal(np.log(0.6), 0.6, n), 0, np.minimum(n_c, n_g)))
    return n_c.astype(np.float32), n_g.astype(np.float32), n_cg.astype(np.float32)


def entrenar(n: int = 200_000, w: int = VENTANA, epocas: int = 4000, semilla: int = 0,
             tasa: float = 0.02) -> tuple[dict, dict]:
    """Entrena la parte densa (Adam, entropia cruzada) con etiquetas de la regla."""
    n_c, n_g, n_cg = datos_entrenamiento(n, w, semilla)
    f = caracteristicas(n_c, n_g, n_cg, w)
    y = regla(n_c, n_g, n_cg, w).astype(np.float32)
    rng = np.random.default_rng(semilla + 1)
    p = {"w1": (rng.normal(0, 0.8, (OCULTAS, 2))).astype(np.float32), "b1": np.zeros(OCULTAS, np.float32),
         "w2": (rng.normal(0, 0.8, (1, OCULTAS))).astype(np.float32), "b2": np.zeros(1, np.float32)}
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2 = 0.9, 0.999
    perdidas = []
    for t in range(1, epocas + 1):
        a = f @ p["w1"].T + p["b1"]
        h = np.maximum(a, 0)
        z = (h @ p["w2"].T + p["b2"]).reshape(-1)
        q = 1 / (1 + np.exp(-np.clip(z, -60, 60)))
        dz = ((q - y) / len(y)).reshape(-1, 1)
        g = {"w2": dz.T @ h, "b2": dz.sum(axis=0)}
        dh = (dz @ p["w2"]) * (a > 0)
        g["w1"] = dh.T @ f
        g["b1"] = dh.sum(axis=0)
        for k in p:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            p[k] = p[k] - tasa * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + 1e-8)
        if t % 500 == 0:
            perdidas.append(float(-np.mean(y * np.log(q + 1e-9) + (1 - y) * np.log(1 - q + 1e-9))))
    acierto = float(((densa(f, p) > UMBRAL) == (y > 0.5)).mean())
    return {k: v.astype(np.float32) for k, v in p.items()}, {"muestras": n, "epocas": epocas,
                                                            "acierto_entrenamiento": acierto,
                                                            "perdidas": perdidas}


# ---------------------------------------------------------------------------
# Cuantizacion INT8 (simulada en numpy, igual que linear_quantize_weights)
# ---------------------------------------------------------------------------

def cuantizar_int8(p: dict) -> tuple[dict, dict]:
    """Pesos INT8 simetricos por tensor; devuelve (pesos decuantizados, enteros y escalas)."""
    dq, enteros = {}, {}
    for k, v in p.items():
        maximo = float(np.abs(v).max()) or 1.0
        escala = maximo / 127.0
        q = np.clip(np.round(v / escala), -127, 127).astype(np.int8)
        enteros[k] = {"q": q, "escala": escala}
        dq[k] = (q.astype(np.float32) * escala).astype(np.float32)
    return dq, enteros


def guardar(ruta: str, p: dict, w: int = VENTANA) -> None:
    np.savez(ruta, ventana=np.array([w]), **p)


def cargar(ruta: str) -> tuple[dict, int]:
    d = np.load(ruta)
    return {k: d[k] for k in ("w1", "b1", "w2", "b2")}, int(d["ventana"][0])


def operaciones_por_ventana(w: int = VENTANA) -> int:
    """Operaciones aritmeticas aproximadas por ventana (para estimar ops/W)."""
    extraccion = 2 * w + 2 * w + 3 * (w - 1) + (w - 1)  # conv 1x1 (2 canales), suma, conv k=2, relu+suma
    rasgos = 12
    densa_ops = 2 * (2 * OCULTAS) + OCULTAS + 2 * OCULTAS + 4
    return extraccion + rasgos + densa_ops
