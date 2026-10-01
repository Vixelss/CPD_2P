"""Validacion de integridad (CONTEXTO.md, seccion 7.5).

1. Por unidad: CRC32 devuelto por el worker contra la tabla .huellas.
2. Global: los bytes hechos suman el tamano del espacio, sin huecos ni solapes.
3. Contra la referencia secuencial guardada, si existe.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np


def verificar_crc(tabla: dict | None, u_ini: int, u_fin: int, crcs: list[int] | None) -> str | None:
    """None si los CRC coinciden; si no, el motivo del rechazo."""
    if tabla is None:
        return None
    if crcs is None:
        return "el worker no devolvio CRC"
    esperados = tabla["crc"][u_ini:u_fin]
    if len(crcs) != len(esperados):
        return "numero de CRC distinto (%d contra %d)" % (len(crcs), len(esperados))
    malas = [u_ini + i for i, (a, b) in enumerate(zip(crcs, esperados)) if int(a) != int(b)]
    if malas:
        return "CRC distinto en las unidades %s" % malas[:10]
    return None


def cobertura(rangos: list[tuple[int, int]], n_unidades: int) -> dict:
    """Comprueba que los rangos hechos cubren cada unidad exactamente una vez."""
    cuenta = np.zeros(n_unidades, dtype=np.int64)
    for a, b in rangos:
        cuenta[a:b] += 1
    huecos = np.flatnonzero(cuenta == 0)
    solapes = np.flatnonzero(cuenta > 1)
    return {"ok": bool(huecos.size == 0 and solapes.size == 0),
            "huecos": huecos[:20].tolist(), "solapes": solapes[:20].tolist(),
            "num_huecos": int(huecos.size), "num_solapes": int(solapes.size)}


def firma_parametros(params: dict) -> str:
    """Hash corto de los parametros que afectan al resultado."""
    texto = json.dumps(params, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(texto.encode()).hexdigest()[:10]


def ruta_referencia(carpeta: str, huella: str, operacion: str, params: dict) -> str:
    """Ruta del resultado de referencia secuencial para (archivo, operacion, parametros)."""
    return os.path.join(carpeta, "%s_%s_%s.json" % (huella, operacion, firma_parametros(params)))


def guardar_referencia(ruta: str, clave: dict, extra: dict | None = None) -> None:
    """Guarda la parte comparable de un resultado secuencial."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    temporal = ruta + ".parcial"
    with open(temporal, "w", encoding="utf-8") as f:
        json.dump({"clave": clave, **(extra or {})}, f)
    os.replace(temporal, ruta)


def leer_referencia(ruta: str) -> dict | None:
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _normalizar(x):
    # JSON convierte tuplas en listas y claves en texto: se normaliza igual
    return json.loads(json.dumps(x, sort_keys=True))


def comparar(clave: dict, referencia: dict, ruta: str = "", max_dif: int = 20) -> list[str]:
    """Lista de discrepancias entre un resultado y la referencia (vacia = coincide)."""
    a, b = _normalizar(clave), _normalizar(referencia)
    salida: list[str] = []

    def rec(x, y, camino):
        if len(salida) >= max_dif:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    salida.append("%s/%s: falta en %s" % (camino, k, "resultado" if k not in x else "referencia"))
                else:
                    rec(x[k], y[k], "%s/%s" % (camino, k))
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y) and len(x) > 0 \
                and all(isinstance(e, (list, dict)) for e in x):
            for i, (e, f) in enumerate(zip(x, y)):
                rec(e, f, "%s[%d]" % (camino, i))
        elif x != y:
            texto = "%s: %s != %s" % (camino, str(x)[:80], str(y)[:80])
            salida.append(texto)

    rec(a, b, ruta)
    return salida
