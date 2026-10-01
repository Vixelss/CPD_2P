"""Genera FASTA "ensuciados" con invalidos en cantidad y tipo conocidos.

Portado de referencias/P1.2/ensuciar.py. Los errores se inyectan SUSTITUYENDO
bytes de las lineas de secuencia (nunca en cabeceras ni saltos), asi el
archivo conserva tamano y estructura. Escribe un .esperado.json con lo
inyectado: el conteo del ensuciado menos el del original debe dar
exactamente eso.

Uso: python -m herramientas.ensuciar origen.fna destino.fna --mb 50 --tasa 100 --tipo mixto
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from herramientas.fasta_crudo import leer_recorte, mascara_secuencia

# Ningun caracter es base, N, IUPAC ni U (uracilo)
CATALOGO = {
    "letras": b"XZJOQE",
    "digitos": b"0123456789",
    "simbolos": b"@#$%*?",
    "espacios": b" \t",
}
TIPOS = tuple(CATALOGO)


def alfabeto(tipo: str) -> bytes:
    """Bytes candidatos para el tipo pedido ('mixto' mezcla todos)."""
    if tipo == "mixto":
        return b"".join(CATALOGO[t] for t in TIPOS)
    if tipo not in CATALOGO:
        raise ValueError("tipo desconocido: %s" % tipo)
    return CATALOGO[tipo]


def ensuciar(origen: str, destino: str, mb: float | None = None, tasa: float = 100,
             tipo: str = "mixto", semilla: int = 1) -> dict:
    """Escribe el ensuciado y su .esperado.json; devuelve lo esperado."""
    from pdn.comun.formato import nombre_byte

    crudo = np.frombuffer(leer_recorte(origen, mb), dtype=np.uint8).copy()
    mascara = mascara_secuencia(crudo)
    candidatos = np.flatnonzero(mascara)
    n = min(len(candidatos), max(1, int(round(len(candidatos) * tasa / 1e6))))
    rng = np.random.default_rng(semilla)
    elegidos = np.sort(rng.choice(candidatos, size=n, replace=False))
    letras = np.frombuffer(alfabeto(tipo), dtype=np.uint8)
    nuevos = letras[rng.integers(0, len(letras), size=n)]
    crudo[elegidos] = nuevos
    # Posicion de cada byte inyectado en coordenadas del .seq
    pos_seq = np.cumsum(mascara) - 1
    por_byte: dict[str, int] = {}
    for b in nuevos:
        por_byte[nombre_byte(int(b))] = por_byte.get(nombre_byte(int(b)), 0) + 1
    with open(destino, "wb") as f:
        f.write(crudo.tobytes())
    esperado = {"inyectados": int(n), "por_byte": por_byte, "tipo": tipo, "tasa": tasa,
                "semilla": semilla, "posiciones_seq": pos_seq[elegidos].tolist(),
                "bytes": nuevos.tolist()}
    with open(os.path.splitext(destino)[0] + ".esperado.json", "w", encoding="utf-8") as f:
        json.dump(esperado, f)
    return esperado


def main() -> None:
    ap = argparse.ArgumentParser(description="Inyecta invalidos conocidos en un FASTA")
    ap.add_argument("origen")
    ap.add_argument("destino")
    ap.add_argument("--mb", type=float, default=None)
    ap.add_argument("--tasa", type=float, default=100, help="errores por millon de bytes de secuencia")
    ap.add_argument("--tipo", default="mixto", choices=TIPOS + ("mixto",))
    ap.add_argument("--semilla", type=int, default=1)
    a = ap.parse_args()
    e = ensuciar(a.origen, a.destino, a.mb, a.tasa, a.tipo, a.semilla)
    print("Inyectados %d invalidos: %s" % (e["inyectados"], e["por_byte"]))


if __name__ == "__main__":
    main()
