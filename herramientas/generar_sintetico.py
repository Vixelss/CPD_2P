"""Generador de FASTA sinteticos con respuesta conocida (CONTEXTO.md, seccion 16.2).

La respuesta esperada (.esperado.json) se calcula aqui con Python puro
(Counter, expresiones regulares y bucles por ventana), por un camino
independiente de pdn/operaciones, para que las pruebas no comparen el
codigo consigo mismo.

Uso:
    python -m herramientas.generar_sintetico salida.fna --mb 2 --semilla 7
    python -m herramientas.generar_sintetico salida.fna --crlf
    python -m herramientas.generar_sintetico --par A.fna B.fna --mb 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import numpy as np

ANCHO = 80
PATRONES = ["TATAAA", "GAATTC", "GGATCC", "CCGG", "GATTACA", "ACNGT"]
INVALIDOS = b"XZJOQE0123456789@#$%*? \t" + bytes([0xC3, 0xA9, 0x00, 0x7F])
IUPAC = b"RYSWKMBDHVryswkmbdhv"
BORDES_UNIDAD = (4096, 65536, 1 << 20, 4 << 20)
_COMPL = str.maketrans("ACGTN", "TGCAN")


def _rc(p: str) -> str:
    return p.translate(_COMPL)[::-1]


def _cabeceras(n: int, estilo: str) -> list[str]:
    # Cabeceras al estilo GenBank (CM...) o RefSeq (NC_...) para los cromosomas
    salida = []
    for i in range(n):
        if i < 3:
            crom = str(i + 1)
            if estilo == "genbank":
                salida.append(">CM%06d.2 Homo sapiens chromosome %s, GRCh38 reference "
                              "primary assembly" % (663 + i, crom))
            else:
                salida.append(">NC_%06d.11 Homo sapiens chromosome %s, GRCh38.p14 Primary "
                              "Assembly" % (i + 1, crom))
        elif i == 3:
            salida.append(">%s Homo sapiens mitochondrion, complete genome"
                          % ("J01415.2" if estilo == "genbank" else "NC_012920.1"))
        else:
            salida.append(">%s%06d.1 Homo sapiens unplaced genomic contig"
                          % ("KI" if estilo == "genbank" else "NT_", 270000 + i))
    return salida


def _longitudes(total: int, rng: np.random.Generator) -> list[int]:
    # Registros grandes, medianos y diminutos; ninguno multiplo de 80 salvo azar
    fijos = [37, 81, 160, 5]
    resto = total - sum(fijos)
    pesos = rng.random(6) + 0.2
    grandes = [int(resto * p / pesos.sum()) for p in pesos]
    grandes[-1] += resto - sum(grandes)
    largos = grandes[:3] + [fijos[0]] + grandes[3:5] + fijos[1:] + grandes[5:]
    return [max(1, x) for x in largos]


def construir_secuencia(total: int, rng: np.random.Generator) -> bytearray:
    """Secuencia con minusculas, tramos de N, IUPAC e invalidos."""
    seq = rng.choice(np.frombuffer(b"ACGT", dtype=np.uint8), size=total)
    # Islas ricas en CG para que haya zonas positivas
    for _ in range(max(1, total // 50000)):
        a = int(rng.integers(0, max(1, total - 600)))
        seq[a:a + 600] = rng.choice(np.frombuffer(b"CGCGCGAT", dtype=np.uint8), size=len(seq[a:a + 600]))
    # Tramos en minuscula (soft-masking)
    for _ in range(max(1, total // 20000)):
        a = int(rng.integers(0, total))
        seq[a:a + int(rng.integers(50, 3000))] |= 0x20
    # Tramos de N
    for _ in range(max(1, total // 100000)):
        a = int(rng.integers(0, total))
        seq[a:a + int(rng.integers(10, 2000))] = ord("N") if rng.random() < 0.8 else ord("n")
    # IUPAC sueltos e invalidos sueltos
    for _ in range(max(3, total // 200000)):
        seq[int(rng.integers(0, total))] = IUPAC[int(rng.integers(0, len(IUPAC)))]
    for _ in range(max(5, total // 100000)):
        seq[int(rng.integers(0, total))] = INVALIDOS[int(rng.integers(0, len(INVALIDOS)))]
    return bytearray(seq.tobytes())


def insertar_patrones(seq: bytearray, largos: list[int], rng: np.random.Generator) -> list[dict]:
    """Inserta patrones en bordes de unidad y cruzando limites de registro."""
    total = len(seq)
    inserciones = []
    # Cruzando bordes de unidad de varios tamanos
    for tam in BORDES_UNIDAD:
        for borde in range(tam, total, tam):
            p = PATRONES[int(rng.integers(0, len(PATRONES)))].replace("N", "A")
            corte = int(rng.integers(1, len(p)))
            ini = borde - corte
            if 0 <= ini and ini + len(p) <= total:
                seq[ini:ini + len(p)] = p.encode()
                inserciones.append({"patron": p, "posicion": ini, "motivo": "borde_unidad_%d" % tam})
            # Y justo empezando en el borde
            if borde + len(p) <= total:
                seq[borde:borde + len(p)] = p.lower().encode()
                inserciones.append({"patron": p, "posicion": borde, "motivo": "inicio_unidad_%d" % tam})
    # Cruzando limites de registro (no deben contarse)
    acumulado = 0
    for largo in largos[:-1]:
        acumulado += largo
        p = "GATTACA"
        ini = acumulado - 3
        if ini >= 0 and ini + len(p) <= total:
            seq[ini:ini + len(p)] = p.encode()
            inserciones.append({"patron": p, "posicion": ini, "motivo": "cruza_registro"})
    return inserciones


def escribir_fasta(ruta: str, cabeceras: list[str], seq: bytes, largos: list[int],
                   crlf: bool = False, ancho: int = ANCHO) -> None:
    """Escribe el FASTA con lineas de ancho fijo."""
    fin = b"\r\n" if crlf else b"\n"
    with open(ruta, "wb") as f:
        pos = 0
        for cab, largo in zip(cabeceras, largos):
            f.write(cab.encode("ascii") + fin)
            reg = seq[pos:pos + largo]
            for i in range(0, len(reg), ancho):
                f.write(reg[i:i + ancho] + fin)
            pos += largo


def esperado_de(cabeceras: list[str], seq: bytes, largos: list[int],
                ventana: int = 200, paso: int = 200) -> dict:
    """Respuesta esperada calculada con Python puro."""
    hist = [0] * 256
    for b, c in Counter(seq).items():
        hist[b] = c
    validos = set(b"ACGTNacgtn" + IUPAC)
    invalidos = [[i, b] for i, b in enumerate(seq) if b not in validos]

    registros = []
    pos = 0
    for cab, largo in zip(cabeceras, largos):
        registros.append({"cabecera": cab, "inicio": pos, "largo": largo})
        pos += largo

    # Patrones, por registro, con expresiones regulares solapadas
    pat_esp = {"patrones": PATRONES, "complemento_inverso": True, "conteos": {},
               "por_registro": {}, "posiciones": {}}
    texto = seq.decode("latin-1").upper()
    for p in PATRONES:
        rc = _rc(p)
        mas_total = menos_total = 0
        union_pos = []
        por_reg = {}
        for n, r in enumerate(registros):
            t = texto[r["inicio"]:r["inicio"] + r["largo"]]
            def buscar(q: str) -> set[int]:
                rx = re.compile("(?=%s)" % q.replace("N", "[ACGT]"))
                return {m.start() for m in rx.finditer(t)}
            mas = buscar(p)
            menos = mas if rc == p else buscar(rc)
            mas_total += len(mas)
            menos_total += len(menos)
            u = sorted(mas | menos)
            if u:
                por_reg[str(n)] = len(u)
            for q in u:
                union_pos.append([r["inicio"] + q, ("+" if q in mas else "") + ("-" if q in menos else "")])
        pat_esp["conteos"][p] = {"+": mas_total, "-": menos_total, "total": len(union_pos)}
        pat_esp["por_registro"][p] = por_reg
        pat_esp["posiciones"][p] = sorted(union_pos)

    # Zonas, ventana por ventana
    zon = {"ventana": ventana, "paso": paso, "evaluadas": 0, "no_evaluables": 0,
           "positivas": 0, "por_registro": {}, "zonas": []}
    acgt = set(b"ACGTacgt")
    for n, r in enumerate(registros):
        ev = po = ventanas = 0
        k = r["inicio"]
        while k + ventana <= r["inicio"] + r["largo"]:
            ventanas += 1
            trozo = seq[k:k + ventana]
            if all(b in acgt for b in trozo):
                u = trozo.upper()
                nc, ng, ncg = u.count(b"C"), u.count(b"G"), u.count(b"CG")
                ev += 1
                if 2 * (nc + ng) > ventana and 10 * ncg * ventana > 6 * nc * ng:
                    po += 1
                    zon["zonas"].append([k, n, nc, ng, ncg])
            else:
                zon["no_evaluables"] += 1
            k += paso
        zon["evaluadas"] += ev
        zon["positivas"] += po
        # Solo aparecen los registros que tienen al menos una ventana
        if ventanas:
            zon["por_registro"][str(n)] = [ev, po]

    return {"seq_total": len(seq), "registros": registros,
            "conteo": {"hist": hist, "invalidos_pos": invalidos},
            "patrones": pat_esp, "zonas": zon}


def generar(ruta: str, mb: float = 1.0, semilla: int = 7, crlf: bool = False,
            ancho: int = ANCHO, estilo: str = "refseq") -> dict:
    """Genera un FASTA sintetico y su .esperado.json. Devuelve lo esperado."""
    rng = np.random.default_rng(semilla)
    total = max(2000, int(mb * 1024 * 1024))
    largos = _longitudes(total, rng)
    cabeceras = _cabeceras(len(largos), estilo)
    seq = construir_secuencia(sum(largos), rng)
    inserciones = insertar_patrones(seq, largos, rng)
    escribir_fasta(ruta, cabeceras, bytes(seq), largos, crlf, ancho)
    esperado = esperado_de(cabeceras, bytes(seq), largos)
    esperado["inserciones"] = inserciones
    esperado["semilla"] = semilla
    esperado["crlf"] = crlf
    with open(_ruta_esperado(ruta), "w", encoding="utf-8") as f:
        json.dump(esperado, f)
    return esperado


def _ruta_esperado(ruta: str) -> str:
    return os.path.splitext(ruta)[0] + ".esperado.json"


def _leer_fasta(ruta: str) -> tuple[list[str], list[bytes]]:
    cabeceras, regs = [], []
    with open(ruta, "rb") as f:
        actual: list[bytes] = []
        for linea in f:
            linea = linea.rstrip(b"\r\n")
            if linea.startswith(b">"):
                if cabeceras:
                    regs.append(b"".join(actual))
                cabeceras.append(linea.decode("ascii"))
                actual = []
            else:
                actual.append(linea)
        regs.append(b"".join(actual))
    return cabeceras, regs


def generar_par(ruta_a: str, ruta_b: str, mb: float = 1.0, semilla: int = 11,
                reales: int = 300, caso: int = 120, con_n: int = 80) -> dict:
    """Par A/B en el mismo orden con sustituciones conocidas por categoria."""
    generar(ruta_a, mb, semilla, estilo="genbank")
    cab_a, regs_a = _leer_fasta(ruta_a)
    rng = np.random.default_rng(semilla + 1)
    seq = bytearray(b"".join(regs_a))
    largos = [len(r) for r in regs_a]
    es_base = np.flatnonzero(np.isin(np.frombuffer(bytes(seq), dtype=np.uint8),
                                     np.frombuffer(b"ACGTacgt", dtype=np.uint8)))
    elegidas = rng.choice(es_base, size=reales + caso + con_n, replace=False)
    mutaciones = []
    for n, p in enumerate(elegidas):
        p = int(p)
        orig = seq[p]
        if n < reales:
            otras = [b for b in b"ACGT" if b != (orig & 0xDF)]
            nuevo = otras[int(rng.integers(0, 3))]
            cat = "reales"
        elif n < reales + caso:
            nuevo = orig ^ 0x20
            cat = "solo_caso"
        else:
            nuevo = ord("N")
            cat = "con_n"
        seq[p] = nuevo
        mutaciones.append([p, cat])
    cab_b = _cabeceras(len(largos), "refseq")
    escribir_fasta(ruta_b, cab_b, bytes(seq), largos)
    esperado = {"reales": reales, "solo_caso": caso, "con_n": con_n,
                "total": reales + caso + con_n, "comparadas": sum(largos),
                "parejas": len(largos), "mutaciones": sorted(mutaciones)}
    with open(_ruta_esperado(ruta_b), "w", encoding="utf-8") as f:
        json.dump(esperado, f)
    return esperado


def generar_par_reordenado(ruta_a: str, ruta_b: str, mb: float = 1.0, semilla: int = 13) -> dict:
    """Par como GenBank contra RefSeq: mismos registros en distinto orden.

    B lleva los registros de A permutados (con cabeceras de otro estilo) y A
    tiene un registro extra sin pareja. Emparejando: cero diferencias.
    """
    generar(ruta_a, mb, semilla, estilo="genbank")
    cab_a, regs_a = _leer_fasta(ruta_a)
    # Registro extra solo en A, con longitud unica
    extra = b"ACGT" * 7 + b"A"
    cab_a.append(">KI999999.1 Homo sapiens unplaced extra contig")
    regs_a.append(extra)
    escribir_fasta(ruta_a, cab_a, b"".join(regs_a), [len(r) for r in regs_a])
    with open(_ruta_esperado(ruta_a), "w", encoding="utf-8") as f:
        json.dump(esperado_de(cab_a, b"".join(regs_a), [len(r) for r in regs_a]), f)
    cab_b = _cabeceras(len(regs_a) - 1, "refseq")
    orden = list(range(len(regs_a) - 1))
    orden = orden[3:] + orden[:3]  # los cromosomas al final, como RefSeq
    cab_b = [cab_b[i] for i in orden]
    regs_b = [regs_a[i] for i in orden]
    escribir_fasta(ruta_b, cab_b, b"".join(regs_b), [len(r) for r in regs_b])
    a = b"".join(regs_a)
    b = b"".join(regs_b)
    n = min(len(a), len(b))
    pos_total = sum(1 for x, y in zip(a[:n], b[:n]) if x != y)
    esperado = {"emparejado": {"parejas": len(regs_b), "sin_pareja_a": 1, "sin_pareja_b": 0,
                               "comparadas": sum(len(r) for r in regs_b), "total": 0},
                "posicional": {"comparadas": n, "total": pos_total}}
    with open(_ruta_esperado(ruta_b), "w", encoding="utf-8") as f:
        json.dump(esperado, f)
    return esperado


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera FASTA sinteticos con respuesta conocida")
    ap.add_argument("salida", nargs="?", help="ruta del .fna a generar")
    ap.add_argument("--mb", type=float, default=1.0)
    ap.add_argument("--semilla", type=int, default=7)
    ap.add_argument("--crlf", action="store_true", help="finales de linea \\r\\n")
    ap.add_argument("--par", nargs=2, metavar=("A", "B"), help="genera un par con sustituciones")
    ap.add_argument("--reordenado", nargs=2, metavar=("A", "B"), help="par con registros reordenados")
    args = ap.parse_args()
    if args.par:
        e = generar_par(args.par[0], args.par[1], args.mb, args.semilla)
        print("Par generado: %d diferencias esperadas" % e["total"])
    elif args.reordenado:
        generar_par_reordenado(args.reordenado[0], args.reordenado[1], args.mb, args.semilla)
        print("Par reordenado generado")
    elif args.salida:
        e = generar(args.salida, args.mb, args.semilla, args.crlf)
        print("Generado %s: %d bytes de secuencia, %d registros"
              % (args.salida, e["seq_total"], len(e["registros"])))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
