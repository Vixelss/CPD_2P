"""Ensuciado y par generado a partir de un FASTA: el sistema reproduce lo esperado."""

from __future__ import annotations

import numpy as np

from herramientas import ensuciar, generar_par
from herramientas.fasta_crudo import mascara_secuencia
from pdn.comun.formato import resumir_histograma
from pdn.operaciones.trabajo import Trabajo, ejecutar_secuencial
from pdn.preparacion.fasta_a_seq import preparar


def test_mascara_secuencia():
    crudo = np.frombuffer(b">cab ACGT\nACG\r\nT\n>x\nGG", dtype=np.uint8)
    assert bytes(crudo[mascara_secuencia(crudo)]) == b"ACGTGG"


def test_ensuciar(tmp_path, sintetico):
    prep, esp = sintetico
    destino = str(tmp_path / "sucio.fna")
    e = ensuciar.ensuciar(prep.ruta_fna, destino, tasa=500, tipo="mixto", semilla=4)
    sucio = preparar(destino)
    t = Trabajo("conteo", {"k_invalidos": 100000}, sucio.indice())
    r = t.finalizar(ejecutar_secuencial(t, sucio.mapear(), tam_tarea=4096))
    original = resumir_histograma(esp["conteo"]["hist"])
    # Lo inyectado puede caer sobre un invalido que ya estaba
    from pdn.comun.formato import CLASE_INVALIDO, TABLA_CLASES
    orig = prep.mapear()
    pisados = sum(1 for p in e["posiciones_seq"] if TABLA_CLASES[orig[p]] == CLASE_INVALIDO)
    assert r["invalidos_total"] == original["invalidos_total"] - pisados + e["inyectados"]
    # Cada byte inyectado aparece en su posicion del .seq
    seq = sucio.mapear()
    assert [int(seq[p]) for p in e["posiciones_seq"]] == e["bytes"]
    pos_invalidos = {x["posicion_global"] for x in r["primeros_invalidos"]}
    assert set(e["posiciones_seq"]) <= pos_invalidos
    # Lo inyectado reemplaza secuencia: el total de bytes no cambia
    assert r["total"] == esp["seq_total"]


def test_generar_par(tmp_path, sintetico):
    prep, _ = sintetico
    a, b = str(tmp_path / "A.fna"), str(tmp_path / "B.fna")
    e = generar_par.generar(prep.ruta_fna, a, b, reales=50, caso=20, con_n=10)
    pa, pb = preparar(a), preparar(b)
    t = Trabajo("comparacion", {"tope": 1000}, pa.indice(), pb.indice(), 4096)
    r = ejecutar_secuencial(t, pa.mapear(), pb.mapear(), tam_tarea=8192)
    assert (r["reales"], r["solo_caso"], r["con_n"]) == (50, 20, 10)
    cats = {0: "solo_caso", 1: "con_n", 2: "reales"}
    assert sorted([p[0], cats[p[4]]] for p in r["posiciones"]) == e["mutaciones"]
