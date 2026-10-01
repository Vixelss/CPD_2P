"""Prueba 16.3.1: preparacion (.seq, .idx, .huellas) y regeneracion."""

from __future__ import annotations

import os
import shutil
import time
import zlib

import numpy as np
import pytest

from pdn.comun.huellas import leer_huellas
from pdn.preparacion.emparejar import emparejar
from pdn.preparacion.fasta_a_seq import ErrorEntrada, preparar
from pdn.preparacion.indice import Indice


def test_seq_e_indice(sintetico):
    prep, esp = sintetico
    seq = prep.mapear()
    assert seq.shape[0] == esp["seq_total"]
    indice = prep.indice()
    assert [(r.cabecera, r.inicio, r.largo) for r in indice.registros] == [
        (r["cabecera"], r["inicio"], r["largo"]) for r in esp["registros"]]
    assert not np.isin(seq, np.frombuffer(b"\r\n", dtype=np.uint8)).any()


def test_crlf_da_el_mismo_seq(sintetico, sintetico_crlf):
    a, _ = sintetico
    b, _ = sintetico_crlf
    with open(a.ruta_seq, "rb") as fa, open(b.ruta_seq, "rb") as fb:
        assert fa.read() == fb.read()
    assert a.indice().a_dict()["registros"] == b.indice().a_dict()["registros"]


def test_huellas_por_unidad(sintetico_grande):
    prep, _ = sintetico_grande
    tabla = leer_huellas(prep.ruta_huellas)
    with open(prep.ruta_seq, "rb") as f:
        datos = f.read()
    tam = tabla["tam_unidad"]
    esperados = [zlib.crc32(datos[i:i + tam]) & 0xFFFFFFFF for i in range(0, len(datos), tam)]
    assert tabla["crc"] == esperados
    assert len(esperados) == 3  # 9 MiB en unidades de 4 MiB


def test_reutiliza_y_regenera(tmp_path, sintetico):
    prep, _ = sintetico
    ruta = str(tmp_path / "copia.fna")
    shutil.copy(prep.ruta_fna, ruta)
    p1 = preparar(ruta)
    assert not p1.reutilizado
    p2 = preparar(ruta)
    assert p2.reutilizado
    # Cambiar el contenido (mismo tamano) obliga a regenerar
    with open(ruta, "r+b") as f:
        f.seek(os.path.getsize(ruta) - 5)
        f.write(b"GGGG\n")
    os.utime(ruta, ns=(time.time_ns(), time.time_ns()))
    p3 = preparar(ruta)
    assert not p3.reutilizado
    # Si falta el .origen, tampoco se reutiliza
    os.remove(os.path.splitext(ruta)[0] + ".origen")
    assert not preparar(ruta).reutilizado


def test_tam_unidad_distinto_regenera(tmp_path, sintetico):
    prep, _ = sintetico
    ruta = str(tmp_path / "u.fna")
    shutil.copy(prep.ruta_fna, ruta)
    preparar(ruta)
    p = preparar(ruta, tam_unidad=65536)
    assert not p.reutilizado
    assert leer_huellas(p.ruta_huellas)["tam_unidad"] == 65536


def test_errores_de_entrada(tmp_path):
    with pytest.raises(ErrorEntrada):
        preparar(str(tmp_path / "no_existe.fna"))
    vacio = tmp_path / "vacio.fna"
    vacio.write_bytes(b"")
    with pytest.raises(ErrorEntrada):
        preparar(str(vacio))
    solo_cab = tmp_path / "cab.fna"
    solo_cab.write_bytes(b">uno\n>dos\n")
    with pytest.raises(ErrorEntrada):
        preparar(str(solo_cab))


def test_sin_cabecera_y_sin_salto_final(tmp_path):
    ruta = tmp_path / "raro.fna"
    ruta.write_bytes(b"ACGT\nAC\n>r2 algo\nGGG")
    p = preparar(str(ruta))
    assert open(p.ruta_seq, "rb").read() == b"ACGTACGGG"
    regs = p.indice().registros
    assert [(r.inicio, r.largo) for r in regs] == [(0, 6), (6, 3)]


def test_localizar_fila_columna(sintetico):
    prep, esp = sintetico
    indice = prep.indice()
    r = esp["registros"][1]
    loc = indice.localizar(r["inicio"] + 165)
    assert loc["registro"] == 1 and loc["posicion"] == 165
    assert (loc["fila"], loc["columna"]) == (3, 6)


def test_emparejar_reordenado(par_reordenado):
    a, b, esp = par_reordenado
    parejas, solo_a, solo_b = emparejar(a.indice().registros, b.indice().registros)
    assert len(parejas) == esp["emparejado"]["parejas"]
    assert len(solo_a) == esp["emparejado"]["sin_pareja_a"]
    assert len(solo_b) == esp["emparejado"]["sin_pareja_b"]
    assert sum(1 for p in parejas if p.criterio == "cromosoma") == 4
    for p in parejas:
        assert p.a.largo == p.b.largo


def test_indice_ida_y_vuelta(tmp_path, sintetico):
    prep, _ = sintetico
    i1 = prep.indice()
    i1.guardar(str(tmp_path / "x.idx"))
    i2 = Indice.leer(str(tmp_path / "x.idx"))
    assert i1.a_dict() == i2.a_dict()
