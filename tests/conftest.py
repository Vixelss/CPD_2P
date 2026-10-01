"""Fixtures comunes: datos sinteticos con respuesta conocida."""

from __future__ import annotations

import json
import os

import pytest

from herramientas import generar_sintetico
from pdn.preparacion.fasta_a_seq import preparar


def _leer(ruta: str) -> dict:
    with open(os.path.splitext(ruta)[0] + ".esperado.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def carpeta_datos(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("datos"))


@pytest.fixture(scope="session")
def sintetico(carpeta_datos):
    """FASTA sintetico de 1 MiB con finales LF, preparado."""
    ruta = os.path.join(carpeta_datos, "sint_lf.fna")
    generar_sintetico.generar(ruta, mb=1.0, semilla=7)
    return preparar(ruta), _leer(ruta)


@pytest.fixture(scope="session")
def sintetico_crlf(carpeta_datos):
    """El mismo contenido que 'sintetico' pero con finales CRLF."""
    ruta = os.path.join(carpeta_datos, "sint_crlf.fna")
    generar_sintetico.generar(ruta, mb=1.0, semilla=7, crlf=True)
    return preparar(ruta), _leer(ruta)


@pytest.fixture(scope="session")
def sintetico_grande(carpeta_datos):
    """FASTA de 9 MiB para que haya varias unidades de 4 MiB."""
    ruta = os.path.join(carpeta_datos, "sint_9mb.fna")
    generar_sintetico.generar(ruta, mb=9.0, semilla=21)
    return preparar(ruta), _leer(ruta)


@pytest.fixture(scope="session")
def par_mutado(carpeta_datos):
    a = os.path.join(carpeta_datos, "par_A.fna")
    b = os.path.join(carpeta_datos, "par_B.fna")
    esperado = generar_sintetico.generar_par(a, b, mb=1.0, semilla=11)
    return preparar(a), preparar(b), esperado


@pytest.fixture(scope="session")
def par_reordenado(carpeta_datos):
    a = os.path.join(carpeta_datos, "reord_A.fna")
    b = os.path.join(carpeta_datos, "reord_B.fna")
    esperado = generar_sintetico.generar_par_reordenado(a, b, mb=0.5, semilla=13)
    return preparar(a), preparar(b), esperado
