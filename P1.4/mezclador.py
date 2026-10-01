# -*- coding: utf-8 -*-
"""
mezclador.py

Genera un archivo nuevo intercalando las filas de las dos cadenas: fila 1 de
A, fila 1 de B, fila 2 de A, fila 2 de B, y asi hasta que una de las dos se
acaba. Lo hace SOLO la CPU, con n procesos.

QUE ES UNA FILA
La misma rejilla de 80 caracteres sobre la que se informan las diferencias
(comparador.ANCHO_LINEA). Asi la fila 7 del archivo mezclado es la fila 7 de
la tabla de diferencias, y las dos lineas de cada pareja quedan una encima
de la otra, caracter con caracter.

FORMATO
    A      1: NNNNACGTACGT...
    B      1: NNNNACGTACGA...
    A      2: TTGACCAGTA...
    B      2: TTGACCAGTA...

El numero de fila se rellena con espacios hasta el ancho del mayor. No es
estetica: hace que TODAS las lineas midan lo mismo (salvo la ultima pareja,
que puede ser una fila incompleta), y entonces la posicion en el archivo de
salida de cualquier fila se calcula con una multiplicacion. Eso es lo que
permite que cada proceso escriba su tramo directamente en su sitio, sin
esperar a los demas ni pasar los datos por el padre.

COMO SE PARALELIZA
El archivo de salida se crea con su tamano final y cada proceso lo abre por
su cuenta. Las filas se reparten en n tramos contiguos; cada proceso arma el
suyo en lotes en RAM, construyendo las lineas con numpy (prefijo, digitos
del numero de fila, 80 bases y salto de linea) sin un solo bucle de Python
por fila, y escribe cada lote directamente en su posicion del archivo. La ultima pareja, que puede ser incompleta, la
escribe el padre al final.

Las cadenas de entrada llegan igual que al motor de CPU: por ruta del .seq,
mapeadas en cada hijo, sin copias.

API publica:
    ruta_por_defecto(cadena_a, cadena_b)          -> ruta de salida
    mezclar(cadena_a, cadena_b, destino, procesos) -> dict con el resumen
"""

import multiprocessing
import os
import shutil
import time

import numpy as np

import comparador
import motor_cpu
from comparador import ErrorEntrada, ANCHO_LINEA


# Filas que un proceso construye de una vez. Acota la memoria de trabajo de
# cada hijo a unas pocas decenas de MB aunque la salida pese gigas.
_LOTE_FILAS = 256 * 1024

# Carpeta donde se deja el archivo mezclado si no se indica otra.
CARPETA_SALIDA = 'resultados'

# Espacio libre que se deja de margen en el disco, ademas del archivo.
_MARGEN_DISCO = 64 << 20

_estado = {}


# ---------------------------------------------------------------------------
# Geometria del archivo de salida
# ---------------------------------------------------------------------------

def _geometria(largo_a, largo_b):
    """Calcula filas, ancho del numero y largo de linea de la salida.

    Devuelve (filas, ancho_numero, largo_linea), donde filas es cuantas
    parejas se escriben (las de la cadena con menos filas) y largo_linea lo
    que mide una linea completa, salto incluido.
    """
    filas_a = (largo_a + ANCHO_LINEA - 1) // ANCHO_LINEA
    filas_b = (largo_b + ANCHO_LINEA - 1) // ANCHO_LINEA
    filas = min(filas_a, filas_b)
    ancho_numero = len(str(max(filas, 1)))
    # 'A ' + numero + ': ' + 80 bases + '\n'
    largo_linea = 2 + ancho_numero + 2 + ANCHO_LINEA + 1
    return filas, ancho_numero, largo_linea


def _linea(letra, fila, ancho_numero, bases):
    """Una linea suelta como bytes. Solo se usa para la ultima pareja."""
    return (('%s %*d: ' % (letra, ancho_numero, fila)).encode('ascii')
            + bytes(bases) + b'\n')


def ruta_por_defecto(cadena_a, cadena_b, carpeta=CARPETA_SALIDA):
    """resultados/mezcla_<A>_<B>.txt, con los nombres de los FASTA."""
    nombre_a = os.path.splitext(os.path.basename(cadena_a.ruta_origen))[0]
    nombre_b = os.path.splitext(os.path.basename(cadena_b.ruta_origen))[0]
    return os.path.join(carpeta, 'mezcla_%s_%s.txt' % (nombre_a, nombre_b))


# ---------------------------------------------------------------------------
# Trabajo de los procesos hijos
# ---------------------------------------------------------------------------

def _iniciar_worker(ruta_a, desp_a, ruta_b, desp_b, ruta_salida,
                    ancho_numero, largo_linea):
    """Mapea las dos cadenas y abre la salida para escribir, en el hijo.

    Cada hijo tiene su propio descriptor del archivo de salida; como cada uno
    escribe en una zona distinta, no hace falta ningun cerrojo.
    """
    _estado['a'] = np.memmap(ruta_a, dtype=np.uint8, mode='r')[desp_a:]
    _estado['b'] = np.memmap(ruta_b, dtype=np.uint8, mode='r')[desp_b:]
    _estado['salida'] = open(ruta_salida, 'r+b')
    _estado['ancho'] = ancho_numero
    _estado['linea'] = largo_linea


def _escribir_filas(inicio, fin):
    """Escribe las parejas de filas completas [inicio, fin) en la salida.

    Cada pareja ocupa 2 * largo_linea bytes y empieza en
    inicio * 2 * largo_linea, asi que no hace falta saber nada de los otros
    procesos.
    """
    a, b, salida = _estado['a'], _estado['b'], _estado['salida']
    ancho, linea = _estado['ancho'], _estado['linea']
    bloque_lleno = np.empty((_LOTE_FILAS, 2, linea), dtype=np.uint8)
    inicio_bases = 2 + ancho + 2

    for desde in range(inicio, fin, _LOTE_FILAS):
        hasta = min(desde + _LOTE_FILAS, fin)
        cuantas = hasta - desde

        # El lote se arma en RAM como (filas, 2 lineas, bytes por linea) y
        # se escribe de una vez en su sitio. Escribir directamente sobre un
        # mapeo del archivo desde doce procesos resulto mas lento que con
        # uno solo: el sistema vacia las paginas sucias de forma sincrona.
        bloque = bloque_lleno[:cuantas]

        bloque[:, 0, 0] = ord('A')
        bloque[:, 1, 0] = ord('B')
        bloque[:, :, 1] = ord(' ')

        # Numero de fila (empieza en 1), alineado a la derecha con espacios.
        numeros = np.arange(desde + 1, hasta + 1, dtype=np.int64)
        for k in range(ancho):
            potencia = 10 ** (ancho - 1 - k)
            digito = ((numeros // potencia) % 10 + ord('0')).astype(np.uint8)
            if k < ancho - 1:
                digito[numeros < potencia] = ord(' ')
            bloque[:, :, 2 + k] = digito[:, None]

        bloque[:, :, 2 + ancho] = ord(':')
        bloque[:, :, 3 + ancho] = ord(' ')

        tramo = slice(desde * ANCHO_LINEA, hasta * ANCHO_LINEA)
        bloque[:, 0, inicio_bases:inicio_bases + ANCHO_LINEA] = \
            a[tramo].reshape(cuantas, ANCHO_LINEA)
        bloque[:, 1, inicio_bases:inicio_bases + ANCHO_LINEA] = \
            b[tramo].reshape(cuantas, ANCHO_LINEA)

        bloque[:, :, linea - 1] = ord('\n')

        salida.seek(desde * 2 * linea)
        salida.write(memoryview(bloque).cast('B'))

    salida.flush()
    return fin - inicio


def _calentar(_):
    return True


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def mezclar(cadena_a, cadena_b, destino=None, procesos=None, progreso=None):
    """Genera el archivo mezclado con la CPU y devuelve un resumen.

    Parametros:
        cadena_a, cadena_b : objetos Cadena de comparador.preparar().
        destino            : ruta del archivo; por defecto ruta_por_defecto().
        procesos           : procesos de CPU; se ajusta con
                             motor_cpu.procesos_validos igual que al comparar.
        progreso           : funcion opcional progreso(hechos, total).

    Devuelve un dict con ruta, filas, bytes, procesos, tiempo_s, mb_por_s,
    preparacion_s (arranque de los procesos, fuera de tiempo_s),
    sobrantes (filas que la cadena mas larga no llego a escribir) y aviso.
    Lanza ErrorEntrada si no hay filas o no cabe en el disco.
    """
    if destino is None:
        destino = ruta_por_defecto(cadena_a, cadena_b)

    filas, ancho, linea = _geometria(cadena_a.largo, cadena_b.largo)
    if filas == 0:
        raise ErrorEntrada('No hay filas que mezclar: una cadena esta vacia.')

    completas = filas - 1
    ultima = completas
    bases_a = cadena_a.datos[ultima * ANCHO_LINEA:(ultima + 1) * ANCHO_LINEA]
    bases_b = cadena_b.datos[ultima * ANCHO_LINEA:(ultima + 1) * ANCHO_LINEA]
    cola = (_linea('A', filas, ancho, bases_a)
            + _linea('B', filas, ancho, bases_b))
    total_bytes = completas * 2 * linea + len(cola)

    carpeta = os.path.dirname(os.path.abspath(destino))
    if not os.path.isdir(carpeta):
        os.makedirs(carpeta)
    libre = shutil.disk_usage(carpeta).free
    if total_bytes + _MARGEN_DISCO > libre:
        raise ErrorEntrada(
            'El archivo mezclado ocuparia %.1f MB y en el disco solo quedan '
            '%.1f MB libres.\nDesmarca "Generar archivo mezclado" o libera '
            'espacio.' % (total_bytes / 1048576.0, libre / 1048576.0))

    filas_a = cadena_a.filas
    filas_b = cadena_b.filas
    sobrantes = abs(filas_a - filas_b)

    if procesos is None:
        _, procesos = motor_cpu.detectar_nucleos()
    procesos, aviso = motor_cpu.procesos_validos(procesos, total_bytes)
    procesos = max(1, min(procesos, completas or 1))

    temporal = destino + '.parcial'
    preparacion = 0.0
    pool = None
    try:
        with open(temporal, 'wb') as f:
            f.truncate(total_bytes)

        limites = [(completas * i) // procesos for i in range(procesos + 1)]
        tareas = [(limites[i], limites[i + 1]) for i in range(procesos)]
        argumentos = (cadena_a.ruta_seq, cadena_a.desplazamiento,
                      cadena_b.ruta_seq, cadena_b.desplazamiento,
                      temporal, ancho, linea)

        # -- preparacion, fuera del cronometro, igual que en motor_cpu -----
        if completas and procesos > 1:
            inicio_preparacion = time.perf_counter()
            pool = multiprocessing.Pool(processes=procesos,
                                        initializer=_iniciar_worker,
                                        initargs=argumentos)
            pool.map(_calentar, range(procesos * 3), chunksize=1)
            preparacion = time.perf_counter() - inicio_preparacion

        # -- medicion ------------------------------------------------------
        arranque = time.perf_counter()
        if completas and pool is not None:
            pendiente = pool.starmap_async(_escribir_filas, tareas)
            while not pendiente.ready():
                pendiente.wait(0.1)
                if progreso is not None:
                    progreso(0, filas)
            pendiente.get()
            # Los hijos tienen el archivo abierto; hay que cerrarlos antes
            # de renombrarlo.
            pool.close()
            pool.join()
            pool = None
        elif completas:
            _iniciar_worker(*argumentos)
            try:
                _escribir_filas(*tareas[0])
            finally:
                _estado['salida'].close()
                _estado.clear()

        # La ultima pareja, que puede ser una fila incompleta.
        with open(temporal, 'r+b') as f:
            f.seek(completas * 2 * linea)
            f.write(cola)
        transcurrido = time.perf_counter() - arranque

        if os.path.exists(destino):
            os.remove(destino)
        os.rename(temporal, destino)
    except Exception:
        if pool is not None:
            pool.terminate()
            pool.join()
        if os.path.exists(temporal):
            try:
                os.remove(temporal)
            except OSError:
                pass
        raise

    if progreso is not None:
        progreso(filas, filas)

    mb = total_bytes / 1048576.0
    return {
        'ruta': destino,
        'filas': filas,
        'filas_a': filas_a,
        'filas_b': filas_b,
        'sobrantes': sobrantes,
        'bytes': total_bytes,
        'procesos': procesos,
        'tiempo_s': round(transcurrido, 4),
        'preparacion_s': round(preparacion, 4),
        'mb_por_s': round(mb / transcurrido, 2) if transcurrido else 0.0,
        'aviso': aviso,
    }


def formatear(resumen):
    """Texto de varias lineas para el registro de la ventana y la consola."""
    lineas = [
        'Archivo mezclado (CPU x%d): %s' % (resumen['procesos'],
                                           resumen['ruta']),
        '  %s parejas de filas, %.1f MB en %.3f s (%.1f MB/s), '
        'preparacion %.2f s'
        % (format(resumen['filas'], ',').replace(',', '.'),
           resumen['bytes'] / 1048576.0, resumen['tiempo_s'],
           resumen['mb_por_s'], resumen['preparacion_s']),
    ]
    if resumen['sobrantes']:
        mas_larga = 'A' if resumen['filas_a'] > resumen['filas_b'] else 'B'
        lineas.append('  La cadena %s tenia %d filas mas; no se escriben '
                      '(se para en la mas corta).'
                      % (mas_larga, resumen['sobrantes']))
    if resumen.get('aviso'):
        lineas.append('  AVISO: %s' % resumen['aviso'])
    return '\n'.join(lineas)


if __name__ == '__main__':
    import sys

    multiprocessing.freeze_support()
    if len(sys.argv) < 3:
        print('Uso: python mezclador.py <cadena_a.fna> <cadena_b.fna> '
              '[procesos] [salida]')
        sys.exit(0)
    try:
        ca = comparador.preparar(sys.argv[1])
        cb = comparador.preparar(sys.argv[2])
        n = int(sys.argv[3]) if len(sys.argv) > 3 else None
        salida = sys.argv[4] if len(sys.argv) > 4 else None
        print(formatear(mezclar(ca, cb, salida, n)))
    except ErrorEntrada as error:
        print('ERROR: %s' % error)
        sys.exit(1)
