# -*- coding: utf-8 -*-
"""
mezclador_gpu.py

La misma mezcla que mezclador.py (fila de A, fila de B, alternadas, hasta
que se acaba la cadena mas corta), pero armando las lineas en la GPU con un
kernel CUDA. El archivo que sale es identico byte a byte al de la CPU.

COMO SE PARALELIZA
Un hilo por BYTE de salida. Como todas las lineas miden lo mismo (ver
mezclador.py), cada hilo deduce de su indice que fila, que linea (A o B) y
que columna le toca, y de ahi que caracter escribir: la letra, un digito del
numero de fila, los dos puntos, una base o el salto de linea. Ningun hilo
depende de otro, y hilos consecutivos escriben bytes consecutivos, asi que
la escritura en la memoria de la tarjeta es perfectamente coalescida.

POR QUE VA POR LOTES
Con los genomas la salida pesa 7 GB y la tarjeta tiene 6. Se procesan lotes
de filas: se copian a la tarjeta los dos trozos de entrada, el kernel arma
las lineas y el resultado vuelve a la RAM, desde donde la CPU lo escribe en
su posicion del archivo. Con dos streams y dos juegos de buffers, mientras
la CPU escribe un lote la tarjeta ya esta armando el siguiente.

La ultima pareja de filas, que puede ser incompleta, la escribe la CPU
igual que en mezclador.py.

API publica:
    ruta_por_defecto(cadena_a, cadena_b)                -> ruta de salida
    mezclar_gpu(cadena_a, cadena_b, destino, lote_mb, bloques)
                                                        -> dict con resumen
"""

import os
import shutil
import time

import numpy as np

import mezclador
import motor_gpu
from comparador import ErrorEntrada, ANCHO_LINEA
from motor_gpu import cuda, HILOS_BLOQUE


if motor_gpu._HAY_CUDA:

    @cuda.jit
    def k_mezclar(a, b, salida, n, fila0, ancho, linea):
        """Arma n bytes de salida a partir de los lotes a y b.

        Parametros de dispositivo:
            a, b    : bases del lote, 80 por fila, de cada cadena.
            salida  : bytes de salida del lote (2 lineas por fila).
            n       : cuantos bytes de salida son utiles en este lote.
            fila0   : numero (desde 0) de la primera fila del lote.
            ancho   : cifras del numero de fila.
            linea   : bytes por linea, salto incluido.
        """
        j = cuda.grid(1)
        paso = cuda.gridsize(1)
        pareja = 2 * linea
        while j < n:
            fila = j // pareja
            resto = j - fila * pareja
            lado = resto // linea           # 0 = A, 1 = B
            col = resto - lado * linea

            if col == 0:
                c = 65 + lado               # 'A' o 'B'
            elif col == 1 or col == 3 + ancho:
                c = 32                      # ' '
            elif col < 2 + ancho:
                # Digito k (desde la izquierda) del numero de fila.
                numero = fila0 + fila + 1
                potencia = 1
                for _ in range(ancho - 1 - (col - 2)):
                    potencia *= 10
                if numero < potencia and potencia > 1:
                    c = 32                  # relleno a la izquierda
                else:
                    c = 48 + (numero // potencia) % 10
            elif col == 2 + ancho:
                c = 58                      # ':'
            elif col == linea - 1:
                c = 10                      # '\n'
            else:
                base = fila * ANCHO_LINEA + (col - 4 - ancho)
                if lado == 0:
                    c = a[base]
                else:
                    c = b[base]
            salida[j] = c
            j += paso


def ruta_por_defecto(cadena_a, cadena_b, carpeta=mezclador.CARPETA_SALIDA):
    """resultados/mezcla_<A>_<B>_GPU.txt."""
    base, ext = os.path.splitext(
        mezclador.ruta_por_defecto(cadena_a, cadena_b, carpeta))
    return base + '_GPU' + ext


def mezclar_gpu(cadena_a, cadena_b, destino=None, lote_mb=motor_gpu.LOTE_MB,
                bloques=motor_gpu.BLOQUES, progreso=None):
    """Genera el archivo mezclado con la GPU y devuelve un resumen.

    lote_mb es cuantos MB de CADA cadena se suben por viaje; bloques, cuantos
    bloques de 256 hilos se lanzan. Los dos se validan con las mismas
    funciones que el motor de comparacion y el ajuste se explica en 'aviso'.
    """
    if not motor_gpu.gpu_disponible():
        raise ErrorEntrada(motor_gpu.motivo_no_disponible())

    if destino is None:
        destino = ruta_por_defecto(cadena_a, cadena_b)

    filas, ancho, linea = mezclador._geometria(cadena_a.largo, cadena_b.largo)
    if filas == 0:
        raise ErrorEntrada('No hay filas que mezclar: una cadena esta vacia.')

    completas = filas - 1
    va, vb = cadena_a.datos, cadena_b.datos
    ultima = slice(completas * ANCHO_LINEA, (completas + 1) * ANCHO_LINEA)
    cola = (mezclador._linea('A', filas, ancho, va[ultima])
            + mezclador._linea('B', filas, ancho, vb[ultima]))
    total_bytes = completas * 2 * linea + len(cola)

    carpeta = os.path.dirname(os.path.abspath(destino))
    if not os.path.isdir(carpeta):
        os.makedirs(carpeta)
    libre = shutil.disk_usage(carpeta).free
    if total_bytes + mezclador._MARGEN_DISCO > libre:
        raise ErrorEntrada(
            'El archivo mezclado ocuparia %.1f MB y en el disco solo quedan '
            '%.1f MB libres.' % (total_bytes / 1048576.0, libre / 1048576.0))

    datos_gpu = motor_gpu.info_gpu()
    lote_mb, aviso_lote = motor_gpu.lote_valido(
        lote_mb, datos_gpu.get('vram_libre_mb'))
    bloques, aviso_bloques = motor_gpu.bloques_validos(bloques)
    filas_lote = max(1, (int(lote_mb) << 20) // ANCHO_LINEA)
    entrada_lote = filas_lote * ANCHO_LINEA
    salida_lote = filas_lote * 2 * linea

    # -- preparacion, fuera del cronometro --------------------------------
    inicio_preparacion = time.perf_counter()
    # Compilar el kernel con un lanzamiento minimo.
    d_mini = cuda.device_array(2 * linea, dtype=np.uint8)
    d_uno = cuda.to_device(np.zeros(ANCHO_LINEA, dtype=np.uint8))
    k_mezclar[1, 32](d_uno, d_uno, d_mini, 2 * linea, 0, ancho, linea)
    cuda.synchronize()
    del d_mini, d_uno

    n_buffers = 2
    h_a = [cuda.pinned_array(entrada_lote, dtype=np.uint8)
           for _ in range(n_buffers)]
    h_b = [cuda.pinned_array(entrada_lote, dtype=np.uint8)
           for _ in range(n_buffers)]
    h_out = [cuda.pinned_array(salida_lote, dtype=np.uint8)
             for _ in range(n_buffers)]
    d_a = [cuda.device_array(entrada_lote, dtype=np.uint8)
           for _ in range(n_buffers)]
    d_b = [cuda.device_array(entrada_lote, dtype=np.uint8)
           for _ in range(n_buffers)]
    d_out = [cuda.device_array(salida_lote, dtype=np.uint8)
             for _ in range(n_buffers)]
    streams = [cuda.stream() for _ in range(n_buffers)]
    preparacion = time.perf_counter() - inicio_preparacion

    temporal = destino + '.parcial'
    pendiente = [None] * n_buffers      # (posicion, bytes) por escribir
    try:
        with open(temporal, 'wb') as f:
            f.truncate(total_bytes)

            def volcar(turno):
                if pendiente[turno] is not None:
                    posicion, cuantos = pendiente[turno]
                    f.seek(posicion)
                    f.write(memoryview(h_out[turno][:cuantos]))
                    pendiente[turno] = None

            # -- medicion -------------------------------------------------
            arranque = time.perf_counter()
            turno = 0
            fila0 = 0
            while fila0 < completas:
                cuantas = min(filas_lote, completas - fila0)
                stream = streams[turno]

                # Esperar al lote anterior de este turno y escribirlo antes
                # de reutilizar sus buffers.
                stream.synchronize()
                volcar(turno)

                n_in = cuantas * ANCHO_LINEA
                n_out = cuantas * 2 * linea
                desde = fila0 * ANCHO_LINEA
                h_a[turno][:n_in] = va[desde:desde + n_in]
                h_b[turno][:n_in] = vb[desde:desde + n_in]
                d_a[turno][:n_in].copy_to_device(h_a[turno][:n_in],
                                                 stream=stream)
                d_b[turno][:n_in].copy_to_device(h_b[turno][:n_in],
                                                 stream=stream)

                malla = min(bloques,
                            (n_out + HILOS_BLOQUE - 1) // HILOS_BLOQUE)
                k_mezclar[malla, HILOS_BLOQUE, stream](
                    d_a[turno], d_b[turno], d_out[turno], n_out, fila0,
                    ancho, linea)
                d_out[turno][:n_out].copy_to_host(h_out[turno][:n_out],
                                                  stream=stream)
                pendiente[turno] = (fila0 * 2 * linea, n_out)

                fila0 += cuantas
                if progreso is not None:
                    progreso(fila0, filas)
                turno = (turno + 1) % n_buffers

            # Los dos ultimos lotes, en orden.
            for _ in range(n_buffers):
                streams[turno].synchronize()
                volcar(turno)
                turno = (turno + 1) % n_buffers

            f.seek(completas * 2 * linea)
            f.write(cola)
            transcurrido = time.perf_counter() - arranque

        if os.path.exists(destino):
            os.remove(destino)
        os.rename(temporal, destino)
    except Exception:
        if os.path.exists(temporal):
            try:
                os.remove(temporal)
            except OSError:
                pass
        raise
    finally:
        # Soltar la VRAM siempre, tambien si algo fallo a mitad.
        del d_a, d_b, d_out, h_a, h_b, h_out

    if progreso is not None:
        progreso(filas, filas)

    mb = total_bytes / 1048576.0
    avisos = [x for x in (aviso_lote, aviso_bloques) if x]
    return {
        'ruta': destino,
        'filas': filas,
        'filas_a': cadena_a.filas,
        'filas_b': cadena_b.filas,
        'sobrantes': abs(cadena_a.filas - cadena_b.filas),
        'bytes': total_bytes,
        'bloques': bloques,
        'lote_mb': lote_mb,
        'tiempo_s': round(transcurrido, 4),
        'preparacion_s': round(preparacion, 4),
        'mb_por_s': round(mb / transcurrido, 2) if transcurrido else 0.0,
        'aviso': ' '.join(avisos) or None,
    }


def formatear(resumen):
    """Texto para el registro de la ventana y la consola."""
    lineas = [
        'Archivo mezclado (GPU %d bloques, lote %d MB): %s'
        % (resumen['bloques'], resumen['lote_mb'], resumen['ruta']),
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
    import comparador

    if len(sys.argv) < 3:
        print('Uso: python mezclador_gpu.py <cadena_a.fna> <cadena_b.fna> '
              '[bloques] [lote_mb] [salida]')
        sys.exit(0)
    try:
        ca = comparador.preparar(sys.argv[1])
        cb = comparador.preparar(sys.argv[2])
        bl = int(sys.argv[3]) if len(sys.argv) > 3 else motor_gpu.BLOQUES
        lm = int(sys.argv[4]) if len(sys.argv) > 4 else motor_gpu.LOTE_MB
        salida = sys.argv[5] if len(sys.argv) > 5 else None
        print(formatear(mezclar_gpu(ca, cb, salida, lm, bl)))
    except ErrorEntrada as error:
        print('ERROR: %s' % error)
        sys.exit(1)
