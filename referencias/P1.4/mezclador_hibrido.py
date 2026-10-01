# -*- coding: utf-8 -*-
"""
mezclador_hibrido.py

La misma mezcla que mezclador.py y mezclador_gpu.py, pero con la CPU y la
GPU TRABAJANDO A LA VEZ sobre el mismo archivo. El archivo que sale es
identico byte a byte al de las otras dos versiones.

COMO SE DECIDE QUE LE TOCA A CADA UNA: REPARTO DINAMICO POR LOTES
Las filas se dividen en lotes de LOTE_FILAS filas y hay un contador
compartido con el siguiente lote libre, protegido por un cerrojo. Hay n
hilos que manejan un proceso de CPU cada uno y un hilo que maneja la GPU.
Cada uno, cuando termina un lote, pide el siguiente al contador. Nadie fija
el reparto de antemano: la plataforma mas rapida vuelve antes a pedir y se
lleva mas lotes. Al final se informa cuantas filas hizo cada una.

Es el mismo patron que el motor hibrido del P1.3 (contador compartido y un
hilo por plataforma), aplicado aqui a la mezcla.

POR QUE SE PUEDE ESCRIBIR EN CUALQUIER ORDEN
Todas las lineas miden lo mismo (ver mezclador.py), asi que el lote k va
siempre en la posicion k * LOTE_FILAS * 2 * largo_linea, lo haya armado la
CPU o la GPU y termine antes o despues que los demas.

API publica:
    ruta_por_defecto(cadena_a, cadena_b)                     -> ruta
    mezclar_hibrido(cadena_a, cadena_b, destino, procesos, bloques)
                                                              -> dict
"""

import multiprocessing
import os
import shutil
import threading
import time

import numpy as np

import mezclador
import mezclador_gpu
import motor_cpu
import motor_gpu
from comparador import ErrorEntrada, ANCHO_LINEA
from motor_gpu import cuda, HILOS_BLOQUE


# Filas por lote del reparto (unos 640 KB de cada cadena). Lotes pequenos
# para que haya muchos que repartir (320 en el par de 200 MB): con lotes
# grandes los hilos de CPU se llevaban casi todos al arrancar y la GPU se
# quedaba sin trabajo.
LOTE_FILAS = 8 * 1024

# Tope de procesos de CPU en el modo combinado. Medido sobre 200 MB: con 1-3
# procesos junto a la GPU el total baja a ~0,5 s; con 4 o mas sube a 0,8-1 s,
# porque los procesos de CPU compiten con la GPU por escribir en el disco y
# le quitan lotes que ella haria mas rapido.
PROCESOS_MAX = 2


def ruta_por_defecto(cadena_a, cadena_b, carpeta=mezclador.CARPETA_SALIDA):
    """resultados/mezcla_<A>_<B>_HIBRIDO.txt."""
    base, ext = os.path.splitext(
        mezclador.ruta_por_defecto(cadena_a, cadena_b, carpeta))
    return base + '_HIBRIDO' + ext


class _Reparto:
    """Contador compartido de lotes: entrega el siguiente rango de filas."""

    def __init__(self, completas):
        self.completas = completas
        self.siguiente = 0
        self.cerrojo = threading.Lock()
        self.parar = False

    def pedir(self):
        """Devuelve (inicio, fin) del siguiente lote, o None si no quedan."""
        with self.cerrojo:
            if self.parar or self.siguiente >= self.completas:
                return None
            inicio = self.siguiente
            self.siguiente = min(inicio + LOTE_FILAS, self.completas)
            return inicio, self.siguiente


def _hilo_cpu(reparto, pool, hechas, errores):
    """Maneja un proceso de CPU: pide lotes hasta que no queden."""
    try:
        while True:
            lote = reparto.pedir()
            if lote is None:
                return
            pool.apply(mezclador._escribir_filas, lote)
            hechas.append(lote[1] - lote[0])
    except Exception as error:                        # pragma: no cover
        reparto.parar = True
        errores.append(error)


def _hilo_gpu(reparto, recursos, va, vb, ruta, ancho, linea, bloques,
              hechas, errores):
    """Maneja la GPU: pide lotes y los arma con k_mezclar.

    Usa dos streams y dos juegos de buffers, como mezclador_gpu: mientras se
    escribe al disco un lote ya armado, la tarjeta arma el siguiente.
    """
    h_a, h_b, h_out, d_a, d_b, d_out, streams = recursos
    pendiente = [None, None]
    try:
        with open(ruta, 'r+b') as f:
            def volcar(turno):
                if pendiente[turno] is not None:
                    posicion, cuantos, filas = pendiente[turno]
                    f.seek(posicion)
                    f.write(memoryview(h_out[turno][:cuantos]))
                    hechas.append(filas)
                    pendiente[turno] = None

            turno = 0
            while True:
                stream = streams[turno]
                stream.synchronize()
                volcar(turno)

                lote = reparto.pedir()
                if lote is None:
                    break
                fila0, fin = lote
                cuantas = fin - fila0
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
                mezclador_gpu.k_mezclar[malla, HILOS_BLOQUE, stream](
                    d_a[turno], d_b[turno], d_out[turno], n_out, fila0,
                    ancho, linea)
                d_out[turno][:n_out].copy_to_host(h_out[turno][:n_out],
                                                  stream=stream)
                pendiente[turno] = (fila0 * 2 * linea, n_out, cuantas)
                turno = 1 - turno

            # El otro buffer puede tener todavia un lote sin escribir.
            for t in (0, 1):
                streams[t].synchronize()
                volcar(t)
    except Exception as error:                        # pragma: no cover
        reparto.parar = True
        errores.append(error)


def mezclar_hibrido(cadena_a, cadena_b, destino=None, procesos=None,
                    bloques=motor_gpu.BLOQUES, progreso=None):
    """Genera el archivo mezclado con CPU y GPU a la vez.

    procesos: procesos de CPU (se valida con motor_cpu.procesos_validos).
    bloques : bloques de GPU (se valida con motor_gpu.bloques_validos).
    Devuelve un dict con ruta, filas, tiempo y el reparto entre CPU y GPU.
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

    if procesos is None:
        _, procesos = motor_cpu.detectar_nucleos()
    procesos, aviso_cpu = motor_cpu.procesos_validos(procesos, total_bytes)
    if procesos > PROCESOS_MAX:
        aviso_cpu = ('En el modo CPU + GPU se usan %d procesos de CPU en vez '
                     'de %d: con mas, compiten con la GPU por el disco y el '
                     'total es mas lento.' % (PROCESOS_MAX, procesos))
        procesos = PROCESOS_MAX
    bloques, aviso_gpu = motor_gpu.bloques_validos(bloques)

    temporal = destino + '.parcial'
    pool = None
    recursos = None
    try:
        with open(temporal, 'wb') as f:
            f.truncate(total_bytes)

        # -- preparacion, fuera del cronometro ----------------------------
        inicio_preparacion = time.perf_counter()
        pool = multiprocessing.Pool(
            processes=procesos, initializer=mezclador._iniciar_worker,
            initargs=(cadena_a.ruta_seq, cadena_a.desplazamiento,
                      cadena_b.ruta_seq, cadena_b.desplazamiento,
                      temporal, ancho, linea))
        pool.map(mezclador._calentar, range(procesos * 3), chunksize=1)

        # Compilar el kernel con un lanzamiento minimo.
        d_uno = cuda.to_device(np.zeros(ANCHO_LINEA, dtype=np.uint8))
        d_mini = cuda.device_array(2 * linea, dtype=np.uint8)
        mezclador_gpu.k_mezclar[1, 32](d_uno, d_uno, d_mini, 2 * linea, 0,
                                       ancho, linea)
        cuda.synchronize()
        del d_uno, d_mini

        n_in = LOTE_FILAS * ANCHO_LINEA
        n_out = LOTE_FILAS * 2 * linea
        recursos = (
            [cuda.pinned_array(n_in, dtype=np.uint8) for _ in range(2)],
            [cuda.pinned_array(n_in, dtype=np.uint8) for _ in range(2)],
            [cuda.pinned_array(n_out, dtype=np.uint8) for _ in range(2)],
            [cuda.device_array(n_in, dtype=np.uint8) for _ in range(2)],
            [cuda.device_array(n_in, dtype=np.uint8) for _ in range(2)],
            [cuda.device_array(n_out, dtype=np.uint8) for _ in range(2)],
            [cuda.stream() for _ in range(2)],
        )
        preparacion = time.perf_counter() - inicio_preparacion

        # -- medicion -----------------------------------------------------
        reparto = _Reparto(completas)
        filas_cpu, filas_gpu, errores = [], [], []
        hilos = [threading.Thread(target=_hilo_cpu,
                                  args=(reparto, pool, filas_cpu, errores))
                 for _ in range(procesos)]
        hilos.append(threading.Thread(
            target=_hilo_gpu,
            args=(reparto, recursos, va, vb, temporal, ancho, linea,
                  bloques, filas_gpu, errores)))

        arranque = time.perf_counter()
        for hilo in hilos:
            hilo.start()
        while any(h.is_alive() for h in hilos):
            hilos[-1].join(0.1)
            if progreso is not None:
                progreso(sum(filas_cpu) + sum(filas_gpu), filas)
        for hilo in hilos:
            hilo.join()
        if errores:
            raise errores[0]

        # Los procesos tienen el archivo abierto: cerrarlos antes de
        # escribir la cola y renombrar.
        pool.close()
        pool.join()
        pool = None

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
    finally:
        del recursos

    if progreso is not None:
        progreso(filas, filas)

    hechas_cpu, hechas_gpu = sum(filas_cpu), sum(filas_gpu)
    mb = total_bytes / 1048576.0
    return {
        'ruta': destino,
        'filas': filas,
        'filas_a': cadena_a.filas,
        'filas_b': cadena_b.filas,
        'sobrantes': abs(cadena_a.filas - cadena_b.filas),
        'bytes': total_bytes,
        'procesos': procesos,
        'bloques': bloques,
        'filas_cpu': hechas_cpu,
        'filas_gpu': hechas_gpu,
        'lotes_cpu': len(filas_cpu),
        'lotes_gpu': len(filas_gpu),
        'tiempo_s': round(transcurrido, 4),
        'preparacion_s': round(preparacion, 4),
        'mb_por_s': round(mb / transcurrido, 2) if transcurrido else 0.0,
        'aviso': ' '.join(x for x in (aviso_cpu, aviso_gpu) if x) or None,
    }


def formatear(resumen):
    """Texto para el registro de la ventana y la consola."""
    total = max(1, resumen['filas_cpu'] + resumen['filas_gpu'])
    lineas = [
        'Archivo mezclado (CPU x%d + GPU %d bloques): %s'
        % (resumen['procesos'], resumen['bloques'], resumen['ruta']),
        '  %s parejas de filas, %.1f MB en %.3f s (%.1f MB/s), '
        'preparacion %.2f s'
        % (format(resumen['filas'], ',').replace(',', '.'),
           resumen['bytes'] / 1048576.0, resumen['tiempo_s'],
           resumen['mb_por_s'], resumen['preparacion_s']),
        '  Reparto: CPU %d lotes (%.1f %% de las filas), '
        'GPU %d lotes (%.1f %%)'
        % (resumen['lotes_cpu'], 100.0 * resumen['filas_cpu'] / total,
           resumen['lotes_gpu'], 100.0 * resumen['filas_gpu'] / total),
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

    multiprocessing.freeze_support()
    if len(sys.argv) < 3:
        print('Uso: python mezclador_hibrido.py <a.fna> <b.fna> '
              '[procesos] [bloques] [salida]')
        sys.exit(0)
    try:
        ca = comparador.preparar(sys.argv[1])
        cb = comparador.preparar(sys.argv[2])
        n = int(sys.argv[3]) if len(sys.argv) > 3 else None
        bl = int(sys.argv[4]) if len(sys.argv) > 4 else motor_gpu.BLOQUES
        salida = sys.argv[5] if len(sys.argv) > 5 else None
        print(formatear(mezclar_hibrido(ca, cb, salida, n, bl)))
    except ErrorEntrada as error:
        print('ERROR: %s' % error)
        sys.exit(1)
