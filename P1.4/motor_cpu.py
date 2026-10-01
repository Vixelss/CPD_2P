# -*- coding: utf-8 -*-
"""
motor_cpu.py

Comparacion de dos cadenas de ADN sobre los nucleos del procesador. Es la
columna "CPU" de la rubrica del P1.4.

QUE HACE
Recorre las dos secuencias a la vez y anota cada posicion en la que los
caracteres no coinciden. De cada diferencia sale su fila y su columna dentro
del FASTA, que es lo que pide el enunciado.

COMO SE PARALELIZA
La secuencia ya viene limpia y mapeada desde un archivo .seq (ver
comparador.preparar), de modo que el reparto es trivial: se corta el rango en
n tramos de tamano parecido y cada proceso compara el suyo. No hay costuras
que cuadrar como en los proyectos anteriores, porque aqui no se interpreta el
formato: el caracter de la posicion i de la cadena A se compara contra el de
la posicion i de la cadena B, y donde caiga el corte da igual.

POR QUE LOS HIJOS MAPEAN EL ARCHIVO EN VEZ DE RECIBIR LOS DATOS
En Windows multiprocessing usa 'spawn': cada argumento que se le pasa a un
proceso hijo se serializa con pickle y viaja por una tuberia. Mandar dos
cadenas de 3.3 GB a doce hijos serian 79 GB de copias antes de empezar a
trabajar.

Por eso a los hijos se les pasa solo la RUTA del .seq y su tamano. Cada uno
lo abre con numpy.memmap, que no copia nada: el sistema operativo hace que
las mismas paginas fisicas del archivo aparezcan en los doce procesos. Los
doce comparten un unico ejemplar de cada cadena.

POR QUE EL POOL SE CREA FUERA DEL CRONOMETRO
Comparar dos bytes es una instruccion. Sobre 200 MB la operacion completa
dura centesimas de segundo, mientras que arrancar doce procesos en Windows
cuesta del orden de medio segundo. Medido en este proyecto: con el arranque
dentro del reloj, la version de doce procesos salia ochenta veces MAS LENTA
que la de uno, lo que no dice nada sobre el paralelismo y todo sobre el coste
de 'spawn'.

Por eso el Pool se crea, se calienta con tareas vacias y se le deja mapear
los archivos ANTES de arrancar el reloj. El coste se publica aparte, en
'preparacion_s'. Es la misma decision que se tomo en el motor hibrido del
P1.3 y por el mismo motivo.

RESPUESTAS A LA RUBRICA, COLUMNA "CPU"
    LIBRERIA    multiprocessing (y numpy para la comparacion vectorizada)
    INSTRUCCION multiprocessing.Pool(processes=n).starmap(...)
    Ver evidencias.py, que es donde se recogen formalmente.

API publica:
    detectar_nucleos()                        -> (fisicos, logicos)
    comparar(a, b)                            -> (dict_resultado, tiempo)
    comparar_paralelo(a, b, procesos)         -> (dict_resultado, tiempo)
    comparar_rango(va, vb, inicio, fin)       -> posiciones distintas
"""

import multiprocessing
import os
import time

import numpy as np

import comparador
from comparador import ErrorEntrada


try:
    import psutil
    _HAY_PSUTIL = True
except ImportError:
    psutil = None
    _HAY_PSUTIL = False


# Tamano minimo de tramo por proceso. Por debajo de esto, mantener un proceso
# ocupado cuesta mas que el trabajo que se le da.
_MIN_TRAMO = 4 << 20        # 4 MB

# Trozo con el que se recorre cada tramo dentro de un proceso. Acota la
# memoria de trabajo: el vector booleano intermedio nunca pasa de este
# tamano, aunque se comparen dos genomas de 3.3 GB.
_TROZO_COMPARACION = 64 << 20   # 64 MB

# Estado que el inicializador del Pool deja en cada proceso hijo.
_estado = {}


# ---------------------------------------------------------------------------
# Deteccion de hardware
# ---------------------------------------------------------------------------

def detectar_nucleos():
    """Devuelve la tupla (nucleos_fisicos, nucleos_logicos).

    Usa psutil si esta instalado. Si no lo esta, cae a la libreria estandar y
    estima los fisicos como la mitad de los logicos cuando hay indicios de
    hyper-threading.
    """
    logicos = os.cpu_count() or 1
    fisicos = None

    if _HAY_PSUTIL:
        try:
            fisicos = psutil.cpu_count(logical=False)
            logicos = psutil.cpu_count(logical=True) or logicos
        except Exception:
            fisicos = None

    if not fisicos:
        fisicos = logicos // 2 if logicos > 1 and logicos % 2 == 0 else logicos

    return int(fisicos), int(logicos)


def procesos_validos(pedidos, largo=None):
    """Ajusta un numero de procesos a lo que esta maquina puede dar de si.

    El enunciado habla de "CPU n nucleos", asi que n es un parametro que el
    usuario elige. Pero pedir mas procesos que nucleos logicos no acelera
    nada: los procesos de sobra se turnan en los mismos nucleos y solo anaden
    cambios de contexto. Y pedir mas procesos que tramos de trabajo deja
    procesos arrancados sin nada que hacer.

    Por eso el numero se recorta en los dos extremos. Devuelve
    (procesos_ajustados, motivo) donde motivo es None si no hubo que tocar
    nada, o un texto explicando el ajuste para mostrarselo al usuario.
    """
    _, logicos = detectar_nucleos()

    try:
        pedidos = int(pedidos)
    except (TypeError, ValueError):
        return logicos, ('"%s" no es un numero de procesos valido; se usan '
                         'los %d nucleos logicos.' % (pedidos, logicos))

    if pedidos < 1:
        return 1, ('No se puede trabajar con %d procesos; se usa 1.'
                   % pedidos)

    if pedidos > logicos:
        return logicos, (
            'Este procesador tiene %d nucleos logicos, asi que %d procesos '
            'no cabrian: se turnarian en los mismos nucleos y solo anadirian '
            'cambios de contexto. Se usan %d.'
            % (logicos, pedidos, logicos))

    if largo is not None and largo > 0:
        maximo = max(1, largo // _MIN_TRAMO)
        if pedidos > maximo:
            return maximo, (
                'Las cadenas son de %.1f MB: con %d procesos cada uno '
                'recibiria menos de %d MB y arrancarlo costaria mas que el '
                'trabajo. Se usan %d.'
                % (largo / (1024.0 * 1024.0), pedidos,
                   _MIN_TRAMO // (1024 * 1024), maximo))

    return pedidos, None


# ---------------------------------------------------------------------------
# Nucleo de comparacion
# ---------------------------------------------------------------------------

def comparar_rango(va, vb, inicio, fin, tope=None):
    """Cuenta y localiza las diferencias entre va y vb en [inicio, fin).

    Devuelve (cuantas, primeras_posiciones, tramos). El conteo es SIEMPRE
    exacto; la lista de posiciones se corta en 'tope' elementos, y 'tramos'
    es una lista de (byte_inicial, diferencias_en_ese_trozo).

    'tramos' sale gratis: el rango ya se recorre por trozos, asi que solo hay
    que anotar el conteo de cada uno. Sirve para dibujar en que zona del
    archivo se concentran las diferencias, que es la unica grafica del
    proyecto que habla del problema y no del hardware. Se calcula asi y no a
    partir de la lista de posiciones porque esa esta cortada en las primeras
    mil: dibujarla daria la impresion falsa de que todas las diferencias
    estan al principio.

    POR QUE SE CORTA LA LISTA Y SE TRABAJA POR TROZOS
    Los dos genomas del proyecto divergen a partir de cierto punto, y al
    compararlos enteros aparecen mil millones de diferencias. Guardarlas
    todas serian 8.6 GB de indices, y devolverlas desde un proceso hijo
    reventaba la tuberia de multiprocessing con un error de Windows
    ('el parametro no es correcto', al intentar enviar 5.3 GB de una vez).

    Ademas nadie necesita mil millones de posiciones: el informe publica el
    total y las primeras. Por eso el rango se recorre en trozos acotados, se
    va sumando el conteo y solo se conservan las primeras 'tope' posiciones.
    El consumo de memoria queda acotado sea cual sea el tamano de la entrada,
    que es lo que hace que el programa no se caiga con los archivos grandes.

    Las posiciones son absolutas, ya desplazadas por 'inicio', para que el
    padre pueda concatenar los tramos sin corregir nada.
    """
    if tope is None:
        tope = comparador.MAX_DETALLE

    cuantas = 0
    recogidas = []
    tramos = []
    faltan = max(0, int(tope))

    for arranque in range(inicio, fin, _TROZO_COMPARACION):
        remate = min(arranque + _TROZO_COMPARACION, fin)
        distintos = va[arranque:remate] != vb[arranque:remate]

        # count_nonzero no reserva memoria; flatnonzero si. Se pregunta
        # primero cuantas hay y solo se piden las posiciones si de verdad
        # hacen falta para completar el tope.
        aqui = int(np.count_nonzero(distintos))
        cuantas += aqui
        tramos.append((arranque, aqui))

        if aqui and faltan:
            posiciones = np.flatnonzero(distintos[:]) + arranque
            recorte = posiciones[:faltan]
            recogidas.append(recorte)
            faltan -= recorte.size

    if recogidas:
        primeras = np.concatenate(recogidas).astype(np.int64)
    else:
        primeras = np.empty(0, dtype=np.int64)

    return cuantas, primeras, tramos


def _iniciar_worker(ruta_a, desp_a, ruta_b, desp_b, largo):
    """Inicializador del Pool: mapea los dos archivos .seq en el hijo.

    El hijo no recibe los datos, recibe las rutas. Mapear es practicamente
    gratis y no duplica memoria: las paginas son las mismas para todos.

    Los desplazamientos permiten comparar dos TRAMOS que empiezan en sitios
    distintos de sus respectivos archivos, que es lo que hace falta cuando
    las mismas secuencias estan guardadas en distinto orden. Se mapea el
    archivo completo y se recorta la vista; recortar un mapeo no copia nada.
    """
    entero_a = np.memmap(ruta_a, dtype=np.uint8, mode='r')
    entero_b = np.memmap(ruta_b, dtype=np.uint8, mode='r')
    _estado['a'] = entero_a[desp_a:desp_a + largo]
    _estado['b'] = entero_b[desp_b:desp_b + largo]


def _trabajo(inicio, fin, tope):
    """Tarea de un proceso hijo: comparar su tramo y devolver el hallazgo.

    Devuelve (cuantas, primeras_posiciones, tramos). Lo que viaja de vuelta
    al padre son un entero, como mucho 'tope' indices y un par de numeros por
    cada trozo de 64 MB; nunca el tramo comparado ni la lista completa de
    diferencias.
    """
    return comparar_rango(_estado['a'], _estado['b'], inicio, fin, tope)


def _calentar(_):
    """Tarea vacia que obliga a un proceso hijo a terminar de arrancar.

    En Windows, Pool() vuelve enseguida pero los hijos todavia tienen que
    levantar un interprete nuevo, reimportar los modulos y mapear los
    archivos. Si eso cayera dentro del cronometro, sobre una operacion que
    dura milisegundos lo taparia por completo.
    """
    return True


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def comparar(cadena_a, cadena_b, detalle=None, progreso=None):
    """Compara las dos cadenas en un solo proceso. Linea base del speedup.

    Devuelve (dict_resultado, tiempo_en_segundos).
    """
    largo = min(cadena_a.largo, cadena_b.largo)
    tope = comparador.MAX_DETALLE if detalle is None else max(0, detalle)

    arranque = time.perf_counter()
    total, posiciones, tramos = comparar_rango(cadena_a.datos,
                                               cadena_b.datos, 0, largo,
                                               tope)
    transcurrido = time.perf_counter() - arranque

    if progreso is not None:
        progreso(largo, largo)

    resultado = comparador.resumir(total, posiciones, cadena_a, cadena_b,
                                   largo, 'CPU x1', detalle=detalle)
    resultado['reparto'] = sorted(tramos)
    resultado['cpu'] = {'procesos': 1, 'preparacion_s': 0.0}
    return resultado, transcurrido


def comparar_paralelo(cadena_a, cadena_b, procesos=None, detalle=None,
                      progreso=None):
    """Compara las dos cadenas repartiendo el trabajo entre varios procesos.

    Parametros:
        cadena_a, cadena_b : objetos Cadena de comparador.preparar().
        procesos           : numero de procesos; por defecto los logicos.
                             Se ajusta solo si el valor no es viable.
        detalle            : cuantas diferencias guardar con fila y columna.
        progreso           : funcion opcional progreso(hechos, total).

    Devuelve (dict_resultado, tiempo_en_segundos). El tiempo mide solo la
    comparacion; el arranque de los procesos se publica en 'preparacion_s'.
    """
    largo = min(cadena_a.largo, cadena_b.largo)
    if largo == 0:
        raise ErrorEntrada('No hay nada que comparar: una cadena esta vacia.')

    if procesos is None:
        _, procesos = detectar_nucleos()
    procesos, aviso = procesos_validos(procesos, largo)

    if procesos <= 1:
        resultado, tiempo = comparar(cadena_a, cadena_b, detalle=detalle,
                                     progreso=progreso)
        if aviso:
            resultado['cpu']['aviso'] = aviso
        return resultado, tiempo

    # Cortes con enteros, para que cubran el rango entero sin huecos.
    tope = comparador.MAX_DETALLE if detalle is None else max(0, detalle)
    limites = [(largo * i) // procesos for i in range(procesos + 1)]
    tareas = [(limites[i], limites[i + 1], tope) for i in range(procesos)]

    # -- preparacion, deliberadamente fuera del cronometro ----------------
    inicio_preparacion = time.perf_counter()
    pool = multiprocessing.Pool(
        processes=procesos,
        initializer=_iniciar_worker,
        initargs=(cadena_a.ruta_seq, cadena_a.desplazamiento,
                  cadena_b.ruta_seq, cadena_b.desplazamiento, largo))
    try:
        pool.map(_calentar, range(procesos * 3), chunksize=1)
        preparacion = time.perf_counter() - inicio_preparacion

        # -- medicion -----------------------------------------------------
        arranque = time.perf_counter()
        if progreso is None:
            # Sin barra de progreso se espera de golpe. Sondear el resultado
            # cada pocos milisegundos tiene un coste que aqui si se nota: la
            # comparacion dura del orden de diez milisegundos, asi que un
            # sondeo de diez le anadiria un cincuenta por ciento de tiempo
            # que no es trabajo.
            partes = pool.starmap(_trabajo, tareas)
        else:
            pendiente = pool.starmap_async(_trabajo, tareas)
            while not pendiente.ready():
                pendiente.wait(0.05)
                progreso(0, largo)
            partes = pendiente.get()
        transcurrido = time.perf_counter() - arranque
    finally:
        pool.close()
        pool.join()

    # Los tramos vienen en orden y cada uno trae su conteo exacto y sus
    # primeras posiciones, absolutas. Sumar los conteos da el total exacto;
    # concatenar y recortar da las primeras diferencias del conjunto, porque
    # los tramos estan ordenados de izquierda a derecha.
    total = sum(parte[0] for parte in partes)
    trozos = [parte[1] for parte in partes if parte[1].size]
    if trozos:
        posiciones = np.concatenate(trozos)[:tope]
    else:
        posiciones = np.empty(0, dtype=np.int64)

    # Los tramos de todos los procesos, ordenados por posicion: es el reparto
    # de las diferencias a lo largo del archivo.
    reparto = sorted(t for parte in partes for t in parte[2])

    if progreso is not None:
        progreso(largo, largo)

    resultado = comparador.resumir(total, posiciones, cadena_a, cadena_b,
                                   largo, 'CPU x%d' % procesos,
                                   detalle=detalle)
    resultado['reparto'] = reparto
    resultado['cpu'] = {
        'procesos': procesos,
        'preparacion_s': round(preparacion, 4),
    }
    if aviso:
        resultado['cpu']['aviso'] = aviso
    return resultado, transcurrido


# ---------------------------------------------------------------------------
# Prueba rapida por linea de comandos
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    multiprocessing.freeze_support()

    fisicos, logicos = detectar_nucleos()
    print('Nucleos fisicos: %d   Nucleos logicos: %d' % (fisicos, logicos))

    if len(sys.argv) < 3:
        print('')
        print('Uso: python motor_cpu.py <cadena_a.fna> <cadena_b.fna> '
              '[procesos]')
        sys.exit(0)

    try:
        print('')
        print('Preparando las dos cadenas (fuera de la medicion)...')
        ca = comparador.preparar(sys.argv[1])
        cb = comparador.preparar(sys.argv[2])
        for aviso in comparador.validar_par(ca, cb):
            print('  AVISO: %s' % aviso)
    except ErrorEntrada as error:
        print('')
        print('ERROR: %s' % error)
        sys.exit(1)

    print('  A: %d bases  (%s)'
          % (ca.largo, 'cache' if ca.desde_cache
             else '%.2f s' % ca.preparacion_s))
    print('  B: %d bases  (%s)'
          % (cb.largo, 'cache' if cb.desde_cache
             else '%.2f s' % cb.preparacion_s))

    n = int(sys.argv[3]) if len(sys.argv) > 3 else logicos
    mb = min(ca.largo, cb.largo) / (1024.0 * 1024.0)

    print('')
    res, t = comparar(ca, cb)
    print('CPU x1     : %8.3f s  (%8.1f MB/s)' % (t, mb / t if t else 0))

    res, t = comparar_paralelo(ca, cb, procesos=n)
    print('CPU x%-2d    : %8.3f s  (%8.1f MB/s)   preparacion %.2f s'
          % (res['cpu']['procesos'], t, mb / t if t else 0,
             res['cpu']['preparacion_s']))
    if res['cpu'].get('aviso'):
        print('  AVISO: %s' % res['cpu']['aviso'])

    print('')
    print(comparador.formatear_diferencias(res))
