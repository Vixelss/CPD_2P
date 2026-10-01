# -*- coding: utf-8 -*-
"""
motor_hibrido.py

CPU y GPU trabajando en paralelo sobre el mismo archivo, que es justo lo que
pide el P1.3. No es "primero la CPU y luego la GPU, y comparamos": eso fue el
P1.2. Aqui las dos plataformas procesan a la vez tramos distintos del mismo
FASTA y al final se suman sus histogramas.

COMO SE REPARTE EL TRABAJO
El archivo se corta en trozos de tamano fijo, numerados del 0 en adelante.
Hay un unico contador compartido que dice cual es el siguiente trozo sin
asignar. Cada trabajador, sea un proceso de CPU o el hilo que conduce la
GPU, repite siempre el mismo ciclo: reclama trabajo, lo procesa y vuelve a
por mas, hasta que se acaba.

Se hace asi y no repartiendo el archivo por porcentajes al empezar porque un
reparto fijo obliga a saber de antemano cuanto mas rapida es una plataforma
que la otra, y eso no se sabe: depende del tamano del archivo, de si viene
de cache, de lo que este haciendo el resto del sistema. Con un reparto fijo
mal calibrado el conjunto va mas lento que la GPU sola, porque la GPU acaba
su mitad y se queda parada esperando a la CPU.

LA RACION ADAPTATIVA
Que cada trabajador pida un trozo suelto no basta, y se comprobo midiendo:
con doce procesos de CPU y un hilo de GPU, los trece piden a la vez en el
instante cero y se reparten trece trozos antes de que nadie haya demostrado
nada. Si el archivo da dieciseis trozos, el reparto ya esta decidido por el
orden de llegada y no por la velocidad de cada uno.

Por eso cada trabajador no pide un trozo sino una racion, y ajusta su tamano
segun lo que acaba de medir de si mismo: tras cada entrega calcula su propia
velocidad y pide la cantidad de trozos que espera despachar en el proximo
cuarto de segundo. Todos empiezan pidiendo uno, en igualdad de condiciones.
La GPU, que despacha unos mil MB por segundo, se estabiliza pidiendo raciones
grandes; cada proceso de CPU, que va muy por debajo, se queda pidiendo de uno
en uno. Nadie fija esa proporcion en el codigo: sale de la medicion, se
readapta sola si el archivo viene de cache o si el sistema se carga, y tiene
el efecto secundario de que la GPU recibe tramos largos y contiguos, que es
justo como mejor aprovecha la transferencia por PCIe.

POR QUE ESTE MODULO NO IMPORTA motor_gpu ARRIBA
En Windows multiprocessing usa el metodo 'spawn': cada proceso hijo arranca
un interprete nuevo y reimporta el modulo donde vive la funcion que va a
ejecutar, es decir este. Si motor_gpu estuviera en los imports de la
cabecera, los doce procesos de CPU cargarian numba y todo el stack de CUDA
para no usarlo jamas. En el P1.2 se midio el efecto de ese import de mas:
contar_paralelo sobre un archivo de 2 MB pasaba de 1.34 s a 5.51 s. Aqui el
dano seria doble, porque ese retraso se le cargaria a la mitad de CPU del
motor y falsearia el reparto. Por eso motor_gpu se importa dentro de
contar_hibrido, que los hijos no ejecutan nunca.

API publica:
    contar_hibrido(ruta, ...)   -> (dict_resultado, tiempo)
    resumen_reparto(registro)   -> quien proceso cuanto
"""

import math
import multiprocessing
import os
import threading
import time

import numpy as np

import motor_cpu
from motor_cpu import BINS, resumir_histograma


# Tamano por defecto de cada trozo repartible. Es el grano del reparto: con
# trozos muy grandes la ultima asignacion deja a alguien trabajando solo
# mientras el resto ya termino, y con trozos muy pequenos se paga demasiadas
# veces el coste fijo de abrir el archivo y posicionarse.
#
# Se eligio 16 MB despues de medir el problema contrario: con trozos de 64 MB
# un archivo de 500 MB solo da ocho piezas, y trece trabajadores pidiendo a
# la vez las agotan en la primera ronda. El reparto necesita bastantes mas
# piezas que trabajadores para que la velocidad de cada uno llegue a notarse.
# Que el trozo sea pequeno no penaliza a la GPU porque la racion adaptativa
# le junta varios consecutivos en una sola peticion.
TROZO_MB = 16

# Cuanto trabajo pide cada trabajador de una vez: el que espera despachar en
# este tiempo, segun la velocidad que acaba de medirse a si mismo. Un cuarto
# de segundo es el compromiso entre reaccionar rapido a un cambio de ritmo y
# no volver al contador compartido cada dos por tres.
OBJETIVO_RACION_S = 0.25

# Techo de la racion. Evita que un trabajador muy rapido se lleve de golpe
# tanto archivo que el resto se quede sin nada que hacer al final.
MAX_RACION = 32

# Estado que el inicializador del Pool deja en cada proceso hijo.
_estado = {}


# ---------------------------------------------------------------------------
# Reparto
# ---------------------------------------------------------------------------

def _reclamar(contador, n_trozos, cuantos=1):
    """Reserva hasta 'cuantos' trozos consecutivos para quien llama.

    Devuelve (primer_trozo, cuantos_concedidos), o (-1, 0) si ya no queda
    trabajo. Los trozos concedidos son siempre contiguos, de modo que quien
    recibe una racion grande puede recorrerla como un unico tramo de archivo
    en vez de como varias lecturas sueltas.

    Es el unico punto de sincronizacion entre la CPU y la GPU en todo el
    motor. El lock protege un incremento de un entero, asi que aunque lo
    peleen trece trabajadores a la vez el coste es irrelevante frente a los
    megabytes que cuesta atender cada peticion.
    """
    with contador.get_lock():
        primero = contador.value
        if primero >= n_trozos:
            return -1, 0
        ultimo = min(primero + max(1, cuantos), n_trozos)
        contador.value = ultimo
        return primero, ultimo - primero


def _limites(indice, cuantos, tam_trozo, tamano):
    """Rango de bytes [inicio, fin) de una racion de trozos consecutivos."""
    inicio = indice * tam_trozo
    return inicio, min((indice + cuantos) * tam_trozo, tamano)


def _siguiente_racion(bytes_hechos, segundos, tam_trozo):
    """Cuantos trozos pedir la proxima vez, segun la velocidad recien medida.

    Se convierte el ritmo observado en la cantidad de trabajo que cabe en
    OBJETIVO_RACION_S. Un trabajador lento se queda en uno; uno rapido sube
    hasta el techo. Si la medicion todavia no es fiable, se pide uno.
    """
    if segundos <= 0 or bytes_hechos <= 0 or tam_trozo <= 0:
        return 1
    velocidad = bytes_hechos / segundos
    trozos = int(velocidad * OBJETIVO_RACION_S) // tam_trozo
    return max(1, min(MAX_RACION, trozos))


def _iniciar_worker(ruta, contador, avance, n_trozos, tam_trozo, tamano, reloj):
    """Inicializador del Pool: deja en el hijo todo lo que necesita saber.

    'reloj' es un valor compartido donde el padre escribe el instante exacto
    del arranque. Se pasa asi y no como numero suelto porque los procesos se
    crean antes de que empiece la medicion, de modo que cuando este
    inicializador corre todavia no existe ese instante.
    """
    _estado.update({
        'ruta': ruta, 'contador': contador, 'avance': avance,
        'n_trozos': n_trozos, 'tam_trozo': tam_trozo,
        'tamano': tamano, 'reloj': reloj,
    })


def _calentar(_):
    """Tarea vacia que obliga a un proceso hijo a terminar de arrancar.

    En Windows, Process.start() vuelve enseguida pero el hijo todavia tiene
    que levantar un interprete nuevo y reimportar los modulos, lo que lleva
    unas decimas de segundo. Si no se espera a que todos esten listos, la GPU
    arranca sola y se lleva los primeros trozos por pura ventaja de salida,
    que es un artefacto del arranque y no una diferencia de rendimiento.
    """
    time.sleep(0.02)
    return True


def _trabajo_cpu(etiqueta):
    """Bucle de un proceso de CPU: reclamar trozo, contarlo, repetir.

    Devuelve su histograma parcial de 256 enteros y el registro de que
    trozos atendio y cuando. La secuencia nunca viaja entre procesos: cada
    hijo abre el archivo por su cuenta y solo manda numeros de vuelta.
    """
    ruta = _estado['ruta']
    contador = _estado['contador']
    avance = _estado['avance']
    n_trozos = _estado['n_trozos']
    tam_trozo = _estado['tam_trozo']
    tamano = _estado['tamano']
    t0 = _estado['reloj'].value

    total = np.zeros(BINS, dtype=np.int64)
    registro = []
    bytes_hechos = 0
    trozos_hechos = 0
    racion = 1

    while True:
        indice, cuantos = _reclamar(contador, n_trozos, racion)
        if indice < 0:
            break

        inicio, fin = _limites(indice, cuantos, tam_trozo, tamano)
        marca = time.perf_counter() - t0
        total += motor_cpu.contar_rango(ruta, inicio, fin)
        cierre = time.perf_counter() - t0

        procesados = fin - inicio
        bytes_hechos += procesados
        trozos_hechos += cuantos
        registro.append({
            'trozo': indice, 'trozos': cuantos, 'plataforma': 'CPU',
            'trabajador': etiqueta, 'inicio_s': round(marca, 4),
            'fin_s': round(cierre, 4), 'bytes': procesados,
        })

        with avance.get_lock():
            avance.value += procesados

        racion = _siguiente_racion(procesados, cierre - marca, tam_trozo)

    return {
        'histograma': [int(x) for x in total],
        'trozos': trozos_hechos,
        'bytes': bytes_hechos,
        'registro': registro,
    }


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def contar_hibrido(ruta, procesos=None, trozo_mb=TROZO_MB, lote_mb=64,
                   usar_cpu=True, usar_gpu=True, progreso=None):
    """Cuenta el archivo con la CPU y la GPU trabajando a la vez.

    Parametros:
        ruta      : archivo FASTA a procesar.
        procesos  : procesos de CPU; por defecto, los nucleos logicos.
        trozo_mb  : grano del reparto, en MB.
        lote_mb   : tamano de lote que usa la GPU dentro de cada trozo.
        usar_cpu  : si es False, solo trabaja la GPU.
        usar_gpu  : si es False, solo trabaja la CPU.
        progreso  : funcion opcional progreso(bytes_hechos, bytes_totales).

    Devuelve (dict_resultado, tiempo_en_segundos). El diccionario trae ademas
    la clave 'hibrido' con el reparto que salio, el censo por SM de la GPU y
    el registro trozo a trozo, que es lo que permite dibujar la linea de
    tiempo y demostrar que las dos plataformas estuvieron activas a la vez.

    El tiempo incluye la creacion de los procesos de CPU, igual que en
    motor_cpu.contar_paralelo: es un coste real de paralelizar y dejarlo
    fuera haria que el motor hibrido pareciese mejor de lo que es.
    """
    tamano = os.path.getsize(ruta)
    tam_trozo = max(1, int(trozo_mb)) * 1024 * 1024
    n_trozos = max(1, math.ceil(tamano / tam_trozo))

    if procesos is None:
        _, procesos = motor_cpu.detectar_nucleos()
    procesos = max(1, int(procesos))

    # Import tardio a proposito: ver la explicacion de la cabecera.
    contexto = None
    if usar_gpu:
        import motor_gpu
        if motor_gpu.gpu_disponible():
            # La compilacion de los kernels queda fuera del cronometro.
            motor_gpu.precalentar()
        else:
            usar_gpu = False

    if not usar_cpu and not usar_gpu:
        raise ValueError('Hay que dejar trabajar al menos a una plataforma.')

    contador = multiprocessing.Value('q', 0)
    avance = multiprocessing.Value('q', 0)
    reloj = multiprocessing.Value('d', 0.0)

    total_gpu = {'histograma': None, 'trozos': 0, 'bytes': 0, 'registro': [],
                 'censo_sm': [], 'error': None}

    def conducir_gpu():
        """Hilo que mantiene la tarjeta pidiendo trozos hasta agotarlos.

        Va en un hilo y no en un proceso porque el contexto de CUDA pertenece
        al proceso que lo creo y no se puede compartir con un hijo. Como el
        hilo pasa casi todo su tiempo esperando a la tarjeta, el GIL no
        estorba: lo suelta en cada llamada al driver.
        """
        nonlocal contexto
        try:
            import motor_gpu
            t0 = reloj.value
            racion = 1
            while True:
                indice, cuantos = _reclamar(contador, n_trozos, racion)
                if indice < 0:
                    break

                inicio, fin = _limites(indice, cuantos, tam_trozo, tamano)
                marca = time.perf_counter() - t0
                contexto.procesar_rango(ruta, inicio, fin)
                # Se espera a la tarjeta antes de anotar el cierre porque los
                # kernels se encolan sin bloquear: sin esta espera el tiempo
                # apuntado seria el de encolar, no el de calcular.
                contexto.sincronizar()
                cierre = time.perf_counter() - t0

                procesados = fin - inicio
                total_gpu['bytes'] += procesados
                total_gpu['trozos'] += cuantos
                total_gpu['registro'].append({
                    'trozo': indice, 'trozos': cuantos, 'plataforma': 'GPU',
                    'trabajador': 'GPU', 'inicio_s': round(marca, 4),
                    'fin_s': round(cierre, 4), 'bytes': procesados,
                })

                with avance.get_lock():
                    avance.value += procesados

                racion = _siguiente_racion(procesados, cierre - marca,
                                           tam_trozo)

            total_gpu['histograma'] = contexto.histograma()
            total_gpu['censo_sm'] = contexto.censo_sm()
        except Exception as error:                    # pragma: no cover
            total_gpu['error'] = '%s: %s' % (type(error).__name__, error)
        finally:
            if contexto is not None:
                contexto.cerrar()

    # -- preparacion, deliberadamente fuera del cronometro ----------------
    # Levantar doce procesos en Windows cuesta del orden de medio segundo, y
    # reservar los buffers de pagina bloqueada de la tarjeta otro tanto. Si
    # ese arranque cayera dentro de la medicion, la plataforma que estuviera
    # lista antes se llevaria los primeros trozos por ventaja de salida y el
    # reparto medido no diria nada sobre el rendimiento de cada una. Es el
    # mismo criterio con el que se saca la compilacion de los kernels.
    inicio_preparacion = time.perf_counter()

    pool = None
    if usar_cpu:
        pool = multiprocessing.Pool(
            processes=procesos,
            initializer=_iniciar_worker,
            initargs=(ruta, contador, avance, n_trozos, tam_trozo,
                      tamano, reloj))
        # Tareas vacias para obligar a todos los hijos a terminar de nacer.
        pool.map(_calentar, range(procesos * 3), chunksize=1)

    if usar_gpu:
        import motor_gpu
        contexto = motor_gpu.ContextoGPU(lote_mb=lote_mb)

    preparacion = time.perf_counter() - inicio_preparacion

    # -- linea de salida comun -------------------------------------------
    partes_cpu = []
    t0 = time.perf_counter()
    reloj.value = t0

    hilo_gpu = None
    if usar_gpu:
        hilo_gpu = threading.Thread(target=conducir_gpu, daemon=True)
        hilo_gpu.start()

    try:
        if pool is not None:
            # Se lanza una tarea por proceso y cada una se queda en su bucle
            # reclamando trozos. No se reparten tareas por adelantado a
            # proposito: el reparto tiene que salir de quien vuelve antes.
            pendiente = pool.starmap_async(_trabajo_cpu,
                                           [('cpu%d' % i,)
                                            for i in range(procesos)])

            while not pendiente.ready():
                pendiente.wait(0.1)
                if progreso is not None:
                    progreso(avance.value, tamano)

            partes_cpu = pendiente.get()

        if hilo_gpu is not None:
            while hilo_gpu.is_alive():
                hilo_gpu.join(0.1)
                if progreso is not None:
                    progreso(avance.value, tamano)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    transcurrido = time.perf_counter() - t0

    if total_gpu['error']:
        raise RuntimeError('Fallo el hilo de GPU: %s' % total_gpu['error'])

    if progreso is not None:
        progreso(tamano, tamano)

    # Suma de todos los histogramas parciales: los de cada proceso de CPU y
    # el que la GPU acumulo dentro de la tarjeta.
    total = [0] * BINS
    for parte in partes_cpu:
        for i in range(BINS):
            total[i] += parte['histograma'][i]
    if total_gpu['histograma'] is not None:
        for i in range(BINS):
            total[i] += int(total_gpu['histograma'][i])

    registro = []
    for parte in partes_cpu:
        registro.extend(parte['registro'])
    registro.extend(total_gpu['registro'])
    registro.sort(key=lambda r: r['inicio_s'])

    bytes_cpu = sum(p['bytes'] for p in partes_cpu)
    trozos_cpu = sum(p['trozos'] for p in partes_cpu)

    resultado = resumir_histograma(total)
    resultado['hibrido'] = {
        'trozo_mb': trozo_mb,
        'lote_mb': lote_mb,
        'preparacion_s': round(preparacion, 4),
        'procesos': procesos if usar_cpu else 0,
        'n_trozos': n_trozos,
        'trozos_cpu': trozos_cpu,
        'trozos_gpu': total_gpu['trozos'],
        'bytes_cpu': bytes_cpu,
        'bytes_gpu': total_gpu['bytes'],
        'censo_sm': total_gpu['censo_sm'],
        'registro': registro,
        'solape_s': solape(registro),
    }
    return resultado, transcurrido


# ---------------------------------------------------------------------------
# Lectura del reparto
# ---------------------------------------------------------------------------

def solape(registro):
    """Segundos durante los cuales la CPU y la GPU estuvieron activas a la vez.

    Es la cifra que demuestra literalmente el enunciado del proyecto. Se
    construye la union de los intervalos de cada plataforma por separado y se
    mide cuanto se cruzan. Si saliera cero, el motor estaria alternando en
    vez de trabajar en paralelo.
    """
    def union(plataforma):
        tramos = sorted((r['inicio_s'], r['fin_s']) for r in registro
                        if r['plataforma'] == plataforma)
        unidos = []
        for inicio, fin in tramos:
            if unidos and inicio <= unidos[-1][1]:
                unidos[-1][1] = max(unidos[-1][1], fin)
            else:
                unidos.append([inicio, fin])
        return unidos

    cpu = union('CPU')
    gpu = union('GPU')
    if not cpu or not gpu:
        return 0.0

    total = 0.0
    i = j = 0
    while i < len(cpu) and j < len(gpu):
        inicio = max(cpu[i][0], gpu[j][0])
        fin = min(cpu[i][1], gpu[j][1])
        if fin > inicio:
            total += fin - inicio
        if cpu[i][1] < gpu[j][1]:
            i += 1
        else:
            j += 1
    return round(total, 4)


def resumen_reparto(datos):
    """Texto corto con el reparto que salio entre las dos plataformas."""
    total_bytes = datos['bytes_cpu'] + datos['bytes_gpu']
    if not total_bytes:
        return '(sin datos)'

    pct_cpu = 100.0 * datos['bytes_cpu'] / total_bytes
    lineas = [
        'Trozos de %d MB: %d en total' % (datos['trozo_mb'], datos['n_trozos']),
        '  CPU x%-2d  %3d trozos  %8.1f MB  %5.1f %%'
        % (datos['procesos'], datos['trozos_cpu'],
           datos['bytes_cpu'] / (1024 * 1024), pct_cpu),
        '  GPU      %3d trozos  %8.1f MB  %5.1f %%'
        % (datos['trozos_gpu'], datos['bytes_gpu'] / (1024 * 1024),
           100.0 - pct_cpu),
        '  Trabajando a la vez: %.2f s' % datos['solape_s'],
    ]
    return '\n'.join(lineas)


# ---------------------------------------------------------------------------
# Prueba rapida por linea de comandos
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    multiprocessing.freeze_support()

    if len(sys.argv) < 2:
        print('Uso: python motor_hibrido.py <archivo.fna> [procesos] [trozo_mb]')
        sys.exit(0)

    archivo = sys.argv[1]
    n_proc = int(sys.argv[2]) if len(sys.argv) > 2 else None
    trozo = int(sys.argv[3]) if len(sys.argv) > 3 else TROZO_MB

    tam = os.path.getsize(archivo)
    print('Archivo: %s  (%.1f MB)' % (archivo, tam / (1024 * 1024)))
    print('')

    res, t = contar_hibrido(archivo, procesos=n_proc, trozo_mb=trozo)
    print('Hibrido CPU+GPU: %.3f s  (%.1f MB/s)'
          % (t, tam / (1024 * 1024) / t if t else 0))
    print('  A=%d C=%d G=%d T=%d N=%d invalidos=%d'
          % (res['A'], res['C'], res['G'], res['T'],
             res['N'], res['invalidos']))
    print('')
    print(resumen_reparto(res['hibrido']))

    censo = res['hibrido']['censo_sm']
    if censo:
        total = sum(censo)
        print('')
        print('Reparto dentro de la GPU, entre sus %d SMs:' % len(censo))
        for i, v in enumerate(censo):
            pct = 100.0 * v / total if total else 0.0
            print('  SM %2d  %14d  %5.2f %%' % (i, v, pct))
