# -*- coding: utf-8 -*-
"""
motor_gpu.py

Comparacion de dos cadenas de ADN sobre la tarjeta grafica, con kernels CUDA
escritos a mano usando Numba. Es la columna "GPU" de la rubrica del P1.4.

QUE HACE EL KERNEL
Un hilo por posicion, con grid-stride loop para cubrir lotes mas grandes que
la malla. Cada hilo compara un caracter de la cadena A contra el de la misma
posicion de la cadena B. Si difieren:

  - suma uno al contador del bloque, que vive en memoria compartida;
  - reserva una ranura atomica en el vector de capturas y apunta la posicion,
    mientras quede sitio.

Al terminar su bucle, el hilo 0 de cada bloque vuelca el contador del bloque
al contador global de una sola vez. Se hace asi y no sumando al contador
global por cada diferencia porque, con dos genomas que difieren en mil
millones de posiciones, mil millones de sumas atomicas sobre la MISMA
direccion de memoria global serializarian el kernel por completo: todos los
hilos se pondrian en fila para tocar el mismo entero.

Es la misma idea que hacia viable el histograma del P1.2 y del P1.3, aplicada
a un contador en vez de a 256 bins: acumular barato y local, y pagar la
sincronizacion cara una sola vez por bloque.

POR QUE POR LOTES
Los dos genomas ocupan 3.1 GB de secuencia cada uno y la RTX 4050 tiene 6 GB
de VRAM, de los que Windows ya reserva una parte. Los dos no caben a la vez.
Se procesan por lotes con dos buffers de memoria pinned y dos streams
alternandose, de modo que la transferencia del lote k+1 se solape con el
computo del lote k. El tamano de lote es configurable.

SOBRE LA LISTA DE POSICIONES
El conteo de diferencias es SIEMPRE exacto: sale de contadores atomicos que
no se pierden ninguna. La lista de posiciones, en cambio, esta acotada por el
tamano del vector de capturas, porque no se pueden traer mil millones de
indices desde la tarjeta.

Mientras las diferencias caben en el vector, las posiciones recogidas son
todas y, una vez ordenadas, son exactamente las mismas que devuelve la CPU.
Cuando no caben, lo que se publica es una muestra, y ademas se informa aparte
de la PRIMERA diferencia real, que se obtiene con un minimo atomico y por
tanto es exacta pase lo que pase. El resultado dice cual de los dos casos se
dio, en vez de dejar que el lector lo suponga.

RESPUESTAS A LA RUBRICA, COLUMNA "GPU"
    LIBRERIA    numba.cuda
    INSTRUCCION @cuda.jit sobre el kernel, y
                k_comparar[bloques, hilos, stream](...) para lanzarlo
    Ver evidencias.py, que es donde se recogen formalmente.

API publica:
    gpu_disponible()                    -> bool
    info_gpu()                          -> dict con datos de la tarjeta
    precalentar()                       -> segundos de compilacion JIT
    recursos_kernel()                   -> ocupacion, registros, memoria
    comparar_gpu(cadena_a, cadena_b)    -> (dict_resultado, tiempo)
"""

import os
import time
import warnings

import numpy as np

import comparador
from comparador import ErrorEntrada


try:
    from numba import cuda
    from numba.core.errors import NumbaPerformanceWarning
    # Las secuencias cortas (un contig de cien mil bases) llenan pocos
    # bloques y Numba avisa de baja ocupacion en cada lanzamiento. Con 701
    # secuencias por comparar eso son cientos de avisos que tapan la salida
    # util. No es un problema: el kernel usa grid-stride y los hilos sin
    # datos salen del bucle de inmediato. Se silencia el aviso, no la causa.
    warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
    # numba-cuda lanza el mismo aviso con su propia clase, que no es la de
    # numba; se filtra tambien por el texto. Con el campo de bloques es
    # esperable: pedir 20 bloques es justo lo que se quiere medir.
    warnings.filterwarnings('ignore', message='.*Grid size')
    _HAY_CUDA = True
except ImportError:                                   # pragma: no cover
    cuda = None
    _HAY_CUDA = False

try:
    import pynvml
    _HAY_NVML = True
except ImportError:                                   # pragma: no cover
    pynvml = None
    _HAY_NVML = False


# Valores por defecto, todos ajustables desde la app y el benchmark.
LOTE_MB = 64                # tamano de cada lote enviado a la tarjeta
HILOS_BLOQUE = 256          # hilos por bloque
BLOQUES = 1024              # bloques del grid-stride loop

# Limites admitidos para el numero de bloques. Es el "GPU n unidades" del
# enunciado: cada bloque se asigna entero a un multiprocesador (SM), asi que
# con n bloques hasta 20 se ocupan n de los 20 SMs de la RTX 4050, y por
# encima se reparten varios por SM hasta llenarla. Mas de 1024 no acelera
# nada en esta tarjeta: 20 SMs x 6 bloques de 256 hilos ya la saturan.
BLOQUES_MIN = 1
BLOQUES_MAX = 1024

# Capacidad del vector de capturas por lote. Con un millon de ranuras caben
# todas las diferencias de cualquier prueba razonable; los genomas completos
# lo desbordan y entonces el resultado lo dice.
MAX_CAPTURA = 1 << 20

# Limites admitidos para el tamano de lote, en MB. Por debajo de 1 MB se
# multiplican los lanzamientos de kernel y no se aprovecha el PCIe; por
# encima de 512 MB se tarda demasiado en llenar el lote antes de empezar a
# computar, y ademas peligra la VRAM de una tarjeta modesta.
LOTE_MIN_MB = 1
LOTE_MAX_MB = 512


# ---------------------------------------------------------------------------
# Disponibilidad e informacion del hardware
# ---------------------------------------------------------------------------

def gpu_disponible():
    """Indica si hay una GPU CUDA utilizable en esta maquina."""
    if not _HAY_CUDA:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def motivo_no_disponible():
    """Explica en castellano por que no se puede usar la GPU, o None."""
    if not _HAY_CUDA:
        return ('No esta instalada la libreria numba-cuda, que es la que '
                'permite escribir kernels para la tarjeta desde Python.')
    try:
        if not cuda.is_available():
            return ('No se detecta ninguna tarjeta NVIDIA con CUDA en este '
                    'equipo. La comparacion en GPU queda desactivada; la CPU '
                    'y la NPU siguen funcionando.')
    except Exception as error:
        return 'El controlador de CUDA dio un error: %s' % error
    return None


def info_gpu():
    """Devuelve un diccionario con las caracteristicas de la GPU activa.

    Si no hay GPU o falta alguna libreria devuelve {'disponible': False} en
    lugar de lanzar excepcion, para que la app pueda seguir funcionando.
    """
    datos = {'disponible': False}
    if not gpu_disponible():
        datos['motivo'] = motivo_no_disponible()
        return datos

    try:
        dispositivo = cuda.get_current_device()
        nombre = dispositivo.name
        if isinstance(nombre, bytes):
            nombre = nombre.decode('utf-8', 'replace')
        libre, total = cuda.current_context().get_memory_info()
        mayor, menor = dispositivo.compute_capability
        datos.update({
            'disponible': True,
            'nombre': nombre,
            'compute_capability': '%d.%d' % (mayor, menor),
            'sms': dispositivo.MULTIPROCESSOR_COUNT,
            'max_hilos_bloque': dispositivo.MAX_THREADS_PER_BLOCK,
            'warp': dispositivo.WARP_SIZE,
            'vram_total_mb': total // (1024 * 1024),
            'vram_libre_mb': libre // (1024 * 1024),
        })
    except Exception as error:                        # pragma: no cover
        datos['motivo'] = str(error)

    return datos


def lote_valido(pedido, vram_libre_mb=None):
    """Ajusta el tamano de lote a algo que la tarjeta pueda sostener.

    Devuelve (lote_mb, motivo). El motivo es None si no hubo que tocar nada.
    Se comprueba tambien contra la VRAM libre, porque el motor reserva dos
    buffers de dispositivo por cada cadena: pedir un lote de 512 MB en una
    tarjeta con 300 MB libres no falla al validar sino en mitad del trabajo,
    con un error de CUDA que no dice nada a quien usa el programa.
    """
    try:
        pedido = int(pedido)
    except (TypeError, ValueError):
        return LOTE_MB, ('"%s" no es un tamano de lote valido; se usan %d MB.'
                         % (pedido, LOTE_MB))

    if pedido < LOTE_MIN_MB:
        return LOTE_MIN_MB, ('Un lote de %d MB es demasiado pequeno; se usa '
                             '%d MB.' % (pedido, LOTE_MIN_MB))

    if pedido > LOTE_MAX_MB:
        return LOTE_MAX_MB, ('Un lote de %d MB es mas de lo que compensa; se '
                             'usa %d MB.' % (pedido, LOTE_MAX_MB))

    if vram_libre_mb:
        # Cuatro buffers de dispositivo (dos cadenas x dos turnos) mas el
        # vector de capturas, con un margen para el resto del sistema.
        necesita = pedido * 4 + 16
        if necesita > vram_libre_mb * 0.8:
            cabe = max(LOTE_MIN_MB, int(vram_libre_mb * 0.8 - 16) // 4)
            return cabe, ('Un lote de %d MB no cabe en los %d MB libres de la '
                          'tarjeta; se usa %d MB.'
                          % (pedido, vram_libre_mb, cabe))

    return pedido, None


def bloques_validos(pedido):
    """Ajusta el numero de bloques a un valor con sentido.

    Devuelve (bloques, motivo). El motivo es None si no hubo que tocar nada.
    """
    try:
        pedido = int(pedido)
    except (TypeError, ValueError):
        return BLOQUES, ('"%s" no es un numero de bloques valido; se usan %d.'
                         % (pedido, BLOQUES))

    if pedido < BLOQUES_MIN:
        return BLOQUES_MIN, ('Hace falta al menos %d bloque; se usa %d.'
                             % (BLOQUES_MIN, BLOQUES_MIN))

    if pedido > BLOQUES_MAX:
        return BLOQUES_MAX, ('%d bloques no caben con provecho en la tarjeta; '
                             'se usan %d.' % (pedido, BLOQUES_MAX))

    return pedido, None


# ---------------------------------------------------------------------------
# Kernel CUDA
# ---------------------------------------------------------------------------

if _HAY_CUDA:

    @cuda.jit
    def k_comparar(a, b, n, desplazamiento, contador, capturas, cuantas_cap,
                   primera):
        """Compara dos lotes posicion a posicion y anota las diferencias.

        Parametros de dispositivo:
            a, b            : los dos lotes de secuencia, mismo tamano.
            n               : cuantos bytes del lote son utiles.
            desplazamiento  : posicion absoluta del primer byte del lote, para
                              que las posiciones que se anoten sean las del
                              archivo y no las del lote.
            contador        : entero de 64 bits con el total de diferencias.
            capturas        : vector donde se apuntan posiciones concretas.
            cuantas_cap     : cuantas ranuras del vector se han usado.
            primera         : minimo atomico, la primera diferencia real.
        """
        # Contador del bloque en memoria compartida. Es lo que evita que mil
        # millones de hilos se pongan en fila sobre el mismo entero global.
        local = cuda.shared.array(1, dtype=np.int64)

        if cuda.threadIdx.x == 0:
            local[0] = 0
        cuda.syncthreads()

        i = cuda.grid(1)
        paso = cuda.gridsize(1)
        while i < n:
            if a[i] != b[i]:
                cuda.atomic.add(local, 0, 1)
                posicion = desplazamiento + i
                # Minimo atomico: da la primera diferencia del archivo de
                # forma exacta, quepa o no la posicion en el vector.
                cuda.atomic.min(primera, 0, posicion)
                ranura = cuda.atomic.add(cuantas_cap, 0, 1)
                if ranura < capturas.size:
                    capturas[ranura] = posicion
            i += paso

        cuda.syncthreads()
        # Una sola suma global por bloque, en vez de una por diferencia.
        if cuda.threadIdx.x == 0 and local[0] != 0:
            cuda.atomic.add(contador, 0, local[0])


# ---------------------------------------------------------------------------
# Precalentamiento del compilador JIT
# ---------------------------------------------------------------------------

_precalentado = False


def precalentar():
    """Compila el kernel sobre un lote minusculo y descarta el resultado.

    Numba compila la primera vez que se lanza un kernel, y eso tarda entre
    medio segundo y dos segundos. Si no se saca del cronometro, ese tiempo se
    le carga a la GPU y la comparativa contra la CPU queda falseada: sobre
    entradas pequenas la compilacion pesaria mas que el trabajo real.

    Devuelve los segundos que costo compilar, o 0.0 si ya estaba hecho.
    """
    global _precalentado
    if _precalentado or not gpu_disponible():
        return 0.0

    inicio = time.perf_counter()
    muestra_a = np.frombuffer(b'ACGTACGTNNNN', dtype=np.uint8).copy()
    muestra_b = np.frombuffer(b'ACGTTCGTNNNN', dtype=np.uint8).copy()

    d_a = cuda.to_device(muestra_a)
    d_b = cuda.to_device(muestra_b)
    d_cont = cuda.to_device(np.zeros(1, dtype=np.int64))
    d_cap = cuda.device_array(16, dtype=np.int64)
    d_ncap = cuda.to_device(np.zeros(1, dtype=np.int64))
    d_pri = cuda.to_device(np.full(1, np.iinfo(np.int64).max, dtype=np.int64))

    # La muestra es minuscula a proposito, asi que Numba avisa de baja
    # ocupacion. Aqui no importa: solo se quiere compilar.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        k_comparar[1, 32](d_a, d_b, muestra_a.size, 0, d_cont, d_cap,
                          d_ncap, d_pri)
        cuda.synchronize()

    _precalentado = True
    return time.perf_counter() - inicio


def recursos_kernel():
    """Registros, memoria compartida y ocupacion teorica del kernel.

    Es el dato que responde al nivel "por nucleos" de los criterios de la
    rubrica: no cuanto se uso la tarjeta en conjunto, sino cuanto cabe dentro
    de cada multiprocesador.
    """
    if not gpu_disponible():
        return None
    precalentar()
    try:
        dispositivo = cuda.get_current_device()
        firma = k_comparar.overloads
        definicion = list(firma.values())[0] if firma else None
        if definicion is None:
            return None

        registros = getattr(definicion, 'regs_per_thread', 0) or 0
        compartida = getattr(definicion, 'shared_mem_per_block', 0) or 0

        warp = dispositivo.WARP_SIZE
        max_hilos_sm = getattr(dispositivo, 'MAX_THREADS_PER_MULTIPROCESSOR',
                               2048)
        max_registros_sm = getattr(
            dispositivo, 'MAX_REGISTERS_PER_MULTIPROCESSOR', 65536)
        max_shared_sm = getattr(
            dispositivo, 'MAX_SHARED_MEMORY_PER_MULTIPROCESSOR', 102400)

        warps_maximos = max_hilos_sm // warp
        warps_bloque = HILOS_BLOQUE // warp

        # Cuantos bloques caben por multiprocesador, segun cada recurso.
        por_registros = (max_registros_sm // (registros * HILOS_BLOQUE)
                         if registros else 99)
        por_compartida = (max_shared_sm // compartida) if compartida else 99
        bloques_sm = max(0, min(por_registros, por_compartida,
                                max_hilos_sm // HILOS_BLOQUE))
        warps_activos = min(warps_maximos, bloques_sm * warps_bloque)

        return {
            'sms': dispositivo.MULTIPROCESSOR_COUNT,
            'registros_por_hilo': int(registros),
            'compartida_por_bloque': int(compartida),
            'warp': int(warp),
            'hilos_bloque': HILOS_BLOQUE,
            'warps_activos': int(warps_activos),
            'warps_maximos': int(warps_maximos),
            'ocupacion_pct': (100.0 * warps_activos / warps_maximos
                              if warps_maximos else 0.0),
            'max_shared_sm': int(max_shared_sm),
            'max_registros_sm': int(max_registros_sm),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def comparar_gpu(cadena_a, cadena_b, lote_mb=LOTE_MB,
                 hilos_bloque=HILOS_BLOQUE, bloques=BLOQUES,
                 detalle=None, progreso=None):
    """Compara las dos cadenas sobre la tarjeta grafica.

    Devuelve (dict_resultado, tiempo_en_segundos). El tiempo no incluye la
    compilacion de los kernels ni la reserva de los buffers: ver precalentar()
    y la clave 'preparacion_s' del resultado.
    """
    if not gpu_disponible():
        raise ErrorEntrada(motivo_no_disponible())

    largo = min(cadena_a.largo, cadena_b.largo)
    if largo == 0:
        raise ErrorEntrada('No hay nada que comparar: una cadena esta vacia.')

    datos_gpu = info_gpu()
    lote_mb, aviso_lote = lote_valido(lote_mb, datos_gpu.get('vram_libre_mb'))
    bloques, aviso_bloques = bloques_validos(bloques)
    lote = int(lote_mb) * 1024 * 1024
    tope = comparador.MAX_DETALLE if detalle is None else max(0, detalle)

    # Fuera del cronometro: compilar no es trabajo del algoritmo.
    precalentar()

    # -- preparacion, deliberadamente fuera del cronometro ----------------
    inicio_preparacion = time.perf_counter()
    n_buffers = 2
    h_a = [cuda.pinned_array(lote, dtype=np.uint8) for _ in range(n_buffers)]
    h_b = [cuda.pinned_array(lote, dtype=np.uint8) for _ in range(n_buffers)]
    d_a = [cuda.device_array(lote, dtype=np.uint8) for _ in range(n_buffers)]
    d_b = [cuda.device_array(lote, dtype=np.uint8) for _ in range(n_buffers)]
    streams = [cuda.stream() for _ in range(n_buffers)]

    d_contador = cuda.to_device(np.zeros(1, dtype=np.int64))
    d_capturas = cuda.device_array(MAX_CAPTURA, dtype=np.int64)
    d_ncapturas = cuda.to_device(np.zeros(1, dtype=np.int64))
    d_primera = cuda.to_device(
        np.full(1, np.iinfo(np.int64).max, dtype=np.int64))
    preparacion = time.perf_counter() - inicio_preparacion

    va = cadena_a.datos
    vb = cadena_b.datos
    turno = 0
    hechos = 0

    # -- medicion ---------------------------------------------------------
    arranque = time.perf_counter()
    try:
        posicion = 0
        while posicion < largo:
            utiles = min(lote, largo - posicion)
            stream = streams[turno]

            # El stream de este turno pudo quedarse trabajando en el lote
            # anterior sobre los mismos buffers; hay que esperarlo antes de
            # sobreescribirlos con datos nuevos.
            stream.synchronize()

            # Copia desde el archivo mapeado al buffer de pagina bloqueada.
            # La memoria pinned es obligatoria para que la transferencia a la
            # tarjeta sea asincrona; sobre memoria paginable normal seria
            # sincrona y no habria solapamiento que ganar.
            h_a[turno][:utiles] = va[posicion:posicion + utiles]
            h_b[turno][:utiles] = vb[posicion:posicion + utiles]

            d_a[turno][:utiles].copy_to_device(h_a[turno][:utiles],
                                               stream=stream)
            d_b[turno][:utiles].copy_to_device(h_b[turno][:utiles],
                                               stream=stream)

            # Se lanzan los bloques pedidos, salvo que el lote sea tan
            # pequeno que no tenga datos para todos. No hay suelo: si se piden
            # 4 bloques se lanzan 4, porque eso es justo lo que se quiere
            # medir. El kernel usa grid-stride, asi que con pocos bloques
            # cada hilo simplemente recorre mas posiciones.
            malla = min(bloques, (utiles + hilos_bloque - 1) // hilos_bloque)

            k_comparar[malla, hilos_bloque, stream](
                d_a[turno], d_b[turno], utiles, posicion,
                d_contador, d_capturas, d_ncapturas, d_primera)

            posicion += utiles
            hechos += utiles
            if progreso is not None:
                progreso(hechos, largo)

            turno = (turno + 1) % n_buffers

        # Se espera a que la tarjeta termine todo lo encolado antes de leer
        # el resultado: los kernels se lanzan sin bloquear, asi que sin esta
        # espera se estaria midiendo el tiempo de encolar, no el de calcular.
        for stream in streams:
            stream.synchronize()
        cuda.synchronize()

        total = int(d_contador.copy_to_host()[0])
        usadas = int(d_ncapturas.copy_to_host()[0])
        primera = int(d_primera.copy_to_host()[0])
        transcurrido = time.perf_counter() - arranque

        cuantas = min(usadas, MAX_CAPTURA)
        capturadas = (d_capturas[:cuantas].copy_to_host() if cuantas
                      else np.empty(0, dtype=np.int64))
    finally:
        # Los buffers de la tarjeta se sueltan siempre, tambien si algo fallo
        # a mitad: dejarlos colgados agota la VRAM a la segunda corrida.
        del d_a, d_b, d_capturas, d_contador, d_ncapturas, d_primera
        del h_a, h_b

    # Los hilos de la GPU terminan en orden arbitrario, asi que las
    # posiciones capturadas llegan desordenadas y hay que ordenarlas antes de
    # publicarlas. Mientras quepan todas, ordenarlas las deja exactamente
    # iguales a las que devuelve la CPU, que es lo que permite verificar un
    # motor contra el otro.
    capturadas.sort()
    posiciones = capturadas[:tope]

    completo = total <= MAX_CAPTURA
    resultado = comparador.resumir(total, posiciones, cadena_a, cadena_b,
                                   largo, 'GPU', detalle=detalle)
    resultado['gpu'] = {
        'lote_mb': lote_mb,
        'hilos_bloque': hilos_bloque,
        'bloques': bloques,
        'preparacion_s': round(preparacion, 4),
        'capturas_usadas': cuantas,
        'capturas_maximas': MAX_CAPTURA,
        'lista_completa': completo,
        'primera_diferencia': primera if total else None,
        'dispositivo': datos_gpu.get('nombre', ''),
        'sms': datos_gpu.get('sms'),
    }
    avisos = [a for a in (aviso_lote, aviso_bloques) if a]
    if avisos:
        resultado['gpu']['aviso'] = ' '.join(avisos)
    if not completo:
        resultado['gpu']['nota'] = (
            'Hay %d diferencias, mas de las %d que caben en el vector de '
            'capturas de la tarjeta. El total es exacto y la primera '
            'diferencia tambien (%d), pero las posiciones listadas son una '
            'muestra, no las primeras.'
            % (total, MAX_CAPTURA, primera))
    return resultado, transcurrido


def uso_gpu():
    """Lectura puntual de utilizacion, memoria y temperatura de la tarjeta.

    Devuelve None si pynvml no esta instalado o la consulta falla, para que el
    resto del programa siga funcionando sin el panel de recursos.
    """
    if not _HAY_NVML:
        return None
    try:
        pynvml.nvmlInit()
        manejador = pynvml.nvmlDeviceGetHandleByIndex(0)
        uso = pynvml.nvmlDeviceGetUtilizationRates(manejador)
        memoria = pynvml.nvmlDeviceGetMemoryInfo(manejador)
        temperatura = pynvml.nvmlDeviceGetTemperature(
            manejador, pynvml.NVML_TEMPERATURE_GPU)
        return {
            'uso_gpu': uso.gpu,
            'uso_memoria': uso.memory,
            'vram_usada_mb': memoria.used // (1024 * 1024),
            'vram_total_mb': memoria.total // (1024 * 1024),
            'temperatura': temperatura,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prueba rapida por linea de comandos
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    datos = info_gpu()
    if not datos['disponible']:
        print('GPU no disponible: %s' % datos.get('motivo'))
        sys.exit(1)

    print('GPU               : %s' % datos['nombre'])
    print('Compute capability: %s   SMs: %d   Warp: %d'
          % (datos['compute_capability'], datos['sms'], datos['warp']))
    print('VRAM              : %d MB libres de %d MB'
          % (datos['vram_libre_mb'], datos['vram_total_mb']))

    if len(sys.argv) < 3:
        print('')
        print('Uso: python motor_gpu.py <cadena_a.fna> <cadena_b.fna> '
              '[lote_mb] [bloques]')
        sys.exit(0)

    print('')
    print('Compilacion JIT del kernel: %.3f s (fuera de la medicion)'
          % precalentar())

    recursos = recursos_kernel()
    if recursos:
        print('Kernel: %d registros por hilo, %d B compartidos por bloque, '
              'ocupacion %d/%d warps (%.0f %%)'
              % (recursos['registros_por_hilo'],
                 recursos['compartida_por_bloque'],
                 recursos['warps_activos'], recursos['warps_maximos'],
                 recursos['ocupacion_pct']))

    try:
        ca = comparador.preparar(sys.argv[1])
        cb = comparador.preparar(sys.argv[2])
        for aviso in comparador.validar_par(ca, cb):
            print('AVISO: %s' % aviso)
    except ErrorEntrada as error:
        print('ERROR: %s' % error)
        sys.exit(1)

    mb = min(ca.largo, cb.largo) / (1024.0 * 1024.0)
    lote = int(sys.argv[3]) if len(sys.argv) > 3 else LOTE_MB
    bloques = int(sys.argv[4]) if len(sys.argv) > 4 else BLOQUES

    res, t = comparar_gpu(ca, cb, lote_mb=lote, bloques=bloques)
    print('')
    print('GPU %d bloques, lote %d MB : %8.3f s  (%8.1f MB/s)   '
          'preparacion %.2f s'
          % (res['gpu']['bloques'], res['gpu']['lote_mb'], t,
             mb / t if t else 0,
             res['gpu']['preparacion_s']))
    if res['gpu'].get('aviso'):
        print('  AVISO: %s' % res['gpu']['aviso'])
    print('')
    print(comparador.formatear_diferencias(res))
    if res['gpu'].get('nota'):
        print('')
        print('  NOTA: %s' % res['gpu']['nota'])
