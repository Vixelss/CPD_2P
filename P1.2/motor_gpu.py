# -*- coding: utf-8 -*-
"""
motor_gpu.py

Motor de conteo de bases sobre archivos FASTA grandes, ejecutado en GPU
NVIDIA con Numba CUDA. Es la contraparte de motor_cpu.py: recibe el mismo
archivo, aplica las mismas reglas y devuelve el mismo diccionario, de modo
que la unica diferencia entre los dos sea el hardware que ejecuta.

API publica:
    gpu_disponible()                       -> bool
    info_gpu()                             -> dict con datos de la tarjeta
    contar_gpu(ruta, ...)                  -> (dict_resultado, tiempo)

POR QUE TRES KERNELS
El formato FASTA mezcla lineas de cabecera (empiezan con '>') con lineas de
secuencia, y las cabeceras no se deben contar. La tentacion es limpiarlas en
el host antes de enviar el lote a la GPU, pero eso seria hacer trampa: la
CPU estaria haciendo parte del trabajo de la GPU y los tiempos dejarian de
ser comparables. Por eso el descarte ocurre dentro de la tarjeta:

  k_marcar_cabeceras : un hilo por byte. Marca las posiciones donde empieza
                       una cabecera, es decir un '>' precedido de salto de
                       linea (o el primer byte del lote, porque los lotes
                       siempre empiezan en linea nueva). Las posiciones se
                       acumulan con un contador atomico.
  k_borrar_cabeceras : un bloque por cabecera encontrada. El hilo 0 busca
                       donde termina la linea y el resto del bloque la
                       sobreescribe en paralelo con saltos de linea, que el
                       histograma ya ignora. Las cabeceras son pocas y
                       cortas, asi que cuesta muy poco.
  k_histograma       : un hilo por byte con grid-stride loop. Cada bloque
                       acumula en un histograma de 256 bins en memoria
                       compartida (rapida, dentro del SM) y solo al final
                       vuelca su resultado al histograma global. Sin la
                       memoria compartida, millones de hilos chocarian
                       atomicamente contra los mismos 4 bins globales y el
                       kernel se serializaria.

El kernel de borrado se lanza con una malla fija y cada bloque consulta en
memoria de dispositivo cuantas cabeceras hay. Asi no hace falta copiar ese
numero al host entre kernel y kernel, que obligaria a sincronizar y romperia
el solapamiento entre transferencia y computo.

POR QUE POR LOTES
El genoma humano ocupa 3.11 GB y la RTX 4050 tiene 6 GB de VRAM, de los
cuales Windows ya reserva una parte. Cargar el archivo entero no es viable,
y ademas seria mala idea: conviene ir enviando trozos mientras la GPU
trabaja en el anterior. Se usan dos buffers de memoria pinned (pagina
bloqueada, requisito para que la copia sea asincrona) y dos streams CUDA
alternandose, de forma que la transferencia del lote k+1 se solape con el
computo del lote k. El tamano de lote es configurable porque es justamente
la perilla de balance de carga que pide el enunciado.
"""

import os
import time
import warnings

import numpy as np

from motor_cpu import BINS, resumir_histograma

try:
    from numba import cuda
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
LOTE_MB = 64                # tamano de cada lote enviado a la GPU
HILOS_BLOQUE = 256          # hilos por bloque en el histograma
BLOQUES_HIST = 1024         # bloques del grid-stride loop del histograma
BLOQUES_BORRADO = 512       # bloques del kernel que borra cabeceras
MIN_BLOQUES = 32            # suelo de bloques para no dejar la GPU vacia
MAX_CABECERAS = 1 << 20     # capacidad del vector de cabeceras por lote

# Bytes con significado propio dentro del formato.
COD_MAYOR = 62              # '>' marca el inicio de una cabecera
COD_SALTO = 10              # '\n' separa lineas y el histograma lo ignora

# Cola de seguridad al buscar el ultimo salto de linea de un lote. Las
# lineas de secuencia FASTA son de 80 caracteres, con 4096 sobra de lejos.
_COLA_BUSQUEDA = 4096


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


def info_gpu():
    """Devuelve un diccionario con las caracteristicas de la GPU activa.

    Si no hay GPU o falta alguna libreria, devuelve un diccionario con
    'disponible' en False en lugar de lanzar excepcion, para que la app
    pueda seguir funcionando solo con CPU.
    """
    datos = {'disponible': False}
    if not gpu_disponible():
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
        datos['error'] = str(error)

    return datos


# ---------------------------------------------------------------------------
# Kernels CUDA
# ---------------------------------------------------------------------------

if _HAY_CUDA:

    @cuda.jit
    def k_marcar_cabeceras(datos, n, posiciones, contador):
        """Marca donde empieza cada linea de cabecera dentro del lote.

        Un hilo por byte, con grid-stride loop para cubrir lotes mas grandes
        que la malla. Una cabecera empieza en un '>' que este al principio
        del lote o justo despues de un salto de linea; asi un '>' que
        apareciera en medio de una linea no se confunde con una cabecera.
        """
        i = cuda.grid(1)
        paso = cuda.gridsize(1)
        while i < n:
            if datos[i] == COD_MAYOR and (i == 0 or datos[i - 1] == COD_SALTO):
                # Reserva atomica de una ranura en el vector de posiciones.
                ranura = cuda.atomic.add(contador, 0, 1)
                if ranura < posiciones.size:
                    posiciones[ranura] = i
            i += paso

    @cuda.jit
    def k_borrar_cabeceras(datos, n, posiciones, contador):
        """Sobreescribe con saltos de linea cada cabecera encontrada.

        Un bloque por cabecera. El hilo 0 recorre la linea hasta dar con su
        final y deja el limite en memoria compartida; despues todos los
        hilos del bloque borran el rango en paralelo. Se escribe el codigo
        del salto de linea porque el histograma ya lo descarta, de modo que
        la cabecera desaparece sin necesidad de mover datos.
        """
        fin = cuda.shared.array(1, dtype=np.int64)

        total = contador[0]
        if total > posiciones.size:
            total = posiciones.size

        # Malla fija: cada bloque atiende varias cabeceras si hacen falta.
        indice = cuda.blockIdx.x
        salto = cuda.gridDim.x
        while indice < total:
            inicio = posiciones[indice]

            if cuda.threadIdx.x == 0:
                j = inicio
                while j < n and datos[j] != COD_SALTO:
                    j += 1
                fin[0] = j
            cuda.syncthreads()

            limite = fin[0]
            p = inicio + cuda.threadIdx.x
            while p < limite:
                datos[p] = COD_SALTO
                p += cuda.blockDim.x
            cuda.syncthreads()

            indice += salto

    @cuda.jit
    def k_histograma(datos, n, global_hist):
        """Histograma de 256 bins sobre el lote, ya sin cabeceras.

        Cada bloque acumula primero en memoria compartida y solo al final
        suma su parcial al histograma global. Esto reduce la contencion
        atomica de millones de hilos a una sola ronda de 256 sumas por
        bloque, que es lo que hace viable el kernel.
        """
        local = cuda.shared.array(BINS, dtype=np.uint32)

        # Puesta a cero del histograma compartido, con paso por si el
        # bloque tiene menos de 256 hilos.
        t = cuda.threadIdx.x
        while t < BINS:
            local[t] = 0
            t += cuda.blockDim.x
        cuda.syncthreads()

        i = cuda.grid(1)
        paso = cuda.gridsize(1)
        while i < n:
            cuda.atomic.add(local, datos[i], 1)
            i += paso
        cuda.syncthreads()

        # Volcado del parcial del bloque al histograma global.
        t = cuda.threadIdx.x
        while t < BINS:
            valor = local[t]
            if valor != 0:
                cuda.atomic.add(global_hist, t, valor)
            t += cuda.blockDim.x


# ---------------------------------------------------------------------------
# Precalentamiento del compilador JIT
# ---------------------------------------------------------------------------

_precalentado = False


def precalentar():
    """Compila los tres kernels sobre un lote minusculo y descarta el result.

    Numba compila cada kernel la primera vez que se lanza, y esa compilacion
    tarda alrededor de un segundo. Si no se saca del cronometro, ese segundo
    se le carga a la GPU y la comparativa contra la CPU queda falseada: en
    archivos pequenos la compilacion pesaria mas que el trabajo real. Por eso
    el benchmark y la aplicacion llaman a esta funcion antes de medir.

    Devuelve los segundos que costo compilar, que son un dato interesante
    para el informe, o 0.0 si ya estaba precalentado.
    """
    global _precalentado
    if _precalentado or not gpu_disponible():
        return 0.0

    inicio = time.perf_counter()

    muestra = np.frombuffer(b'>cabecera de prueba\nACGTacgtNNXZ\n',
                            dtype=np.uint8).copy()
    d_buf = cuda.to_device(muestra)
    d_pos = cuda.device_array(16, dtype=np.int64)
    d_cont = cuda.to_device(np.zeros(1, dtype=np.int64))
    d_hist = cuda.to_device(np.zeros(BINS, dtype=np.uint64))

    n = muestra.size
    # La muestra es minuscula a proposito, asi que Numba avisa de baja
    # ocupacion de la GPU. Aqui no importa: solo se quiere compilar.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        k_marcar_cabeceras[1, 32](d_buf, n, d_pos, d_cont)
        k_borrar_cabeceras[1, 32](d_buf, n, d_pos, d_cont)
        k_histograma[1, 32](d_buf, n, d_hist)
        cuda.synchronize()

    _precalentado = True
    return time.perf_counter() - inicio


# ---------------------------------------------------------------------------
# Utilidades de lectura
# ---------------------------------------------------------------------------

def _recortar_a_linea(buffer, leidos):
    """Recorta el lote para que termine justo despues de un salto de linea.

    Devuelve cuantos bytes del buffer se deben procesar. Los bytes sobrantes
    se devolveran al archivo con un seek hacia atras, de forma que ninguna
    linea quede partida entre dos lotes y ninguna cabecera se pierda.
    """
    if leidos == 0:
        return 0

    # Busqueda hacia atras solo en la cola: con lineas de 80 caracteres el
    # salto aparece en los primeros bytes revisados.
    desde = max(0, leidos - _COLA_BUSQUEDA)
    cola = buffer[desde:leidos].tobytes()
    corte = cola.rfind(bytes([COD_SALTO]))
    if corte >= 0:
        return desde + corte + 1

    # Caso raro (linea larguisima): se revisa el lote completo.
    corte = buffer[:leidos].tobytes().rfind(bytes([COD_SALTO]))
    if corte >= 0:
        return corte + 1

    # No hay ningun salto de linea: se procesa el lote entero tal cual.
    return leidos


# ---------------------------------------------------------------------------
# API publica de conteo
# ---------------------------------------------------------------------------

def contar_gpu(ruta, lote_mb=LOTE_MB, hilos_bloque=HILOS_BLOQUE,
               bloques=BLOQUES_HIST, progreso=None, medir_fases=False):
    """Cuenta las bases del archivo usando la GPU.

    Parametros:
        ruta         : archivo FASTA a procesar.
        lote_mb      : tamano en MB de cada lote enviado a la tarjeta.
        hilos_bloque : hilos por bloque de los kernels.
        bloques      : bloques del grid-stride loop del histograma.
        progreso     : funcion opcional progreso(bytes_leidos, bytes_totales).
        medir_fases  : si es True sincroniza para cronometrar lectura,
                       transferencia y computo por separado. Da informacion
                       util para el informe pero impide el solapamiento, por
                       lo que el tiempo total sale peor. Para medir
                       rendimiento real debe quedarse en False.

    Devuelve (dict_resultado, tiempo_en_segundos). El diccionario es el
    mismo que devuelve motor_cpu, mas la clave 'gpu' con los parametros y
    los tiempos por fase.
    """
    if not gpu_disponible():
        raise RuntimeError('No hay GPU CUDA disponible en esta maquina.')

    # Fuera del cronometro: compilar los kernels no es trabajo del algoritmo.
    precalentar()

    tamano = os.path.getsize(ruta)
    lote = max(1, int(lote_mb)) * 1024 * 1024

    # Dos buffers pinned y dos streams para solapar copia y computo. La
    # memoria pinned es obligatoria: sobre memoria paginable normal la copia
    # seria sincrona y no habria ningun solapamiento que ganar.
    n_buffers = 2
    h_buffers = [cuda.pinned_array(lote, dtype=np.uint8)
                 for _ in range(n_buffers)]
    d_buffers = [cuda.device_array(lote, dtype=np.uint8)
                 for _ in range(n_buffers)]
    d_posiciones = [cuda.device_array(MAX_CABECERAS, dtype=np.int64)
                    for _ in range(n_buffers)]
    d_contadores = [cuda.device_array(1, dtype=np.int64)
                    for _ in range(n_buffers)]
    streams = [cuda.stream() for _ in range(n_buffers)]

    # Histograma global acumulado de todo el archivo. En uint64 porque un
    # solo bin puede superar los 4.290 millones en archivos grandes.
    d_hist = cuda.to_device(np.zeros(BINS, dtype=np.uint64))

    ceros = np.zeros(1, dtype=np.int64)
    tiempos = {'lectura': 0.0, 'transferencia': 0.0, 'computo': 0.0}
    leidos_total = 0
    turno = 0

    arranque = time.perf_counter()
    with open(ruta, 'rb') as f:
        while True:
            h_buf = h_buffers[turno]
            d_buf = d_buffers[turno]
            stream = streams[turno]

            # El stream de este turno pudo quedarse trabajando en el lote
            # anterior sobre el mismo buffer; hay que esperarlo antes de
            # sobreescribirlo con datos nuevos.
            stream.synchronize()

            t0 = time.perf_counter()
            leidos = f.readinto(h_buf)
            if medir_fases:
                tiempos['lectura'] += time.perf_counter() - t0

            if not leidos:
                break

            # Se procesan solo lineas completas y se devuelve el resto.
            utiles = _recortar_a_linea(h_buf, leidos)
            if utiles < leidos:
                f.seek(utiles - leidos, os.SEEK_CUR)
            if utiles == 0:
                break

            t1 = time.perf_counter()
            d_buf[:utiles].copy_to_device(h_buf[:utiles], stream=stream)
            d_contadores[turno].copy_to_device(ceros, stream=stream)
            if medir_fases:
                stream.synchronize()
                tiempos['transferencia'] += time.perf_counter() - t1

            t2 = time.perf_counter()
            malla = min(bloques, (utiles + hilos_bloque - 1) // hilos_bloque)
            # Suelo de bloques: el ultimo lote de un archivo puede quedar muy
            # pequeno despues de recortarlo a linea completa, y con una malla
            # de uno o dos bloques la tarjeta queda casi vacia y Numba avisa
            # de baja ocupacion. Lanzar bloques de sobra no cuesta nada
            # porque los kernels usan grid-stride: los hilos que no tienen
            # datos que procesar salen del bucle de inmediato.
            malla = max(MIN_BLOQUES, malla)

            k_marcar_cabeceras[malla, hilos_bloque, stream](
                d_buf, utiles, d_posiciones[turno], d_contadores[turno])
            k_borrar_cabeceras[BLOQUES_BORRADO, hilos_bloque, stream](
                d_buf, utiles, d_posiciones[turno], d_contadores[turno])
            k_histograma[malla, hilos_bloque, stream](d_buf, utiles, d_hist)
            if medir_fases:
                stream.synchronize()
                tiempos['computo'] += time.perf_counter() - t2

            leidos_total += utiles
            if progreso is not None:
                progreso(leidos_total, tamano)

            turno = (turno + 1) % n_buffers

    # Se espera a que todos los streams terminen antes de leer el resultado.
    for stream in streams:
        stream.synchronize()
    cuda.synchronize()

    hist = d_hist.copy_to_host()
    transcurrido = time.perf_counter() - arranque

    if progreso is not None:
        progreso(tamano, tamano)

    resultado = resumir_histograma(hist)
    resultado['gpu'] = {
        'lote_mb': lote_mb,
        'hilos_bloque': hilos_bloque,
        'bloques': bloques,
        'bytes_procesados': leidos_total,
        'tiempos_fase': tiempos if medir_fases else None,
    }
    return resultado, transcurrido


def uso_gpu():
    """Lectura puntual de utilizacion, memoria y temperatura de la GPU.

    Devuelve None si pynvml no esta instalado o la consulta falla, para que
    el resto del programa siga funcionando sin el panel de recursos.
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
        print('No hay GPU CUDA disponible.')
        sys.exit(1)

    print('GPU              : %s' % datos['nombre'])
    print('Compute capability: %s   SMs: %d   Warp: %d'
          % (datos['compute_capability'], datos['sms'], datos['warp']))
    print('VRAM             : %d MB libres de %d MB'
          % (datos['vram_libre_mb'], datos['vram_total_mb']))

    if len(sys.argv) < 2:
        print('')
        print('Uso: python motor_gpu.py <archivo.fna> [lote_mb]')
        sys.exit(0)

    archivo = sys.argv[1]
    mb = int(sys.argv[2]) if len(sys.argv) > 2 else LOTE_MB

    print('')
    print('Compilacion JIT de los kernels: %.3f s (fuera de la medicion)'
          % precalentar())

    res, t = contar_gpu(archivo, lote_mb=mb, medir_fases=True)
    tam = os.path.getsize(archivo)
    print('')
    print('GPU lote=%d MB : %8.3f s   (%.1f MB/s)'
          % (mb, t, tam / (1024 * 1024) / t if t else 0))
    print('  A=%d C=%d G=%d T=%d N=%d invalidos=%d'
          % (res['A'], res['C'], res['G'], res['T'],
             res['N'], res['invalidos']))
    if res['detalle']:
        print('  detalle invalidos: %s' % res['detalle'])
    fases = res['gpu']['tiempos_fase']
    if fases:
        print('  fases: lectura %.3f s | transferencia %.3f s | computo %.3f s'
              % (fases['lectura'], fases['transferencia'], fases['computo']))
