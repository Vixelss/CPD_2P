# -*- coding: utf-8 -*-
"""
motor_gpu.py

Motor de conteo de bases sobre archivos FASTA grandes, ejecutado en GPU
NVIDIA con Numba CUDA. Es la mitad de GPU del motor hibrido del P1.3.

API publica:
    gpu_disponible()                     -> bool
    info_gpu()                           -> dict con datos de la tarjeta
    ContextoGPU                          -> tarjeta lista para varios tramos
    contar_gpu(ruta, ...)                -> (dict_resultado, tiempo)
    recursos_kernel()                    -> ocupacion y recursos por SM

POR QUE TRES KERNELS
El formato FASTA mezcla lineas de cabecera (empiezan con '>') con lineas de
secuencia, y las cabeceras no se deben contar. La tentacion es limpiarlas en
el host antes de enviar el lote a la GPU, pero eso seria hacer trampa: la
CPU estaria haciendo parte del trabajo de la GPU y los tiempos dejarian de
ser comparables. En el P1.3 seria peor todavia, porque el reparto de carga
del motor hibrido decide a quien darle el siguiente trozo segun lo rapido
que va cada uno. Por eso el descarte ocurre dentro de la tarjeta:

  k_marcar_cabeceras : un hilo por byte. Marca las posiciones donde empieza
                       una cabecera, es decir un '>' precedido de salto de
                       linea (o el primer byte del lote, porque los lotes
                       siempre empiezan en linea nueva).
  k_borrar_cabeceras : un bloque por cabecera encontrada. El hilo 0 busca
                       donde termina la linea y el resto del bloque la
                       sobreescribe en paralelo con saltos de linea, que el
                       histograma ya ignora.
  k_histograma       : un hilo por byte con grid-stride loop. Cada bloque
                       acumula en un histograma de 256 bins en memoria
                       compartida (rapida, dentro del SM) y solo al final
                       vuelca su resultado al histograma global.

EL CENSO POR SM
El tercer kernel lleva ademas la cuenta de cuantos bytes proceso cada
multiprocesador de la tarjeta. Cada hilo pregunta en que SM se esta
ejecutando leyendo el registro especial %smid, y al terminar su parte suma
su cuota al contador de ese SM. Es una sola suma atomica por hilo al final
del bucle, no una por byte, de modo que el coste sobre el tiempo medido es
despreciable y el reparto entre los SMs de la tarjeta queda medido de verdad
en lugar de supuesto.

POR QUE POR LOTES
El genoma humano ocupa 3.11 GB y la RTX 4050 tiene 6 GB de VRAM, de los
cuales Windows ya reserva una parte. Se usan dos buffers de memoria pinned
(pagina bloqueada, requisito para que la copia sea asincrona) y dos streams
CUDA alternandose, de forma que la transferencia del lote k+1 se solape con
el computo del lote k.
"""

import os
import time
import warnings

import numpy as np

from motor_cpu import BINS, resumir_histograma

try:
    from numba import cuda, types
    from numba.extending import intrinsic
    from llvmlite import ir
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
MAX_SM = 256                # capacidad del censo por multiprocesador

# Bytes con significado propio dentro del formato.
COD_MAYOR = 62              # '>' marca el inicio de una cabecera
COD_SALTO = 10              # salto de linea; el histograma lo ignora

# Cola de seguridad al buscar el ultimo salto de linea de un lote. Las
# lineas de secuencia FASTA son de 80 caracteres, con 4096 sobra de lejos.
_COLA_BUSQUEDA = 4096

# Holgura al final del buffer para cerrar la ultima linea de un tramo sin
# partirla. Con lineas de 80 caracteres el margen es enorme; si aun asi no
# cupiera se avisa con una excepcion clara en vez de contar mal.
_COLA_LINEA = 1 << 20


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

    Si no hay GPU o falta alguna libreria devuelve 'disponible' en False en
    lugar de lanzar excepcion, para que la app pueda seguir solo con CPU.
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
            'max_hilos_sm': dispositivo.MAX_THREADS_PER_MULTIPROCESSOR,
            'max_shared_sm': dispositivo.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            'max_registros_sm': dispositivo.MAX_REGISTERS_PER_MULTIPROCESSOR,
            'warp': dispositivo.WARP_SIZE,
            'vram_total_mb': total // (1024 * 1024),
            'vram_libre_mb': libre // (1024 * 1024),
        })
    except Exception as error:                        # pragma: no cover
        datos['error'] = str(error)

    return datos


def numero_sms():
    """Cuantos multiprocesadores tiene la tarjeta, o 0 si no hay GPU."""
    if not gpu_disponible():
        return 0
    try:
        return int(cuda.get_current_device().MULTIPROCESSOR_COUNT)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Kernels CUDA
# ---------------------------------------------------------------------------

if _HAY_CUDA:

    @intrinsic
    def smid(typingctx):
        """Identificador del multiprocesador donde corre el hilo que llama.

        CUDA expone el dato en el registro especial %smid, al que no se llega
        desde Python: hay que bajar a PTX. Numba permite inyectar una
        instruccion de ensamblador dentro del kernel declarando un intrinsic,
        que es lo que hace esta funcion. Es la pieza que convierte "la GPU
        trabajo al 74 por ciento" en "cada uno de los SMs proceso esta
        cantidad de bytes".
        """
        sig = types.uint32()

        def codegen(context, builder, signature, args):
            fnty = ir.FunctionType(ir.IntType(32), [])
            asm = ir.InlineAsm(fnty, "mov.u32 $0, %smid;", "=r",
                               side_effect=True)
            return builder.call(asm, [])

        return sig, codegen

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
        final y deja el limite en memoria compartida; despues todos los hilos
        del bloque borran el rango en paralelo. Se escribe el codigo del
        salto de linea porque el histograma ya lo descarta, de modo que la
        cabecera desaparece sin necesidad de mover datos.
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
    def k_histograma(datos, n, global_hist, por_sm):
        """Histograma de 256 bins, anotando el trabajo de cada SM.

        Cada bloque acumula primero en memoria compartida y solo al final
        suma su parcial al histograma global. Esto reduce la contencion
        atomica de millones de hilos a una sola ronda de 256 sumas por
        bloque, que es lo que hace viable el kernel.

        El censo por SM se lleva en una variable local y se vuelca una sola
        vez al terminar el bucle, no en cada iteracion: asi la medicion no le
        cuesta tiempo al trabajo que esta midiendo.
        """
        local = cuda.shared.array(BINS, dtype=np.uint32)

        # Puesta a cero del histograma compartido, con paso por si el bloque
        # tiene menos de 256 hilos.
        t = cuda.threadIdx.x
        while t < BINS:
            local[t] = 0
            t += cuda.blockDim.x
        cuda.syncthreads()

        i = cuda.grid(1)
        paso = cuda.gridsize(1)
        vistos = 0
        while i < n:
            cuda.atomic.add(local, datos[i], 1)
            vistos += 1
            i += paso
        cuda.syncthreads()

        # Volcado del parcial del bloque al histograma global.
        t = cuda.threadIdx.x
        while t < BINS:
            valor = local[t]
            if valor != 0:
                cuda.atomic.add(global_hist, t, valor)
            t += cuda.blockDim.x

        if vistos != 0:
            cuda.atomic.add(por_sm, smid(), vistos)


# ---------------------------------------------------------------------------
# Precalentamiento del compilador JIT
# ---------------------------------------------------------------------------

_precalentado = False


def precalentar():
    """Compila los tres kernels sobre un lote minusculo y descarta el result.

    Numba compila cada kernel la primera vez que se lanza, y esa compilacion
    tarda alrededor de un segundo. Si no se saca del cronometro, ese segundo
    se le carga a la GPU. En el motor hibrido seria peor: el reparto de carga
    creeria que la tarjeta es lentisima y le daria casi todo el trabajo a la
    CPU. Por eso siempre se llama a esta funcion antes de medir.

    Devuelve los segundos que costo compilar, o 0.0 si ya estaba hecho.
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
    d_sm = cuda.to_device(np.zeros(MAX_SM, dtype=np.uint64))

    n = muestra.size
    # La muestra es minuscula a proposito, asi que Numba avisa de baja
    # ocupacion de la GPU. Aqui no importa: solo se quiere compilar.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        k_marcar_cabeceras[1, 32](d_buf, n, d_pos, d_cont)
        k_borrar_cabeceras[1, 32](d_buf, n, d_pos, d_cont)
        k_histograma[1, 32](d_buf, n, d_hist, d_sm)
        cuda.synchronize()

    _precalentado = True
    return time.perf_counter() - inicio


def recursos_kernel():
    """Recursos que consume el kernel del histograma dentro de un SM.

    Es la lectura de memoria a nivel de nucleo de GPU: cuantos registros
    gasta cada hilo, cuanta memoria compartida cada bloque, y cuantos warps
    caben simultaneamente en un multiprocesador con ese gasto. La ocupacion
    resultante es la fraccion del SM que el kernel consigue mantener llena.
    """
    if not gpu_disponible():
        return None
    precalentar()

    datos = info_gpu()
    if not datos.get('disponible'):
        return None

    def leer(nombre, por_defecto=0):
        # Numba devuelve un diccionario indexado por firma compilada; el
        # kernel solo tiene una, pero se toma el maximo por prudencia.
        try:
            valores = getattr(k_histograma, nombre)()
            if isinstance(valores, dict):
                return max(valores.values()) if valores else por_defecto
            return valores
        except Exception:
            return por_defecto

    registros = leer('get_regs_per_thread')
    compartida = leer('get_shared_mem_per_block')

    warp = datos['warp']
    max_warps = datos['max_hilos_sm'] // warp

    # Cuantos warps caben de verdad, tomando el recurso mas restrictivo.
    limite_registros = max_warps
    if registros:
        limite_registros = datos['max_registros_sm'] // (registros * warp)
    limite_compartida = max_warps
    if compartida:
        bloques_sm = datos['max_shared_sm'] // compartida
        limite_compartida = bloques_sm * (HILOS_BLOQUE // warp)

    warps_activos = max(0, min(max_warps, limite_registros, limite_compartida))

    return {
        'sms': datos['sms'],
        'registros_por_hilo': registros,
        'compartida_por_bloque': compartida,
        'warp': warp,
        'warps_activos': warps_activos,
        'warps_maximos': max_warps,
        'ocupacion_pct': (100.0 * warps_activos / max_warps)
                         if max_warps else 0.0,
        'max_shared_sm': datos['max_shared_sm'],
        'max_registros_sm': datos['max_registros_sm'],
    }


# ---------------------------------------------------------------------------
# Utilidades de lectura
# ---------------------------------------------------------------------------

def _recortar_a_linea(buffer, leidos):
    """Recorta el lote para que termine justo despues de un salto de linea.

    Devuelve cuantos bytes del buffer se deben procesar. Los bytes sobrantes
    se devuelven al archivo con un seek hacia atras, de forma que ninguna
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
# Tarjeta preparada para procesar varios tramos
# ---------------------------------------------------------------------------

class ContextoGPU:
    """Reserva la tarjeta una vez y procesa sobre ella los tramos que lleguen.

    En el P1.2 cada medicion abria y cerraba su propia reserva de memoria,
    porque solo se procesaba un archivo entero por corrida. Aqui no sirve: el
    motor hibrido le pide a la GPU un trozo cada pocas decimas de segundo, y
    reservar dos buffers de memoria pinned en cada uno costaria mas que el
    propio trabajo. Se reserva al construir y se libera al cerrar.

    El histograma y el censo por SM se acumulan dentro de la tarjeta entre
    tramos y se descargan al host una sola vez, al final.
    """

    def __init__(self, lote_mb=LOTE_MB, hilos_bloque=HILOS_BLOQUE,
                 bloques=BLOQUES_HIST):
        if not gpu_disponible():
            raise RuntimeError('No hay GPU CUDA disponible en esta maquina.')

        precalentar()

        self.lote = max(1, int(lote_mb)) * 1024 * 1024
        self.hilos_bloque = hilos_bloque
        self.bloques = bloques
        self.sms = numero_sms()

        # Dos buffers pinned y dos streams para solapar copia y computo. La
        # memoria pinned es obligatoria: sobre memoria paginable la copia
        # seria sincrona y no habria ningun solapamiento que ganar. Cada
        # buffer lleva holgura al final para cerrar la ultima linea de un
        # tramo sin partirla.
        self.n_buffers = 2
        self.h_buffers = [cuda.pinned_array(self.lote + _COLA_LINEA,
                                            dtype=np.uint8)
                          for _ in range(self.n_buffers)]
        self.d_buffers = [cuda.device_array(self.lote + _COLA_LINEA,
                                            dtype=np.uint8)
                          for _ in range(self.n_buffers)]
        self.d_posiciones = [cuda.device_array(MAX_CABECERAS, dtype=np.int64)
                             for _ in range(self.n_buffers)]
        self.d_contadores = [cuda.device_array(1, dtype=np.int64)
                             for _ in range(self.n_buffers)]
        self.streams = [cuda.stream() for _ in range(self.n_buffers)]

        # Acumuladores globales. El histograma va en uint64 porque un solo
        # bin puede superar los 4.290 millones en archivos grandes.
        self.d_hist = cuda.to_device(np.zeros(BINS, dtype=np.uint64))
        self.d_sm = cuda.to_device(np.zeros(MAX_SM, dtype=np.uint64))

        self._ceros = np.zeros(1, dtype=np.int64)
        self._turno = 0
        self.bytes_procesados = 0
        self.tramos = 0

    # -- procesamiento ----------------------------------------------------

    def procesar_rango(self, ruta, inicio, fin, progreso=None):
        """Cuenta el tramo [inicio, fin) del archivo sobre la tarjeta.

        Aplica exactamente la misma regla de bordes que
        motor_cpu._contar_rango: si el tramo no empieza en el byte 0 se
        descarta su primera linea partida, que pertenece al tramo anterior, y
        el ultimo lote se extiende hasta cerrar la linea en curso. Gracias a
        eso da igual que un trozo lo tome la CPU y el siguiente la GPU:
        ningun nucleotido se pierde ni se cuenta dos veces.

        Devuelve los bytes que realmente proceso.
        """
        hechos = 0

        with open(ruta, 'rb') as f:
            if inicio > 0:
                # Se retrocede un byte a proposito, igual que en la CPU: si en
                # 'inicio' empieza una linea nueva el byte anterior es el
                # salto y readline solo lo consume; si 'inicio' cae dentro de
                # una linea, readline descarta el trozo que ya conto el tramo
                # anterior.
                f.seek(inicio - 1)
                f.readline()
            posicion = f.tell()

            while posicion < fin:
                h_buf = self.h_buffers[self._turno]
                d_buf = self.d_buffers[self._turno]
                stream = self.streams[self._turno]

                # El stream de este turno pudo quedarse trabajando en el lote
                # anterior sobre el mismo buffer; hay que esperarlo antes de
                # sobreescribirlo con datos nuevos.
                stream.synchronize()

                por_leer = min(self.lote, fin - posicion)
                leidos = f.readinto(memoryview(h_buf)[:por_leer])
                if not leidos:
                    break

                if f.tell() >= fin:
                    # Ultimo lote del tramo. Solo se completa la linea si de
                    # verdad quedo cortada, es decir si el ultimo byte leido
                    # no es ya un salto de linea. La comprobacion no es un
                    # detalle: readline() colocado justo despues de un salto
                    # devuelve la linea siguiente entera, que pertenece al
                    # tramo del vecino, y esa linea acabaria contada dos
                    # veces. Como los cortes caen cada 16 MB y las lineas
                    # miden 81 bytes, la coincidencia es rara y el error solo
                    # aparecia en algunas corridas, cuando el tramo que
                    # terminaba en salto le tocaba a la GPU.
                    if leidos > 0 and h_buf[leidos - 1] != COD_SALTO:
                        resto = f.readline()
                        if resto:
                            if len(resto) > _COLA_LINEA:
                                raise ValueError(
                                    'Linea de %d bytes, mayor que la holgura '
                                    'del buffer; sube _COLA_LINEA.'
                                    % len(resto))
                            h_buf[leidos:leidos + len(resto)] = np.frombuffer(
                                resto, dtype=np.uint8)
                            leidos += len(resto)
                    utiles = leidos
                else:
                    # Lote intermedio: se recorta a linea completa y lo que
                    # sobra se devuelve al archivo con un seek hacia atras.
                    utiles = _recortar_a_linea(h_buf, leidos)
                    if utiles < leidos:
                        f.seek(utiles - leidos, os.SEEK_CUR)

                posicion = f.tell()
                if utiles == 0:
                    break

                d_buf[:utiles].copy_to_device(h_buf[:utiles], stream=stream)
                self.d_contadores[self._turno].copy_to_device(self._ceros,
                                                              stream=stream)

                malla = min(self.bloques,
                            (utiles + self.hilos_bloque - 1)
                            // self.hilos_bloque)
                # Suelo de bloques: el ultimo lote de un tramo puede quedar
                # muy pequeno despues de recortarlo, y con una malla de uno o
                # dos bloques la tarjeta queda casi vacia. Lanzar bloques de
                # sobra no cuesta nada porque los kernels usan grid-stride:
                # los hilos sin datos salen del bucle de inmediato.
                malla = max(MIN_BLOQUES, malla)

                k_marcar_cabeceras[malla, self.hilos_bloque, stream](
                    d_buf, utiles, self.d_posiciones[self._turno],
                    self.d_contadores[self._turno])
                k_borrar_cabeceras[BLOQUES_BORRADO, self.hilos_bloque, stream](
                    d_buf, utiles, self.d_posiciones[self._turno],
                    self.d_contadores[self._turno])
                k_histograma[malla, self.hilos_bloque, stream](
                    d_buf, utiles, self.d_hist, self.d_sm)

                hechos += utiles
                self.bytes_procesados += utiles
                if progreso is not None:
                    progreso(utiles)

                self._turno = (self._turno + 1) % self.n_buffers

        self.tramos += 1
        return hechos

    # -- resultados -------------------------------------------------------

    def sincronizar(self):
        """Espera a que la tarjeta termine todo lo encolado."""
        for stream in self.streams:
            stream.synchronize()
        cuda.synchronize()

    def histograma(self):
        """Descarga el histograma acumulado de todos los tramos."""
        self.sincronizar()
        return self.d_hist.copy_to_host()

    def censo_sm(self):
        """Bytes procesados por cada multiprocesador de la tarjeta.

        Devuelve una lista con una entrada por SM. No es una estimacion ni un
        reparto teorico: es la cuenta que llevaron los propios hilos dentro
        de cada multiprocesador.
        """
        self.sincronizar()
        censo = self.d_sm.copy_to_host()
        return [int(x) for x in censo[:self.sms]]

    def cerrar(self):
        """Libera la reserva de la tarjeta."""
        try:
            self.sincronizar()
        except Exception:
            pass
        self.h_buffers = []
        self.d_buffers = []
        self.d_posiciones = []
        self.d_contadores = []
        self.streams = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cerrar()
        return False


# ---------------------------------------------------------------------------
# API publica de conteo
# ---------------------------------------------------------------------------

def contar_gpu(ruta, lote_mb=LOTE_MB, hilos_bloque=HILOS_BLOQUE,
               bloques=BLOQUES_HIST, progreso=None):
    """Cuenta las bases del archivo entero usando solo la GPU.

    Devuelve (dict_resultado, tiempo_en_segundos). El diccionario es el mismo
    que devuelve motor_cpu, mas la clave 'gpu' con los parametros usados y el
    censo de bytes por multiprocesador.
    """
    tamano = os.path.getsize(ruta)

    # Fuera del cronometro: compilar los kernels no es trabajo del algoritmo.
    precalentar()

    contexto = ContextoGPU(lote_mb, hilos_bloque, bloques)
    leidos = [0]

    def avanzar(bytes_lote):
        leidos[0] += bytes_lote
        if progreso is not None:
            progreso(leidos[0], tamano)

    arranque = time.perf_counter()
    contexto.procesar_rango(ruta, 0, tamano, avanzar)
    hist = contexto.histograma()
    transcurrido = time.perf_counter() - arranque

    censo = contexto.censo_sm()
    contexto.cerrar()

    if progreso is not None:
        progreso(tamano, tamano)

    resultado = resumir_histograma(hist)
    resultado['gpu'] = {
        'lote_mb': lote_mb,
        'hilos_bloque': hilos_bloque,
        'bloques': bloques,
        'bytes_procesados': leidos[0],
        'censo_sm': censo,
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

    print('')
    print('Compilacion JIT de los kernels: %.3f s (fuera de la medicion)'
          % precalentar())

    recursos = recursos_kernel()
    if recursos:
        print('Recursos por SM  : %d registros/hilo, %d B compartidos/bloque'
              % (recursos['registros_por_hilo'],
                 recursos['compartida_por_bloque']))
        print('Ocupacion        : %d de %d warps (%.1f %%)'
              % (recursos['warps_activos'], recursos['warps_maximos'],
                 recursos['ocupacion_pct']))

    if len(sys.argv) < 2:
        print('')
        print('Uso: python motor_gpu.py <archivo.fna> [lote_mb]')
        sys.exit(0)

    archivo = sys.argv[1]
    mb = int(sys.argv[2]) if len(sys.argv) > 2 else LOTE_MB

    res, t = contar_gpu(archivo, lote_mb=mb)
    tam = os.path.getsize(archivo)
    print('')
    print('GPU lote=%d MB : %8.3f s   (%.1f MB/s)'
          % (mb, t, tam / (1024 * 1024) / t if t else 0))
    print('  A=%d C=%d G=%d T=%d N=%d invalidos=%d'
          % (res['A'], res['C'], res['G'], res['T'],
             res['N'], res['invalidos']))

    censo = res['gpu']['censo_sm']
    total = sum(censo)
    print('')
    print('Reparto entre los %d multiprocesadores:' % len(censo))
    for i, v in enumerate(censo):
        pct = 100.0 * v / total if total else 0.0
        print('  SM %2d  %14d  %5.2f %%  %s'
              % (i, v, pct, '#' * int(pct * 3)))
