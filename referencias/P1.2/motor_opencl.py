# -*- coding: utf-8 -*-
"""
motor_opencl.py

Tercer motor de conteo, escrito en OpenCL. Existe para cubrir el punto extra
del enunciado que pide un benchmark sobre diferentes GPUs, discreta frente a
integrada.

POR QUE HACE FALTA UN TERCER MOTOR
El motor de GPU principal esta escrito con Numba CUDA, y CUDA solo funciona
sobre tarjetas NVIDIA. La grafica integrada del procesador es Intel, de modo
que con CUDA no hay manera de medirla. OpenCL, en cambio, es un estandar
abierto que ambos fabricantes implementan: en este equipo expone las dos
tarjetas, la NVIDIA GeForce RTX 4050 con 20 unidades de computo y la Intel
UHD Graphics con 48.

Que el mismo codigo OpenCL corra en las dos tarjetas tiene una ventaja que va
mas alla de poder medir la integrada: hace que la comparacion entre ambas sea
limpia. Si se comparase el kernel CUDA de la NVIDIA contra un kernel distinto
de la Intel, no se sabria cuanto de la diferencia viene del hardware y cuanto
de la implementacion. Aqui el algoritmo es identico y solo cambia el
dispositivo.

Los tres kernels reproducen exactamente la estrategia de motor_gpu.py:
marcar las cabeceras, borrarlas dentro del dispositivo y calcular el
histograma de 256 bins acumulando primero en memoria local.

DIFERENCIA DE ACUMULACION RESPECTO AL MOTOR CUDA
El histograma global se mantiene en enteros de 32 bits y se descarga al host
al terminar cada lote, donde se acumula en enteros de Python, que no tienen
limite de tamano. El motor CUDA usa enteros de 64 bits en el dispositivo y
descarga una sola vez. Se hizo asi porque las sumas atomicas de 64 bits son
una extension opcional de OpenCL que no todas las implementaciones ofrecen, y
descargar 256 enteros por lote cuesta muy poco. La contrapartida es que hay
una sincronizacion por lote, de modo que este motor no solapa transferencia y
computo. No es un problema para su proposito, que es comparar dos tarjetas
entre si bajo las mismas condiciones.

API publica:
    opencl_disponible()          -> bool
    listar_dispositivos()        -> lista de dicts con las tarjetas
    contar_opencl(ruta, ...)     -> (dict_resultado, tiempo)
"""

import os
import time

import numpy as np

from motor_cpu import BINS, resumir_histograma

try:
    import pyopencl as cl
    _HAY_OPENCL = True
except ImportError:                                   # pragma: no cover
    cl = None
    _HAY_OPENCL = False


LOTE_MB = 64
GRUPO = 256                 # elementos de trabajo por grupo
GRUPOS = 1024               # grupos lanzados en el bucle con paso
MAX_CABECERAS = 1 << 18

COD_SALTO = 10
_COLA_BUSQUEDA = 4096


FUENTE = """
#define COD_MAYOR 62
#define COD_SALTO 10

/* Marca donde empieza cada linea de cabecera dentro del lote. */
__kernel void k_marcar(__global const uchar *datos,
                       const uint n,
                       __global uint *posiciones,
                       __global uint *contador,
                       const uint capacidad)
{
    uint i = get_global_id(0);
    uint paso = get_global_size(0);
    while (i < n) {
        if (datos[i] == COD_MAYOR && (i == 0 || datos[i - 1] == COD_SALTO)) {
            uint ranura = atomic_inc(contador);
            if (ranura < capacidad) {
                posiciones[ranura] = i;
            }
        }
        i += paso;
    }
}

/* Sobreescribe cada cabecera con saltos de linea, que no se cuentan. */
__kernel void k_borrar(__global uchar *datos,
                       const uint n,
                       __global const uint *posiciones,
                       __global const uint *contador,
                       const uint capacidad)
{
    __local uint fin;

    uint total = contador[0];
    if (total > capacidad) {
        total = capacidad;
    }

    uint indice = get_group_id(0);
    uint salto = get_num_groups(0);
    uint local_id = get_local_id(0);
    uint tam_grupo = get_local_size(0);

    while (indice < total) {
        uint inicio = posiciones[indice];

        if (local_id == 0) {
            uint j = inicio;
            while (j < n && datos[j] != COD_SALTO) {
                j++;
            }
            fin = j;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        uint limite = fin;
        for (uint p = inicio + local_id; p < limite; p += tam_grupo) {
            datos[p] = COD_SALTO;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        indice += salto;
    }
}

/* Histograma de 256 bins acumulando primero en memoria local. */
__kernel void k_histograma(__global const uchar *datos,
                           const uint n,
                           __global uint *hist)
{
    __local uint local_hist[256];

    uint t = get_local_id(0);
    uint tam_grupo = get_local_size(0);

    for (uint k = t; k < 256; k += tam_grupo) {
        local_hist[k] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint i = get_global_id(0);
    uint paso = get_global_size(0);
    while (i < n) {
        atomic_inc(&local_hist[datos[i]]);
        i += paso;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = t; k < 256; k += tam_grupo) {
        uint valor = local_hist[k];
        if (valor != 0) {
            atomic_add(&hist[k], valor);
        }
    }
}
"""


# ---------------------------------------------------------------------------
# Descubrimiento de dispositivos
# ---------------------------------------------------------------------------

def opencl_disponible():
    """Indica si hay al menos un dispositivo OpenCL utilizable."""
    if not _HAY_OPENCL:
        return False
    try:
        return bool(listar_dispositivos())
    except Exception:
        return False


def listar_dispositivos():
    """Devuelve la lista de GPUs OpenCL disponibles en el equipo.

    Cada elemento trae el indice con el que se selecciona el dispositivo, su
    nombre, la plataforma a la que pertenece, el numero de unidades de computo
    y la memoria global. Se filtran los dispositivos que no son GPU para que
    la comparacion sea entre tarjetas.
    """
    if not _HAY_OPENCL:
        return []

    dispositivos = []
    try:
        plataformas = cl.get_platforms()
    except Exception:
        return []

    for plataforma in plataformas:
        try:
            encontrados = plataforma.get_devices(
                device_type=cl.device_type.GPU)
        except Exception:
            continue
        for aparato in encontrados:
            # La grafica integrada comparte la memoria del sistema, asi que
            # el atributo host_unified_memory la distingue de la discreta.
            try:
                integrada = bool(aparato.host_unified_memory)
            except Exception:
                integrada = False
            dispositivos.append({
                'indice': len(dispositivos),
                'nombre': aparato.name.strip(),
                'plataforma': plataforma.name.strip(),
                'unidades': aparato.max_compute_units,
                'memoria_mb': aparato.global_mem_size // (1024 * 1024),
                'max_grupo': aparato.max_work_group_size,
                'integrada': integrada,
                'tipo': 'integrada' if integrada else 'discreta',
                '_aparato': aparato,
            })
    return dispositivos


def _elegir(dispositivo):
    """Resuelve el dispositivo pedido por indice, por nombre o por tipo."""
    lista = listar_dispositivos()
    if not lista:
        raise RuntimeError('No hay dispositivos OpenCL disponibles.')

    if dispositivo is None:
        return lista[0]
    if isinstance(dispositivo, int):
        if 0 <= dispositivo < len(lista):
            return lista[dispositivo]
        raise ValueError('Indice de dispositivo fuera de rango: %d'
                         % dispositivo)

    texto = str(dispositivo).lower()
    for datos in lista:
        if texto in ('integrada', 'discreta') and datos['tipo'] == texto:
            return datos
        if texto in datos['nombre'].lower():
            return datos
    raise ValueError('No se encontro el dispositivo: %s' % dispositivo)


# ---------------------------------------------------------------------------
# Compilacion
# ---------------------------------------------------------------------------

_cache = {}


def _preparar(datos_dispositivo):
    """Crea el contexto, la cola y el programa, y los guarda en cache.

    Compilar el programa OpenCL cuesta tiempo y no forma parte del algoritmo,
    igual que la compilacion JIT del motor CUDA. Se hace una sola vez por
    dispositivo y queda fuera de cualquier medicion.
    """
    clave = (datos_dispositivo['plataforma'], datos_dispositivo['nombre'])
    if clave in _cache:
        return _cache[clave]

    contexto = cl.Context(devices=[datos_dispositivo['_aparato']])
    cola = cl.CommandQueue(contexto)
    programa = cl.Program(contexto, FUENTE).build()

    # Los objetos de kernel se crean una sola vez y se reutilizan en todos
    # los lotes. Acceder a ellos como atributo del programa, con
    # programa.k_marcar(...), parece comodo pero construye un kernel nuevo en
    # cada llamada; con un archivo de 3 GB eso son decenas de construcciones
    # inutiles que se le cargarian al tiempo de OpenCL y ensuciarian la
    # comparacion contra CUDA. El propio pyopencl lo avisa con
    # RepeatedKernelRetrieval.
    kernels = {
        'marcar': cl.Kernel(programa, 'k_marcar'),
        'borrar': cl.Kernel(programa, 'k_borrar'),
        'histograma': cl.Kernel(programa, 'k_histograma'),
    }

    _cache[clave] = (contexto, cola, programa, kernels)
    return _cache[clave]


def precalentar(dispositivo=None):
    """Compila el programa para el dispositivo indicado y devuelve el coste."""
    if not opencl_disponible():
        return 0.0
    inicio = time.perf_counter()
    _preparar(_elegir(dispositivo))
    return time.perf_counter() - inicio


# ---------------------------------------------------------------------------
# Conteo
# ---------------------------------------------------------------------------

def contar_opencl(ruta, dispositivo=None, lote_mb=LOTE_MB, grupo=GRUPO,
                  grupos=GRUPOS, progreso=None):
    """Cuenta las bases del archivo sobre un dispositivo OpenCL.

    'dispositivo' acepta un indice, un fragmento del nombre de la tarjeta o
    las palabras 'integrada' y 'discreta'.

    Devuelve (dict_resultado, tiempo_en_segundos), con el mismo formato que
    los otros dos motores.
    """
    if not opencl_disponible():
        raise RuntimeError('No hay dispositivos OpenCL en esta maquina.')

    datos_dispositivo = _elegir(dispositivo)
    contexto, cola, programa, kernels = _preparar(datos_dispositivo)

    # El tamano de grupo no puede superar el maximo del dispositivo.
    grupo = min(int(grupo), int(datos_dispositivo['max_grupo']))

    tamano = os.path.getsize(ruta)
    lote = max(1, int(lote_mb)) * 1024 * 1024

    memoria = cl.mem_flags
    d_datos = cl.Buffer(contexto, memoria.READ_WRITE, size=lote)
    d_pos = cl.Buffer(contexto, memoria.READ_WRITE, size=MAX_CABECERAS * 4)
    d_cont = cl.Buffer(contexto, memoria.READ_WRITE, size=4)
    d_hist = cl.Buffer(contexto, memoria.READ_WRITE, size=BINS * 4)

    buffer_host = np.empty(lote, dtype=np.uint8)
    hist_lote = np.empty(BINS, dtype=np.uint32)
    ceros_hist = np.zeros(BINS, dtype=np.uint32)
    cero = np.zeros(1, dtype=np.uint32)

    # Acumulador en enteros de Python: no puede desbordarse nunca.
    total = [0] * BINS
    leidos_total = 0

    arranque = time.perf_counter()
    with open(ruta, 'rb') as f:
        while True:
            leidos = f.readinto(buffer_host)
            if not leidos:
                break

            utiles = _recortar_a_linea(buffer_host, leidos)
            if utiles < leidos:
                f.seek(utiles - leidos, os.SEEK_CUR)
            if utiles == 0:
                break

            cl.enqueue_copy(cola, d_datos, buffer_host[:utiles])
            cl.enqueue_copy(cola, d_cont, cero)
            cl.enqueue_copy(cola, d_hist, ceros_hist)

            # El numero total de elementos de trabajo debe ser multiplo del
            # tamano de grupo, requisito de OpenCL.
            necesarios = min(grupos * grupo,
                             ((utiles + grupo - 1) // grupo) * grupo)
            globales = (max(grupo, necesarios),)
            locales = (grupo,)

            n32 = np.uint32(utiles)
            cap32 = np.uint32(MAX_CABECERAS)

            kernels['marcar'](cola, globales, locales,
                              d_datos, n32, d_pos, d_cont, cap32)
            kernels['borrar'](cola, (grupos * grupo,), locales,
                              d_datos, n32, d_pos, d_cont, cap32)
            kernels['histograma'](cola, globales, locales,
                                  d_datos, n32, d_hist)

            cl.enqueue_copy(cola, hist_lote, d_hist)
            cola.finish()

            for i in range(BINS):
                total[i] += int(hist_lote[i])

            leidos_total += utiles
            if progreso is not None:
                progreso(leidos_total, tamano)

    transcurrido = time.perf_counter() - arranque

    if progreso is not None:
        progreso(tamano, tamano)

    resultado = resumir_histograma(total)
    resultado['opencl'] = {
        'dispositivo': datos_dispositivo['nombre'],
        'plataforma': datos_dispositivo['plataforma'],
        'tipo': datos_dispositivo['tipo'],
        'unidades': datos_dispositivo['unidades'],
        'lote_mb': lote_mb,
        'grupo': grupo,
        'bytes_procesados': leidos_total,
    }
    return resultado, transcurrido


def _recortar_a_linea(buffer, leidos):
    """Recorta el lote para que termine justo despues de un salto de linea."""
    if leidos == 0:
        return 0
    desde = max(0, leidos - _COLA_BUSQUEDA)
    cola = buffer[desde:leidos].tobytes()
    corte = cola.rfind(bytes([COD_SALTO]))
    if corte >= 0:
        return desde + corte + 1
    corte = buffer[:leidos].tobytes().rfind(bytes([COD_SALTO]))
    if corte >= 0:
        return corte + 1
    return leidos


# ---------------------------------------------------------------------------
# Prueba rapida por linea de comandos
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    lista = listar_dispositivos()
    if not lista:
        print('No hay dispositivos OpenCL disponibles.')
        sys.exit(1)

    print('Dispositivos OpenCL encontrados:')
    for datos in lista:
        print('  [%d] %-36s %-22s %2d unidades  %6d MB  %s'
              % (datos['indice'], datos['nombre'], datos['plataforma'],
                 datos['unidades'], datos['memoria_mb'], datos['tipo']))

    if len(sys.argv) < 2:
        print('')
        print('Uso: python motor_opencl.py <archivo.fna> [dispositivo] [lote_mb]')
        sys.exit(0)

    archivo = sys.argv[1]
    cual = sys.argv[2] if len(sys.argv) > 2 else None
    mb = int(sys.argv[3]) if len(sys.argv) > 3 else LOTE_MB
    if cual is not None and cual.isdigit():
        cual = int(cual)

    print('')
    for datos in (lista if cual is None else [_elegir(cual)]):
        coste = precalentar(datos['indice'])
        res, t = contar_opencl(archivo, dispositivo=datos['indice'],
                               lote_mb=mb)
        tam = os.path.getsize(archivo)
        print('%-36s %-10s %8.3f s  %8.1f MB/s   (compilacion %.3f s)'
              % (datos['nombre'], datos['tipo'], t,
                 tam / (1024 * 1024) / t if t else 0, coste))
        print('    A=%d C=%d G=%d T=%d N=%d iupac=%d inval=%d'
              % (res['A'], res['C'], res['G'], res['T'], res['N'],
                 res['ambiguos'], res['invalidos']))
