# -*- coding: utf-8 -*-
"""
motor_cpu.py

Motor de conteo de bases sobre archivos FASTA grandes, ejecutado en CPU.
Hereda la estructura del motor_adn.py del proyecto P1.1 (division del
archivo en rangos de bytes, cada proceso abre el archivo por su cuenta y
solo devuelve numeros) y le anade lo que exige el P1.2: conteo de N y
deteccion de caracteres invalidos con su detalle.

API publica:
    detectar_nucleos()                        -> (fisicos, logicos)
    contar_secuencial(ruta, progreso=None)    -> (dict_resultado, tiempo)
    contar_paralelo(ruta, n, progreso=None)   -> (dict_resultado, tiempo)

DECISION DE DISENO IMPORTANTE
El motor calcula siempre un histograma completo de 256 bins, uno por cada
valor de byte posible, y de ahi deriva todos los conteos. Es exactamente el
mismo algoritmo que ejecuta motor_gpu.py. Se hace asi a proposito: si la
CPU usara un atajo distinto al de la GPU, los tiempos no serian comparables
y la comparativa del proyecto no valdria nada. La unica variable entre los
dos motores debe ser el hardware que ejecuta, no el algoritmo.

Reglas de conteo:
  - Las lineas de cabecera (empiezan con '>') se ignoran por completo.
  - A, C, G y T cuentan; las minusculas del soft-masking cuentan igual.
  - N es base desconocida: se cuenta aparte, NO es un caracter invalido.
  - Los saltos de linea no son datos, se ignoran.
  - Cualquier otro caracter dentro de una linea de secuencia es INVALIDO.
"""

import multiprocessing
import os
import time

import numpy as np

try:
    import psutil
    _HAY_PSUTIL = True
except ImportError:
    psutil = None
    _HAY_PSUTIL = False


# Tamano de bloque de lectura. 8 MB da buen equilibrio entre llamadas al
# sistema operativo y memoria ocupada por proceso.
_BLOQUE = 8 * 1024 * 1024

# Tamano minimo de rango por proceso. Evita que una misma linea abarque un
# rango completo y termine contada dos veces en archivos muy pequenos.
_MIN_RANGO = 64 * 1024

# Numero de bins del histograma: un byte puede valer de 0 a 255.
BINS = 256

# Codigos ASCII que interesan.
COD_A, COD_C, COD_G, COD_T, COD_N = 65, 67, 71, 84, 78
COD_a, COD_c, COD_g, COD_t, COD_n = 97, 99, 103, 116, 110

# Bytes que no son datos de secuencia y por tanto no se juzgan: el salto de
# linea de Unix y el retorno de carro de Windows.
IGNORADOS = (10, 13)

# Bases reconocidas, en mayuscula y minuscula.
CONOCIDOS = (COD_A, COD_a, COD_C, COD_c, COD_G, COD_g,
             COD_T, COD_t, COD_N, COD_n)

# Codigos IUPAC de ambiguedad. Son notacion estandar y legitima del formato
# FASTA para posiciones donde la secuenciacion no llego a resolver una base
# concreta pero si acoto las posibilidades:
#   R = A o G (purina)          Y = C o T (pirimidina)
#   S = G o C (enlace fuerte)   W = A o T (enlace debil)
#   K = G o T (ceto)            M = A o C (amino)
#   B = C, G o T (no A)         D = A, G o T (no C)
#   H = A, C o T (no G)         V = A, C o G (no T)
# En el genoma humano GRCh38 aparecen 103 de estos codigos repartidos por
# los 3.11 GB del archivo. Contarlos como caracteres invalidos seria un
# error: no son basura, son informacion parcial. Por eso el motor los cuenta
# en su propia categoria, separados tanto de las bases como de los errores.
IUPAC = 'RYSWKMBDHV'
CODIGOS_IUPAC = tuple([ord(c) for c in IUPAC] +
                      [ord(c.lower()) for c in IUPAC])

# Contador compartido entre procesos, lo inyecta el inicializador del Pool.
_contador_compartido = None


# ---------------------------------------------------------------------------
# Deteccion de hardware
# ---------------------------------------------------------------------------

def detectar_nucleos():
    """Devuelve la tupla (nucleos_fisicos, nucleos_logicos).

    Usa psutil si esta instalado. Si no lo esta, cae a la libreria estandar
    y estima los fisicos como la mitad de los logicos cuando hay indicios
    de hyper-threading.
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


# ---------------------------------------------------------------------------
# Interpretacion del histograma
# ---------------------------------------------------------------------------

def nombre_byte(codigo):
    """Nombre legible de un byte para mostrarlo en informes y tablas."""
    if 32 <= codigo < 127:
        return chr(codigo)
    return '\\x%02x' % codigo


def resumir_histograma(hist):
    """Traduce un histograma de 256 bins al diccionario de resultados.

    Recibe cualquier secuencia indexable de 256 enteros y devuelve el dict
    que usan por igual el motor de CPU y el de GPU. Esta funcion es la unica
    que conoce el significado de cada byte, de modo que los dos motores
    interpretan sus resultados exactamente igual.
    """
    hist = [int(x) for x in hist]

    a = hist[COD_A] + hist[COD_a]
    c = hist[COD_C] + hist[COD_c]
    g = hist[COD_G] + hist[COD_g]
    t = hist[COD_T] + hist[COD_t]
    n = hist[COD_N] + hist[COD_n]

    # Se clasifica todo lo demas en dos grupos distintos: los codigos IUPAC
    # de ambiguedad, que son validos aunque no sean una base concreta, y los
    # caracteres realmente invalidos, que no pertenecen al formato.
    detalle = {}
    detalle_iupac = {}
    invalidos = 0
    ambiguos = 0

    for codigo in range(BINS):
        cuenta = hist[codigo]
        if not cuenta or codigo in CONOCIDOS or codigo in IGNORADOS:
            continue
        if codigo in CODIGOS_IUPAC:
            ambiguos += cuenta
            # Las minusculas se agrupan con su mayuscula, igual que se hace
            # con las bases: el soft-masking no cambia el significado.
            detalle_iupac[chr(codigo).upper()] = \
                detalle_iupac.get(chr(codigo).upper(), 0) + cuenta
        else:
            invalidos += cuenta
            detalle[nombre_byte(codigo)] = cuenta

    return {
        'A': a, 'C': c, 'G': g, 'T': t,
        'N': n,
        'bases': a + c + g + t,          # bases realmente identificadas
        'ambiguos': ambiguos,            # codigos IUPAC, validos pero no base
        'detalle_iupac': detalle_iupac,
        'invalidos': invalidos,          # caracteres que no son del formato
        'detalle': detalle,
        'histograma': hist,
    }


def bins_significativos(hist):
    """Devuelve el histograma sin los bytes que no son datos de secuencia.

    Sirve para comparar resultados entre el motor de CPU y el de GPU. Los dos
    coinciden siempre en los bins que importan, pero NO en el bin del salto
    de linea, y es correcto que asi sea: el motor de CPU elimina las lineas
    de cabecera del buffer, mientras que el de GPU las sobreescribe con
    saltos de linea porque borrar de verdad obligaria a mover datos dentro de
    la tarjeta. La cabecera desaparece igual en ambos casos, pero el segundo
    deja saltos de linea de mas. Como los saltos no se cuentan como nada, la
    diferencia no afecta a ningun resultado; simplemente no se debe comparar
    ese bin.
    """
    return tuple(int(hist[i]) for i in range(BINS) if i not in IGNORADOS)


def sumar_histogramas(partes):
    """Suma bin a bin los histogramas devueltos por varios procesos."""
    total = [0] * BINS
    for parte in partes:
        for i in range(BINS):
            total[i] += int(parte[i])
    return total


# ---------------------------------------------------------------------------
# Nucleo de conteo
# ---------------------------------------------------------------------------

def _histograma_bloque(bloque):
    """Devuelve el histograma de 256 bins de un bloque de bytes.

    El bloque siempre llega alineado a lineas completas, por lo que se puede
    descartar de forma segura cualquier linea de cabecera que contenga.
    """
    if b'>' in bloque:
        # Ruta lenta: el bloque incluye cabeceras. Se parte en lineas y se
        # eliminan las que empiezan con el signo mayor que. Ocurre pocas
        # veces porque hay pocas cabeceras frente a millones de lineas.
        lineas = bloque.split(b'\n')
        bloque = b'\n'.join(l for l in lineas if not l.startswith(b'>'))

    # Una sola pasada en C sobre el bloque. Es el equivalente exacto del
    # kernel de histograma que corre en la GPU.
    datos = np.frombuffer(bloque, dtype=np.uint8)
    return np.bincount(datos, minlength=BINS)


def _contar_rango(ruta, inicio, fin, reportar=None):
    """Histograma del archivo entre los bytes [inicio, fin).

    El rango se ajusta a lineas completas: si no empieza en el byte 0 se
    descarta la linea partida inicial (la cuenta el rango anterior) y el
    ultimo bloque se extiende hasta el fin de linea. Asi ningun nucleotido
    se pierde ni se cuenta dos veces.

    'reportar' es una funcion opcional que recibe los bytes procesados en
    cada bloque para alimentar el progreso.
    """
    total = np.zeros(BINS, dtype=np.int64)

    with open(ruta, 'rb') as f:
        if inicio > 0:
            # Se retrocede un byte a proposito. Si en 'inicio' empieza una
            # linea nueva, el byte anterior es el salto de linea y readline
            # solo consume ese salto, con lo que no se pierde la linea. Si
            # 'inicio' cae dentro de una linea, readline descarta el trozo
            # sobrante, que ya lo conto el rango anterior.
            f.seek(inicio - 1)
            f.readline()
        posicion = f.tell()

        while posicion < fin:
            bloque = f.read(min(_BLOQUE, fin - posicion))
            if not bloque:
                break

            # Se completa la ultima linea para no cortar una linea a la mitad.
            if not bloque.endswith(b'\n'):
                resto = f.readline()
                if resto:
                    bloque += resto

            posicion = f.tell()
            total += _histograma_bloque(bloque)

            if reportar is not None:
                reportar(len(bloque))

    return total


def _iniciar_worker(contador):
    """Inicializador del Pool: guarda el contador compartido en el hijo."""
    global _contador_compartido
    _contador_compartido = contador


def _trabajo(ruta, inicio, fin):
    """Tarea que ejecuta cada proceso hijo sobre su rango de bytes.

    Devuelve una lista de 256 enteros. Es lo unico que viaja de vuelta al
    proceso padre: la secuencia nunca se transfiere entre procesos.
    """
    contador = _contador_compartido

    if contador is None:
        reportar = None
    else:
        def reportar(leidos):
            # Un incremento por bloque de 8 MB, el coste del lock es minimo.
            with contador.get_lock():
                contador.value += leidos

    return [int(x) for x in _contar_rango(ruta, inicio, fin, reportar)]


# ---------------------------------------------------------------------------
# API publica de conteo
# ---------------------------------------------------------------------------

def contar_secuencial(ruta, progreso=None):
    """Cuenta recorriendo el archivo en un solo proceso.

    Devuelve (dict_resultado, tiempo_en_segundos).
    """
    tamano = os.path.getsize(ruta)

    if progreso is None:
        reportar = None
    else:
        acumulado = [0]

        def reportar(leidos):
            acumulado[0] += leidos
            progreso(acumulado[0], tamano)

    inicio = time.perf_counter()
    hist = _contar_rango(ruta, 0, tamano, reportar)
    transcurrido = time.perf_counter() - inicio

    if progreso is not None:
        progreso(tamano, tamano)

    return resumir_histograma(hist), transcurrido


def contar_paralelo(ruta, n_procesos, progreso=None):
    """Cuenta repartiendo el archivo entre varios procesos.

    El archivo se divide en n rangos de bytes de tamano similar y cada
    proceso lee y cuenta el suyo de forma independiente. No se transfiere
    la secuencia entre procesos: cada hijo abre el archivo por su cuenta y
    solo devuelve su histograma de 256 enteros.

    Devuelve (dict_resultado, tiempo_en_segundos). El tiempo incluye la
    creacion de los procesos, que es parte real del coste de paralelizar.
    """
    tamano = os.path.getsize(ruta)
    n = max(1, int(n_procesos))

    # Con archivos muy pequenos se reduce el numero de rangos para que
    # ninguno quede por debajo del minimo seguro.
    if tamano // n < _MIN_RANGO:
        n = max(1, tamano // _MIN_RANGO)

    # Cortes calculados con enteros para que cubran el archivo completo.
    limites = [(tamano * i) // n for i in range(n + 1)]
    tareas = [(ruta, limites[i], limites[i + 1]) for i in range(n)]

    contador = multiprocessing.Value('q', 0)

    inicio = time.perf_counter()
    with multiprocessing.Pool(processes=n,
                              initializer=_iniciar_worker,
                              initargs=(contador,)) as pool:
        pendiente = pool.starmap_async(_trabajo, tareas)

        # Mientras los hijos trabajan, el padre solo consulta el contador.
        while not pendiente.ready():
            pendiente.wait(0.1)
            if progreso is not None:
                progreso(contador.value, tamano)

        partes = pendiente.get()
    transcurrido = time.perf_counter() - inicio

    if progreso is not None:
        progreso(tamano, tamano)

    return resumir_histograma(sumar_histogramas(partes)), transcurrido


# ---------------------------------------------------------------------------
# Prueba rapida por linea de comandos
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    multiprocessing.freeze_support()

    fisicos, logicos = detectar_nucleos()
    print('Nucleos fisicos: %d   Nucleos logicos: %d' % (fisicos, logicos))
    print('psutil disponible: %s' % ('si' if _HAY_PSUTIL else 'no'))

    if len(sys.argv) < 2:
        print('')
        print('Uso: python motor_cpu.py <archivo.fna> [n_procesos]')
        sys.exit(0)

    archivo = sys.argv[1]
    procesos = int(sys.argv[2]) if len(sys.argv) > 2 else logicos

    res_s, t_s = contar_secuencial(archivo)
    print('')
    print('Secuencial    : %8.3f s' % t_s)
    print('  A=%d C=%d G=%d T=%d N=%d invalidos=%d'
          % (res_s['A'], res_s['C'], res_s['G'], res_s['T'],
             res_s['N'], res_s['invalidos']))
    if res_s['detalle']:
        print('  detalle invalidos: %s' % res_s['detalle'])

    res_p, t_p = contar_paralelo(archivo, procesos)
    print('Paralelo (%2d) : %8.3f s' % (procesos, t_p))
    print('  A=%d C=%d G=%d T=%d N=%d invalidos=%d'
          % (res_p['A'], res_p['C'], res_p['G'], res_p['T'],
             res_p['N'], res_p['invalidos']))

    iguales = res_s['histograma'] == res_p['histograma']
    print('')
    print('Histogramas identicos : %s' % iguales)
    if t_p > 0:
        print('Speedup               : %.2fx' % (t_s / t_p))
