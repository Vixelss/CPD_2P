# -*- coding: utf-8 -*-
"""
evidencias.py

Recoge en un solo sitio las evidencias que pide la rubrica del P1.3, que
esta planteada como una matriz: seis criterios por tres niveles de detalle
crecientes, CPU, GPU y nucleos de GPU.

Los tres primeros criterios (programa, libreria e instruccion) son
declarativos: dicen donde esta el codigo de cada nivel y con que se hace.
Los tres ultimos (uso, temperatura y memoria) son mediciones en vivo, y se
rellenan con lo que el monitor y los motores acaban de leer.

Nada de lo que sale de aqui esta escrito a mano en el informe: la aplicacion
lo muestra en pantalla y el generador del documento lo toma de la misma
funcion, de modo que informe y programa no pueden contradecirse.
"""

import os
import textwrap


# Los tres niveles de la rubrica, en el orden en que aparecen en la tabla.
NIVELES = ('CPU', 'GPU', 'NUCLEOS')

ROTULOS = {
    'CPU': 'CPU',
    'GPU': 'GPU',
    'NUCLEOS': 'GPU x nucleos',
}


# ---------------------------------------------------------------------------
# Criterios declarativos
# ---------------------------------------------------------------------------

# 1) PROGRAMA: donde vive el codigo de cada nivel.
PROGRAMA = {
    'CPU': 'motor_cpu.py\n'
           'contar_paralelo() reparte rangos de bytes entre procesos y\n'
           'contar_rango() cuenta uno con numpy.bincount.',
    'GPU': 'motor_gpu.py\n'
           'Tres kernels encadenados: marcar cabeceras, borrarlas dentro\n'
           'de la tarjeta y calcular el histograma de 256 bins.',
    'NUCLEOS': 'motor_gpu.py -> k_histograma\n'
               'Cada hilo acumula en memoria compartida de su propio SM y\n'
               'anota en que multiprocesador se ejecuto.',
}

# 2) LIBRERIA: que biblioteca habilita cada nivel.
LIBRERIA = {
    'CPU': 'multiprocessing (biblioteca estandar)\n'
           'numpy para el conteo vectorizado dentro de cada proceso.',
    'GPU': 'numba.cuda\n'
           'Compila a PTX los kernels escritos en Python y gestiona\n'
           'memoria, streams y lanzamientos.',
    'NUCLEOS': 'numba.extending.intrinsic + llvmlite.ir\n'
               'Permiten inyectar ensamblador PTX dentro del kernel, que es\n'
               'la unica via para consultar el multiprocesador en curso.',
}

# 3) INSTRUCCION: la instruccion concreta que pone a trabajar cada nivel.
INSTRUCCION = {
    'CPU': 'multiprocessing.Pool(processes=n)\n'
           'pool.starmap_async(_trabajo_cpu, tareas)',
    'GPU': '@cuda.jit sobre la funcion del kernel\n'
           'k_histograma[bloques, hilos, stream](datos, n, hist, por_sm)',
    'NUCLEOS': 'mov.u32 $0, %smid;   (ensamblador PTX)\n'
               'cuda.atomic.add(por_sm, smid(), vistos)',
}


# ---------------------------------------------------------------------------
# Formato de las mediciones
# ---------------------------------------------------------------------------

def _pct(valor, decimales=1):
    """Porcentaje formateado, o None si la magnitud no se llego a medir."""
    if valor in ('', None):
        return None
    return ('%.' + str(decimales) + 'f %%') % valor


def _reparto_nucleos(valores, etiqueta='nucleo'):
    """Resume una lista de porcentajes por nucleo en dos lineas legibles.

    Se muestran el minimo, el maximo y la media porque lo que interesa de un
    reparto no es cada valor suelto sino si esta equilibrado. La lista
    completa se dibuja aparte, en la pestana de nucleos.
    """
    if not valores:
        return None
    menor = min(valores)
    mayor = max(valores)
    medio = sum(valores) / len(valores)
    return ('%d %ss activos\nmedia %.2f %%, entre %.2f %% y %.2f %%'
            % (len(valores), etiqueta, medio, menor, mayor))


def censo_a_porcentajes(censo):
    """Convierte el censo de bytes por SM en porcentajes del total."""
    total = sum(censo) if censo else 0
    if not total:
        return []
    return [100.0 * v / total for v in censo]


# ---------------------------------------------------------------------------
# Construccion de la matriz
# ---------------------------------------------------------------------------

def matriz(recursos=None, censo_sm=None, recursos_kernel=None,
           info_gpu=None, nucleos_cpu=None):
    """Arma la matriz completa de la rubrica.

    Parametros, todos opcionales: lo que no se haya medido todavia
    simplemente no aparece, en lugar de rellenarse con un cero que se leeria
    como una medicion real.

        recursos        : resumen de monitor.Monitor.
        censo_sm        : bytes procesados por cada SM, de motor_gpu.
        recursos_kernel : ocupacion y registros, de motor_gpu.
        info_gpu        : caracteristicas de la tarjeta.
        nucleos_cpu     : uso medio por nucleo logico, del monitor.

    Devuelve una lista de criterios; cada uno trae su nombre y un diccionario
    de celdas indexado por nivel. Una celda ausente se dibuja vacia.
    """
    recursos = recursos or {}
    info_gpu = info_gpu or {}

    filas = [
        {'criterio': '1) PROGRAMA', 'celdas': dict(PROGRAMA)},
        {'criterio': '2) LIBRERIA', 'celdas': dict(LIBRERIA)},
        {'criterio': '3) INSTRUCCION', 'celdas': dict(INSTRUCCION)},
    ]

    # 4) Uso por nucleo -----------------------------------------------------
    uso = {}
    if nucleos_cpu:
        uso['CPU'] = ('%d nucleos logicos\nmedia %.1f %%, entre %.1f %% y '
                      '%.1f %%'
                      % (len(nucleos_cpu), sum(nucleos_cpu) / len(nucleos_cpu),
                         min(nucleos_cpu), max(nucleos_cpu)))
    elif _pct(recursos.get('cpu_medio')):
        uso['CPU'] = 'media %s' % _pct(recursos.get('cpu_medio'))

    if _pct(recursos.get('gpu_medio')):
        uso['GPU'] = ('media %s\nmaximo %s'
                      % (_pct(recursos.get('gpu_medio')),
                         _pct(recursos.get('gpu_max'))))

    porcentajes = censo_a_porcentajes(censo_sm or [])
    reparto = _reparto_nucleos(porcentajes, 'SM')
    if reparto:
        uso['NUCLEOS'] = reparto
    filas.append({'criterio': '4) % USO POR NUCLEO', 'celdas': uso})

    # 5) Temperatura --------------------------------------------------------
    temperatura = {}
    if recursos.get('cpu_temp_medio') not in ('', None):
        temperatura['CPU'] = ('media %.1f C\nmaximo %.1f C'
                              % (recursos['cpu_temp_medio'],
                                 recursos.get('cpu_temp_max',
                                              recursos['cpu_temp_medio'])))
    if recursos.get('gpu_temp_max') not in ('', None):
        temperatura['GPU'] = ('media %.1f C\nmaximo %.0f C'
                              % (recursos.get('gpu_temp_medio',
                                              recursos['gpu_temp_max']),
                                 recursos['gpu_temp_max']))
    filas.append({'criterio': '5) TEMPERATURA', 'celdas': temperatura})

    # 6) Uso de memoria -----------------------------------------------------
    memoria = {}
    if recursos.get('ram_medio') not in ('', None):
        texto = 'RAM del sistema %s' % _pct(recursos['ram_medio'])
        if recursos.get('proc_mb_max') not in ('', None):
            texto += '\nprograma y procesos %d MB' % recursos['proc_mb_max']
        memoria['CPU'] = texto

    if recursos.get('vram_mb_max') not in ('', None) and info_gpu.get('vram_total_mb'):
        total = info_gpu['vram_total_mb']
        usada = recursos['vram_mb_max']
        memoria['GPU'] = ('VRAM %d MB de %d MB (%.1f %%)'
                          % (usada, total, 100.0 * usada / total))

    if recursos_kernel:
        memoria['NUCLEOS'] = (
            '%d registros por hilo\n'
            '%d B compartidos por bloque\n'
            'ocupacion %d/%d warps por SM (%.0f %%)'
            % (recursos_kernel['registros_por_hilo'],
               recursos_kernel['compartida_por_bloque'],
               recursos_kernel['warps_activos'],
               recursos_kernel['warps_maximos'],
               recursos_kernel['ocupacion_pct']))
    filas.append({'criterio': '6) % USO DE MEMORIA', 'celdas': memoria})

    return filas


# ---------------------------------------------------------------------------
# Salida en texto
# ---------------------------------------------------------------------------

def partir(celda, ancho):
    """Reparte el contenido de una celda en lineas que quepan en 'ancho'.

    Se respetan los saltos de linea que ya trae el texto, y cada uno de esos
    tramos se ajusta al ancho de la columna. Sin esto, una celda larga se
    desborda sobre la columna vecina y la tabla deja de leerse.
    """
    if not celda:
        return ['']
    lineas = []
    for tramo in celda.split('\n'):
        if not tramo:
            lineas.append('')
            continue
        lineas.extend(textwrap.wrap(tramo, ancho) or [''])
    return lineas


def texto(filas, ancho=36):
    """Dibuja la matriz como tabla de texto, para consola y para el informe."""
    lineas = []
    cabecera = '%-21s %s' % ('CRITERIO',
                             ' '.join(('%-*s' % (ancho, ROTULOS[n]))
                                      for n in NIVELES))
    lineas.append(cabecera)
    lineas.append('=' * len(cabecera))

    for fila in filas:
        trozos = [partir(fila['celdas'].get(n, ''), ancho) for n in NIVELES]
        alto = max(len(t) for t in trozos)
        for i in range(alto):
            etiqueta = fila['criterio'] if i == 0 else ''
            partes = ['%-*s' % (ancho, t[i] if i < len(t) else '')
                      for t in trozos]
            lineas.append(('%-21s %s' % (etiqueta, ' '.join(partes))).rstrip())
        lineas.append('-' * len(cabecera))

    return '\n'.join(lineas)


def guardar(filas, destino):
    """Escribe la matriz en un archivo de texto."""
    carpeta = os.path.dirname(destino)
    if carpeta and not os.path.isdir(carpeta):
        os.makedirs(carpeta)
    with open(destino, 'w', encoding='utf-8') as f:
        f.write('MATRIZ DE EVIDENCIAS  -  P1.3 Computacion CPU & GPU Paralela\n')
        f.write('=' * 78 + '\n\n')
        f.write(texto(filas))
        f.write('\n')
    return destino


# ---------------------------------------------------------------------------
# Prueba rapida por linea de comandos
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time

    import monitor as mod_monitor

    print('Tomando lecturas durante unos segundos...')
    with mod_monitor.Monitor() as m:
        time.sleep(6)
    resumen = m.resumen()

    censo = None
    recursos = None
    datos = {}
    try:
        import motor_gpu
        if motor_gpu.gpu_disponible():
            datos = motor_gpu.info_gpu()
            recursos = motor_gpu.recursos_kernel()
    except Exception:
        pass

    print('')
    print(texto(matriz(recursos=resumen, censo_sm=censo,
                       recursos_kernel=recursos, info_gpu=datos,
                       nucleos_cpu=resumen.get('cpu_por_nucleo'))))
