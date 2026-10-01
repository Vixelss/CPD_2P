# -*- coding: utf-8 -*-
"""
evidencias.py

Recoge en un solo sitio las evidencias que pide la rubrica del P1.4, que esta
planteada como una matriz: seis criterios por tres plataformas, CPU, GPU y
NPU.

Los tres primeros criterios (programa, libreria e instruccion) son
declarativos: dicen donde esta el codigo de cada plataforma y con que se hace.
Los tres ultimos (uso por nucleo, temperatura y memoria) son mediciones en
vivo, y se rellenan con lo que el monitor y los motores acaban de leer.

REGLA QUE NO SE DEBE ROMPER
Una celda que no se pudo medir se deja VACIA, nunca en cero. Un cero se
leeria como "se midio y salio cero", que es una afirmacion distinta. Esto
importa especialmente en la columna de NPU: en un equipo sin NPU las celdas
de medicion salen vacias con su explicacion, y eso es exactamente lo que hay
que ensenar en el informe.

CELDAS QUE NO EXISTEN EN EL HARDWARE
Dos casillas de la rubrica piden algo que el hardware no publica, y el
programa lo dice en vez de dejarlas en blanco:

  - Temperatura por multiprocesador de GPU: la tarjeta tiene un unico sensor
    termico de die. NVML no expone temperatura por SM porque ese sensor no
    existe fisicamente.
  - Temperatura de la NPU: la NPU esta integrada en el mismo die que el
    procesador y comparte su sensor. No hay lectura independiente.

Decirlo con su motivo es una respuesta tecnica; dejarlo en blanco parece una
omision.
"""

import os
import textwrap


# Las tres plataformas de la rubrica, en el orden en que aparecen.
NIVELES = ('CPU', 'GPU', 'NPU')

ROTULOS = {
    'CPU': 'CPU',
    'GPU': 'GPU',
    'NPU': 'NPU',
}


# ---------------------------------------------------------------------------
# Criterios declarativos
# ---------------------------------------------------------------------------

# 1) PROGRAMA: donde vive el codigo de cada plataforma.
PROGRAMA = {
    'CPU': 'motor_cpu.py\n'
           'comparar_paralelo() corta el rango en n tramos y cada proceso\n'
           'compara el suyo; comparar_rango() enfrenta los dos vectores\n'
           'con una sola operacion vectorizada de numpy.',
    'GPU': 'motor_gpu.py -> k_comparar\n'
           'Un hilo por posicion con grid-stride loop. Cada bloque cuenta\n'
           'en memoria compartida y vuelca su total una sola vez al\n'
           'contador global.',
    'NPU': 'motor_npu.py -> construir_modelo()\n'
           'La comparacion se expresa como un grafo de red neuronal:\n'
           'resta, cuadrado, umbral y n capas de convolucion que suman\n'
           'las diferencias en arbol.',
}

# 2) LIBRERIA: que biblioteca habilita cada plataforma.
LIBRERIA = {
    'CPU': 'multiprocessing (biblioteca estandar)\n'
           'numpy para la comparacion vectorizada dentro de cada proceso.',
    'GPU': 'numba.cuda\n'
           'Compila a PTX el kernel escrito en Python y gestiona memoria,\n'
           'streams y lanzamientos.',
    'NPU': 'onnxruntime\n'
           'Carga el grafo y lo despacha al acelerador mediante el\n'
           'proveedor de ejecucion del fabricante (QNN en Qualcomm,\n'
           'OpenVINO en Intel, VitisAI en AMD).',
}

# 3) INSTRUCCION: la instruccion concreta que pone a trabajar cada plataforma.
INSTRUCCION = {
    'CPU': 'multiprocessing.Pool(processes=n)\n'
           'pool.starmap(_trabajo, tareas)',
    'GPU': '@cuda.jit sobre la funcion del kernel\n'
           'k_comparar[bloques, hilos, stream](a, b, n, ...)',
    'NPU': 'onnxruntime.InferenceSession(modelo, providers=[...])\n'
           'sesion.run(None, {"A": lote_a, "B": lote_b})',
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


# ---------------------------------------------------------------------------
# Construccion de la matriz
# ---------------------------------------------------------------------------

def matriz(recursos=None, recursos_kernel=None, info_gpu=None,
           info_npu=None, nucleos_cpu=None, datos_npu=None,
           por_plataforma=None, datos_gpu=None):
    """Arma la matriz completa de la rubrica.

    Parametros, todos opcionales: lo que no se haya medido todavia
    simplemente no aparece, en lugar de rellenarse con un cero que se leeria
    como una medicion real.

        recursos        : resumen de monitor.Monitor.
        recursos_kernel : ocupacion y registros, de motor_gpu.
        info_gpu        : caracteristicas de la tarjeta.
        info_npu        : proveedor elegido, de motor_npu.
        nucleos_cpu     : uso medio por nucleo logico, del monitor.
        datos_npu       : bloque 'npu' del ultimo resultado de comparacion.
        por_plataforma  : {'CPU': resumen, 'GPU': resumen, 'NPU': resumen},
                          cada uno medido SOLO mientras trabajaba esa
                          plataforma. Si se da, cada columna usa el suyo y
                          una plataforma que no corrio queda vacia.
        datos_gpu       : bloque 'gpu' del ultimo resultado de comparacion.

    POR QUE POR PLATAFORMA
    Un unico resumen de toda la sesion mezcla los ratos en que trabajaba cada
    plataforma con los ratos en que no hacia nada, y la matriz decia "CPU al
    59 %" mientras la grafica, medida solo durante la CPU, decia 95 %. Las dos
    tienen que salir de las mismas muestras.

    Devuelve una lista de criterios; cada uno trae su nombre y un diccionario
    de celdas indexado por plataforma. Una celda ausente se dibuja vacia.
    """
    recursos = recursos or {}
    info_gpu = info_gpu or {}
    info_npu = info_npu or {}
    datos_npu = datos_npu or {}
    datos_gpu = datos_gpu or {}
    if por_plataforma is None:
        rec_cpu = rec_gpu = rec_npu = recursos
    else:
        rec_cpu = por_plataforma.get('CPU') or {}
        rec_gpu = por_plataforma.get('GPU') or {}
        rec_npu = por_plataforma.get('NPU') or {}

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
    elif _pct(rec_cpu.get('cpu_medio')):
        uso['CPU'] = 'media %s' % _pct(rec_cpu.get('cpu_medio'))

    if _pct(rec_gpu.get('gpu_medio')):
        detalle = 'media %s\nmaximo %s' % (_pct(rec_gpu.get('gpu_medio')),
                                           _pct(rec_gpu.get('gpu_max')))
        if info_gpu.get('sms'):
            detalle += ('\n%d multiprocesadores, %d hilos por bloque'
                        % (info_gpu['sms'],
                           (recursos_kernel or {}).get('hilos_bloque', 256)))
        if datos_gpu.get('bloques'):
            detalle += '\n%d bloques lanzados' % datos_gpu['bloques']
        uso['GPU'] = detalle

    if _pct(rec_npu.get('npu_medio')):
        uso['NPU'] = ('media %s\nmaximo %s\n%d capas del grafo'
                      % (_pct(rec_npu.get('npu_medio')),
                         _pct(rec_npu.get('npu_max')),
                         datos_npu.get('capas', 0)))
    elif datos_npu.get('capas'):
        # No hay contador de NPU en este equipo, pero si se ejecuto el grafo.
        # Se publica lo que si se sabe y se dice por que falta lo demas.
        uso['NPU'] = ('%d capas del grafo, proveedor %s\n'
                      '(este equipo no publica contador de uso de NPU)'
                      % (datos_npu['capas'],
                         datos_npu.get('proveedor', '?')))
    filas.append({'criterio': '4) % USO POR NUCLEO', 'celdas': uso})

    # 5) Temperatura --------------------------------------------------------
    temperatura = {}
    if rec_cpu.get('cpu_temp_medio') not in ('', None):
        temperatura['CPU'] = ('media %.1f C\nmaximo %.1f C'
                              % (rec_cpu['cpu_temp_medio'],
                                 rec_cpu.get('cpu_temp_max',
                                              rec_cpu['cpu_temp_medio'])))
    if rec_gpu.get('gpu_temp_max') not in ('', None):
        temperatura['GPU'] = ('media %.1f C\nmaximo %.0f C\n'
                              '(sensor unico de die: la tarjeta no publica\n'
                              'temperatura por multiprocesador)'
                              % (rec_gpu.get('gpu_temp_medio',
                                              rec_gpu['gpu_temp_max']),
                                 rec_gpu['gpu_temp_max']))

    # La NPU no tiene sensor propio: va integrada en el mismo die que el
    # procesador y comparte su zona termica. Se publica esa lectura y se
    # explica, que es la respuesta correcta y no una casilla en blanco.
    if rec_npu.get('cpu_temp_medio') not in ('', None):
        temperatura['NPU'] = (
            'media %.1f C  (del paquete)\n'
            'La NPU esta integrada en el mismo die que el procesador\n'
            'y comparte su sensor termico: no existe lectura separada.'
            % rec_npu['cpu_temp_medio'])
    filas.append({'criterio': '5) TEMPERATURA', 'celdas': temperatura})

    # 6) Uso de memoria -----------------------------------------------------
    memoria = {}
    if rec_cpu.get('ram_medio') not in ('', None):
        texto = 'RAM del sistema %s' % _pct(rec_cpu['ram_medio'])
        if rec_cpu.get('proc_mb_max') not in ('', None):
            texto += '\nprograma y procesos %d MB' % rec_cpu['proc_mb_max']
        memoria['CPU'] = texto

    if (rec_gpu.get('vram_mb_max') not in ('', None)
            and info_gpu.get('vram_total_mb')):
        total = info_gpu['vram_total_mb']
        usada = rec_gpu['vram_mb_max']
        texto = ('VRAM %d MB de %d MB (%.1f %%)'
                 % (usada, total, 100.0 * usada / total))
        if recursos_kernel:
            texto += ('\n%d registros por hilo\n'
                      '%d B compartidos por bloque\n'
                      'ocupacion %d/%d warps por SM (%.0f %%)'
                      % (recursos_kernel['registros_por_hilo'],
                         recursos_kernel['compartida_por_bloque'],
                         recursos_kernel['warps_activos'],
                         recursos_kernel['warps_maximos'],
                         recursos_kernel['ocupacion_pct']))
        memoria['GPU'] = texto

    if datos_npu.get('lote'):
        # La NPU no tiene memoria propia: usa la RAM del sistema por un bus
        # compartido. Es la diferencia de fondo con la GPU y merece decirse.
        memoria['NPU'] = (
            'Sin memoria dedicada: la NPU usa la RAM del sistema\n'
            'por un bus compartido, sin transferencia por PCIe.\n'
            'Tensor de entrada: 2 x %d bytes por lote'
            % datos_npu['lote'])
    filas.append({'criterio': '6) % USO DE MEMORIA', 'celdas': memoria})

    return filas


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------

def partir(celda, ancho):
    """Parte el texto de una celda en lineas que quepan en el ancho dado."""
    if not celda:
        return ['']
    lineas = []
    for parrafo in str(celda).split('\n'):
        if len(parrafo) <= ancho:
            lineas.append(parrafo)
        else:
            lineas.extend(textwrap.wrap(parrafo, ancho) or [''])
    return lineas or ['']


def texto(filas, ancho=36):
    """Devuelve la matriz como texto tabulado, listo para consola o informe."""
    salida = []
    salida.append('MATRIZ DE EVIDENCIAS  -  P1.4 Computacion NPU Paralela')
    salida.append('=' * 78)
    salida.append('')

    cabecera = '%-22s %s' % ('CRITERIO',
                             ' '.join('%-*s' % (ancho, ROTULOS[n])
                                      for n in NIVELES))
    salida.append(cabecera)
    salida.append('=' * len(cabecera))

    for fila in filas:
        bloques = [partir(fila['celdas'].get(n), ancho) for n in NIVELES]
        alto = max(len(b) for b in bloques)
        for i in range(alto):
            etiqueta = fila['criterio'] if i == 0 else ''
            trozos = [(b[i] if i < len(b) else '') for b in bloques]
            salida.append('%-22s %s'
                          % (etiqueta,
                             ' '.join('%-*s' % (ancho, t) for t in trozos)))
        salida.append('-' * len(cabecera))

    return '\n'.join(salida)


def guardar(filas, destino):
    """Escribe la matriz en un archivo de texto."""
    carpeta = os.path.dirname(destino)
    if carpeta and not os.path.isdir(carpeta):
        os.makedirs(carpeta)
    with open(destino, 'w', encoding='utf-8') as f:
        f.write(texto(filas))
        f.write('\n')
    return destino


# ---------------------------------------------------------------------------
# Prueba rapida por linea de comandos
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import motor_gpu
    import motor_npu

    print(texto(matriz(info_gpu=motor_gpu.info_gpu(),
                       recursos_kernel=motor_gpu.recursos_kernel(),
                       info_npu=motor_npu.info_npu())))
