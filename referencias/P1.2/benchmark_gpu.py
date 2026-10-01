# -*- coding: utf-8 -*-
"""
benchmark_gpu.py

Banco de pruebas comparativo CPU contra GPU para el P1.2. Reune las cuatro
mediciones que pide el enunciado y las exporta a CSV para el informe:

  1. Barrido de CPU        : 1, 2, 4, 8 y 12 procesos, con speedup y
                             eficiencia respecto a la ejecucion de 1 proceso.
  2. Barrido de lote en GPU : distintos tamanos de lote, que es la perilla de
                             balance de carga que pide el punto extra del
                             enunciado.
  3. Escalabilidad          : el mismo procesamiento sobre tamanos crecientes,
                             para ver a partir de que volumen la GPU compensa
                             el coste de transferir por PCIe.
  4. Errores por segundo    : velocidad de identificacion de caracteres
                             invalidos, criterio de comparacion explicito del
                             enunciado.

Todas las corridas verifican ademas que el conteo coincide con el de
referencia. Un motor rapido que cuenta mal no vale nada, asi que cualquier
discrepancia se marca en el CSV y en el resumen por pantalla.

NOTA SOBRE WINDOWS
Igual que verificar.py, esto debe correrse como archivo real por el metodo
'spawn' de multiprocessing, y motor_gpu se importa tarde para que los
procesos hijos no carguen el stack de CUDA sin usarlo.

Uso:
    python benchmark_gpu.py <archivo.fna> [opciones]
"""

import argparse
import csv
import datetime
import multiprocessing
import os
import sys

import motor_cpu
import monitor as mod_monitor

# motor_gpu se importa dentro de las funciones: ver la nota de la cabecera.


# Configuraciones por defecto de cada barrido.
PROCESOS_POR_DEFECTO = (1, 2, 4, 8, 12)
LOTES_POR_DEFECTO = (8, 16, 32, 64, 128, 256)
TAMANOS_ESCALABILIDAD_MB = (50, 200, 500, 1000)


def formato_tamano(n):
    """Convierte un numero de bytes a una cadena legible."""
    for unidad in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unidad == 'GB':
            return '%.1f %s' % (n, unidad)
        n /= 1024.0
    return '%.1f GB' % n


def fila_base(etiqueta, plataforma, resultado, tiempo, tamano):
    """Construye la fila comun a todas las mediciones."""
    mb = tamano / (1024 * 1024)
    return {
        'etiqueta': etiqueta,
        'plataforma': plataforma,
        'bytes': tamano,
        'tamano': formato_tamano(tamano),
        'tiempo_s': round(tiempo, 4),
        'mb_por_s': round(mb / tiempo, 2) if tiempo > 0 else 0.0,
        'A': resultado['A'],
        'C': resultado['C'],
        'G': resultado['G'],
        'T': resultado['T'],
        'N': resultado['N'],
        'ambiguos': resultado['ambiguos'],
        'bases': resultado['bases'],
        'invalidos': resultado['invalidos'],
        'invalidos_por_s': round(resultado['invalidos'] / tiempo, 2)
                           if tiempo > 0 else 0.0,
    }


def _anadir_recursos(fila, resumen):
    """Vuelca al CSV las magnitudes de uso de recursos que se pudieron leer."""
    for clave in ('cpu_medio', 'cpu_max', 'ram_medio', 'ram_mb_max',
                  'proc_mb_medio', 'proc_mb_max',
                  'gpu_medio', 'gpu_max', 'vram_mb_max', 'gpu_temp_max'):
        valor = resumen.get(clave)
        fila[clave] = round(valor, 2) if isinstance(valor, float) else valor
    return fila


# ---------------------------------------------------------------------------
# Barridos
# ---------------------------------------------------------------------------

class _Medidor:
    """Envuelve el muestreo de recursos de una sola configuracion.

    Puede trabajar de dos maneras. Si se le pasa un monitor que ya esta
    corriendo, se limita a marcar el tramo que le corresponde y a resumirlo
    al terminar, sin arrancar ni parar nada. Si no se le pasa ninguno, crea
    el suyo propio y lo gestiona.

    Existe porque nunca debe haber dos monitores muestreando a la vez: la
    referencia interna de psutil.cpu_percent es global al proceso y dos
    muestreadores concurrentes se la corrompen mutuamente. La aplicacion
    grafica ya mantiene un monitor vivo para su panel en directo, asi que le
    cede ese mismo monitor al barrido en lugar de dejar que abra otro.
    """

    def __init__(self, activo=True, monitor=None):
        self.propio = None
        self.prestado = monitor
        self.desde = 0
        if not activo:
            return
        if monitor is not None:
            self.desde = monitor.marcar()
        else:
            self.propio = mod_monitor.Monitor()
            self.propio.iniciar()

    def resumen(self):
        """Resumen de recursos del tramo medido, o None si no es fiable.

        Una ejecucion en GPU sobre un archivo pequeno puede durar menos que
        el intervalo de muestreo, con lo que el tramo se queda sin muestras
        o con una sola. En ese caso se devuelve None y las columnas de
        recursos quedan vacias en el CSV. Publicar un cero seria peor que no
        publicar nada: un cero se lee como "la GPU no se uso", cuando lo que
        ocurrio en realidad es que no dio tiempo a medirla.
        """
        if self.propio is not None:
            self.propio.detener()
            datos = self.propio.resumen()
        elif self.prestado is not None:
            datos = self.prestado.resumen(self.desde)
        else:
            return None

        if datos.get('muestras', 0) < mod_monitor.MINIMO_MUESTRAS:
            return None
        return datos


def barrido_cpu(ruta, procesos=PROCESOS_POR_DEFECTO, referencia=None,
                progreso=None, con_monitor=True, monitor=None):
    """Mide la CPU con distintos numeros de procesos.

    La configuracion de 1 proceso es la linea base del speedup, tal como se
    hizo en el P1.1. El speedup de n procesos es t(1)/t(n) y la eficiencia es
    el speedup dividido entre n, en porcentaje: dice que fraccion de cada
    nucleo se esta aprovechando de verdad.
    """
    tamano = os.path.getsize(ruta)
    filas = []
    base = None

    for n in procesos:
        if progreso is not None:
            progreso('CPU con %d proceso(s)' % n)

        medidor = _Medidor(con_monitor, monitor)
        if n == 1:
            resultado, tiempo = motor_cpu.contar_secuencial(ruta)
        else:
            resultado, tiempo = motor_cpu.contar_paralelo(ruta, n)
        recursos = medidor.resumen()

        if base is None:
            base = tiempo
        if referencia is None:
            referencia = resultado

        fila = fila_base('CPU x%d' % n, 'CPU', resultado, tiempo, tamano)
        fila['procesos'] = n
        fila['lote_mb'] = ''
        fila['speedup'] = round(base / tiempo, 3) if tiempo > 0 else 0.0
        fila['eficiencia_pct'] = round(100.0 * (base / tiempo) / n, 2) \
            if tiempo > 0 else 0.0
        fila['conteo_ok'] = (motor_cpu.bins_significativos(resultado['histograma'])
                             == motor_cpu.bins_significativos(referencia['histograma']))
        if recursos:
            _anadir_recursos(fila, recursos)
        filas.append(fila)

    return filas, referencia


def barrido_gpu(ruta, lotes=LOTES_POR_DEFECTO, referencia=None,
                base_cpu=None, progreso=None, con_monitor=True, monitor=None):
    """Mide la GPU con distintos tamanos de lote.

    El tamano de lote decide cuanta secuencia viaja a la tarjeta de una vez.
    Un lote pequeno desaprovecha el ancho de banda del PCIe y multiplica los
    lanzamientos de kernel; uno demasiado grande gasta VRAM y retrasa el
    inicio del computo, porque hay que llenarlo entero antes de empezar. El
    optimo depende de la maquina y encontrarlo es el punto extra de balance
    de carga del enunciado.
    """
    import motor_gpu

    if not motor_gpu.gpu_disponible():
        return [], referencia

    # La compilacion de los kernels queda fuera de toda medicion.
    jit = motor_gpu.precalentar()

    tamano = os.path.getsize(ruta)
    filas = []

    for lote in lotes:
        if progreso is not None:
            progreso('GPU con lote de %d MB' % lote)

        medidor = _Medidor(con_monitor, monitor)
        resultado, tiempo = motor_gpu.contar_gpu(ruta, lote_mb=lote)
        recursos = medidor.resumen()

        if referencia is None:
            referencia = resultado

        fila = fila_base('GPU lote %d MB' % lote, 'GPU', resultado,
                          tiempo, tamano)
        fila['procesos'] = ''
        fila['lote_mb'] = lote
        # El speedup de la GPU se mide contra la CPU de 1 proceso, que es la
        # misma linea base del barrido de CPU. Asi las dos columnas de
        # speedup del CSV son directamente comparables entre si.
        fila['speedup'] = round(base_cpu / tiempo, 3) \
            if (base_cpu and tiempo > 0) else ''
        fila['eficiencia_pct'] = ''
        fila['conteo_ok'] = (motor_cpu.bins_significativos(resultado['histograma'])
                             == motor_cpu.bins_significativos(referencia['histograma']))
        fila['jit_s'] = round(jit, 3)
        if recursos:
            _anadir_recursos(fila, recursos)
        filas.append(fila)

    return filas, referencia


def escalabilidad(ruta, tamanos_mb=TAMANOS_ESCALABILIDAD_MB, procesos=None,
                  lote_mb=64, progreso=None, carpeta_temp='datos'):
    """Compara CPU y GPU sobre porciones crecientes del mismo archivo.

    Responde a la pregunta que de verdad importa en el informe: la GPU no es
    mas rapida siempre, tiene un coste fijo de transferencia y de lanzamiento
    de kernels que solo se amortiza a partir de cierto volumen de datos. Este
    barrido localiza ese punto de cruce.

    Los recortes se generan una sola vez y se reutilizan en llamadas
    posteriores.
    """
    import motor_gpu

    if procesos is None:
        _, logicos = motor_cpu.detectar_nucleos()
        procesos = logicos

    hay_gpu = motor_gpu.gpu_disponible()
    if hay_gpu:
        motor_gpu.precalentar()

    if not os.path.isdir(carpeta_temp):
        os.makedirs(carpeta_temp)

    tamano_total = os.path.getsize(ruta)
    filas = []

    for mb in tamanos_mb:
        objetivo = mb * 1024 * 1024
        if objetivo > tamano_total:
            continue

        recorte = os.path.join(carpeta_temp, 'recorte_%dMB.fna' % mb)
        if not os.path.exists(recorte) or os.path.getsize(recorte) == 0:
            if progreso is not None:
                progreso('generando recorte de %d MB' % mb)
            _recortar(ruta, recorte, objetivo)

        real = os.path.getsize(recorte)

        if progreso is not None:
            progreso('escalabilidad %d MB: CPU secuencial' % mb)
        res, tiempo = motor_cpu.contar_secuencial(recorte)
        fila = fila_base('%d MB' % mb, 'CPU x1', res, tiempo, real)
        fila['procesos'] = 1
        fila['lote_mb'] = ''
        filas.append(fila)
        base = tiempo

        if progreso is not None:
            progreso('escalabilidad %d MB: CPU paralelo' % mb)
        res, tiempo = motor_cpu.contar_paralelo(recorte, procesos)
        fila = fila_base('%d MB' % mb, 'CPU x%d' % procesos, res, tiempo, real)
        fila['procesos'] = procesos
        fila['lote_mb'] = ''
        fila['speedup'] = round(base / tiempo, 3) if tiempo > 0 else 0.0
        filas.append(fila)

        if hay_gpu:
            if progreso is not None:
                progreso('escalabilidad %d MB: GPU' % mb)
            res, tiempo = motor_gpu.contar_gpu(recorte, lote_mb=lote_mb)
            fila = fila_base('%d MB' % mb, 'GPU', res, tiempo, real)
            fila['procesos'] = ''
            fila['lote_mb'] = lote_mb
            fila['speedup'] = round(base / tiempo, 3) if tiempo > 0 else 0.0
            filas.append(fila)

    return filas


def barrido_tipos_error(ruta, tipos=None, mb=50, tasa=200, lote_mb=64,
                        procesos=None, progreso=None, carpeta_temp='datos',
                        monitor=None, con_monitor=True):
    """Mide la deteccion de errores para cada familia de caracter invalido.

    Cubre el punto extra del enunciado sobre escenarios con diferentes tipos
    de error. Para cada familia se genera, a partir del ADN real, una copia
    con una cantidad conocida de errores de ese tipo, y se comprueba que CPU
    y GPU detectan exactamente esa cantidad. Ademas se calcula la velocidad
    de identificacion de errores por segundo, que es uno de los criterios de
    comparacion que pide el enunciado y que no tiene sentido medir sobre el
    archivo limpio, donde siempre daria cero.

    La columna 'deteccion_ok' es la que importa: dice si el motor reprodujo
    el numero exacto de errores inyectados. Un motor rapido que cuenta mal
    los errores no sirve para nada.
    """
    import ensuciar
    import motor_gpu

    if tipos is None:
        tipos = list(ensuciar.TIPOS) + ['mixto']
    if procesos is None:
        _, procesos = motor_cpu.detectar_nucleos()

    hay_gpu = motor_gpu.gpu_disponible()
    if hay_gpu:
        motor_gpu.precalentar()

    if not os.path.isdir(carpeta_temp):
        os.makedirs(carpeta_temp)

    # Se parte siempre de ADN real, no de una cadena sintetica, para que la
    # prueba sea representativa del archivo que se procesa de verdad.
    base = os.path.join(carpeta_temp, 'recorte_%dMB.fna' % mb)
    if not os.path.exists(base) or os.path.getsize(base) == 0:
        if progreso is not None:
            progreso('preparando base de %d MB' % mb)
        _recortar(ruta, base, mb * 1024 * 1024)

    filas = []

    for tipo in tipos:
        sucio = os.path.join(carpeta_temp, 'sucio_%s_%dMB.fna' % (tipo, mb))
        if not os.path.exists(sucio) or os.path.getsize(sucio) == 0:
            if progreso is not None:
                progreso('generando errores de tipo %s' % tipo)
            esperado = ensuciar.ensuciar(base, sucio, tasa=tasa, tipo=tipo)
            ensuciar.escribir_esperado(sucio, esperado, tipo, tasa, 1234)
            inyectados = esperado['invalidos']
        else:
            # Se recupera del archivo de conteos correctos ya generado.
            inyectados = _leer_invalidos_esperados(sucio)

        tamano = os.path.getsize(sucio)

        for plataforma in ('CPU', 'GPU'):
            if plataforma == 'GPU' and not hay_gpu:
                continue
            if progreso is not None:
                progreso('tipo %s en %s' % (tipo, plataforma))

            medidor = _Medidor(con_monitor, monitor)
            if plataforma == 'CPU':
                resultado, tiempo = motor_cpu.contar_paralelo(sucio, procesos)
            else:
                resultado, tiempo = motor_gpu.contar_gpu(sucio, lote_mb=lote_mb)
            recursos = medidor.resumen()

            fila = fila_base('%s (%s)' % (tipo, plataforma), plataforma,
                             resultado, tiempo, tamano)
            fila['procesos'] = procesos if plataforma == 'CPU' else ''
            fila['lote_mb'] = lote_mb if plataforma == 'GPU' else ''
            fila['tipo_error'] = tipo
            fila['errores_inyectados'] = inyectados
            fila['deteccion_ok'] = (resultado['invalidos'] == inyectados)
            fila['conteo_ok'] = fila['deteccion_ok']
            fila['tipos_distintos'] = len(resultado['detalle'])
            fila['errores'] = True
            if recursos:
                _anadir_recursos(fila, recursos)
            filas.append(fila)

    return filas


def barrido_dispositivos(ruta, lote_mb=64, referencia=None, base_cpu=None,
                         progreso=None, monitor=None, con_monitor=True):
    """Compara todas las unidades de computo disponibles en el equipo.

    Cubre el punto extra del enunciado sobre benchmark en distintas GPUs,
    discreta frente a integrada. Se miden tres caminos:

      - CUDA sobre la tarjeta NVIDIA, que es el motor principal del proyecto.
      - OpenCL sobre esa misma tarjeta NVIDIA.
      - OpenCL sobre la grafica integrada del procesador.

    Incluir las dos primeras no es redundante. Comparar el kernel CUDA contra
    la grafica integrada mezclaria dos diferencias a la vez, la del hardware y
    la de la implementacion, y no se sabria cual pesa mas. Al medir tambien
    OpenCL sobre la NVIDIA quedan aisladas: entre las dos filas de OpenCL solo
    cambia la tarjeta, y entre las dos filas de NVIDIA solo cambia la
    tecnologia.
    """
    import motor_gpu
    try:
        import motor_opencl
    except ImportError:
        motor_opencl = None

    tamano = os.path.getsize(ruta)
    filas = []

    def registrar(etiqueta, plataforma, resultado, tiempo, extra):
        fila = fila_base(etiqueta, plataforma, resultado, tiempo, tamano)
        fila['procesos'] = ''
        fila['lote_mb'] = lote_mb
        fila['speedup'] = round(base_cpu / tiempo, 3) \
            if (base_cpu and tiempo > 0) else ''
        fila['eficiencia_pct'] = ''
        fila['dispositivos'] = True
        fila.update(extra)
        if referencia is not None:
            fila['conteo_ok'] = (
                motor_cpu.bins_significativos(resultado['histograma'])
                == motor_cpu.bins_significativos(referencia['histograma']))
        filas.append(fila)
        return fila

    # 1) CUDA sobre la NVIDIA.
    if motor_gpu.gpu_disponible():
        if progreso is not None:
            progreso('CUDA sobre la GPU discreta')
        motor_gpu.precalentar()
        datos = motor_gpu.info_gpu()
        medidor = _Medidor(con_monitor, monitor)
        resultado, tiempo = motor_gpu.contar_gpu(ruta, lote_mb=lote_mb)
        recursos = medidor.resumen()
        fila = registrar('CUDA %s' % _corto(datos['nombre']), 'GPU',
                         resultado, tiempo,
                         {'tecnologia': 'CUDA', 'tipo_gpu': 'discreta',
                          'dispositivo': datos['nombre'],
                          'unidades': datos['sms']})
        if recursos:
            _anadir_recursos(fila, recursos)
        if referencia is None:
            referencia = resultado

    # 2) y 3) OpenCL sobre cada tarjeta del equipo.
    if motor_opencl is not None and motor_opencl.opencl_disponible():
        for datos in motor_opencl.listar_dispositivos():
            if progreso is not None:
                progreso('OpenCL sobre %s' % _corto(datos['nombre']))
            motor_opencl.precalentar(datos['indice'])
            medidor = _Medidor(con_monitor, monitor)
            resultado, tiempo = motor_opencl.contar_opencl(
                ruta, dispositivo=datos['indice'], lote_mb=lote_mb)
            recursos = medidor.resumen()
            fila = registrar('OpenCL %s' % _corto(datos['nombre']), 'GPU',
                             resultado, tiempo,
                             {'tecnologia': 'OpenCL',
                              'tipo_gpu': datos['tipo'],
                              'dispositivo': datos['nombre'],
                              'unidades': datos['unidades']})
            if recursos:
                _anadir_recursos(fila, recursos)
            if referencia is None:
                referencia = resultado

    return filas, referencia


def _corto(nombre):
    """Acorta el nombre comercial de una tarjeta para que quepa en tablas."""
    limpio = nombre.replace('NVIDIA GeForce ', '').replace('(R)', '')
    limpio = limpio.replace(' Laptop GPU', '').replace('Graphics', 'Gfx')
    return limpio.strip()


def _leer_invalidos_esperados(ruta_sucio):
    """Lee el total de invalidos del archivo .esperado.txt correspondiente."""
    ruta = ruta_sucio + '.esperado.txt'
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding='utf-8') as f:
        for linea in f:
            partes = linea.rstrip('\n').split('\t')
            if partes[0] == 'INVALIDOS' and len(partes) == 2:
                return int(partes[1])
    return None


def _recortar(origen, destino, objetivo):
    """Copia los primeros bytes del origen cortando en una linea completa.

    El recorte tiene que terminar en un salto de linea: si partiera una linea
    de secuencia por la mitad, el archivo dejaria de ser un FASTA valido y los
    conteos de CPU y GPU podrian discrepar en el ultimo fragmento.
    """
    bloque = 8 * 1024 * 1024
    escritos = 0
    with open(origen, 'rb') as entrada, open(destino, 'wb') as salida:
        while escritos < objetivo:
            datos = entrada.read(min(bloque, objetivo - escritos))
            if not datos:
                break
            salida.write(datos)
            escritos += len(datos)
        # Se completa la ultima linea hasta su salto correspondiente.
        resto = entrada.readline()
        if resto:
            salida.write(resto)


# ---------------------------------------------------------------------------
# Exportacion
# ---------------------------------------------------------------------------

COLUMNAS = ['etiqueta', 'plataforma', 'procesos', 'lote_mb', 'tamano', 'bytes',
            'tiempo_s', 'mb_por_s', 'speedup', 'eficiencia_pct',
            'A', 'C', 'G', 'T', 'N', 'ambiguos', 'bases',
            'invalidos', 'invalidos_por_s', 'tipo_error',
            'errores_inyectados', 'tipos_distintos', 'deteccion_ok',
            'tecnologia', 'dispositivo', 'tipo_gpu', 'unidades',
            'conteo_ok', 'jit_s', 'cpu_medio', 'cpu_max', 'ram_medio',
            'ram_mb_max', 'proc_mb_medio', 'proc_mb_max',
            'gpu_medio', 'gpu_max', 'vram_mb_max',
            'gpu_temp_max']


def exportar_csv(filas, ruta_csv):
    """Escribe las filas de resultados en un CSV apto para el informe."""
    if not filas:
        return None

    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS, extrasaction='ignore')
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({clave: fila.get(clave, '')
                               for clave in COLUMNAS})
    return ruta_csv


def formatear_tabla(filas):
    """Devuelve una tabla de texto con lo esencial de cada medicion."""
    if not filas:
        return '(sin resultados)'

    cabecera = ('%-22s %-8s %10s %10s %9s %9s %8s'
                % ('Configuracion', 'Plat.', 'Tiempo s', 'MB/s',
                   'Speedup', 'Efic. %', 'OK'))
    lineas = [cabecera, '-' * len(cabecera)]

    for fila in filas:
        speedup = fila.get('speedup', '')
        eficiencia = fila.get('eficiencia_pct', '')
        lineas.append('%-22s %-8s %10.3f %10.1f %9s %9s %8s'
                      % (fila['etiqueta'], fila['plataforma'],
                         fila['tiempo_s'], fila['mb_por_s'],
                         ('%.2fx' % speedup) if speedup != '' else '-',
                         ('%.1f' % eficiencia) if eficiencia != '' else '-',
                         'si' if fila.get('conteo_ok', True) else 'NO'))
    return '\n'.join(lineas)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    analizador = argparse.ArgumentParser(
        description='Benchmark comparativo CPU contra GPU sobre FASTA.')
    analizador.add_argument('archivo', help='archivo FASTA a procesar')
    analizador.add_argument('--procesos', default=None,
                            help='lista de procesos, por ejemplo 1,2,4,8,12')
    analizador.add_argument('--lotes', default=None,
                            help='lista de tamanos de lote en MB para la GPU')
    analizador.add_argument('--escalabilidad', default=None,
                            help='lista de tamanos en MB, o "no" para omitir')
    analizador.add_argument('--csv', default=None,
                            help='ruta del CSV de salida')
    analizador.add_argument('--sin-monitor', action='store_true',
                            help='no muestrear uso de recursos')
    args = analizador.parse_args()

    if not os.path.exists(args.archivo):
        print('No existe el archivo: %s' % args.archivo)
        return 1

    import motor_gpu

    procesos = _lista_enteros(args.procesos, PROCESOS_POR_DEFECTO)
    lotes = _lista_enteros(args.lotes, LOTES_POR_DEFECTO)

    tamano = os.path.getsize(args.archivo)
    fisicos, logicos = motor_cpu.detectar_nucleos()
    datos_gpu = motor_gpu.info_gpu()

    print('=' * 70)
    print('BENCHMARK CPU CONTRA GPU  -  P1.2 Computacion PDN')
    print('=' * 70)
    print('Archivo : %s  (%s)' % (args.archivo, formato_tamano(tamano)))
    print('CPU     : %d nucleos fisicos / %d logicos' % (fisicos, logicos))
    if datos_gpu['disponible']:
        print('GPU     : %s  (CC %s, %d SMs, %d MB VRAM)'
              % (datos_gpu['nombre'], datos_gpu['compute_capability'],
                 datos_gpu['sms'], datos_gpu['vram_total_mb']))
    else:
        print('GPU     : no disponible')
    print('Monitoreo: %s' % mod_monitor.disponibilidad())
    print('')

    def avisar(texto):
        sys.stdout.write('\r  midiendo: %-45s' % texto)
        sys.stdout.flush()

    con_monitor = not args.sin_monitor

    print('1) Barrido de CPU')
    filas_cpu, referencia = barrido_cpu(args.archivo, procesos,
                                        progreso=avisar,
                                        con_monitor=con_monitor)
    print('\r' + ' ' * 60)
    print(formatear_tabla(filas_cpu))
    print('')

    base_cpu = filas_cpu[0]['tiempo_s'] if filas_cpu else None

    filas_gpu = []
    if datos_gpu['disponible']:
        print('2) Barrido de tamano de lote en GPU')
        filas_gpu, referencia = barrido_gpu(args.archivo, lotes, referencia,
                                            base_cpu, progreso=avisar,
                                            con_monitor=con_monitor)
        print('\r' + ' ' * 60)
        print(formatear_tabla(filas_gpu))
        print('')

        mejor = min(filas_gpu, key=lambda f: f['tiempo_s'])
        print('   Mejor lote: %d MB  (%.3f s, %.1f MB/s)'
              % (mejor['lote_mb'], mejor['tiempo_s'], mejor['mb_por_s']))
        print('')

    filas_esc = []
    if args.escalabilidad != 'no':
        tamanos = _lista_enteros(args.escalabilidad,
                                 TAMANOS_ESCALABILIDAD_MB)
        print('3) Escalabilidad por tamano de archivo')
        filas_esc = escalabilidad(args.archivo, tamanos, progreso=avisar)
        print('\r' + ' ' * 60)
        print(formatear_tabla(filas_esc))
        print('')

    todas = filas_cpu + filas_gpu + filas_esc

    malos = [f for f in todas if not f.get('conteo_ok', True)]
    print('4) Verificacion de conteos')
    if malos:
        print('   ATENCION: %d configuracion(es) dieron un conteo distinto:'
              % len(malos))
        for fila in malos:
            print('     - %s' % fila['etiqueta'])
    else:
        print('   Todas las configuraciones dieron exactamente el mismo')
        print('   conteo. Los tiempos son comparables entre si.')
    if referencia and referencia['invalidos']:
        print('   Caracteres invalidos detectados: %d (%d tipos distintos)'
              % (referencia['invalidos'], len(referencia['detalle'])))
    print('')

    ruta_csv = args.csv
    if ruta_csv is None:
        sello = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        ruta_csv = 'benchmark_%s.csv' % sello
    exportar_csv(todas, ruta_csv)
    print('Resultados exportados a: %s' % ruta_csv)
    return 0


def _lista_enteros(texto, por_defecto):
    """Convierte '1,2,4' en (1, 2, 4). Si no hay texto, usa el valor dado."""
    if not texto:
        return por_defecto
    valores = []
    for parte in texto.replace(' ', '').split(','):
        if parte:
            valores.append(int(parte))
    return tuple(valores) if valores else por_defecto


if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main())
