# -*- coding: utf-8 -*-
"""
benchmark.py

Banco de pruebas del P1.3. Mide el motor hibrido contra cada plataforma por
separado y caracteriza su balanceo de carga.

  1. Comparativa de modos  : CPU sola, GPU sola y las dos a la vez. Es la
                             medicion que justifica el proyecto: si trabajar
                             en paralelo no aporta nada, hay que poder verlo.
  2. Barrido de procesos   : cuantos procesos de CPU conviene sumarle a la
                             GPU. Mas no es siempre mejor, porque todos
                             compiten por el mismo disco.
  3. Barrido de trozo      : el grano del reparto, que es la perilla del
                             balanceo de carga.
  4. Escalabilidad         : a partir de que volumen compensa el conjunto.

Todas las corridas comprueban ademas que el conteo coincide con el de
referencia. Un motor rapido que cuenta mal no vale nada.

NOTA SOBRE WINDOWS
Esto debe correrse como archivo real, no por la entrada estandar. En Windows
multiprocessing usa el metodo 'spawn' y cada proceso hijo reimporta el modulo
principal, asi que necesita un archivo en disco y el guard
if __name__ == '__main__'.

Uso:
    python benchmark.py <archivo.fna> [opciones]
"""

import argparse
import csv
import multiprocessing
import os
import sys

import monitor as mod_monitor
import motor_cpu
import motor_hibrido

# motor_gpu se importa dentro de las funciones: si estuviera aqui, cada
# proceso hijo del Pool cargaria el stack de CUDA para no usarlo.


PROCESOS_POR_DEFECTO = (2, 4, 6, 8, 12)
TROZOS_POR_DEFECTO = (8, 16, 32, 64)
TAMANOS_ESCALABILIDAD_MB = (50, 200, 500, 1000)


def formato_tamano(n):
    """Convierte un numero de bytes a una cadena legible."""
    for unidad in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unidad == 'GB':
            return '%.1f %s' % (n, unidad)
        n /= 1024.0
    return '%.1f GB' % n


# ---------------------------------------------------------------------------
# Muestreo de recursos de una configuracion
# ---------------------------------------------------------------------------

class Medidor:
    """Envuelve el muestreo de recursos de una sola configuracion.

    Puede trabajar de dos maneras. Si se le pasa un monitor que ya esta
    corriendo, se limita a marcar el tramo que le corresponde y a resumirlo
    al terminar. Si no se le pasa ninguno, crea el suyo y lo gestiona.

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

        Una corrida corta puede durar menos que el intervalo de muestreo, con
        lo que el tramo se queda sin muestras o con una sola. En ese caso se
        devuelve None y las columnas de recursos quedan vacias. Publicar un
        cero seria peor que no publicar nada: un cero se lee como "no se
        uso", cuando lo que ocurrio es que no dio tiempo a medirlo.
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


def _anadir_recursos(fila, resumen):
    """Vuelca a la fila las magnitudes de recursos que se pudieron leer."""
    if not resumen:
        return fila
    for clave in ('cpu_medio', 'cpu_max', 'cpu_temp_medio', 'cpu_temp_max',
                  'ram_medio', 'ram_mb_max', 'proc_mb_medio', 'proc_mb_max',
                  'gpu_medio', 'gpu_max', 'vram_mb_max', 'gpu_temp_max'):
        valor = resumen.get(clave)
        fila[clave] = round(valor, 2) if isinstance(valor, float) else valor
    return fila


# ---------------------------------------------------------------------------
# Construccion de filas
# ---------------------------------------------------------------------------

def fila_base(etiqueta, plataforma, resultado, tiempo, tamano):
    """Construye la fila comun a todas las mediciones."""
    mb = tamano / (1024 * 1024)
    fila = {
        'etiqueta': etiqueta,
        'plataforma': plataforma,
        'bytes': tamano,
        'tamano': formato_tamano(tamano),
        'tiempo_s': round(tiempo, 4),
        'mb_por_s': round(mb / tiempo, 2) if tiempo > 0 else 0.0,
        'A': resultado['A'], 'C': resultado['C'],
        'G': resultado['G'], 'T': resultado['T'],
        'N': resultado['N'],
        'ambiguos': resultado['ambiguos'],
        'bases': resultado['bases'],
        'invalidos': resultado['invalidos'],
    }

    datos = resultado.get('hibrido')
    if datos:
        total = datos['bytes_cpu'] + datos['bytes_gpu']
        fila.update({
            'procesos': datos['procesos'],
            'trozo_mb': datos['trozo_mb'],
            'trozos_cpu': datos['trozos_cpu'],
            'trozos_gpu': datos['trozos_gpu'],
            'reparto_cpu_pct': round(100.0 * datos['bytes_cpu'] / total, 2)
                               if total else 0.0,
            'reparto_gpu_pct': round(100.0 * datos['bytes_gpu'] / total, 2)
                               if total else 0.0,
            'solape_s': datos['solape_s'],
            'preparacion_s': datos['preparacion_s'],
        })
        # La linea de tiempo y el censo por SM viajan pegados a la fila para
        # poder dibujarlos y exportarlos. No llegan al CSV principal porque
        # exportar_csv solo escribe las columnas declaradas.
        if datos.get('registro'):
            fila['registro_hibrido'] = datos['registro']
        if datos.get('censo_sm'):
            fila['censo_sm'] = datos['censo_sm']
    return fila


def _coincide(resultado, referencia):
    """Compara los conteos que importan, ignorando los saltos de linea.

    El bin del salto de linea difiere entre plataformas y es correcto que lo
    haga: la CPU elimina las cabeceras del buffer mientras la GPU las
    sobreescribe con saltos, porque borrarlas de verdad obligaria a mover
    datos dentro de la tarjeta. Como los saltos no se cuentan como nada, la
    diferencia no afecta a ningun resultado.
    """
    if referencia is None:
        return True
    return (motor_cpu.bins_significativos(resultado['histograma'])
            == motor_cpu.bins_significativos(referencia['histograma']))


# ---------------------------------------------------------------------------
# Mediciones
# ---------------------------------------------------------------------------

def comparar_modos(ruta, procesos=None, trozo_mb=motor_hibrido.TROZO_MB,
                   lote_mb=64, referencia=None, progreso=None,
                   monitor=None, con_monitor=True):
    """CPU sola, GPU sola y las dos a la vez, sobre el mismo archivo.

    Es la medicion central del proyecto. La linea base de todos los speedup
    es la CPU con un solo proceso, de modo que las tres barras se puedan
    comparar entre si.
    """
    import motor_gpu

    tamano = os.path.getsize(ruta)
    if procesos is None:
        _, procesos = motor_cpu.detectar_nucleos()
    hay_gpu = motor_gpu.gpu_disponible()
    if hay_gpu:
        motor_gpu.precalentar()

    filas = []
    base = None

    def medir(etiqueta, plataforma, funcion, modo):
        nonlocal base, referencia
        if progreso is not None:
            progreso(etiqueta)
        medidor = Medidor(con_monitor, monitor)
        resultado, tiempo = funcion()
        recursos = medidor.resumen()

        if base is None:
            base = tiempo
        if referencia is None:
            referencia = resultado

        fila = fila_base(etiqueta, plataforma, resultado, tiempo, tamano)
        fila['modo'] = modo
        fila['speedup'] = round(base / tiempo, 3) if tiempo > 0 else 0.0
        fila['conteo_ok'] = _coincide(resultado, referencia)
        fila.setdefault('procesos', procesos if plataforma != 'GPU' else '')
        _anadir_recursos(fila, recursos)
        filas.append(fila)
        return fila

    medir('CPU x1', 'CPU', lambda: motor_cpu.contar_secuencial(ruta),
          'cpu_seq')
    medir('CPU x%d' % procesos, 'CPU',
          lambda: motor_cpu.contar_paralelo(ruta, procesos), 'cpu_par')

    if hay_gpu:
        medir('GPU sola', 'GPU',
              lambda: motor_hibrido.contar_hibrido(
                  ruta, trozo_mb=trozo_mb, lote_mb=lote_mb,
                  usar_cpu=False), 'gpu')
        medir('CPU x%d + GPU' % procesos, 'HIBRIDO',
              lambda: motor_hibrido.contar_hibrido(
                  ruta, procesos=procesos, trozo_mb=trozo_mb,
                  lote_mb=lote_mb), 'hibrido')

    return filas, referencia


def barrido_procesos(ruta, procesos=PROCESOS_POR_DEFECTO,
                     trozo_mb=motor_hibrido.TROZO_MB, lote_mb=64,
                     referencia=None, base_cpu=None, progreso=None,
                     monitor=None, con_monitor=True):
    """Cuantos procesos de CPU conviene sumarle a la GPU.

    No es evidente que mas sea mejor: los procesos de CPU y la GPU leen del
    mismo disco, de modo que a partir de cierto punto los procesos nuevos no
    anaden capacidad sino competencia, y le quitan ancho de banda a la
    tarjeta, que es la que mas rendimiento saca de cada byte que recibe.
    """
    tamano = os.path.getsize(ruta)
    filas = []

    for n in procesos:
        if progreso is not None:
            progreso('hibrido con %d proceso(s) de CPU' % n)
        medidor = Medidor(con_monitor, monitor)
        resultado, tiempo = motor_hibrido.contar_hibrido(
            ruta, procesos=n, trozo_mb=trozo_mb, lote_mb=lote_mb)
        recursos = medidor.resumen()

        if referencia is None:
            referencia = resultado

        fila = fila_base('Hibrido x%d' % n, 'HIBRIDO', resultado, tiempo,
                         tamano)
        fila['modo'] = 'barrido_procesos'
        fila['speedup'] = round(base_cpu / tiempo, 3) \
            if (base_cpu and tiempo > 0) else ''
        fila['conteo_ok'] = _coincide(resultado, referencia)
        _anadir_recursos(fila, recursos)
        filas.append(fila)

    return filas, referencia


def barrido_trozo(ruta, trozos=TROZOS_POR_DEFECTO, procesos=None, lote_mb=64,
                  referencia=None, base_cpu=None, progreso=None,
                  monitor=None, con_monitor=True):
    """El grano del reparto, que es la perilla del balanceo de carga.

    Con trozos grandes hay pocas piezas que repartir y el reparto lo decide
    el orden de llegada; con trozos diminutos se paga el coste fijo de
    posicionarse en el archivo demasiadas veces. El optimo esta en medio y
    depende de la maquina.
    """
    tamano = os.path.getsize(ruta)
    if procesos is None:
        _, procesos = motor_cpu.detectar_nucleos()
    filas = []

    for trozo in trozos:
        if progreso is not None:
            progreso('hibrido con trozos de %d MB' % trozo)
        medidor = Medidor(con_monitor, monitor)
        resultado, tiempo = motor_hibrido.contar_hibrido(
            ruta, procesos=procesos, trozo_mb=trozo, lote_mb=lote_mb)
        recursos = medidor.resumen()

        if referencia is None:
            referencia = resultado

        fila = fila_base('Trozo %d MB' % trozo, 'HIBRIDO', resultado, tiempo,
                         tamano)
        fila['modo'] = 'barrido_trozo'
        fila['speedup'] = round(base_cpu / tiempo, 3) \
            if (base_cpu and tiempo > 0) else ''
        fila['conteo_ok'] = _coincide(resultado, referencia)
        _anadir_recursos(fila, recursos)
        filas.append(fila)

    return filas, referencia


def escalabilidad(ruta, tamanos_mb=TAMANOS_ESCALABILIDAD_MB, procesos=None,
                  trozo_mb=motor_hibrido.TROZO_MB, lote_mb=64, progreso=None,
                  carpeta_temp='datos'):
    """Compara los tres modos sobre porciones crecientes del mismo archivo.

    Los recortes se generan una sola vez y se reutilizan despues.
    """
    import motor_gpu

    if procesos is None:
        _, procesos = motor_cpu.detectar_nucleos()
    hay_gpu = motor_gpu.gpu_disponible()
    if hay_gpu:
        motor_gpu.precalentar()

    if not os.path.isdir(carpeta_temp):
        os.makedirs(carpeta_temp)

    total = os.path.getsize(ruta)
    filas = []

    for mb in tamanos_mb:
        objetivo = mb * 1024 * 1024
        if objetivo > total:
            continue

        recorte = os.path.join(carpeta_temp, 'recorte_%dMB.fna' % mb)
        if not os.path.exists(recorte) or os.path.getsize(recorte) == 0:
            if progreso is not None:
                progreso('generando recorte de %d MB' % mb)
            recortar(ruta, recorte, objetivo)

        real = os.path.getsize(recorte)

        if progreso is not None:
            progreso('escalabilidad %d MB: CPU' % mb)
        res, tiempo = motor_cpu.contar_paralelo(recorte, procesos)
        fila = fila_base('%d MB' % mb, 'CPU', res, tiempo, real)
        fila['modo'] = 'escala'
        fila['escala'] = True
        fila['procesos'] = procesos
        filas.append(fila)
        base = tiempo

        if hay_gpu:
            if progreso is not None:
                progreso('escalabilidad %d MB: GPU' % mb)
            res, tiempo = motor_hibrido.contar_hibrido(
                recorte, trozo_mb=trozo_mb, lote_mb=lote_mb, usar_cpu=False)
            fila = fila_base('%d MB' % mb, 'GPU', res, tiempo, real)
            fila['modo'] = 'escala'
            fila['escala'] = True
            fila['speedup'] = round(base / tiempo, 3) if tiempo > 0 else 0.0
            filas.append(fila)

            if progreso is not None:
                progreso('escalabilidad %d MB: hibrido' % mb)
            res, tiempo = motor_hibrido.contar_hibrido(
                recorte, procesos=procesos, trozo_mb=trozo_mb,
                lote_mb=lote_mb)
            fila = fila_base('%d MB' % mb, 'HIBRIDO', res, tiempo, real)
            fila['modo'] = 'escala'
            fila['escala'] = True
            fila['speedup'] = round(base / tiempo, 3) if tiempo > 0 else 0.0
            filas.append(fila)

    return filas


def recortar(origen, destino, objetivo):
    """Copia los primeros bytes del origen cortando en una linea completa.

    El recorte tiene que terminar en un salto de linea: si partiera una linea
    de secuencia por la mitad, el archivo dejaria de ser un FASTA valido.
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
        resto = entrada.readline()
        if resto:
            salida.write(resto)


# ---------------------------------------------------------------------------
# Exportacion
# ---------------------------------------------------------------------------

COLUMNAS = ['etiqueta', 'plataforma', 'modo', 'procesos', 'trozo_mb',
            'tamano', 'bytes', 'tiempo_s', 'mb_por_s', 'speedup',
            'trozos_cpu', 'trozos_gpu', 'reparto_cpu_pct', 'reparto_gpu_pct',
            'solape_s', 'preparacion_s',
            'A', 'C', 'G', 'T', 'N', 'ambiguos', 'bases', 'invalidos',
            'conteo_ok', 'cpu_medio', 'cpu_max', 'cpu_temp_medio',
            'cpu_temp_max', 'ram_medio', 'ram_mb_max', 'proc_mb_medio',
            'proc_mb_max', 'gpu_medio', 'gpu_max', 'vram_mb_max',
            'gpu_temp_max']


def exportar_csv(filas, ruta_csv):
    """Escribe las filas de resultados en un CSV apto para el informe."""
    if not filas:
        return None
    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS,
                                  extrasaction='ignore')
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({c: fila.get(c, '') for c in COLUMNAS})
    return ruta_csv


def exportar_registro(filas, ruta_csv):
    """Guarda la linea de tiempo trozo a trozo de las corridas hibridas.

    Es la evidencia directa de que las dos plataformas trabajaron a la vez:
    cada linea dice que trozo se proceso, quien lo hizo y entre que instantes.
    """
    lineas = []
    for fila in filas:
        registro = fila.get('registro_hibrido')
        if not registro:
            continue
        for apunte in registro:
            fila_nueva = dict(apunte)
            fila_nueva['configuracion'] = fila['etiqueta']
            lineas.append(fila_nueva)

    if not lineas:
        return None

    columnas = ['configuracion', 'plataforma', 'trabajador', 'trozo',
                'trozos', 'bytes', 'inicio_s', 'fin_s']
    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        escritor = csv.DictWriter(f, fieldnames=columnas,
                                  extrasaction='ignore')
        escritor.writeheader()
        for linea in lineas:
            escritor.writerow({c: linea.get(c, '') for c in columnas})
    return ruta_csv


def formatear_tabla(filas):
    """Devuelve una tabla de texto con lo esencial de cada medicion."""
    if not filas:
        return '(sin resultados)'

    cabecera = ('%-18s %-8s %9s %9s %8s %14s %8s %6s'
                % ('Configuracion', 'Plat.', 'Tiempo s', 'MB/s', 'Speedup',
                   'Reparto C/G', 'Solape', 'OK'))
    lineas = [cabecera, '-' * len(cabecera)]

    for fila in filas:
        speedup = fila.get('speedup', '')
        if fila.get('reparto_cpu_pct') not in ('', None):
            reparto = '%3.0f%% / %3.0f%%' % (fila['reparto_cpu_pct'],
                                             fila['reparto_gpu_pct'])
        else:
            reparto = '-'
        solape = fila.get('solape_s')
        lineas.append('%-18s %-8s %9.3f %9.1f %8s %14s %8s %6s'
                      % (fila['etiqueta'], fila['plataforma'],
                         fila['tiempo_s'], fila['mb_por_s'],
                         ('%.2fx' % speedup) if speedup != '' else '-',
                         reparto,
                         ('%.2f s' % solape) if solape not in ('', None)
                         else '-',
                         'si' if fila.get('conteo_ok', True) else 'NO'))
    return '\n'.join(lineas)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    analizador = argparse.ArgumentParser(
        description='Banco de pruebas del motor hibrido CPU+GPU.')
    analizador.add_argument('archivo', help='archivo FASTA a procesar')
    analizador.add_argument('--procesos', default=None,
                            help='lista de procesos, por ejemplo 2,4,8,12')
    analizador.add_argument('--trozos', default=None,
                            help='lista de tamanos de trozo en MB')
    analizador.add_argument('--escalabilidad', default=None,
                            help='lista de tamanos en MB, o "no" para omitir')
    analizador.add_argument('--csv', default='resultados/benchmark.csv')
    analizador.add_argument('--sin-monitor', action='store_true',
                            help='no muestrear uso de recursos')
    args = analizador.parse_args()

    if not os.path.exists(args.archivo):
        print('No existe el archivo: %s' % args.archivo)
        return 1

    import motor_gpu

    tamano = os.path.getsize(args.archivo)
    fisicos, logicos = motor_cpu.detectar_nucleos()
    datos_gpu = motor_gpu.info_gpu()
    con_monitor = not args.sin_monitor

    print('=' * 72)
    print('BANCO DE PRUEBAS CPU + GPU EN PARALELO  -  P1.3 Computacion PDN')
    print('=' * 72)
    print('Archivo : %s  (%s)' % (args.archivo, formato_tamano(tamano)))
    print('CPU     : %d nucleos fisicos / %d logicos' % (fisicos, logicos))
    if datos_gpu['disponible']:
        print('GPU     : %s  (CC %s, %d SMs, %d MB VRAM)'
              % (datos_gpu['nombre'], datos_gpu['compute_capability'],
                 datos_gpu['sms'], datos_gpu['vram_total_mb']))
    else:
        print('GPU     : no disponible')
    print('')

    def avisar(texto):
        sys.stdout.write('\r  midiendo: %-50s' % texto)
        sys.stdout.flush()

    print('1) Comparativa de modos')
    filas_modos, referencia = comparar_modos(args.archivo, progreso=avisar,
                                             con_monitor=con_monitor)
    print('\r' + ' ' * 64)
    print(formatear_tabla(filas_modos))
    print('')

    base_cpu = filas_modos[0]['tiempo_s'] if filas_modos else None

    procesos = _lista(args.procesos, PROCESOS_POR_DEFECTO)
    print('2) Cuantos procesos de CPU sumarle a la GPU')
    filas_proc, referencia = barrido_procesos(
        args.archivo, procesos, referencia=referencia, base_cpu=base_cpu,
        progreso=avisar, con_monitor=con_monitor)
    print('\r' + ' ' * 64)
    print(formatear_tabla(filas_proc))
    if filas_proc:
        mejor = min(filas_proc, key=lambda f: f['tiempo_s'])
        print('   Mejor: %s (%.3f s)' % (mejor['etiqueta'],
                                         mejor['tiempo_s']))
    print('')

    trozos = _lista(args.trozos, TROZOS_POR_DEFECTO)
    print('3) Grano del reparto')
    filas_trozo, referencia = barrido_trozo(
        args.archivo, trozos, referencia=referencia, base_cpu=base_cpu,
        progreso=avisar, con_monitor=con_monitor)
    print('\r' + ' ' * 64)
    print(formatear_tabla(filas_trozo))
    if filas_trozo:
        mejor = min(filas_trozo, key=lambda f: f['tiempo_s'])
        print('   Mejor: %s (%.3f s)' % (mejor['etiqueta'],
                                         mejor['tiempo_s']))
    print('')

    filas_esc = []
    if args.escalabilidad != 'no':
        tamanos = _lista(args.escalabilidad, TAMANOS_ESCALABILIDAD_MB)
        print('4) Escalabilidad por tamano')
        filas_esc = escalabilidad(args.archivo, tamanos, progreso=avisar)
        print('\r' + ' ' * 64)
        print(formatear_tabla(filas_esc))
        print('')

    todas = filas_modos + filas_proc + filas_trozo + filas_esc

    malos = [f for f in todas if not f.get('conteo_ok', True)]
    print('5) Verificacion de conteos')
    if malos:
        print('   ATENCION: %d configuracion(es) con conteo distinto:'
              % len(malos))
        for fila in malos:
            print('     - %s' % fila['etiqueta'])
    else:
        print('   Todas las configuraciones dieron exactamente el mismo')
        print('   conteo. Los tiempos son comparables entre si.')
    print('')

    carpeta = os.path.dirname(args.csv)
    if carpeta and not os.path.isdir(carpeta):
        os.makedirs(carpeta)
    exportar_csv(todas, args.csv)
    print('Resultados exportados a: %s' % args.csv)
    return 0


def _lista(texto, por_defecto):
    """Convierte '1,2,4' en (1, 2, 4). Si no hay texto, usa el valor dado."""
    if not texto:
        return por_defecto
    valores = [int(p) for p in texto.replace(' ', '').split(',') if p]
    return tuple(valores) if valores else por_defecto


if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main())
