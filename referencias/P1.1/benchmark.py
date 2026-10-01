# -*- coding: utf-8 -*-
"""
benchmark.py

Barrido automatico de rendimiento del motor de conteo de ADN.

Ejecuta el mismo archivo FASTA con 1, 2, 4, 8 y 12 procesos y calcula
para cada configuracion el tiempo, el speedup y la eficiencia. La
configuracion de 1 proceso es la linea base del speedup.

Ademas verifica que TODAS las configuraciones devuelvan exactamente el
mismo conteo de A, C, G y T. Si alguna difiere se reporta, porque un
speedup con un resultado incorrecto no vale de nada.

Uso por linea de comandos:
    python benchmark.py archivo.fna
    python benchmark.py archivo.fna --csv resultados.csv
    python benchmark.py archivo.fna --procesos 1,2,4,6,8,12
    python benchmark.py archivo.fna --sin-secuencial

Uso como modulo (lo hace app.py):
    resumen = ejecutar_barrido(ruta)
    exportar_csv(resumen, 'resultados.csv')
"""

import argparse
import csv
import datetime
import os
import sys

import motor_adn

# Configuraciones del barrido pedidas por el proyecto.
PROCESOS_POR_DEFECTO = (1, 2, 4, 8, 12)


def formato_tamano(n):
    """Devuelve un tamano en bytes con una unidad legible."""
    valor = float(n)
    for unidad in ('B', 'KB', 'MB', 'GB', 'TB'):
        if valor < 1024.0 or unidad == 'TB':
            return '%.2f %s' % (valor, unidad)
        valor /= 1024.0


def ejecutar_barrido(ruta, procesos=PROCESOS_POR_DEFECTO,
                     incluir_secuencial=True, progreso=None, aviso=None):
    """Corre el barrido completo y devuelve un resumen con los resultados.

    Parametros:
        ruta                archivo FASTA a procesar
        procesos            lista con los numeros de procesos a probar
        incluir_secuencial  si tambien se mide contar_secuencial
        progreso            funcion progreso(leidos, total) para la barra
        aviso               funcion aviso(texto) para el estado en pantalla

    Devuelve un diccionario con:
        archivo, tamano, fisicos, logicos, fecha
        resultados      lista de filas, una por configuracion
        conteo_base     conteo de referencia (el de la primera medicion)
        coinciden       True si todas las configuraciones dieron lo mismo
        discrepancias   lista de etiquetas que NO coincidieron
    """
    if not os.path.isfile(ruta):
        raise FileNotFoundError('No existe el archivo: %s' % ruta)

    tamano = os.path.getsize(ruta)
    fisicos, logicos = motor_adn.detectar_nucleos()

    def anunciar(texto):
        if aviso is not None:
            aviso(texto)

    mediciones = []

    # 1) Medicion secuencial de referencia.
    if incluir_secuencial:
        anunciar('Ejecutando version secuencial...')
        conteo, tiempo = motor_adn.contar_secuencial(ruta, progreso)
        mediciones.append(('Secuencial', 1, conteo, tiempo))

    # 2) Barrido paralelo.
    for n in procesos:
        anunciar('Ejecutando version paralela con %d procesos...' % n)
        conteo, tiempo = motor_adn.contar_paralelo(ruta, n, progreso)
        mediciones.append(('Paralelo %d' % n, n, conteo, tiempo))

    if not mediciones:
        raise ValueError('No hay ninguna configuracion que ejecutar')

    # 3) Linea base del speedup: la configuracion de 1 proceso.
    tiempo_base = None
    for etiqueta, n, _conteo, tiempo in mediciones:
        if etiqueta == 'Paralelo 1':
            tiempo_base = tiempo
            break
    if tiempo_base is None:
        # Si el usuario excluyo la configuracion de 1 proceso, se usa la
        # primera medicion disponible como referencia.
        tiempo_base = mediciones[0][3]

    # 4) Verificacion de que todos los conteos coinciden.
    conteo_base = mediciones[0][2]
    discrepancias = []

    resultados = []
    for etiqueta, n, conteo, tiempo in mediciones:
        coincide = (conteo == conteo_base)
        if not coincide:
            discrepancias.append(etiqueta)

        speedup = (tiempo_base / tiempo) if tiempo > 0 else 0.0
        eficiencia = (speedup / n * 100.0) if n > 0 else 0.0
        total = sum(conteo.values())

        resultados.append({
            'etiqueta': etiqueta,
            'procesos': n,
            'tiempo': tiempo,
            'speedup': speedup,
            'eficiencia': eficiencia,
            'conteo': dict(conteo),
            'total': total,
            'mb_por_segundo': (tamano / (1024.0 * 1024.0) / tiempo) if tiempo > 0 else 0.0,
            'coincide': coincide,
        })

    anunciar('Barrido terminado.')

    return {
        'archivo': ruta,
        'tamano': tamano,
        'fisicos': fisicos,
        'logicos': logicos,
        'fecha': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'resultados': resultados,
        'conteo_base': conteo_base,
        'coinciden': not discrepancias,
        'discrepancias': discrepancias,
    }


def exportar_csv(resumen, ruta_csv):
    """Escribe los resultados del barrido en un archivo CSV."""
    columnas = ['configuracion', 'procesos', 'tiempo_s', 'speedup',
                'eficiencia_pct', 'mb_por_segundo', 'A', 'C', 'G', 'T',
                'total_bases', 'conteo_coincide', 'archivo', 'tamano_bytes',
                'nucleos_fisicos', 'nucleos_logicos', 'fecha']

    # newline='' es obligatorio en Windows para que csv no meta lineas vacias.
    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        escritor = csv.writer(f)
        escritor.writerow(columnas)

        for fila in resumen['resultados']:
            conteo = fila['conteo']
            escritor.writerow([
                fila['etiqueta'],
                fila['procesos'],
                '%.4f' % fila['tiempo'],
                '%.4f' % fila['speedup'],
                '%.2f' % fila['eficiencia'],
                '%.2f' % fila['mb_por_segundo'],
                conteo['A'], conteo['C'], conteo['G'], conteo['T'],
                fila['total'],
                'si' if fila['coincide'] else 'NO',
                os.path.basename(resumen['archivo']),
                resumen['tamano'],
                resumen['fisicos'],
                resumen['logicos'],
                resumen['fecha'],
            ])

    return ruta_csv


def formatear_tabla(resumen):
    """Devuelve el resumen como texto tabulado, listo para imprimir."""
    lineas = []
    lineas.append('')
    lineas.append('Archivo : %s (%s)' % (os.path.basename(resumen['archivo']),
                                         formato_tamano(resumen['tamano'])))
    lineas.append('Nucleos : %d fisicos / %d logicos'
                  % (resumen['fisicos'], resumen['logicos']))
    lineas.append('Fecha   : %s' % resumen['fecha'])
    lineas.append('')

    cabecera = '%-14s %9s %10s %11s %12s %10s' % (
        'Configuracion', 'Procesos', 'Tiempo(s)', 'Speedup',
        'Eficiencia', 'MB/s')
    lineas.append(cabecera)
    lineas.append('-' * len(cabecera))

    for fila in resumen['resultados']:
        lineas.append('%-14s %9d %10.3f %10.2fx %11.1f%% %10.1f' % (
            fila['etiqueta'], fila['procesos'], fila['tiempo'],
            fila['speedup'], fila['eficiencia'], fila['mb_por_segundo']))

    lineas.append('')
    conteo = resumen['conteo_base']
    total = sum(conteo.values())
    lineas.append('Conteo de nucleotidos')
    for base in ('A', 'C', 'G', 'T'):
        porcentaje = (conteo[base] / total * 100.0) if total else 0.0
        lineas.append('  %s : %16d  (%5.2f%%)' % (base, conteo[base], porcentaje))
    lineas.append('  %s : %16d' % ('Total', total))
    lineas.append('')

    if resumen['coinciden']:
        lineas.append('VERIFICACION: todas las configuraciones dieron el mismo conteo.')
    else:
        lineas.append('VERIFICACION FALLIDA: estas configuraciones difieren del')
        lineas.append('conteo de referencia: %s' % ', '.join(resumen['discrepancias']))

    return '\n'.join(lineas)


def _parsear_procesos(texto):
    """Convierte '1,2,4,8,12' en la tupla (1, 2, 4, 8, 12)."""
    valores = []
    for parte in texto.split(','):
        parte = parte.strip()
        if not parte:
            continue
        n = int(parte)
        if n < 1:
            raise argparse.ArgumentTypeError('el numero de procesos debe ser >= 1')
        valores.append(n)
    if not valores:
        raise argparse.ArgumentTypeError('lista de procesos vacia')
    return tuple(valores)


def main():
    parser = argparse.ArgumentParser(
        description='Barrido de rendimiento secuencial contra paralelo.')
    parser.add_argument('archivo', help='archivo FASTA a procesar')
    parser.add_argument('--csv', default=None,
                        help='ruta del CSV de salida (por defecto se genera una)')
    parser.add_argument('--procesos', type=_parsear_procesos,
                        default=PROCESOS_POR_DEFECTO,
                        help='lista de configuraciones, por ejemplo 1,2,4,8,12')
    parser.add_argument('--sin-secuencial', action='store_true',
                        help='omite la medicion secuencial de referencia')
    args = parser.parse_args()

    fisicos, logicos = motor_adn.detectar_nucleos()
    print('Nucleos detectados: %d fisicos / %d logicos' % (fisicos, logicos))
    print('Archivo: %s (%s)' % (args.archivo,
                                formato_tamano(os.path.getsize(args.archivo))))
    print('Configuraciones: %s' % ', '.join(str(n) for n in args.procesos))

    def aviso(texto):
        print('  -> %s' % texto)

    resumen = ejecutar_barrido(args.archivo,
                               procesos=args.procesos,
                               incluir_secuencial=not args.sin_secuencial,
                               aviso=aviso)

    print(formatear_tabla(resumen))

    ruta_csv = args.csv
    if ruta_csv is None:
        base = os.path.splitext(os.path.basename(args.archivo))[0]
        marca = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        ruta_csv = 'benchmark_%s_%s.csv' % (base, marca)

    exportar_csv(resumen, ruta_csv)
    print('Resultados exportados a: %s' % ruta_csv)

    return 0 if resumen['coinciden'] else 1


if __name__ == '__main__':
    # multiprocessing en Windows exige este guard.
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
