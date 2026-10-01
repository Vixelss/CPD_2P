# -*- coding: utf-8 -*-
"""
informe.py

Genera de una sola pasada todo el material del P1.3: corre el analisis
completo sobre el archivo indicado y deja en disco el CSV con las
mediciones, la linea de tiempo del reparto, la matriz de evidencias de la
rubrica, las graficas y un resumen en texto listo para citar.

Reutiliza las mismas funciones de dibujo que usa la aplicacion grafica, de
modo que las imagenes del informe son exactamente las que se ven en
pantalla y no una version paralela que pudiera divergir con el tiempo.

Uso:
    python informe.py <archivo.fna> [--salida carpeta] [--rapido]
"""

import argparse
import multiprocessing
import os
import sys
import time
import tkinter as tk

import benchmark
import evidencias
import monitor as mod_monitor
import motor_cpu
import motor_hibrido


def generar(ruta, salida='resultados', rapido=False):
    """Corre el analisis completo y escribe todo el material en la carpeta."""
    if not os.path.isdir(salida):
        os.makedirs(salida)

    procesos = (2, 4, 8) if rapido else benchmark.PROCESOS_POR_DEFECTO
    trozos = (16, 32) if rapido else benchmark.TROZOS_POR_DEFECTO
    tamanos = (50, 200) if rapido else benchmark.TAMANOS_ESCALABILIDAD_MB

    def avisar(texto):
        sys.stdout.write('\r  %-58s' % texto)
        sys.stdout.flush()

    print('Archivo: %s (%s)'
          % (ruta, benchmark.formato_tamano(os.path.getsize(ruta))))
    print('')

    # Un solo monitor para todo el informe, prestado a cada medicion. Dos
    # monitores simultaneos se corromperian las lecturas de psutil.
    monitor = mod_monitor.Monitor()
    monitor.iniciar()
    # La temperatura del procesador se lee en un hilo lento; se le da un
    # momento para que la primera lectura llegue antes de empezar a medir.
    time.sleep(mod_monitor.INTERVALO_TEMPERATURA)

    try:
        print('1/4 comparativa de modos')
        filas_modos, referencia = benchmark.comparar_modos(
            ruta, progreso=avisar, monitor=monitor)
        print('\r' + benchmark.formatear_tabla(filas_modos))
        print('')

        base = filas_modos[0]['tiempo_s'] if filas_modos else None

        print('2/4 cuantos procesos de CPU sumarle a la GPU')
        filas_proc, referencia = benchmark.barrido_procesos(
            ruta, procesos=procesos, referencia=referencia, base_cpu=base,
            progreso=avisar, monitor=monitor)
        print('\r' + benchmark.formatear_tabla(filas_proc))
        print('')

        print('3/4 grano del reparto')
        filas_trozo, referencia = benchmark.barrido_trozo(
            ruta, trozos=trozos, referencia=referencia, base_cpu=base,
            progreso=avisar, monitor=monitor)
        print('\r' + benchmark.formatear_tabla(filas_trozo))
        print('')

        print('4/4 escalabilidad por tamano')
        filas_esc = benchmark.escalabilidad(ruta, tamanos_mb=tamanos,
                                            progreso=avisar)
        for fila in filas_esc:
            fila['escala'] = True
        print('\r' + benchmark.formatear_tabla(filas_esc))
        print('')

        recursos = monitor.resumen()
    finally:
        monitor.detener()

    todas = filas_modos + filas_proc + filas_trozo + filas_esc

    # El censo por SM y la linea de tiempo se toman de la corrida hibrida de
    # la comparativa de modos, que es la mas representativa.
    registro = []
    for fila in todas:
        if fila.get('registro_hibrido'):
            registro = fila['registro_hibrido']
            break

    censo = _censo_de(ruta, todas)

    recursos_kernel = None
    datos_gpu = {}
    try:
        import motor_gpu
        if motor_gpu.gpu_disponible():
            datos_gpu = motor_gpu.info_gpu()
            recursos_kernel = motor_gpu.recursos_kernel()
    except Exception:
        pass

    matriz = evidencias.matriz(recursos=recursos, censo_sm=censo,
                               recursos_kernel=recursos_kernel,
                               info_gpu=datos_gpu,
                               nucleos_cpu=recursos.get('cpu_por_nucleo'))

    ruta_csv = os.path.join(salida, 'benchmark.csv')
    benchmark.exportar_csv(todas, ruta_csv)

    ruta_reg = os.path.join(salida, 'registro_reparto.csv')
    benchmark.exportar_registro(todas, ruta_reg)

    ruta_evid = os.path.join(salida, 'evidencias.txt')
    evidencias.guardar(matriz, ruta_evid)

    ruta_png = os.path.join(salida, 'ejecucion.png')
    ruta_nuc = os.path.join(salida, 'nucleos.png')
    dibujar(todas, referencia, registro, censo, recursos, ruta_png, ruta_nuc)

    ruta_txt = os.path.join(salida, 'resumen.txt')
    escribir_resumen(ruta_txt, ruta, filas_modos, filas_proc, filas_trozo,
                     filas_esc, referencia, recursos, censo, recursos_kernel,
                     matriz)

    print('Generado:')
    for archivo in (ruta_csv, ruta_reg, ruta_evid, ruta_png, ruta_nuc,
                    ruta_txt):
        print('  %s' % archivo)
    return todas


def _censo_de(ruta, filas):
    """Recupera el censo por SM de la corrida de GPU de la comparativa."""
    for fila in filas:
        censo = fila.get('censo_sm')
        if censo:
            return censo
    # La comparativa no guarda el censo en la fila, asi que se hace una
    # pasada corta de GPU sola para tenerlo. Es barata comparada con todo lo
    # que ya se ha medido y garantiza que la cifra publicada viene de una
    # ejecucion real sobre este mismo archivo.
    try:
        import motor_gpu
        if not motor_gpu.gpu_disponible():
            return None
        resultado, _ = motor_hibrido.contar_hibrido(ruta, usar_cpu=False)
        return resultado['hibrido'].get('censo_sm')
    except Exception:
        return None


def dibujar(filas, referencia, registro, censo, recursos, destino,
            destino_nucleos):
    """Dibuja las graficas reutilizando el codigo de la aplicacion.

    Se usa la misma ventana que ve el usuario, solo que sin mostrarla, para
    que las imagenes del informe sean exactamente las de la aplicacion.
    """
    import interfaz

    raiz = tk.Tk()
    raiz.withdraw()
    app = interfaz.Aplicacion(raiz)
    app.filas.extend(filas)
    app.ultimo_registro = registro or []
    app.ultimo_censo = censo
    app.ultimos_recursos = recursos or {}
    if referencia is not None:
        app._mostrar_conteo(referencia)
    app._dibujar_graficas()
    app._dibujar_nucleos()
    app.figura.savefig(destino, dpi=120, facecolor='white')
    app.figura_nuc.savefig(destino_nucleos, dpi=120, facecolor='white')

    # Se marca el cierre antes de destruir para que los bucles que se
    # reprograman con after no intenten encolarse sobre un interprete que ya
    # no existe.
    app.cerrando = True
    if app.monitor:
        app.monitor.detener()
        app.monitor = None
    raiz.destroy()


def escribir_resumen(destino, ruta, filas_modos, filas_proc, filas_trozo,
                     filas_esc, referencia, recursos, censo, recursos_kernel,
                     matriz):
    """Escribe en texto plano las cifras que se citan en el informe."""
    lineas = []
    lineas.append('RESUMEN DE RESULTADOS  -  P1.3 Computacion CPU & GPU '
                  'Paralela')
    lineas.append('=' * 68)
    lineas.append('Archivo: %s' % ruta)
    lineas.append('Tamano : %s'
                  % benchmark.formato_tamano(os.path.getsize(ruta)))
    lineas.append('')

    fisicos, logicos = motor_cpu.detectar_nucleos()
    lineas.append('CPU: %d nucleos fisicos / %d logicos' % (fisicos, logicos))
    try:
        import motor_gpu
        datos = motor_gpu.info_gpu()
        if datos['disponible']:
            lineas.append('GPU: %s, compute capability %s, %d SMs, %d MB VRAM'
                          % (datos['nombre'], datos['compute_capability'],
                             datos['sms'], datos['vram_total_mb']))
    except Exception:
        pass
    lineas.append('')

    if referencia:
        lineas.append('CONTEO')
        for clave, nombre in (('A', 'A'), ('C', 'C'), ('G', 'G'), ('T', 'T')):
            lineas.append('  %s = %d' % (nombre, referencia[clave]))
        lineas.append('  Total de bases = %d' % referencia['bases'])
        lineas.append('  N (desconocidas) = %d' % referencia['N'])
        lineas.append('  Codigos IUPAC de ambiguedad = %d %s'
                      % (referencia['ambiguos'],
                         referencia['detalle_iupac'] or ''))
        lineas.append('  Caracteres invalidos = %d %s'
                      % (referencia['invalidos'], referencia['detalle'] or ''))
        lineas.append('')

    for titulo, filas in (('COMPARATIVA DE MODOS', filas_modos),
                          ('PROCESOS DE CPU JUNTO A LA GPU', filas_proc),
                          ('GRANO DEL REPARTO', filas_trozo),
                          ('ESCALABILIDAD', filas_esc)):
        if not filas:
            continue
        lineas.append(titulo)
        lineas.append(benchmark.formatear_tabla(filas))
        lineas.append('')

    hibridas = [f for f in filas_modos + filas_proc + filas_trozo
                if f['plataforma'] == 'HIBRIDO']
    if hibridas:
        lineas.append('TRABAJO SIMULTANEO')
        for fila in hibridas:
            lineas.append('  %-16s CPU %5.1f %% / GPU %5.1f %%   '
                          'a la vez %5.2f s de %5.2f s'
                          % (fila['etiqueta'], fila['reparto_cpu_pct'],
                             fila['reparto_gpu_pct'], fila['solape_s'],
                             fila['tiempo_s']))
        lineas.append('')

    if recursos.get('cpu_por_nucleo'):
        nucleos = recursos['cpu_por_nucleo']
        lineas.append('USO POR NUCLEO LOGICO DE CPU')
        lineas.append('  ' + '  '.join('%d:%.0f%%' % (i, v)
                                       for i, v in enumerate(nucleos)))
        lineas.append('  media %.1f %%, entre %.1f %% y %.1f %%'
                      % (sum(nucleos) / len(nucleos), min(nucleos),
                         max(nucleos)))
        lineas.append('')

    if censo:
        porcentajes = evidencias.censo_a_porcentajes(censo)
        lineas.append('REPARTO ENTRE LOS MULTIPROCESADORES DE LA GPU')
        for i, (bytes_sm, pct) in enumerate(zip(censo, porcentajes)):
            lineas.append('  SM %2d  %14d bytes  %5.2f %%'
                          % (i, bytes_sm, pct))
        lineas.append('  media %.2f %%, entre %.2f %% y %.2f %%'
                      % (sum(porcentajes) / len(porcentajes),
                         min(porcentajes), max(porcentajes)))
        lineas.append('')

    if recursos_kernel:
        lineas.append('RECURSOS DEL KERNEL DENTRO DE CADA SM')
        lineas.append('  %d registros por hilo, %d B de memoria compartida '
                      'por bloque' % (recursos_kernel['registros_por_hilo'],
                                      recursos_kernel['compartida_por_bloque']))
        lineas.append('  ocupacion %d de %d warps por SM (%.1f %%)'
                      % (recursos_kernel['warps_activos'],
                         recursos_kernel['warps_maximos'],
                         recursos_kernel['ocupacion_pct']))
        lineas.append('')

    lineas.append('MATRIZ DE EVIDENCIAS')
    lineas.append(evidencias.texto(matriz))
    lineas.append('')

    if filas_modos:
        mejor = min(filas_modos, key=lambda f: f['tiempo_s'])
        lineas.append('COMPARATIVA FINAL')
        for fila in filas_modos:
            lineas.append('  %-16s %8.3f s  %8.1f MB/s'
                          % (fila['etiqueta'], fila['tiempo_s'],
                             fila['mb_por_s']))
        lineas.append('  Configuracion mas rapida: %s (%.3f s)'
                      % (mejor['etiqueta'], mejor['tiempo_s']))

    with open(destino, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lineas) + '\n')


def main():
    analizador = argparse.ArgumentParser(
        description='Genera el material del informe del P1.3.')
    analizador.add_argument('archivo')
    analizador.add_argument('--salida', default='resultados')
    analizador.add_argument('--rapido', action='store_true',
                            help='menos configuraciones, para pruebas')
    args = analizador.parse_args()

    if not os.path.exists(args.archivo):
        print('No existe el archivo: %s' % args.archivo)
        return 1

    generar(args.archivo, args.salida, args.rapido)
    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main())
