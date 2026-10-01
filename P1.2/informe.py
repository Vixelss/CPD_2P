# -*- coding: utf-8 -*-
"""
informe.py

Genera de una sola pasada el material del informe: corre el benchmark
completo sobre el archivo indicado y deja en disco el CSV con todas las
mediciones y una imagen PNG con las cuatro graficas, lista para pegar en el
documento.

Reutiliza las mismas funciones de dibujo que usa la aplicacion grafica, de
modo que las graficas del informe son exactamente las que se ven en pantalla.

Uso:
    python informe.py <archivo.fna> [--salida carpeta] [--rapido]
"""

import argparse
import multiprocessing
import os
import sys
import tkinter as tk

import benchmark_gpu
import motor_cpu


def generar(ruta, salida='resultados', rapido=False):
    """Corre el benchmark completo y escribe CSV mas PNG en la carpeta dada."""
    if not os.path.isdir(salida):
        os.makedirs(salida)

    procesos = (1, 2, 4, 8) if rapido else benchmark_gpu.PROCESOS_POR_DEFECTO
    lotes = (32, 64, 128) if rapido else benchmark_gpu.LOTES_POR_DEFECTO
    tamanos = (50, 200) if rapido else benchmark_gpu.TAMANOS_ESCALABILIDAD_MB

    def avisar(texto):
        sys.stdout.write('\r  %-55s' % texto)
        sys.stdout.flush()

    print('Archivo: %s (%s)'
          % (ruta, benchmark_gpu.formato_tamano(os.path.getsize(ruta))))
    print('')

    print('1/3 barrido de CPU')
    filas_cpu, referencia = barrido_seguro(
        benchmark_gpu.barrido_cpu, ruta, procesos=procesos, progreso=avisar)
    print('\r' + benchmark_gpu.formatear_tabla(filas_cpu))
    print('')

    base = filas_cpu[0]['tiempo_s'] if filas_cpu else None

    print('2/3 barrido de lote en GPU')
    filas_gpu, referencia = benchmark_gpu.barrido_gpu(
        ruta, lotes=lotes, referencia=referencia, base_cpu=base,
        progreso=avisar)
    print('\r' + benchmark_gpu.formatear_tabla(filas_gpu))
    print('')

    print('3/5 escalabilidad por tamano')
    filas_esc = benchmark_gpu.escalabilidad(ruta, tamanos_mb=tamanos,
                                            progreso=avisar)
    for fila in filas_esc:
        fila['escala'] = True
    print('\r' + benchmark_gpu.formatear_tabla(filas_esc))
    print('')

    print('4/5 GPU discreta frente a integrada')
    filas_disp, referencia = benchmark_gpu.barrido_dispositivos(
        ruta, referencia=referencia, base_cpu=base, progreso=avisar)
    print('\r' + benchmark_gpu.formatear_tabla(filas_disp))
    print('')

    print('5/5 escenarios con distintos tipos de error')
    filas_err = benchmark_gpu.barrido_tipos_error(ruta, progreso=avisar)
    print('\r' + benchmark_gpu.formatear_tabla(filas_err))
    print('')

    todas = filas_cpu + filas_gpu + filas_esc + filas_disp + filas_err

    ruta_csv = os.path.join(salida, 'benchmark.csv')
    benchmark_gpu.exportar_csv(todas, ruta_csv)

    ruta_png = os.path.join(salida, 'graficas.png')
    ruta_rec = os.path.join(salida, 'recursos.png')
    dibujar(todas, referencia, ruta_png, ruta_rec)

    # Resumen en texto, para copiar directamente al informe.
    ruta_txt = os.path.join(salida, 'resumen.txt')
    escribir_resumen(ruta_txt, ruta, filas_cpu, filas_gpu, filas_esc,
                     filas_disp, filas_err, referencia)

    print('Generado:')
    for archivo in (ruta_csv, ruta_png, ruta_rec, ruta_txt):
        print('  %s' % archivo)
    return todas


def barrido_seguro(funcion, ruta, **kwargs):
    """Envoltura que deja pasar los argumentos por nombre sin sorpresas."""
    return funcion(ruta, **kwargs)


def dibujar(filas, referencia, destino, destino_recursos=None):
    """Dibuja las graficas reutilizando el codigo de la aplicacion.

    Se usa la misma ventana que ve el usuario, solo que sin mostrarla, para
    que las imagenes del informe sean exactamente las de la aplicacion y no
    una version paralela que pudiera divergir.
    """
    import interfaz

    raiz = tk.Tk()
    raiz.withdraw()
    app = interfaz.Aplicacion(raiz)
    app.filas.extend(filas)
    if referencia is not None:
        app._mostrar_conteo(referencia)
    app._dibujar_graficas()
    app.figura.savefig(destino, dpi=120, facecolor='white')
    if destino_recursos:
        app.figura_rec.savefig(destino_recursos, dpi=120, facecolor='white')

    # Se marca el cierre antes de destruir para que los bucles que se
    # reprograman con after no intenten encolarse sobre un interprete que ya
    # no existe.
    app.cerrando = True
    raiz.destroy()


def escribir_resumen(destino, ruta, filas_cpu, filas_gpu, filas_esc,
                     filas_disp, filas_err, referencia):
    """Escribe en texto plano las cifras que se citan en el informe."""
    lineas = []
    lineas.append('RESUMEN DE RESULTADOS  -  P1.2 Computacion GPU Paralela')
    lineas.append('=' * 62)
    lineas.append('Archivo: %s' % ruta)
    lineas.append('Tamano : %s'
                  % benchmark_gpu.formato_tamano(os.path.getsize(ruta)))
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
        lineas.append('  A = %d' % referencia['A'])
        lineas.append('  C = %d' % referencia['C'])
        lineas.append('  G = %d' % referencia['G'])
        lineas.append('  T = %d' % referencia['T'])
        lineas.append('  Total de bases = %d' % referencia['bases'])
        lineas.append('  N (desconocidas) = %d' % referencia['N'])
        lineas.append('  Codigos IUPAC de ambiguedad = %d %s'
                      % (referencia['ambiguos'],
                         referencia['detalle_iupac'] or ''))
        lineas.append('  Caracteres invalidos = %d %s'
                      % (referencia['invalidos'],
                         referencia['detalle'] or ''))
        lineas.append('')

    for titulo, filas in (('BARRIDO DE CPU', filas_cpu),
                          ('BARRIDO DE LOTE EN GPU', filas_gpu),
                          ('ESCALABILIDAD', filas_esc),
                          ('GPU DISCRETA FRENTE A INTEGRADA', filas_disp),
                          ('ESCENARIOS DE ERROR', filas_err)):
        if not filas:
            continue
        lineas.append(titulo)
        lineas.append(benchmark_gpu.formatear_tabla(filas))
        lineas.append('')

    if filas_disp:
        lineas.append('DETALLE DE LAS UNIDADES DE COMPUTO')
        for fila in filas_disp:
            lineas.append('  %-26s %-8s %-10s %2d unidades  %8.3f s  %8.1f MB/s'
                          % (fila['dispositivo'], fila['tecnologia'],
                             fila['tipo_gpu'], fila['unidades'],
                             fila['tiempo_s'], fila['mb_por_s']))
        integradas = [f for f in filas_disp if f['tipo_gpu'] == 'integrada']
        discretas = [f for f in filas_disp
                     if f['tipo_gpu'] == 'discreta'
                     and f['tecnologia'] == 'OpenCL']
        if integradas and discretas:
            lineas.append('  Con el mismo kernel OpenCL, la discreta fue '
                          '%.2fx mas rapida que la integrada.'
                          % (integradas[0]['tiempo_s'] /
                             discretas[0]['tiempo_s']))
        lineas.append('')

    if filas_err:
        lineas.append('VELOCIDAD DE IDENTIFICACION DE ERRORES')
        for fila in filas_err:
            lineas.append('  %-20s %-4s  %6d de %6d errores  %14s por segundo'
                          % (fila['etiqueta'], fila['plataforma'],
                             fila['invalidos'], fila['errores_inyectados'],
                             '{:,.0f}'.format(fila['invalidos_por_s'])))
        fallos = [f for f in filas_err if not f['deteccion_ok']]
        lineas.append('  Escenarios correctos: %d de %d'
                      % (len(filas_err) - len(fallos), len(filas_err)))
        lineas.append('')

    if referencia:
        lineas.append('USO DE RECURSOS POR CONFIGURACION')
        for fila in filas_cpu + filas_gpu:
            lineas.append('  %-16s CPU %6s %%  GPU %6s %%  RAM proceso %6s MB'
                          '  VRAM %6s MB  %4s C'
                          % (fila['etiqueta'],
                             fila.get('cpu_medio', 'n/d'),
                             fila.get('gpu_medio', 'n/d'),
                             fila.get('proc_mb_max', 'n/d'),
                             fila.get('vram_mb_max', 'n/d'),
                             fila.get('gpu_temp_max', 'n/d')))
        lineas.append('')

    if filas_cpu and filas_gpu:
        mejor_cpu = min(filas_cpu, key=lambda f: f['tiempo_s'])
        mejor_gpu = min(filas_gpu, key=lambda f: f['tiempo_s'])
        lineas.append('COMPARATIVA FINAL')
        lineas.append('  Mejor CPU : %s en %.3f s (%.1f MB/s)'
                      % (mejor_cpu['etiqueta'], mejor_cpu['tiempo_s'],
                         mejor_cpu['mb_por_s']))
        lineas.append('  Mejor GPU : %s en %.3f s (%.1f MB/s)'
                      % (mejor_gpu['etiqueta'], mejor_gpu['tiempo_s'],
                         mejor_gpu['mb_por_s']))
        if mejor_gpu['tiempo_s'] > 0:
            lineas.append('  La GPU fue %.2fx mas rapida que la mejor CPU'
                          % (mejor_cpu['tiempo_s'] / mejor_gpu['tiempo_s']))
        base = filas_cpu[0]['tiempo_s']
        if mejor_gpu['tiempo_s'] > 0:
            lineas.append('  La GPU fue %.2fx mas rapida que la CPU secuencial'
                          % (base / mejor_gpu['tiempo_s']))

    with open(destino, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lineas) + '\n')


def main():
    analizador = argparse.ArgumentParser(
        description='Genera CSV y graficas del benchmark para el informe.')
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
