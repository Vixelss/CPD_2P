# -*- coding: utf-8 -*-
"""
capturas.py

Abre la aplicacion, ejecuta una comparacion de verdad y captura sus pestanas
como imagenes para el informe.

REGLA QUE NO SE DEBE ROMPER
Este programa no inventa valores nunca. Todo lo que aparece en las imagenes
sale de una ejecucion real sobre los archivos que se indiquen. Si una
magnitud no se pudo medir, la captura muestra el panel como quedo, sin
rellenarlo. Una imagen destinada a un informe academico tiene que ensenar lo
que el programa hizo, no lo que se esperaba que hiciera.

Sirve ademas como prueba de la interfaz: si la ventana se rompe al procesar,
este programa falla en vez de sacar una captura a medias.

Uso:
    python capturas.py                          (usa el par de datos\\)
    python capturas.py <a.fna> <b.fna>
    python capturas.py <a.fna> <b.fna> --modo cpu
"""

import argparse
import os
import sys
import time
import tkinter as tk

SALIDA_POR_DEFECTO = 'resultados'

# Margen que Windows deja alrededor de la ventana. Sin recortarlo, la captura
# sale con una franja del escritorio por los bordes.
_MARGEN = 8


def _preparar_dpi():
    """Avisa a Windows de que el programa entiende el escalado de pantalla.

    Sin esto, en una pantalla al 125 por ciento las coordenadas que devuelve
    Tk no coinciden con las que usa la captura, y la imagen sale desplazada y
    recortada por un lado.
    """
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def capturar(raiz, destino):
    """Guarda una imagen de la ventana tal y como esta en este momento."""
    from PIL import ImageGrab

    raiz.update_idletasks()
    raiz.update()
    time.sleep(0.35)

    x = raiz.winfo_rootx() - _MARGEN
    y = raiz.winfo_rooty() - _MARGEN
    ancho = raiz.winfo_width() + _MARGEN * 2
    alto = raiz.winfo_height() + _MARGEN * 2

    imagen = ImageGrab.grab(bbox=(max(0, x), max(0, y), x + ancho, y + alto),
                            all_screens=True)
    imagen.save(destino)
    return destino


def generar(ruta_a, ruta_b, salida=SALIDA_POR_DEFECTO, modo='todas',
            procesos=None, lote=64, capas=8):
    """Arranca la ventana, corre una comparacion real y captura las pestanas."""
    _preparar_dpi()

    if not os.path.isdir(salida):
        os.makedirs(salida)

    # Import tardio: igual que en app_p14, para no cargar la interfaz en los
    # procesos hijos de multiprocessing.
    from interfaz import Aplicacion

    raiz = tk.Tk()
    app = Aplicacion(raiz)

    app.ruta_a.set(os.path.abspath(ruta_a))
    app.ruta_b.set(os.path.abspath(ruta_b))
    app._describir_par()
    app.modo.set(modo)
    if procesos:
        app.procesos.set(str(procesos))
    app.lote_gpu.set(str(lote))
    app.capas_npu.set(str(capas))

    raiz.update()
    capturar(raiz, os.path.join(salida, 'ui_inicio.png'))
    print('  ui_inicio.png')

    # Se lanza la comparacion y se espera a que el hilo trabajador termine,
    # bombeando los eventos de Tk mientras tanto. No se puede usar un
    # mainloop() normal porque entonces no habria manera de saber cuando
    # parar para capturar.
    app.comparar()
    limite = time.time() + 1800
    while (app.trabajando or not app.resultados) and time.time() < limite:
        raiz.update()
        time.sleep(0.05)

    if not app.resultados:
        raiz.destroy()
        raise RuntimeError('La comparacion no produjo ningun resultado.')

    # Una vuelta mas para que la cola acabe de vaciarse y las graficas se
    # terminen de dibujar antes de fotografiarlas.
    for _ in range(40):
        raiz.update()
        time.sleep(0.05)

    pestanas = [
        (0, 'ui_diferencias.png'),
        (1, 'ui_rendimiento.png'),
        (2, 'ui_nucleos.png'),
        (3, 'ui_evidencias.png'),
        (4, 'ui_registro.png'),
    ]
    for indice, nombre in pestanas:
        app.pestanas.select(indice)
        # Dos vueltas de eventos: la primera le da al lienzo su tamano real
        # y dispara el redibujado, la segunda lo deja pintado. Sin esperar a
        # las dos, la captura coge la grafica a medio ajustar.
        for _ in range(12):
            raiz.update()
            time.sleep(0.05)
        capturar(raiz, os.path.join(salida, nombre))
        print('  %s' % nombre)

    # Se exporta tambien el material de texto, que es lo que se cita en el
    # informe junto a las imagenes.
    app.pestanas.select(0)
    raiz.update()
    _exportar(app, salida)

    resumen = [
        '%-16s %8.3f s  %8.1f MB/s  %d diferencias  %.6f %% iguales'
        % (r['etiqueta'], r['tiempo'], r['mb_s'], r['diferencias'],
           r['similitud'])
        for r in app.resultados
    ]

    raiz.destroy()
    return resumen


def _exportar(app, salida):
    """Vuelca CSV, matriz de evidencias y graficas sin abrir dialogos."""
    import evidencias

    ruta_csv = os.path.join(salida, 'comparacion.csv')
    with open(ruta_csv, 'w', encoding='utf-8', newline='') as f:
        f.write('plataforma,etiqueta,tiempo_s,mb_por_s,diferencias,'
                'similitud_pct,preparacion_s\n')
        for r in app.resultados:
            f.write('%s,%s,%.4f,%.2f,%d,%.6f,%.4f\n'
                    % (r['plataforma'], r['etiqueta'], r['tiempo'],
                       r['mb_s'], r['diferencias'], r['similitud'],
                       r['preparacion']))
    print('  comparacion.csv')

    ruta_dif = os.path.join(salida, 'diferencias.csv')
    primero = app.resultados[0]['resultado']
    with open(ruta_dif, 'w', encoding='utf-8', newline='') as f:
        f.write('posicion,fila,columna,cadena_a,cadena_b\n')
        for d in primero['detalle']:
            f.write('%d,%d,%d,%s,%s\n'
                    % (d['posicion'], d['fila'], d['columna'],
                       d['a'], d['b']))
    print('  diferencias.csv')

    evidencias.guardar(app.matriz_evidencias(),
                       os.path.join(salida, 'evidencias.txt'))
    print('  evidencias.txt')

    app.figura.savefig(os.path.join(salida, 'graficas.png'), dpi=120)
    print('  graficas.png')


def _par_por_defecto():
    """Busca el par de prueba de datos\\, o lo genera si no esta."""
    ruta_a = os.path.join('datos', 'par_10MB_A.fna')
    ruta_b = os.path.join('datos', 'par_10MB_B.fna')
    if os.path.exists(ruta_a) and os.path.exists(ruta_b):
        return ruta_a, ruta_b

    import generar_par
    for candidato in (os.path.join('..', 'P1.3', 'datos', 'recorte_50MB.fna'),
                      os.path.join('..', 'P1.1', 'prueba_50MB.fna')):
        if os.path.exists(candidato):
            print('Generando el par de prueba...')
            datos = generar_par.generar(candidato, salida='datos', mb=10,
                                        diferencias=5000)
            return datos['ruta_a'], datos['ruta_b']
    return None, None


def main():
    p = argparse.ArgumentParser(
        description='Captura la ventana del P1.4 para el informe.')
    p.add_argument('cadena_a', nargs='?', default=None)
    p.add_argument('cadena_b', nargs='?', default=None)
    p.add_argument('--salida', default=SALIDA_POR_DEFECTO)
    p.add_argument('--modo', default='todas',
                   choices=('cpu', 'gpu', 'npu', 'todas'))
    p.add_argument('--procesos', type=int, default=None)
    p.add_argument('--lote', type=int, default=64)
    p.add_argument('--capas', type=int, default=8)
    args = p.parse_args()

    ruta_a, ruta_b = args.cadena_a, args.cadena_b
    if not ruta_a or not ruta_b:
        ruta_a, ruta_b = _par_por_defecto()
    if not ruta_a:
        print('No hay par de prueba y no se pudo generar uno.')
        return 1

    print('Cadena A: %s' % ruta_a)
    print('Cadena B: %s' % ruta_b)
    print('')
    print('Capturando en %s\\ :' % args.salida)

    resumen = generar(ruta_a, ruta_b, salida=args.salida, modo=args.modo,
                      procesos=args.procesos, lote=args.lote,
                      capas=args.capas)

    print('')
    print('Resultados medidos:')
    for linea in resumen:
        print('  %s' % linea)
    return 0


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
