# -*- coding: utf-8 -*-
"""
capturas.py

Abre la aplicacion, ejecuta un procesamiento de verdad y captura sus pestanas
como imagenes para el informe.

REGLA QUE NO SE DEBE ROMPER
Este programa no inventa valores nunca. Una version anterior rellenaba las
barras de recursos con cifras de ejemplo para que no salieran vacias en la
captura, y eso es inaceptable: una imagen destinada a un informe academico
tiene que mostrar lo que el programa midio de verdad. Si una magnitud no se
pudo medir, la captura debe ensenar que no se pudo medir.

Por eso todo lo que aparece en las imagenes sale de una ejecucion real sobre
el archivo que se indique. Si se quiere que las capturas muestren deteccion
de errores, hay que pasarle un archivo que tenga errores; para eso esta la
opcion --sucio, que genera uno a partir del ADN real.

La ventana se pone en primer plano desde el propio Python antes de cada
captura, para que ninguna otra ventana se cuele en el recorte.

Uso:
    python capturas.py <archivo.fna> [--salida carpeta] [--sucio]
"""

import argparse
import ctypes
import multiprocessing
import os
import sys
import tkinter as tk


SALIDA_POR_DEFECTO = 'resultados'

# Margenes del marco de la ventana en Windows 11, en pixeles fisicos.
_BORDE = 9
_TITULO = 40


def _preparar_dpi():
    """Declara el proceso como consciente del escalado de pantalla.

    Sin esto, Windows le miente a Python sobre las coordenadas: en esta
    maquina la pantalla es de 1920x1080 fisicos al 125 por ciento, y un
    proceso no consciente recibe 1536x864. La captura tomaria entonces un
    recorte desplazado y saldria media ventana.
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def capturar(raiz, destino):
    """Guarda una imagen de la ventana completa."""
    from PIL import ImageGrab

    raiz.attributes('-topmost', True)
    raiz.lift()
    raiz.focus_force()
    raiz.update_idletasks()
    raiz.update()
    raiz.after(400)
    raiz.update()

    x = raiz.winfo_rootx()
    y = raiz.winfo_rooty()
    ancho = raiz.winfo_width()
    alto = raiz.winfo_height()
    caja = (x - _BORDE, y - _TITULO, x + ancho + _BORDE, y + alto + _BORDE)

    imagen = ImageGrab.grab(bbox=caja, all_screens=True)
    imagen.save(destino)
    raiz.attributes('-topmost', False)
    print('  %-22s %dx%d' % (os.path.basename(destino),
                             imagen.width, imagen.height))
    return destino


def preparar_archivo_sucio(ruta, carpeta='datos', mb=200, tasa=200):
    """Deriva del ADN real un archivo con errores conocidos.

    Sirve para que las capturas muestren el detector de caracteres invalidos
    encontrando algo. Sobre el genoma limpio el panel mostraria cero, que es
    correcto pero no demuestra nada.
    """
    import benchmark_gpu
    import ensuciar

    if not os.path.isdir(carpeta):
        os.makedirs(carpeta)

    base = os.path.join(carpeta, 'recorte_%dMB.fna' % mb)
    if not os.path.exists(base) or os.path.getsize(base) == 0:
        print('Generando recorte de %d MB del genoma real...' % mb)
        benchmark_gpu._recortar(ruta, base, mb * 1024 * 1024)

    sucio = os.path.join(carpeta, 'sucio_capturas_%dMB.fna' % mb)
    if not os.path.exists(sucio) or os.path.getsize(sucio) == 0:
        print('Inyectando errores conocidos...')
        esperado = ensuciar.ensuciar(base, sucio, tasa=tasa, tipo='mixto')
        ensuciar.escribir_esperado(sucio, esperado, 'mixto', tasa, 1234)
        print('  %d errores de %d tipos distintos'
              % (esperado['invalidos'], len(esperado['detalle'])))
    return sucio


def generar(ruta, salida=SALIDA_POR_DEFECTO, completo=True):
    """Corre la aplicacion sobre el archivo dado y captura sus pestanas."""
    import benchmark_gpu
    import interfaz

    if not os.path.isdir(salida):
        os.makedirs(salida)

    raiz = tk.Tk()
    try:
        from tkinter import ttk
        ttk.Style().theme_use('vista')
    except Exception:
        pass

    app = interfaz.Aplicacion(raiz)
    app.ruta.set(ruta)
    app.etiqueta_tamano.configure(
        text=benchmark_gpu.formato_tamano(os.path.getsize(ruta)))

    print('Ejecutando el analisis sobre %s...' % os.path.basename(ruta))
    if completo:
        app.analisis_completo()
    else:
        app.procesar_una()

    # Se bombea el bucle de eventos hasta que el hilo trabajador termine,
    # igual que haria la ventana en uso normal.
    while app.trabajando:
        raiz.update()
        raiz.after(50)
    raiz.update()
    print('Analisis terminado. Capturando...')

    generadas = []
    for indice in range(len(app.pestanas.tabs())):
        app.pestanas.select(indice)
        raiz.update_idletasks()
        raiz.update()
        nombre = app.pestanas.tab(indice, 'text').strip().lower()
        nombre = nombre.replace(' ', '_')
        generadas.append(capturar(raiz, os.path.join(salida,
                                                     'ui_%s.png' % nombre)))

    # Las figuras tambien se guardan sueltas, en mejor resolucion que la
    # captura de pantalla, por si se quieren usar directamente en el informe.
    app.figura.savefig(os.path.join(salida, 'graficas.png'), dpi=120,
                       facecolor='white')
    app.figura_rec.savefig(os.path.join(salida, 'recursos.png'), dpi=120,
                           facecolor='white')
    benchmark_gpu.exportar_csv(app.filas,
                               os.path.join(salida, 'benchmark.csv'))

    app.cerrando = True
    raiz.destroy()
    print('Listo. %d capturas mas las figuras y el CSV en %s'
          % (len(generadas), salida))
    return generadas


def main():
    analizador = argparse.ArgumentParser(
        description='Captura la aplicacion en funcionamiento.')
    analizador.add_argument('archivo', help='archivo FASTA a procesar')
    analizador.add_argument('--salida', default=SALIDA_POR_DEFECTO)
    analizador.add_argument('--sucio', action='store_true',
                            help='derivar del archivo una copia con errores '
                                 'conocidos, para que las capturas muestren '
                                 'el detector de invalidos encontrando algo')
    analizador.add_argument('--simple', action='store_true',
                            help='una sola pasada en vez del analisis completo')
    args = analizador.parse_args()

    if not os.path.exists(args.archivo):
        print('No existe el archivo: %s' % args.archivo)
        return 1

    ruta = args.archivo
    if args.sucio:
        ruta = preparar_archivo_sucio(args.archivo)

    generar(ruta, args.salida, completo=not args.simple)
    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    _preparar_dpi()
    sys.exit(main())
