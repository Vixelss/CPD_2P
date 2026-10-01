# -*- coding: utf-8 -*-
"""
capturas.py

Abre la aplicacion, ejecuta un procesamiento de verdad y captura sus
pestanas como imagenes para el informe.

REGLA QUE NO SE DEBE ROMPER
Este programa no inventa valores nunca. Todo lo que aparece en las imagenes
sale de una ejecucion real sobre el archivo que se indique. Si una magnitud
no se pudo medir, la captura muestra el panel como quedo, sin rellenarlo.
Una imagen destinada a un informe academico tiene que ensenar lo que el
programa midio.

La ventana se pone en primer plano desde el propio Python antes de cada
captura, para que ninguna otra ventana se cuele en el recorte.

Uso:
    python capturas.py <archivo.fna> [--salida carpeta] [--completo]
"""

import argparse
import ctypes
import json
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
    print('  %-26s %dx%d' % (os.path.basename(destino),
                             imagen.width, imagen.height))
    return destino


def _guardar_datos(app, destino):
    """Guarda en JSON las mediciones que no caben en el CSV de filas.

    Son las que el generador del informe necesita y que son de la corrida
    entera y no de una configuracion concreta: el uso de cada nucleo logico,
    el reparto entre los multiprocesadores de la tarjeta, los recursos que
    consume el kernel y el conteo final. Guardarlas aqui es lo que permite
    que el documento cite exactamente las mismas cifras que se ven en las
    capturas, en lugar de llevarlas escritas a mano.
    """
    resultado = app.ultimo_resultado or {}
    datos = {
        'archivo': app.ruta.get(),
        'recursos': app.ultimos_recursos or {},
        'censo_sm': app.ultimo_censo or [],
        'recursos_kernel': app.recursos_kernel,
        'conteo': {clave: resultado.get(clave)
                   for clave in ('A', 'C', 'G', 'T', 'N', 'bases',
                                 'ambiguos', 'invalidos')},
    }
    with open(destino, 'w', encoding='utf-8') as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)
    print('  %-26s' % os.path.basename(destino))
    return destino


def generar(ruta, salida=SALIDA_POR_DEFECTO, completo=False, procesos=4,
            trozo_mb=16):
    """Corre la aplicacion sobre el archivo dado y captura sus pestanas."""
    import benchmark
    import evidencias
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
        text=benchmark.formato_tamano(os.path.getsize(ruta)))
    app.modo.set('hibrido')
    app.procesos.set(procesos)
    app.trozo_mb.set(trozo_mb)

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

        # Las graficas se dibujaron mientras su pestana estaba oculta, con el
        # lienzo aun sin su tamano definitivo. Al mostrarla, Tk ensena la
        # imagen que habia, que es mas grande que el area visible y sale
        # recortada por los bordes. Redibujar aqui la rehace ya con el tamano
        # real del lienzo.
        for lienzo in (app.lienzo, app.lienzo_nuc):
            try:
                lienzo.draw()
            except Exception:
                pass
        raiz.update_idletasks()
        raiz.update()

        nombre = app.pestanas.tab(indice, 'text').strip().lower()
        nombre = nombre.replace(' ', '_')
        generadas.append(capturar(raiz, os.path.join(salida,
                                                     'ui_%s.png' % nombre)))

    # Las figuras tambien se guardan sueltas, en mejor resolucion que la
    # captura de pantalla, por si se quieren usar directamente en el informe.
    app.figura.savefig(os.path.join(salida, 'ejecucion.png'), dpi=120,
                       facecolor='white')
    app.figura_nuc.savefig(os.path.join(salida, 'nucleos.png'), dpi=120,
                           facecolor='white')
    benchmark.exportar_csv(app.filas, os.path.join(salida, 'benchmark.csv'))
    benchmark.exportar_registro(app.filas,
                                os.path.join(salida,
                                             'registro_reparto.csv'))
    evidencias.guardar(app.matriz_evidencias(),
                       os.path.join(salida, 'evidencias.txt'))
    _guardar_datos(app, os.path.join(salida, 'datos_informe.json'))

    app.cerrando = True
    if app.monitor:
        app.monitor.detener()
        app.monitor = None
    raiz.destroy()
    print('Listo. %d capturas mas las figuras y el CSV en %s'
          % (len(generadas), salida))
    return generadas


def main():
    analizador = argparse.ArgumentParser(
        description='Captura la aplicacion en funcionamiento.')
    analizador.add_argument('archivo', help='archivo FASTA a procesar')
    analizador.add_argument('--salida', default=SALIDA_POR_DEFECTO)
    analizador.add_argument('--completo', action='store_true',
                            help='analisis completo en vez de una pasada')
    analizador.add_argument('--procesos', type=int, default=4)
    analizador.add_argument('--trozo', type=int, default=16)
    args = analizador.parse_args()

    if not os.path.exists(args.archivo):
        print('No existe el archivo: %s' % args.archivo)
        return 1

    generar(args.archivo, args.salida, args.completo, args.procesos,
            args.trozo)
    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    _preparar_dpi()
    sys.exit(main())
