# -*- coding: utf-8 -*-
"""
app_p13.py

Lanzador de la aplicacion grafica del P1.3.

POR QUE ESTE ARCHIVO ESTA CASI VACIO
En Windows multiprocessing usa el metodo 'spawn': cada proceso hijo arranca
un interprete nuevo y vuelve a importar el modulo principal para poder
reconstruir el trabajo. Eso significa que todo lo que este escrito en el
nivel superior de este archivo se ejecuta una vez por cada proceso hijo.

Si la ventana estuviera aqui, los procesos de CPU del motor hibrido
importarian Tkinter, matplotlib, numba y todo el stack de CUDA sin llegar a
usar nada de eso. En el P1.2 se midio el efecto de un solo import de mas:
contar_paralelo sobre un archivo de 2 MB pasaba de 1.34 s a 5.51 s solo por
tener motor_gpu importado arriba. En el P1.3 el dano seria peor, porque ese
retraso se le cargaria a la mitad de CPU del motor justo mientras compite
con la GPU por el reparto, y el resultado del balanceo no diria nada sobre
el rendimiento real de cada plataforma.

Por eso este archivo solo contiene el arranque, y la interfaz de verdad vive
en interfaz.py, que se importa dentro de main(). Los procesos hijos importan
este modulo, no encuentran nada pesado y arrancan en milisegundos.

Uso:
    python app_p13.py
"""

import multiprocessing
import sys


def main():
    # Import tardio a proposito: ver la explicacion de la cabecera.
    from interfaz import ejecutar
    return ejecutar()


if __name__ == '__main__':
    # Obligatorio en Windows antes de crear cualquier proceso hijo.
    multiprocessing.freeze_support()
    sys.exit(main())
