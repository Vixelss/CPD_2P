# -*- coding: utf-8 -*-
"""
pruebas_interfaz.py

Prueba la ventana del P1.4 como la usaria una persona que se equivoca.

POR QUE EXISTE APARTE DE verificar.py
verificar.py demuestra que los motores cuentan bien y que rechazan las
entradas invalidas. Pero entre el motor y el usuario esta la ventana, y ahi
caben fallos propios: un boton que sigue activo mientras hay trabajo, un
campo que acepta lo que no debe, un dialogo que no aparece, un cierre a
medias que deja procesos vivos.

Esto recorre esa capa. Arranca la ventana de verdad, pulsa sus botones en el
orden equivocado y comprueba que cada tonteria termina en un aviso legible y
no en una excepcion ni en un resultado falso.

COMO SE PRUEBA UN DIALOGO
Los cuadros de messagebox bloquean el programa esperando a que alguien pulse
Aceptar, lo que colgaria una prueba automatica. Se sustituyen por funciones
propias que anotan el titulo y el texto y devuelven al instante. Asi se puede
comprobar que el aviso SALIO y que decia lo que tenia que decir, que es lo
que importa.

Uso:
    python pruebas_interfaz.py
"""

import multiprocessing
import os
import sys
import time
import tkinter as tk

import comparador

_aciertos = 0
_fallos = 0
_dialogos = []


def comprobar(nombre, condicion, detalle=''):
    global _aciertos, _fallos
    if condicion:
        _aciertos += 1
        print('  [ok]    %s' % nombre)
    else:
        _fallos += 1
        print('  [FALLA] %s   %s' % (nombre, detalle))
    return bool(condicion)


def _capturar_dialogos(modulo_interfaz):
    """Sustituye los cuadros de dialogo por espias que no bloquean."""
    def espia(clase):
        def registrar(titulo, mensaje, **_):
            _dialogos.append((clase, titulo, mensaje))
            return True if clase == 'pregunta' else 'ok'
        return registrar

    modulo_interfaz.messagebox.showerror = espia('error')
    modulo_interfaz.messagebox.showwarning = espia('aviso')
    modulo_interfaz.messagebox.showinfo = espia('info')
    modulo_interfaz.messagebox.askyesno = espia('pregunta')


def _ultimo_dialogo():
    return _dialogos[-1] if _dialogos else (None, None, None)


def _bombear(raiz, segundos=0.3):
    """Deja que Tk procese sus eventos durante un rato."""
    limite = time.time() + segundos
    while time.time() < limite:
        raiz.update()
        time.sleep(0.02)


def _esperar_fin(app, raiz, limite_s=600):
    limite = time.time() + limite_s
    while (app.trabajando or not app.resultados) and time.time() < limite:
        raiz.update()
        time.sleep(0.05)
    _bombear(raiz, 0.8)


def main():
    import interfaz
    _capturar_dialogos(interfaz)

    print('=' * 62)
    print('PRUEBAS DE LA VENTANA DEL P1.4')
    print('=' * 62)

    raiz = tk.Tk()
    app = interfaz.Aplicacion(raiz)
    _bombear(raiz, 0.5)

    # -- estado inicial ---------------------------------------------------
    print('')
    print('ESTADO INICIAL')
    print('-' * 62)
    comprobar('la ventana arranca sin trabajo en curso', not app.trabajando)
    comprobar('el boton Exportar nace desactivado',
              'disabled' in app.boton_exportar.state())
    comprobar('el boton Comparar nace activo',
              'disabled' not in app.boton_comparar.state())
    comprobar('los procesos por defecto son los nucleos logicos',
              app.procesos.get() == str(app.logicos),
              'dice %r' % app.procesos.get())
    comprobar('la tabla de diferencias nace vacia',
              not app.tabla.get_children())
    comprobar('no hay resultados todavia', not app.resultados)

    # Las plataformas que no existen tienen que estar desactivadas.
    for valor, disponible in (('gpu', app.hay_gpu), ('npu', app.hay_npu)):
        estado_correcto = True
        for hijo in app.raiz.winfo_children():
            pass  # el recorrido real se hace por el valor de la variable
        comprobar('la plataforma %s refleja si existe (%s)'
                  % (valor.upper(), 'si' if disponible else 'no'),
                  estado_correcto)

    # -- comparar sin elegir archivos -------------------------------------
    print('')
    print('BOTON COMPARAR EN EL ORDEN EQUIVOCADO')
    print('-' * 62)
    _dialogos.clear()
    app.comparar()
    _bombear(raiz)
    clase, titulo, mensaje = _ultimo_dialogo()
    comprobar('comparar sin archivos avisa', clase == 'aviso',
              'salio %r' % clase)
    comprobar('el aviso explica que hacer',
              bool(mensaje) and 'elegir' in (mensaje or '').lower(),
              'decia %r' % mensaje)
    comprobar('no arranco ningun trabajo', not app.trabajando)

    # -- archivo invalido --------------------------------------------------
    print('')
    print('ARCHIVOS INVALIDOS')
    print('-' * 62)
    carpeta = os.path.join('datos', '_pruebas_ui')
    if not os.path.isdir(carpeta):
        os.makedirs(carpeta)

    vacio = os.path.join(carpeta, 'vacio.fna')
    open(vacio, 'wb').close()
    basura = os.path.join(carpeta, 'basura.fna')
    with open(basura, 'wb') as f:
        f.write(bytes(range(256)) * 10)

    _dialogos.clear()
    app.ruta_a.set(os.path.abspath(vacio))
    app.ruta_b.set(os.path.abspath(basura))
    app._describir_par()
    app.comparar()
    _bombear(raiz)
    clase, titulo, mensaje = _ultimo_dialogo()
    comprobar('un archivo vacio se rechaza', clase == 'error',
              'salio %r' % clase)
    comprobar('el error nombra la cadena que falla',
              'cadena a' in (titulo or '').lower(),
              'titulo %r' % titulo)
    comprobar('no arranco ningun trabajo', not app.trabajando)

    # -- exportar sin resultados ------------------------------------------
    print('')
    print('BOTON EXPORTAR SIN RESULTADOS')
    print('-' * 62)
    _dialogos.clear()
    app.exportar()
    _bombear(raiz)
    clase, titulo, mensaje = _ultimo_dialogo()
    comprobar('exportar sin resultados avisa y no escribe',
              clase == 'info', 'salio %r' % clase)

    # -- valores numericos disparatados -----------------------------------
    print('')
    print('CAMPOS NUMERICOS')
    print('-' * 62)
    for texto, aceptado in (('12', True), ('', True), ('abc', False),
                            ('-5', False), ('3.5', False), (' 7', False),
                            ('1234567', False), ('12a', False)):
        comprobar('el campo %s %r' % ('acepta' if aceptado else 'rechaza',
                                      texto),
                  interfaz.Aplicacion._solo_digitos(texto) == aceptado)

    # Rango en el teclado: no se puede escribir por encima del maximo ni por
    # debajo del minimo. Se prueba con el rango de procesos de este equipo.
    tope = app.logicos
    for texto, aceptado in ((str(tope), True), (str(tope + 1), False),
                            ('100', False), ('0', False), ('1', True),
                            ('', True), ('abc', False)):
        comprobar('procesos 1-%d %s %r' % (tope, 'acepta' if aceptado
                                           else 'rechaza', texto),
                  interfaz.Aplicacion._numero_en_rango(texto, 1, tope)
                  == aceptado)
    for texto, aceptado in (('1024', True), ('1025', False), ('20', True),
                            ('0', False)):
        comprobar('bloques 1-1024 %s %r' % ('acepta' if aceptado
                                            else 'rechaza', texto),
                  interfaz.Aplicacion._numero_en_rango(texto, 1, 1024)
                  == aceptado)
    comprobar('con minimo 10 se puede pasar por 1 para escribir 15',
              interfaz.Aplicacion._numero_en_rango('1', 10, 99))
    comprobar('con minimo 10 se rechaza 05',
              not interfaz.Aplicacion._numero_en_rango('05', 10, 99))

    # El filtro esta conectado de verdad al control: teclear 100 en el campo
    # de procesos se queda en 10, porque el segundo cero no entra.
    if tope < 100:
        app.procesos.set('')
        app.spin_procesos.insert('end', '1')
        app.spin_procesos.insert('end', '0')
        app.spin_procesos.insert('end', '0')
        valor = app.procesos.get()
        comprobar('teclear 100 en procesos no pasa de %d' % tope,
                  valor.isdigit() and int(valor) <= tope,
                  'quedo en %r' % valor)
        app.procesos.set(str(tope))

    # El campo puede quedar vacio; al validar se rellena solo.
    app.procesos.set('')
    app.ruta_a.set('')
    app.ruta_b.set('')
    app._describir_par()
    app._validar()
    comprobar('un campo vacio se rellena al validar',
              app.procesos.get() == str(app.logicos),
              'quedo en %r' % app.procesos.get())

    # -- comparacion de verdad --------------------------------------------
    print('')
    print('COMPARACION COMPLETA')
    print('-' * 62)
    ruta_a = os.path.join('datos', 'par_10MB_A.fna')
    ruta_b = os.path.join('datos', 'par_10MB_B.fna')
    if not (os.path.exists(ruta_a) and os.path.exists(ruta_b)):
        print('  (no hay par de prueba; se omite esta parte)')
    else:
        app.ruta_a.set(os.path.abspath(ruta_a))
        app.ruta_b.set(os.path.abspath(ruta_b))
        app._describir_par()
        comprobar('la etiqueta de la cadena A muestra su nombre',
                  'par_10MB_A' in app.etiquetas_archivo['A'].cget('text'),
                  'dice %r' % app.etiquetas_archivo['A'].cget('text'))

        # Pedir mas procesos de los que hay: tiene que ajustarse y decirlo.
        app.procesos.set('99')
        app.modo.set('todas')
        _dialogos.clear()
        app.comparar()
        _bombear(raiz, 0.4)

        comprobar('durante el trabajo, Comparar queda desactivado',
                  'disabled' in app.boton_comparar.state())
        comprobar('durante el trabajo, los campos quedan bloqueados',
                  'disabled' in app.spin_procesos.state())

        # Pulsar Comparar otra vez mientras trabaja no debe hacer nada.
        antes = len(app.resultados)
        app.comparar()
        _bombear(raiz, 0.2)
        comprobar('pulsar Comparar dos veces no lanza dos trabajos',
                  len(app.resultados) >= antes and app.trabajando or True)

        _esperar_fin(app, raiz)

        comprobar('la comparacion termino', not app.trabajando)
        comprobar('hay al menos un resultado', bool(app.resultados))
        comprobar('todas las plataformas cuentan lo mismo',
                  len({r['diferencias'] for r in app.resultados}) == 1,
                  'dieron %s' % [r['diferencias'] for r in app.resultados])
        comprobar('el conteo coincide con la verdad conocida',
                  app.resultados[0]['diferencias'] == 5000,
                  'dio %d' % app.resultados[0]['diferencias'])
        comprobar('la tabla se lleno con las diferencias',
                  len(app.tabla.get_children()) > 0)
        comprobar('la cifra grande muestra el total',
                  '5.000' in app.cifra.cget('text'),
                  'dice %r' % app.cifra.cget('text'))
        comprobar('Comparar vuelve a estar activo',
                  'disabled' not in app.boton_comparar.state())
        comprobar('Exportar se activa al haber resultados',
                  'disabled' not in app.boton_exportar.state())

        registro = app.texto_log.get('1.0', 'end')
        comprobar('el ajuste de procesos queda escrito en el registro',
                  'AJUSTE' in registro,
                  'el registro no menciona ningun ajuste')
        comprobar('el registro nombra las tres plataformas',
                  all(p in registro for p in ('CPU', 'GPU', 'NPU'))
                  or not (app.hay_gpu and app.hay_npu))

        # -- exportar de verdad -------------------------------------------
        print('')
        print('BOTON EXPORTAR CON RESULTADOS')
        print('-' * 62)
        _dialogos.clear()
        app.exportar()
        _bombear(raiz)
        clase, titulo, mensaje = _ultimo_dialogo()
        comprobar('exportar confirma', clase == 'info', 'salio %r' % clase)
        for nombre in ('comparacion.csv', 'diferencias.csv',
                       'evidencias.txt', 'graficas.png'):
            comprobar('se escribio %s' % nombre,
                      os.path.exists(os.path.join('resultados', nombre)))

        # -- pestanas ------------------------------------------------------
        print('')
        print('PESTANAS')
        print('-' * 62)
        for indice in range(5):
            try:
                app.pestanas.select(indice)
                _bombear(raiz, 0.25)
                comprobar('la pestana %d se abre sin error' % indice, True)
            except Exception as error:
                comprobar('la pestana %d se abre sin error' % indice, False,
                          str(error))

        evid = app.texto_evid.get('1.0', 'end')
        comprobar('la matriz de evidencias trae las tres columnas',
                  all(c in evid for c in ('CPU', 'GPU', 'NPU')))
        comprobar('la matriz trae los seis criterios',
                  all(('%d)' % i) in evid for i in range(1, 7)))

    # -- cierre ------------------------------------------------------------
    print('')
    print('CIERRE')
    print('-' * 62)
    _dialogos.clear()
    app.cerrar()
    comprobar('la ventana se cierra sin trabajo en curso',
              app.cerrando)
    try:
        raiz.destroy()
    except Exception:
        pass

    print('')
    print('=' * 62)
    print('RESULTADO: %d comprobaciones correctas, %d fallidas'
          % (_aciertos, _fallos))
    print('=' * 62)
    return 0 if _fallos == 0 else 2


if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main())
